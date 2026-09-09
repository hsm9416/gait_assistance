"""Unified sensor frame and the three acquisition sources (spec 1, 29).

Sources
-------
``CsvSensorSource``    replay of a logged CSV file (offline simulation)
``MockSensorSource``   fully synthetic hemiparetic gait (simulation mode)
``PadSensorSource``    live telemetry from the PAD device via ``connect.py``
"""

from __future__ import annotations

import math
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import CHANNELS, DEFAULT_CSV_ALIASES, SensorConfig
from ..utils.filters import EMAFilter
from .encoder import EncoderSample, check_encoder_sample
from .imu import ImuSample, MockImu, check_imu_sample


@dataclass(frozen=True)
class SensorSample:
    """One fully populated sensor frame (spec 1)."""

    timestamp: float
    belt_length: float
    belt_velocity: float
    motor_position: float
    motor_velocity: float
    motor_current: float
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    seq: int = 0

    def channel(self, name: str) -> float:
        """Return the value of the canonical channel ``name``.

        Raises:
            KeyError: if ``name`` is not a known channel.
        """
        if name not in CHANNELS:
            raise KeyError(f"unknown sensor channel: {name!r}")
        return float(getattr(self, name))

    def to_vector(self, channels: Sequence[str]) -> np.ndarray:
        """Return the selected channels as a 1-D float array."""
        return np.array([self.channel(c) for c in channels], dtype=float)

    def is_finite(self) -> bool:
        """Return True when no channel holds NaN or Inf."""
        return all(math.isfinite(self.channel(c)) for c in CHANNELS)

    @property
    def imu(self) -> ImuSample:
        """View of the inertial channels."""
        return ImuSample(
            timestamp=self.timestamp,
            accel_x=self.accel_x,
            accel_y=self.accel_y,
            accel_z=self.accel_z,
            gyro_x=self.gyro_x,
            gyro_y=self.gyro_y,
            gyro_z=self.gyro_z,
        )

    @property
    def encoder(self) -> EncoderSample:
        """View of the belt/motor channels."""
        return EncoderSample(
            timestamp=self.timestamp,
            belt_length=self.belt_length,
            belt_velocity=self.belt_velocity,
            motor_position=self.motor_position,
            motor_velocity=self.motor_velocity,
            motor_current=self.motor_current,
        )

    @classmethod
    def from_mapping(cls, values: Dict[str, float], seq: int = 0) -> "SensorSample":
        """Build a sample from a mapping, defaulting missing channels to 0.0."""
        kwargs = {c: float(values.get(c, 0.0)) for c in CHANNELS}
        return cls(seq=seq, **kwargs)


@dataclass(frozen=True)
class SensorHealth:
    """Result of the per-frame sensor validity checks."""

    ok: bool
    imu_ok: bool
    encoder_ok: bool
    reason: str = ""


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #


class SensorSource(ABC):
    """Abstract producer of :class:`SensorSample` frames."""

    @abstractmethod
    def read(self) -> Optional[SensorSample]:
        """Return the next sample, or ``None`` when no new data is available.

        A source that has permanently run out of data must raise
        :class:`StopIteration`.
        """

    def open(self) -> None:
        """Prepare the source (no-op by default)."""

    def close(self) -> None:
        """Release the source (no-op by default)."""

    def __enter__(self) -> "SensorSource":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def resolve_csv_columns(
    columns: Iterable[str], overrides: Optional[Dict[str, str]] = None
) -> Dict[str, Optional[str]]:
    """Map canonical channel names onto the columns of a CSV file.

    Args:
        columns: column names present in the file.
        overrides: explicit ``canonical -> column`` pairs that win over the
            alias table.

    Returns:
        Mapping from every canonical channel to a column name or ``None``.
    """
    available = {c.strip(): c for c in columns}
    lowered = {c.strip().lower(): c for c in columns}
    overrides = dict(overrides or {})
    mapping: Dict[str, Optional[str]] = {}
    for channel in CHANNELS:
        if channel in overrides:
            mapping[channel] = overrides[channel]
            continue
        found: Optional[str] = None
        for alias in DEFAULT_CSV_ALIASES.get(channel, (channel,)):
            if alias in available:
                found = available[alias]
                break
            if alias.lower() in lowered:
                found = lowered[alias.lower()]
                break
        mapping[channel] = found
    return mapping


class CsvSensorSource(SensorSource):
    """Replay a logged CSV file frame by frame.

    Args:
        path: CSV file to replay.
        config: sensor configuration (column overrides, realtime flag).
        max_samples: optional cap on the number of replayed frames.
    """

    def __init__(
        self,
        path: str,
        config: Optional[SensorConfig] = None,
        *,
        max_samples: Optional[int] = None,
    ) -> None:
        self.path = Path(path)
        self.config = config or SensorConfig()
        self.max_samples = max_samples
        self._samples: List[SensorSample] = []
        self._index = 0
        self._t0_wall = 0.0
        self.missing_channels: Tuple[str, ...] = ()

    def open(self) -> None:
        """Load and convert the whole file into :class:`SensorSample` objects."""
        frame = pd.read_csv(self.path)
        mapping = resolve_csv_columns(frame.columns, self.config.csv_column_map)
        self.missing_channels = tuple(c for c, col in mapping.items() if col is None)
        n = len(frame) if self.max_samples is None else min(len(frame), self.max_samples)
        arrays: Dict[str, np.ndarray] = {}
        for channel, column in mapping.items():
            if column is None:
                arrays[channel] = np.zeros(n, dtype=float)
            else:
                arrays[channel] = (
                    pd.to_numeric(frame[column], errors="coerce")
                    .to_numpy(dtype=float)[:n]
                )
        if mapping["timestamp"] is None:
            dt = 1.0 / max(self.config.sample_rate_hz, 1e-6)
            arrays["timestamp"] = np.arange(n, dtype=float) * dt
        # derive belt velocity when the file does not carry it
        if mapping["belt_velocity"] is None and mapping["belt_length"] is not None:
            arrays["belt_velocity"] = _derivative(
                arrays["belt_length"], arrays["timestamp"]
            )
        self._samples = [
            SensorSample(seq=i, **{c: float(arrays[c][i]) for c in CHANNELS})
            for i in range(n)
        ]
        self._index = 0
        self._t0_wall = time.monotonic()

    def read(self) -> Optional[SensorSample]:
        """Return the next logged frame.

        Raises:
            StopIteration: when the file is exhausted.
        """
        if self._index >= len(self._samples):
            raise StopIteration("CSV replay finished")
        sample = self._samples[self._index]
        if self.config.csv_realtime and self._index > 0:
            target = sample.timestamp - self._samples[0].timestamp
            slack = target - (time.monotonic() - self._t0_wall)
            if slack > 0.0:
                time.sleep(slack)
        self._index += 1
        return sample

    @property
    def samples(self) -> List[SensorSample]:
        """All loaded samples (available after :meth:`open`)."""
        return self._samples

    def __len__(self) -> int:
        return len(self._samples)


class MockSensorSource(SensorSource):
    """Synthetic hemiparetic gait generator (spec 29).

    Args:
        config: sensor configuration (only ``sample_rate_hz`` is used).
        stride_period_s: nominal stride duration.
        swing_ratio: fraction of the stride spent in swing.
        belt_amplitude_mm: peak-to-peak belt excursion.
        motor: optional :class:`~.encoder.MockMotor` whose offset is added to
            the belt length, closing the simulation loop.
        drift_after_s: after this time the gait pattern changes, which is
            useful to exercise OOD detection; ``0`` disables the drift.
        drift_transition_s: duration of the transition into the drifted gait;
            the change is blended so the belt signal stays continuous.
        belt_offset_mm: mean belt length, matching the resting length of the
            real device (its logs sit around -130 mm).
        seed: RNG seed.
    """

    def __init__(
        self,
        config: Optional[SensorConfig] = None,
        *,
        stride_period_s: float = 1.2,
        swing_ratio: float = 0.32,
        belt_amplitude_mm: float = 60.0,
        motor: Optional[object] = None,
        drift_after_s: float = 0.0,
        drift_transition_s: float = 3.0,
        belt_offset_mm: float = -130.0,
        seed: int = 0,
    ) -> None:
        self.config = config or SensorConfig()
        self.stride_period_s = stride_period_s
        self.swing_ratio = swing_ratio
        self.belt_amplitude_mm = belt_amplitude_mm
        self.belt_offset_mm = belt_offset_mm
        self.motor = motor
        self.drift_after_s = drift_after_s
        self.drift_transition_s = max(drift_transition_s, 1e-3)
        self._imu = MockImu(stride_period_s, swing_ratio, seed=seed)
        self._rng = np.random.default_rng(seed)
        self._t = 0.0
        self._seq = 0
        self._phase = 0.0
        self._prev_belt = 0.0

    @property
    def dt(self) -> float:
        """Sampling period in seconds."""
        return 1.0 / max(self.config.sample_rate_hz, 1e-6)

    def drift_fraction(self, t: float) -> float:
        """Smooth 0 -> 1 blend into the drifted gait pattern at time ``t``."""
        if self.drift_after_s <= 0.0:
            return 0.0
        u = (t - self.drift_after_s) / self.drift_transition_s
        if u <= 0.0:
            return 0.0
        if u >= 1.0:
            return 1.0
        return float(u * u * (3.0 - 2.0 * u))  # smoothstep

    def read(self) -> Optional[SensorSample]:
        """Generate the next synthetic frame."""
        dt = self.dt
        t = self._t
        blend = self.drift_fraction(t)
        period = self.stride_period_s * (1.0 + 0.45 * blend)
        amplitude = self.belt_amplitude_mm * (1.0 - 0.55 * blend)
        swing_ratio = self.swing_ratio * (1.0 - 0.40 * blend)
        # integrate the phase so a changing period never makes the signal jump
        self._phase = (self._phase + dt / period) % 1.0
        x = self._phase
        # belt extends through swing and retracts sharply at heel strike
        belt = (
            self.belt_offset_mm
            - amplitude * math.cos(2.0 * math.pi * x)
            + self._rng.normal(0.0, 0.4)
        )
        if self.motor is not None and hasattr(self.motor, "step"):
            belt += float(self.motor.step(dt))
        belt_vel = (belt - self._prev_belt) / dt if self._seq else 0.0
        self._prev_belt = belt
        self._imu.swing_ratio = swing_ratio
        imu = self._imu.sample(t, phase_fraction=x)
        sample = SensorSample(
            timestamp=t,
            belt_length=belt,
            belt_velocity=belt_vel,
            motor_position=belt * 10.0,
            motor_velocity=belt_vel * 10.0,
            motor_current=float(getattr(self.motor, "last_command_a", 0.0)),
            accel_x=imu.accel_x,
            accel_y=imu.accel_y,
            accel_z=imu.accel_z,
            gyro_x=imu.gyro_x,
            gyro_y=imu.gyro_y,
            gyro_z=imu.gyro_z,
            seq=self._seq,
        )
        self._t += dt
        self._seq += 1
        return sample


class PadSensorSource(SensorSource):
    """Live telemetry from the PAD device (``connect.PadController``).

    The import of the hardware library is deferred so that the package stays
    importable on machines without ``pyserial`` or the device attached.

    Args:
        config: sensor configuration (``serial_port``, ``serial_baudrate``).
        controller: an already constructed controller (used by tests).
    """

    def __init__(
        self, config: SensorConfig, controller: Optional[object] = None
    ) -> None:
        self.config = config
        self._controller = controller
        self._seq = 0

    @property
    def controller(self) -> object:
        """The underlying ``PadController``.

        Raises:
            RuntimeError: if the source has not been opened.
        """
        if self._controller is None:
            raise RuntimeError("PadSensorSource is not open")
        return self._controller

    def open(self) -> None:
        """Open the serial link and start the telemetry session."""
        if self._controller is None:
            PadController = _import_pad_controller()
            port = self.config.serial_port
            if not port:
                port = _auto_serial_port()
            self._controller = PadController(
                port=port, baudrate=self.config.serial_baudrate
            )
            self._controller.open()  # type: ignore[attr-defined]
        self._controller.start_session()  # type: ignore[attr-defined]

    def close(self) -> None:
        """Stop the session and close the serial link."""
        if self._controller is None:
            return
        try:
            self._controller.stop_session()  # type: ignore[attr-defined]
        finally:
            self._controller.close()  # type: ignore[attr-defined]

    def read(self) -> Optional[SensorSample]:
        """Poll the link and convert a telemetry frame, if one arrived."""
        data = self.controller.poll()  # type: ignore[attr-defined]
        if data is None:
            return None
        self._seq += 1
        return SensorSample(
            timestamp=float(data.time_s),
            belt_length=float(data.belt_length),
            belt_velocity=float(data.belt_velocity),
            motor_position=float(data.motor_pos),
            motor_velocity=float(data.motor_vel),
            motor_current=float(data.motor_iq_meas),
            accel_x=float(data.accel_x),
            accel_y=float(data.accel_y),
            accel_z=float(data.accel_z),
            gyro_x=float(data.gyro_x),
            gyro_y=float(data.gyro_y),
            gyro_z=float(data.gyro_z),
            seq=int(data.seq),
        )


# --------------------------------------------------------------------------- #
# Manager
# --------------------------------------------------------------------------- #


class SensorManager:
    """Wraps a source with optional filtering and per-frame health checks.

    Args:
        source: the acquisition backend.
        config: sensor configuration.
    """

    def __init__(self, source: SensorSource, config: Optional[SensorConfig] = None) -> None:
        self.source = source
        self.config = config or SensorConfig()
        self._velocity_filter = (
            EMAFilter.from_cutoff(self.config.velocity_filter_hz, self.config.sample_rate_hz)
            if self.config.velocity_filter_hz > 0.0
            else None
        )
        self.latest: Optional[SensorSample] = None
        self.latest_health: SensorHealth = SensorHealth(True, True, True)
        self.finished = False
        self.frame_count = 0

    def open(self) -> None:
        """Open the underlying source."""
        self.source.open()

    def close(self) -> None:
        """Close the underlying source."""
        self.source.close()

    def poll(self) -> Optional[Tuple[SensorSample, SensorHealth]]:
        """Read one frame.

        Returns:
            ``(sample, health)`` or ``None`` when no new frame is available or
            the source is exhausted (in which case :attr:`finished` is set).
        """
        try:
            sample = self.source.read()
        except StopIteration:
            self.finished = True
            return None
        if sample is None:
            return None
        if self._velocity_filter is not None:
            sample = replace(
                sample, belt_velocity=self._velocity_filter.update(sample.belt_velocity)
            )
        health = self.check(sample)
        self.latest = sample
        self.latest_health = health
        self.frame_count += 1
        return sample, health

    def check(self, sample: SensorSample) -> SensorHealth:
        """Run the IMU and encoder validity checks on ``sample``."""
        if not sample.is_finite():
            return SensorHealth(False, False, False, "NaN/Inf in sensor frame")
        imu_ok, imu_reason = check_imu_sample(sample.imu)
        enc_ok, enc_reason = check_encoder_sample(sample.encoder)
        reason = imu_reason or enc_reason
        return SensorHealth(imu_ok and enc_ok, imu_ok, enc_ok, reason)

    def stream(self) -> Iterator[Tuple[SensorSample, SensorHealth]]:
        """Iterate over all frames until the source is exhausted."""
        while not self.finished:
            item = self.poll()
            if item is not None:
                yield item


def _derivative(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Numerical time derivative with a safe fallback for degenerate steps."""
    if values.size < 2:
        return np.zeros_like(values)
    dt = np.diff(times, prepend=times[0] - (times[1] - times[0] or 1e-3))
    dt[dt <= 0.0] = np.nan
    out = np.diff(values, prepend=values[0]) / dt
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _import_pad_controller() -> type:
    """Import ``PadController`` from the repository root next to the package."""
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from connect import PadController  # type: ignore[import-not-found]

    return PadController


def _auto_serial_port() -> str:
    """Return the single available serial port.

    Raises:
        RuntimeError: when zero or several ports are present.
    """
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from pad_external_control_lib import list_serial_ports  # type: ignore

    ports = list_serial_ports()
    if not ports:
        raise RuntimeError("no serial port available; pass sensor.serial_port")
    if len(ports) > 1:
        raise RuntimeError(f"several serial ports available: {ports}")
    return ports[0]


def create_sensor_source(
    mode: str,
    config: SensorConfig,
    *,
    csv_path: Optional[str] = None,
    motor: Optional[object] = None,
    **kwargs: object,
) -> SensorSource:
    """Factory selecting a source from the run mode.

    Args:
        mode: ``"offline"``, ``"simulation"`` or ``"hardware"``.
        config: sensor configuration.
        csv_path: file to replay in offline mode.
        motor: mock motor to close the loop in simulation mode.
        **kwargs: forwarded to :class:`MockSensorSource`.

    Returns:
        The configured source.
    """
    if mode == "offline":
        if not csv_path:
            raise ValueError("offline mode requires csv_path")
        return CsvSensorSource(csv_path, config)
    if mode == "simulation":
        return MockSensorSource(config, motor=motor, **kwargs)  # type: ignore[arg-type]
    if mode == "hardware":
        return PadSensorSource(config)
    raise ValueError(f"unknown run mode: {mode!r}")


__all__ = [
    "CsvSensorSource",
    "MockSensorSource",
    "PadSensorSource",
    "SensorHealth",
    "SensorManager",
    "SensorSample",
    "SensorSource",
    "create_sensor_source",
    "resolve_csv_columns",
]
