"""Safety monitor (spec 25).

Any of the listed conditions forces the assistance to zero and, when the fault
is persistent, the system into ``SAFE_STOP``.  Out-of-distribution strides are
not faults: they only cap the maximum gain, which is handled by
:class:`~..control.assistance_policy.AssistanceGainPolicy`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..config import SafetyConfig
from ..sensors.sensor_manager import SensorSample


class FaultCode(Enum):
    """Reasons the assistance can be inhibited."""

    EMERGENCY_STOP = "EMERGENCY_STOP"
    ENCODER_ERROR = "ENCODER_ERROR"
    IMU_ERROR = "IMU_ERROR"
    NAN_DATA = "NAN_DATA"
    INVALID_SPD = "INVALID_SPD"
    MOTOR_POSITION_LIMIT = "MOTOR_POSITION_LIMIT"
    MOTOR_CURRENT_LIMIT = "MOTOR_CURRENT_LIMIT"
    STALE_DATA = "STALE_DATA"


@dataclass(frozen=True)
class SafetyStatus:
    """Outcome of one safety evaluation."""

    ok: bool
    assist_allowed: bool
    faults: Tuple[FaultCode, ...] = ()
    message: str = ""

    @property
    def fault_names(self) -> str:
        """Comma-separated fault names, empty when there is no fault."""
        return ",".join(f.value for f in self.faults)


class SafetyMonitor:
    """Per-sample and per-stride safety checks with optional fault latching.

    Args:
        config: safety limits.
    """

    def __init__(self, config: Optional[SafetyConfig] = None) -> None:
        self.config = config or SafetyConfig()
        self._latched: List[FaultCode] = []
        self._emergency_stop = False
        self._last_timestamp: Optional[float] = None
        self.status = SafetyStatus(True, True)

    # -- external triggers -------------------------------------------------- #

    def trigger_emergency_stop(self, reason: str = "operator request") -> SafetyStatus:
        """Latch an emergency stop; assistance is inhibited until :meth:`clear`."""
        self._emergency_stop = True
        self._latch(FaultCode.EMERGENCY_STOP)
        self.status = SafetyStatus(False, False, tuple(self._latched), reason)
        return self.status

    def report_invalid_spd(self, reason: str = "invalid SPD matrix") -> SafetyStatus:
        """Latch an invalid-SPD fault raised by the high-level loop."""
        self._latch(FaultCode.INVALID_SPD)
        self.status = SafetyStatus(False, False, tuple(self._latched), reason)
        return self.status

    def clear(self) -> None:
        """Clear every latched fault and the emergency stop."""
        self._latched.clear()
        self._emergency_stop = False
        self._last_timestamp = None
        self.status = SafetyStatus(True, True)

    @property
    def emergency_stop(self) -> bool:
        """Whether an emergency stop is currently latched."""
        return self._emergency_stop

    @property
    def latched_faults(self) -> Tuple[FaultCode, ...]:
        """Faults latched so far."""
        return tuple(self._latched)

    # -- per-sample checks -------------------------------------------------- #

    def check_sample(
        self, sample: SensorSample, *, wall_time: Optional[float] = None
    ) -> SafetyStatus:
        """Validate one sensor frame against the safety envelope.

        Args:
            sample: current sensor frame.
            wall_time: arrival time used for the staleness check; defaults to
                the sample timestamp.

        Returns:
            The :class:`SafetyStatus` for this cycle.
        """
        cfg = self.config
        faults: List[FaultCode] = []
        messages: List[str] = []

        if self._emergency_stop:
            faults.append(FaultCode.EMERGENCY_STOP)
            messages.append("emergency stop active")

        if not sample.is_finite():
            faults.append(FaultCode.NAN_DATA)
            messages.append("NaN/Inf in sensor frame")
        else:
            if not (
                cfg.min_belt_length_mm <= sample.belt_length <= cfg.max_belt_length_mm
            ) or abs(sample.belt_velocity) > cfg.max_belt_velocity_mms:
                faults.append(FaultCode.ENCODER_ERROR)
                messages.append("belt reading out of range")
            if abs(sample.motor_position) > cfg.max_motor_position:
                faults.append(FaultCode.MOTOR_POSITION_LIMIT)
                messages.append("motor position out of range")
            if abs(sample.motor_current) > cfg.max_motor_current_a:
                faults.append(FaultCode.MOTOR_CURRENT_LIMIT)
                messages.append("motor current out of range")
            accel = np.array([sample.accel_x, sample.accel_y, sample.accel_z])
            gyro = np.array([sample.gyro_x, sample.gyro_y, sample.gyro_z])
            if np.any(np.abs(accel) > cfg.max_accel_g) or np.any(
                np.abs(gyro) > cfg.max_gyro_dps
            ):
                faults.append(FaultCode.IMU_ERROR)
                messages.append("IMU saturated")

        arrival = sample.timestamp if wall_time is None else wall_time
        if self._last_timestamp is not None:
            gap = arrival - self._last_timestamp
            if gap > cfg.stale_data_s:
                faults.append(FaultCode.STALE_DATA)
                messages.append(f"telemetry gap {gap:.3f}s")
        self._last_timestamp = arrival

        if cfg.latch_faults:
            for fault in faults:
                self._latch(fault)
            active = tuple(self._latched) if self._latched else ()
        else:
            active = tuple(faults)

        ok = not faults
        self.status = SafetyStatus(
            ok=ok,
            assist_allowed=not active,
            faults=active,
            message="; ".join(messages),
        )
        return self.status

    def check_command(self, current_a: float) -> bool:
        """Return whether a current command is finite and within the limit."""
        if not math.isfinite(current_a):
            return False
        return abs(current_a) <= self.config.max_motor_current_a

    def limit_current(self, current_a: float, limit_a: float) -> float:
        """Clamp a current command, mapping non-finite values to zero."""
        if not math.isfinite(current_a):
            return 0.0
        return float(np.clip(current_a, -abs(limit_a), abs(limit_a)))

    def _latch(self, fault: FaultCode) -> None:
        """Record a fault once."""
        if fault not in self._latched:
            self._latched.append(fault)


__all__ = ["FaultCode", "SafetyMonitor", "SafetyStatus"]
