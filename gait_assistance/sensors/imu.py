"""IMU sample container, health checks and a synthetic IMU (spec 1, 25, 29)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ImuSample:
    """One inertial measurement.

    Accelerations are expressed in g and angular rates in deg/s, matching the
    units produced by :class:`connect.PadSensorData`.
    """

    timestamp: float
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float

    @property
    def accel(self) -> np.ndarray:
        """Acceleration vector ``[ax, ay, az]``."""
        return np.array([self.accel_x, self.accel_y, self.accel_z], dtype=float)

    @property
    def gyro(self) -> np.ndarray:
        """Angular-rate vector ``[gx, gy, gz]``."""
        return np.array([self.gyro_x, self.gyro_y, self.gyro_z], dtype=float)

    @property
    def accel_magnitude(self) -> float:
        """Euclidean norm of the acceleration vector."""
        return float(np.linalg.norm(self.accel))


def check_imu_sample(
    sample: ImuSample, *, max_accel_g: float = 16.0, max_gyro_dps: float = 2000.0
) -> Tuple[bool, str]:
    """Validate one IMU sample.

    Args:
        sample: measurement to validate.
        max_accel_g: saturation limit of the accelerometer.
        max_gyro_dps: saturation limit of the gyroscope.

    Returns:
        ``(ok, reason)``; ``reason`` is empty when the sample is valid.
    """
    values = np.concatenate([sample.accel, sample.gyro])
    if not np.all(np.isfinite(values)):
        return False, "IMU produced NaN/Inf"
    if np.any(np.abs(sample.accel) > max_accel_g):
        return False, "accelerometer saturated"
    if np.any(np.abs(sample.gyro) > max_gyro_dps):
        return False, "gyroscope saturated"
    return True, ""


def is_imu_frozen(samples: List[ImuSample], *, tol: float = 1e-9) -> bool:
    """Return True when every sample in ``samples`` is bit-identical.

    A frozen signal usually means the sensor stopped updating even though the
    link is still delivering frames.
    """
    if len(samples) < 2:
        return False
    ref = np.concatenate([samples[0].accel, samples[0].gyro])
    for s in samples[1:]:
        if np.any(np.abs(np.concatenate([s.accel, s.gyro]) - ref) > tol):
            return False
    return True


class MockImu:
    """Synthetic shank-mounted IMU with a hemiparetic gait signature.

    The signal is a stride-periodic pattern plus a heel-strike acceleration
    spike, which is enough to exercise the whole pipeline without hardware.

    Args:
        stride_period_s: nominal stride duration.
        swing_ratio: fraction of the stride spent in swing.
        noise_std: white-noise standard deviation added to every channel.
        seed: RNG seed.
    """

    def __init__(
        self,
        stride_period_s: float = 1.2,
        swing_ratio: float = 0.35,
        noise_std: float = 0.02,
        seed: int = 0,
    ) -> None:
        self.stride_period_s = stride_period_s
        self.swing_ratio = swing_ratio
        self.noise_std = noise_std
        self._rng = np.random.default_rng(seed)

    def sample(self, t: float, *, phase_fraction: Optional[float] = None) -> ImuSample:
        """Generate an IMU sample for absolute time ``t``.

        Args:
            t: timestamp in seconds.
            phase_fraction: gait-cycle position in ``[0, 1)``; derived from
                ``t`` when omitted.

        Returns:
            A synthetic :class:`ImuSample`.
        """
        x = (t / self.stride_period_s) % 1.0 if phase_fraction is None else phase_fraction
        omega = 2.0 * math.pi
        # heel strike spike at the start of the cycle
        spike = math.exp(-((x % 1.0) ** 2) / (2.0 * 0.02 ** 2))
        ax = 0.35 * math.sin(omega * x) + 0.9 * spike
        ay = -0.98 + 0.25 * math.cos(omega * x)
        az = 0.10 * math.sin(2.0 * omega * x)
        # shank angular rate: large positive peak during swing
        swing_pos = (x - (1.0 - self.swing_ratio)) / max(self.swing_ratio, 1e-6)
        gz = 180.0 * math.sin(math.pi * swing_pos) if 0.0 <= swing_pos <= 1.0 else -25.0
        gx = 12.0 * math.sin(omega * x)
        gy = 8.0 * math.cos(omega * x)
        noise = self._rng.normal(0.0, self.noise_std, 6)
        return ImuSample(
            timestamp=t,
            accel_x=ax + noise[0],
            accel_y=ay + noise[1],
            accel_z=az + noise[2],
            gyro_x=gx + noise[3] * 50.0,
            gyro_y=gy + noise[4] * 50.0,
            gyro_z=gz + noise[5] * 50.0,
        )


__all__ = ["ImuSample", "MockImu", "check_imu_sample", "is_imu_frozen"]
