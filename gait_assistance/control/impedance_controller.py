"""Low-level impedance controller (spec 23).

The Riemannian layer never produces a torque or a current: it only supplies
``assist_gain`` and ``target_belt_length``.  Everything below converts that
set-point into a motor current with a position/velocity impedance law

``F = K * (x_d - x) + D * (v_d - v)``

which is the same law already used by ``impedance_controller.py`` at the root
of this repository; the force-to-current mapping is reused from
``pad_external_control_lib`` when it can be imported.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..config import ImpedanceConfig


def _load_pad_force_map() -> Optional[Callable[[float], float]]:
    """Return the device force-to-current map, or ``None`` when unavailable."""
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from pad_external_control_lib import force_to_current_a  # type: ignore
    except Exception:  # pragma: no cover - hardware library not installed
        return None
    return force_to_current_a


def default_force_to_current(force_n: float) -> float:
    """Fallback affine force-to-current map matching the PAD device.

    The device library uses ``I = (F + 0.04) / 0.87`` for positive forces and
    commands zero current otherwise; this reproduces it without the import.
    """
    if force_n <= 0.0:
        return 0.0
    return (force_n + 0.04) / 0.87


@dataclass(frozen=True)
class MotorCommand:
    """One low-level actuator command with its intermediate quantities."""

    current_a: float
    force_n: float
    position_error_mm: float
    velocity_error_mms: float
    saturated: bool

    def is_finite(self) -> bool:
        """True when every field is a finite number."""
        return all(
            math.isfinite(v)
            for v in (
                self.current_a,
                self.force_n,
                self.position_error_mm,
                self.velocity_error_mms,
            )
        )


class ImpedanceController:
    """Position/velocity impedance law with current saturation.

    Args:
        config: impedance gains and the current limit.
    """

    def __init__(self, config: Optional[ImpedanceConfig] = None) -> None:
        self.config = config or ImpedanceConfig()
        self._force_map: Callable[[float], float] = default_force_to_current
        if self.config.use_pad_force_map:
            device_map = _load_pad_force_map()
            if device_map is not None:
                self._force_map = device_map
        self.last_command: Optional[MotorCommand] = None

    def compute(
        self,
        target_position: float,
        measured_position: float,
        *,
        target_velocity: float = 0.0,
        measured_velocity: float = 0.0,
    ) -> MotorCommand:
        """Compute the current command for one cycle.

        Args:
            target_position: desired belt length (mm).
            measured_position: measured belt length (mm).
            target_velocity: desired belt velocity (mm/s).
            measured_velocity: measured belt velocity (mm/s).

        Returns:
            The :class:`MotorCommand`; non-finite inputs produce a zero command
            rather than propagating NaN into the actuator (spec 25).
        """
        cfg = self.config
        values = (target_position, measured_position, target_velocity, measured_velocity)
        if not all(math.isfinite(float(v)) for v in values):
            command = MotorCommand(0.0, 0.0, 0.0, 0.0, False)
            self.last_command = command
            return command

        position_error = float(target_position) - float(measured_position)
        velocity_error = float(target_velocity) - float(measured_velocity)
        force = cfg.stiffness_n_per_mm * position_error + cfg.damping_n_per_mms * velocity_error

        raw_current = self.force_to_current(force)
        current = float(np.clip(raw_current, -cfg.current_limit_a, cfg.current_limit_a))
        command = MotorCommand(
            current_a=current,
            force_n=force,
            position_error_mm=position_error,
            velocity_error_mms=velocity_error,
            saturated=abs(current - raw_current) > 1e-12,
        )
        self.last_command = command
        return command

    def force_to_current(self, force_n: float) -> float:
        """Convert a force (N) into a motor current (A)."""
        if not math.isfinite(force_n):
            return 0.0
        if self.config.use_pad_force_map:
            return float(self._force_map(float(force_n)))
        return float(force_n) * self.config.current_per_newton

    def zero_command(self) -> MotorCommand:
        """Return (and remember) a zero command, used on safe stop."""
        command = MotorCommand(0.0, 0.0, 0.0, 0.0, False)
        self.last_command = command
        return command


__all__ = ["ImpedanceController", "MotorCommand", "default_force_to_current"]
