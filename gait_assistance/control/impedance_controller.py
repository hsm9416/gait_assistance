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


#: Positive impedance force to keep in reserve when sizing ``assist_force_n``
#: (N).  The ``K*dx + D*dv`` term rides on top of the assist force and would
#: otherwise push the sum over the limit near the peak of the swing profile.
#: 0.8 N covers the largest value measured across every recorded run (swing
#: cycles only; the worst was +0.75 N).
IMPEDANCE_HEADROOM_N: float = 0.8


@dataclass(frozen=True)
class MotorCommand:
    """One low-level actuator command with its intermediate quantities."""

    current_a: float
    force_n: float
    position_error_mm: float
    velocity_error_mms: float
    saturated: bool
    #: the ``K*dx + D*dv`` part, which resists the wearer's own belt motion
    impedance_force_n: float = 0.0
    #: the feed-forward part, which is the assistance the wearer feels
    assist_force_n: float = 0.0

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
        assist_fraction: float = 0.0,
    ) -> MotorCommand:
        """Compute the current command for one cycle.

        The law is the position/velocity impedance plus an optional
        feed-forward assist force::

            F = K * (x_d - x) + D * (v_d - v) + assist_force_n * assist_fraction

        The third term is what the wearer feels as help.  It is kept out of the
        stiffness on purpose: the belt swings about its baseline by roughly the
        distance the assistance displaces the target, so a stiffness large
        enough to make the displaced target felt also turns the device into a
        spring resisting the wearer.

        Args:
            target_position: desired belt length (mm).
            measured_position: measured belt length (mm).
            target_velocity: desired belt velocity (mm/s).
            measured_velocity: measured belt velocity (mm/s).
            assist_fraction: ``assist_gain * swing_profile`` in ``[0, 1]``,
                supplied by the caller; the high-level layer still decides how
                much to assist, this only turns that decision into a force.

        Returns:
            The :class:`MotorCommand`; non-finite inputs produce a zero command
            rather than propagating NaN into the actuator (spec 25).
        """
        cfg = self.config
        values = (target_position, measured_position, target_velocity,
                  measured_velocity, assist_fraction)
        if not all(math.isfinite(float(v)) for v in values):
            command = MotorCommand(0.0, 0.0, 0.0, 0.0, False)
            self.last_command = command
            return command

        position_error = float(target_position) - float(measured_position)
        velocity_error = float(target_velocity) - float(measured_velocity)
        impedance_force = (
            cfg.stiffness_n_per_mm * position_error
            + cfg.damping_n_per_mms * velocity_error
        )
        assist_force = cfg.assist_force_n * float(
            np.clip(assist_fraction, 0.0, 1.0)
        )
        force = impedance_force + assist_force

        raw_current = self.force_to_current(force)
        current = float(np.clip(raw_current, -cfg.current_limit_a, cfg.current_limit_a))
        command = MotorCommand(
            current_a=current,
            force_n=force,
            position_error_mm=position_error,
            velocity_error_mms=velocity_error,
            saturated=abs(current - raw_current) > 1e-12,
            impedance_force_n=impedance_force,
            assist_force_n=assist_force,
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

    def max_force_n(self) -> float:
        """Largest force the configured current limit can actually pass.

        Found by bisecting this controller's own force-to-current map rather
        than inverting a formula, so a custom or device-supplied map gives the
        right answer too.  The map is monotone in the force, which is what
        makes the bisection valid.

        Returns:
            The force (N) whose command lands exactly on
            ``current_limit_a``; ``0`` when the limit is zero.
        """
        limit = float(self.config.current_limit_a)
        if limit <= 0.0:
            return 0.0
        # the device map clamps at its own ceiling (13 A), so a software limit
        # above that is not the binding one: whichever is lower decides
        target = min(limit, self.force_to_current(1e4))
        if target <= 0.0:
            return 0.0
        lo, hi = 0.0, 1.0
        while self.force_to_current(hi) < target and hi < 1e4:
            hi *= 2.0
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            if self.force_to_current(mid) < target:
                lo = mid
            else:
                hi = mid
        return hi

    def assist_force_limit_n(
        self,
        max_gain: float,
        *,
        impedance_headroom_n: float = IMPEDANCE_HEADROOM_N,
    ) -> float:
        """Largest ``assist_force_n`` whose peak still fits under the limit.<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        ``assist_force_n`` and ``current_limit_a`` are configured
        independently, and a force command the limit cannot pass does not make
        the assistance stronger: it flattens the swing profile into a plateau
        at the limit, so what the wearer feels is a square pulse with steep
        edges instead of the commanded sine.  Sizing the force against the
        limit is what keeps the profile's shape.

        Args:
            max_gain: largest gain the policy may command - the peak assist
                force is ``assist_force_n * max_gain``.
            impedance_headroom_n: positive impedance force to leave in reserve.

        Returns:
            The force (N), or ``0`` when the limit cannot even cover the
            headroom.
        """
        if max_gain <= 0.0:
            return 0.0
        return max(self.max_force_n() - float(impedance_headroom_n), 0.0) / float(max_gain)

    def zero_command(self) -> MotorCommand:
        """Return (and remember) a zero command, used on safe stop."""
        command = MotorCommand(0.0, 0.0, 0.0, 0.0, False)
        self.last_command = command
        return command


__all__ = [
    "IMPEDANCE_HEADROOM_N",
    "ImpedanceController",
    "MotorCommand",
    "default_force_to_current",
]
