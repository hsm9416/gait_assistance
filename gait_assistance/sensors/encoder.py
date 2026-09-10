"""Belt encoder / motor state, health checks and motor interfaces.

The physical device exposes belt length and motor state through the same
telemetry frame (see ``connect.PadSensorData``), so the encoder module also
hosts the actuator abstraction used by the low-level loop (spec 23, 29).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class EncoderSample:
    """Belt and motor state at one instant."""

    timestamp: float
    belt_length: float      #: mm
    belt_velocity: float    #: mm/s
    motor_position: float
    motor_velocity: float
    motor_current: float    #: A


def check_encoder_sample(
    sample: EncoderSample,
    *,
    min_belt_length_mm: float = -400.0,
    max_belt_length_mm: float = 400.0,
    max_belt_velocity_mms: float = 2000.0,
    max_motor_position: float = 1.0e6,
    max_motor_current_a: float = 10.0,
) -> Tuple[bool, str]:
    """Validate one encoder/motor sample.

    Returns:
        ``(ok, reason)``; ``reason`` is empty when the sample is valid.
    """
    values = (
        sample.belt_length,
        sample.belt_velocity,
        sample.motor_position,
        sample.motor_velocity,
        sample.motor_current,
    )
    if not all(math.isfinite(v) for v in values):
        return False, "encoder produced NaN/Inf"
    if not min_belt_length_mm <= sample.belt_length <= max_belt_length_mm:
        return False, "belt length out of range"
    if abs(sample.belt_velocity) > max_belt_velocity_mms:
        return False, "belt velocity out of range"
    if abs(sample.motor_position) > max_motor_position:
        return False, "motor position out of range"
    if abs(sample.motor_current) > max_motor_current_a:
        return False, "motor current out of range"
    return True, ""


class MotorInterface(ABC):
    """Actuator abstraction: the only command is a current in amperes."""

    @abstractmethod
    def send_current(self, current_a: float) -> None:
        """Send a current command (A) to the motor."""

    def stop(self) -> None:
        """Command zero current."""
        self.send_current(0.0)

    @property
    def last_command_a(self) -> float:
        """Most recent current command in amperes."""
        return getattr(self, "_last_command_a", 0.0)


class MockMotor(MotorInterface):
    """Simulated motor with a first-order belt response (spec 29).

    The model is deliberately simple: the commanded current produces a force
    that accelerates the belt towards shorter lengths (retraction), with linear
    damping and a spring pulling back to the rest length.

    Args:
        current_limit_a: saturation limit applied to every command.
        rest_length_mm: belt length the passive spring pulls towards.
        mm_per_amp: static belt displacement produced by one ampere.  Positive,
            because the device is unidirectional in the other sense: the repo's
            ``impedance_controller.py`` drives ``belt_length`` towards 0 from a
            resting value near -130 mm using ``force_to_current_a``, which only
            emits current for a positive force.  That loop is stable only if a
            positive current moves the belt towards 0, so the plant sign here
            has to match.  With the sign inverted the simulated loop runs away,
            which is a property of the model, not of the device.
        tau_s: time constant of the belt response.
    """

    def __init__(
        self,
        current_limit_a: float = 1.0,
        rest_length_mm: float = 0.0,
        mm_per_amp: float = 40.0,
        tau_s: float = 0.08,
    ) -> None:
        self.current_limit_a = current_limit_a
        self.rest_length_mm = rest_length_mm
        self.mm_per_amp = mm_per_amp
        self.tau_s = max(tau_s, 1e-3)
        self._last_command_a = 0.0
        self.saturated = False
        self._offset_mm = 0.0

    def send_current(self, current_a: float) -> None:
        """Clamp and store the command; the plant is advanced by :meth:`step`."""
        value = float(current_a)
        if not math.isfinite(value):
            value = 0.0
        clamped = float(np.clip(value, -self.current_limit_a, self.current_limit_a))
        self.saturated = abs(clamped - value) > 1e-12
        self._last_command_a = clamped

    def step(self, dt: float) -> float:
        """Advance the simulated belt by ``dt`` seconds.

        Returns:
            The commanded belt-length offset (mm) contributed by the motor.
        """
        target = self.mm_per_amp * self._last_command_a
        alpha = 1.0 - math.exp(-max(dt, 0.0) / self.tau_s)
        self._offset_mm += alpha * (target - self._offset_mm)
        return self._offset_mm

    @property
    def offset_mm(self) -> float:
        """Current motor-induced belt offset (mm)."""
        return self._offset_mm

    def reset(self) -> None:
        """Zero the command and the plant state."""
        self._last_command_a = 0.0
        self._offset_mm = 0.0
        self.saturated = False


class PadMotor(MotorInterface):
    """Adapter forwarding current commands to a ``connect.PadController``.

    Args:
        controller: an open :class:`connect.PadController` instance.
        current_limit_a: additional software limit on top of the device limit.
    """

    def __init__(self, controller: object, current_limit_a: float = 1.0) -> None:
        self._controller = controller
        self.current_limit_a = current_limit_a
        self._last_command_a = 0.0

    def send_current(self, current_a: float) -> None:
        """Clamp and forward the command to the PAD device."""
        value = float(current_a)
        if not math.isfinite(value):
            value = 0.0
        value = float(np.clip(value, -self.current_limit_a, self.current_limit_a))
        self._last_command_a = value
        self._controller.set_current(value)  # type: ignore[attr-defined]


__all__ = [
    "EncoderSample",
    "MockMotor",
    "MotorInterface",
    "PadMotor",
    "check_encoder_sample",
]
