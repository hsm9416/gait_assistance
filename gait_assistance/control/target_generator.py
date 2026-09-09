"""Swing assistance profile and belt target generation (spec 21, 22).

The *shape* of the assistance over the swing phase and its *magnitude* are
deliberately separate objects: :class:`SwingAssistanceProfile` only knows the
normalised shape ``f(x)`` for ``x`` in ``[0, 1]``, while the magnitude comes
from the high-level assist gain.

**Assistance is applied during swing and nowhere else.**  This is a property of
this module, not of the caller: :meth:`BeltTargetGenerator.generate` evaluates
the profile only when the phase is :attr:`~..gait.phase_detector.GaitPhase.SWING`
and drives the belt to the neutral baseline length in stance, whatever gain the
high-level loop has published.  A gain is therefore a statement about how much
to help the next swing, never a standing pull, and the stance foot is never
loaded by the device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from ..config import AssistConfig
from ..gait.phase_detector import GaitPhase


def sine_profile(x: float | np.ndarray) -> np.ndarray:
    """Half-sine profile ``sin(pi * x)``, the default shape (spec 21)."""
    xc = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return np.sin(np.pi * xc)


def raised_cosine_profile(x: float | np.ndarray) -> np.ndarray:
    """Raised-cosine profile ``0.5 * (1 - cos(2 pi x))``, smoother at the ends."""
    xc = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * xc))


def trapezoid_profile(
    x: float | np.ndarray, rise: float = 0.25, fall: float = 0.25
) -> np.ndarray:
    """Trapezoidal profile with linear ``rise`` and ``fall`` fractions."""
    xc = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    out = np.ones_like(xc)
    out = np.where(xc < rise, xc / max(rise, 1e-6), out)
    out = np.where(xc > 1.0 - fall, (1.0 - xc) / max(fall, 1e-6), out)
    return np.clip(out, 0.0, 1.0)


#: Registry of the available profile shapes.
PROFILES: Dict[str, Callable[[float | np.ndarray], np.ndarray]] = {
    "sine": sine_profile,
    "raised_cosine": raised_cosine_profile,
    "trapezoid": trapezoid_profile,
}


class SwingAssistanceProfile:
    """Normalised assistance shape over the swing phase.

    Args:
        name: profile key from :data:`PROFILES`.

    Raises:
        ValueError: if ``name`` is unknown.
    """

    def __init__(self, name: str = "sine") -> None:
        if name not in PROFILES:
            raise ValueError(f"unknown profile {name!r}; available: {sorted(PROFILES)}")
        self.name = name
        self._fn = PROFILES[name]

    def __call__(self, x: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the profile at swing progress ``x`` in ``[0, 1]``."""
        value = self._fn(x)
        return float(value) if np.isscalar(x) or np.ndim(x) == 0 else value

    def sample(self, n: int = 100) -> np.ndarray:
        """Return the profile sampled on ``n`` uniform points of ``[0, 1]``."""
        return np.asarray(self._fn(np.linspace(0.0, 1.0, n)), dtype=float)


@dataclass(frozen=True)
class BeltTarget:
    """Belt set-point produced for one control cycle."""

    target_belt_length: float
    target_belt_velocity: float
    profile_value: float
    assist_gain: float


class BeltTargetGenerator:
    """Turn (gain, phase, progress) into a belt length set-point (spec 22).

    In swing::

        target = baseline_belt_length - assist_gain * profile(x) * max_retraction

    In stance the profile term is zero, so ``target == baseline_belt_length``
    and the device holds the neutral length instead of assisting.

    Args:
        config: assist configuration (profile name, retraction limit).
    """

    #: The one phase in which the device assists.
    ASSIST_PHASE: GaitPhase = GaitPhase.SWING

    def __init__(self, config: Optional[AssistConfig] = None) -> None:
        self.config = config or AssistConfig()
        self.profile = SwingAssistanceProfile(self.config.profile)
        self._last_target: Optional[float] = None
        self._last_time: Optional[float] = None

    def reset(self) -> None:
        """Forget the previous target (used to restart velocity estimation)."""
        self._last_target = None
        self._last_time = None

    @classmethod
    def assists_in(cls, phase: GaitPhase, phase_progress: Optional[float]) -> bool:
        """Whether the device may assist at this instant.

        Args:
            phase: current gait phase.
            phase_progress: progress inside that phase.

        Returns:
            True only in :attr:`ASSIST_PHASE` and only when the progress is
            known.  Without a progress value there is no defined point on the
            profile, and guessing one would place a retraction at an arbitrary
            moment of the swing, so the answer is no assistance rather than
            assistance at a made-up phase.
        """
        return phase is cls.ASSIST_PHASE and phase_progress is not None

    def generate(
        self,
        baseline_belt_length: float,
        assist_gain: float,
        phase: GaitPhase,
        phase_progress: Optional[float],
        timestamp: Optional[float] = None,
    ) -> BeltTarget:
        """Compute the belt target for the current control cycle.

        Args:
            baseline_belt_length: patient's neutral belt length (mm).
            assist_gain: high-level gain in ``[0, max_gain]``.
            phase: current gait phase; assistance is applied in swing only.
            phase_progress: progress inside the swing phase in ``[0, 1]``;
                ``None`` freezes the profile at 0.
            timestamp: current time, used to differentiate the target.

        Returns:
            The :class:`BeltTarget`.  Outside swing ``profile_value`` is 0 and
            the target is the neutral baseline length; ``assist_gain`` is still
            reported as published, so the log distinguishes "no gain" from
            "gain held but not applied in this phase".
        """
        gain = float(np.clip(assist_gain, 0.0, self.config.max_gain))
        if self.assists_in(phase, phase_progress):
            profile_value = float(self.profile(float(np.clip(phase_progress, 0.0, 1.0))))
        else:
            profile_value = 0.0
        target = baseline_belt_length - gain * profile_value * self.config.max_retraction_mm

        velocity = 0.0
        if (
            timestamp is not None
            and self._last_target is not None
            and self._last_time is not None
            and timestamp > self._last_time
        ):
            velocity = (target - self._last_target) / (timestamp - self._last_time)
        if timestamp is not None:
            self._last_target = target
            self._last_time = timestamp
        return BeltTarget(target, velocity, profile_value, gain)


__all__ = [
    "PROFILES",
    "BeltTarget",
    "BeltTargetGenerator",
    "SwingAssistanceProfile",
    "raised_cosine_profile",
    "sine_profile",
    "trapezoid_profile",
]
