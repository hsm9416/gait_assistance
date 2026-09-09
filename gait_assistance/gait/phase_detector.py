"""Gait phase *estimation* (spec 2).

The detector is a replaceable strategy: any class implementing
:class:`PhaseDetector` can be plugged into the low-level loop through
:func:`create_phase_detector`, so the detection algorithm can be swapped
without touching the rest of the system.

None of the detectors shipped here observes foot-ground contact.

:class:`ScheduledPhaseDetector` - the current default - does not even look at
the sensors.  It emits swing and stance on a fixed, configurable cadence so the
whole pipeline can be exercised while the real swing/stance detection is
developed elsewhere.  Treat everything it produces as a schedule, not a
measurement.

The two signal-driven detectors produce an **estimated** gait phase: the
belt-velocity detector infers it from how the assistance belt travels, the gyro
detector from shank angular rate.  Both are proxies, and neither has been
validated against a ground-truth contact signal, so:

* the phase boundaries are the boundaries of a *belt* or *shank* event, offset
  from true toe-off and heel strike by an unmeasured amount;
* ``swing_time``, ``stance_time`` and ``swing_ratio`` derived from them live on
  an uncalibrated scale and must not be compared with published normative
  values (the belt-velocity detector marks the whole extension interval as
  SWING, which is longer than biomechanical swing);
* the honest use of these quantities is *relative* - comparing a stride to the
  same patient's other strides through the same detector.

Validating against a foot-switch / FSR insole or a foot-mounted IMU is the
outstanding work that would turn these estimates into measurements.  Each class
declares :attr:`PhaseDetector.ground_truth_validated`, which is ``False``
everywhere today; a detector that has been validated should set it and say
against what in :attr:`PhaseDetector.validated_against`.

Plugging in the real detector
-----------------------------

This module is the seam.  A new detector needs to do exactly three things, and
nothing outside it has to change:

1. Subclass :class:`PhaseDetector` and implement :meth:`PhaseDetector.update`,
   returning a :class:`PhaseResult` on every sample.  Emit
   ``GaitEvent.HEEL_STRIKE`` on the sample that starts a stride - that event is
   what the segmenter cuts strides on - and ``GaitEvent.TOE_OFF`` on the
   stance-to-swing transition.  Call :meth:`PhaseDetector._enter` and
   :meth:`PhaseDetector._register_heel_strike` so the inherited stride-period
   and swing-duration statistics stay populated.
2. Set the class attributes that describe its provenance:
   :attr:`PhaseDetector.phase_source` (a short token that reaches the
   ``phase_source`` column of the stride log and the ``estimated_*`` metrics),
   :attr:`PhaseDetector.estimates_from`, and - once it has been checked against
   a contact signal - :attr:`PhaseDetector.ground_truth_validated` together
   with :attr:`PhaseDetector.validated_against`.
3. Register it::

       from gait_assistance.gait.phase_detector import register_phase_detector
       register_phase_detector("fsr", FsrPhaseDetector)

   after which ``--set phase.detector=fsr`` selects it.  Detector-specific
   tuning goes in :class:`~gait_assistance.config.PhaseConfig`.

Everything downstream is already phase-agnostic: the segmenter cuts on heel
strikes, the metrics read the labels, and the belt target applies assistance in
swing only.  Swapping the detector is the whole integration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional, Type

import numpy as np

from ..config import PhaseConfig
from ..sensors.sensor_manager import SensorSample


class GaitPhase(Enum):
    """Minimal gait phase set required by the specification.

    These labels name an *estimated* phase inferred from proximal kinematics,
    not an observed foot-contact state; see the module docstring.
    """

    STANCE = "STANCE"
    SWING = "SWING"


class GaitEvent(Enum):
    """Discrete gait events emitted by a detector."""

    HEEL_STRIKE = "HEEL_STRIKE"
    TOE_OFF = "TOE_OFF"


@dataclass(frozen=True)
class PhaseResult:
    """Output of one detector update: an estimated phase and any event."""

    phase: GaitPhase
    event: Optional[GaitEvent] = None
    #: progress inside the current phase in ``[0, 1]`` when it can be
    #: estimated, otherwise ``None``.  The belt target needs this to place the
    #: assistance inside the swing, so a detector that cannot supply it
    #: produces no assistance at all.
    phase_progress: Optional[float] = None
    #: provenance token of the detector that produced this result; it travels
    #: with the stride into the ``phase_source`` log column
    source: str = ""

    @property
    def heel_strike(self) -> bool:
        """True when this update carried a heel-strike event."""
        return self.event is GaitEvent.HEEL_STRIKE

    @property
    def toe_off(self) -> bool:
        """True when this update carried a toe-off event."""
        return self.event is GaitEvent.TOE_OFF


class PhaseDetector(ABC):
    """Base class for stance/swing detectors.

    Attributes:
        estimates_from: the signal the phase is inferred from, for reports that
            have to state what the phase actually came from.
        ground_truth_validated: whether this detector's events have been
            checked against an independent contact measurement.  ``False`` for
            every detector in this package.
        validated_against: what the validation used, once there is one.
    """

    #: short provenance token written to the ``phase_source`` log column
    phase_source: str = "unspecified"
    #: what the phase is inferred from (proximal proxy, not foot contact)
    estimates_from: str = "unspecified"
    #: no detector here has been checked against a contact ground truth
    ground_truth_validated: bool = False
    #: name of the ground-truth source, once a detector is validated
    validated_against: str = ""
    #: True for a stand-in that reads no sensor at all, so reports can say so
    #: instead of calling a schedule an estimate
    is_placeholder: bool = False

    @classmethod
    def describe(cls) -> str:
        """One-line statement of what this detector's phase is worth."""
        if cls.is_placeholder:
            return (
                f"{cls.__name__}: PLACEHOLDER phase from {cls.estimates_from}; "
                f"the cadence is configured, not measured - nothing derived "
                f"from it describes the patient"
            )
        if cls.ground_truth_validated:
            return (
                f"{cls.__name__}: phase from {cls.estimates_from}, "
                f"validated against {cls.validated_against}"
            )
        return (
            f"{cls.__name__}: ESTIMATED phase from {cls.estimates_from}, "
            f"not validated against foot contact (FSR / foot IMU pending)"
        )

    def _result(
        self, event: Optional[GaitEvent], progress: Optional[float]
    ) -> PhaseResult:
        """Build a :class:`PhaseResult` stamped with this detector's source."""
        return PhaseResult(self._phase, event, progress, self.phase_source)

    def __init__(self, config: Optional[PhaseConfig] = None) -> None:
        self.config = config or PhaseConfig()
        self._phase = GaitPhase.STANCE
        self._phase_start_t = 0.0
        self._last_hs_t: Optional[float] = None
        self._stride_period_s: Optional[float] = None
        self._swing_duration_s: Optional[float] = None

    # -- public API -------------------------------------------------------- #

    @abstractmethod
    def update(self, sample: SensorSample) -> PhaseResult:
        """Feed one sensor frame and return the estimated phase and any event."""

    def reset(self) -> None:
        """Return the detector to its initial state."""
        self._phase = GaitPhase.STANCE
        self._phase_start_t = 0.0
        self._last_hs_t = None
        self._stride_period_s = None
        self._swing_duration_s = None

    @property
    def phase(self) -> GaitPhase:
        """Last detected phase."""
        return self._phase

    @property
    def stride_period_s(self) -> Optional[float]:
        """Most recent heel-strike-to-heel-strike interval."""
        return self._stride_period_s

    @property
    def swing_duration_s(self) -> Optional[float]:
        """Most recent completed swing duration."""
        return self._swing_duration_s

    # -- helpers for subclasses -------------------------------------------- #

    def _enter(self, phase: GaitPhase, t: float) -> None:
        """Record a phase change at time ``t``."""
        if phase is GaitPhase.STANCE and self._phase is GaitPhase.SWING:
            self._swing_duration_s = max(t - self._phase_start_t, 0.0)
        self._phase = phase
        self._phase_start_t = t

    def _register_heel_strike(self, t: float) -> None:
        """Update stride-period statistics on a heel strike at time ``t``."""
        if self._last_hs_t is not None:
            period = t - self._last_hs_t
            if period > 0.0:
                self._stride_period_s = period
        self._last_hs_t = t

    def _progress(self, t: float) -> Optional[float]:
        """Estimate progress inside the current phase in ``[0, 1]``."""
        elapsed = t - self._phase_start_t
        if self._phase is GaitPhase.SWING:
            expected = self._swing_duration_s
        else:
            expected = None
            if self._stride_period_s is not None and self._swing_duration_s is not None:
                expected = max(self._stride_period_s - self._swing_duration_s, 1e-3)
        if expected is None or expected <= 0.0:
            return None
        return float(min(max(elapsed / expected, 0.0), 1.0))


class ScheduledPhaseDetector(PhaseDetector):
    """Fixed-cadence swing/stance placeholder for the real detector.

    It ignores every sensor channel and derives the phase purely from the
    sample timestamp::

        u = (t - t0 - offset) mod period
        u <  stance_duration  ->  STANCE, progress = u / stance_duration
        u >= stance_duration  ->  SWING,  progress = (u - stance) / swing

    A ``HEEL_STRIKE`` is emitted whenever ``u`` wraps to the start of a cycle
    and a ``TOE_OFF`` when it crosses into the swing window, so the segmenter
    cuts one stride per period and the belt target gets a clean swing progress
    to place the assistance in.

    This exists so the manifold, deficit and assistance chain can be run and
    tuned end to end while swing/stance detection is being developed
    separately.  It is a clock: the cadence it reports is the one that was
    configured, and it will agree with a patient's actual gait only by
    coincidence.  Nothing computed from it - swing time, stance time, any
    ratio, any symmetry - describes the patient, which is why it declares its
    own ``phase_source`` and never claims to be an estimate of anything.

    Timing comes from :class:`~gait_assistance.config.PhaseConfig`:
    ``scheduled_period_s``, ``scheduled_swing_ratio`` or the explicit
    ``scheduled_swing_duration_s``, and ``scheduled_offset_s``.
    """

    phase_source = "scheduled"
    estimates_from = "a fixed clock (no sensor input)"
    is_placeholder = True

    def __init__(self, config: Optional[PhaseConfig] = None) -> None:
        super().__init__(config)
        self._t0: Optional[float] = None
        self._cycle_index: Optional[int] = None
        self._in_swing = False

    @property
    def period_s(self) -> float:
        """Configured stride period (s)."""
        return max(float(self.config.scheduled_period_s), 1e-6)

    @property
    def swing_s(self) -> float:
        """Configured swing duration (s)."""
        return self.config.swing_duration_s()

    @property
    def stance_s(self) -> float:
        """Configured stance duration (s)."""
        return self.config.stance_duration_s()

    def reset(self) -> None:
        """Return the detector to its initial state."""
        super().reset()
        self._t0 = None
        self._cycle_index = None
        self._in_swing = False

    def update(self, sample: SensorSample) -> PhaseResult:
        """Place the sample in the scheduled cycle and report the phase.

        Args:
            sample: current sensor frame; only its timestamp is read.

        Returns:
            The :class:`PhaseResult` for this instant.
        """
        t = float(sample.timestamp)
        if self._t0 is None:
            self._t0 = t
        elapsed = t - self._t0 - float(self.config.scheduled_offset_s)
        period = self.period_s
        cycle = int(np.floor(elapsed / period))
        u = elapsed - cycle * period

        stance = self.stance_s
        swing = max(period - stance, 1e-6)
        in_swing = u >= stance

        event: Optional[GaitEvent] = None
        if self._cycle_index is None or cycle != self._cycle_index:
            # A new cycle starts on a heel strike, which is where the
            # segmenter cuts one stride from the next.
            self._cycle_index = cycle
            self._in_swing = False
            self._register_heel_strike(t)
            self._enter(GaitPhase.STANCE, t)
            event = GaitEvent.HEEL_STRIKE
        if in_swing and not self._in_swing:
            self._in_swing = True
            self._enter(GaitPhase.SWING, t)
            # A heel strike and a toe-off cannot land on the same sample
            # unless stance is zero, which the config clamps against.
            event = event or GaitEvent.TOE_OFF

        if in_swing:
            progress = (u - stance) / swing
        else:
            progress = u / max(stance, 1e-6)
        return self._result(event, float(min(max(progress, 0.0), 1.0)))


class BeltVelocityPhaseDetector(PhaseDetector):
    """Belt-velocity state machine ported from ``heelstrike_detect.py``.

    The belt extends while the leg swings forward; the extension reverses
    sharply at heel strike:

    ``STANCE --(v >= loading_vel)--> SWING --(v <= strike_vel)--> [HS] STANCE``

    The events are *belt* events used as a stand-in for gait events.  The
    SWING interval it reports starts at the onset of belt extension, which
    precedes toe-off, so the estimated swing ratio runs high compared with a
    contact-derived one; treat it as a within-patient relative measure until
    the detector is checked against an FSR insole or a foot-mounted IMU.
    """

    phase_source = "belt_derived"
    estimates_from = "belt extension velocity (proximal proxy)"

    def __init__(self, config: Optional[PhaseConfig] = None) -> None:
        super().__init__(config)
        self._loading = False
        self._last_event_t = -1.0e9

    def reset(self) -> None:
        """Return the detector to its initial state."""
        super().reset()
        self._loading = False
        self._last_event_t = -1.0e9

    def update(self, sample: SensorSample) -> PhaseResult:
        """Feed one sensor frame and return the current phase and any event.

        Args:
            sample: current sensor frame.

        Returns:
            The detected :class:`PhaseResult`.
        """
        cfg = self.config
        t = sample.timestamp
        belt = sample.belt_length
        vel = sample.belt_velocity
        event: Optional[GaitEvent] = None

        if not self._loading and vel >= cfg.loading_vel:
            self._loading = True
            self._enter(GaitPhase.SWING, t)
            event = GaitEvent.TOE_OFF
        elif self._loading and vel <= cfg.strike_vel:
            belt_ok = belt <= cfg.min_belt_mm if cfg.min_belt_mm < 0 else belt >= cfg.min_belt_mm
            if belt_ok and (t - self._last_event_t) >= cfg.refractory_s:
                self._loading = False
                self._last_event_t = t
                self._register_heel_strike(t)
                self._enter(GaitPhase.STANCE, t)
                event = GaitEvent.HEEL_STRIKE
        return self._result(event, self._progress(t))


class GyroPhaseDetector(PhaseDetector):
    """Alternative detector driven by the shank angular rate.

    Swing is entered when the gyroscope rate exceeds
    ``gyro_swing_threshold``; heel strike is declared on the following
    negative-going crossing of ``gyro_strike_threshold``.  It exists to show
    that the detector really is interchangeable.

    Shank angular rate is likewise a proxy for foot contact, so this detector's
    phase is an estimate on the same footing as the belt-velocity one.
    """

    phase_source = "gyro_derived"
    estimates_from = "shank angular rate (proximal proxy)"

    def __init__(self, config: Optional[PhaseConfig] = None) -> None:
        super().__init__(config)
        self._swinging = False
        self._last_event_t = -1.0e9

    def reset(self) -> None:
        """Return the detector to its initial state."""
        super().reset()
        self._swinging = False
        self._last_event_t = -1.0e9

    def update(self, sample: SensorSample) -> PhaseResult:
        """Feed one sensor frame and return the current phase and any event."""
        cfg = self.config
        t = sample.timestamp
        rate = sample.channel(cfg.gyro_channel)
        event: Optional[GaitEvent] = None

        if not self._swinging and rate >= cfg.gyro_swing_threshold:
            self._swinging = True
            self._enter(GaitPhase.SWING, t)
            event = GaitEvent.TOE_OFF
        elif self._swinging and rate <= cfg.gyro_strike_threshold:
            if (t - self._last_event_t) >= cfg.refractory_s:
                self._swinging = False
                self._last_event_t = t
                self._register_heel_strike(t)
                self._enter(GaitPhase.STANCE, t)
                event = GaitEvent.HEEL_STRIKE
        return self._result(event, self._progress(t))


#: Registry of the available detector implementations.  A future module adds
#: itself here through :func:`register_phase_detector` rather than by editing
#: this file.
PHASE_DETECTORS: Dict[str, Callable[[Optional[PhaseConfig]], PhaseDetector]] = {
    "scheduled": ScheduledPhaseDetector,
    "belt_velocity": BeltVelocityPhaseDetector,
    "gyro": GyroPhaseDetector,
}


def register_phase_detector(name: str, factory: Type[PhaseDetector]) -> None:
    """Make a detector selectable through ``phase.detector``.

    This is the entry point for the real swing/stance detection once it lands:
    import it, register it under a name, and select that name in the
    configuration.  No other module needs to change.

    Args:
        name: key used by ``phase.detector``.
        factory: a :class:`PhaseDetector` subclass taking an optional
            :class:`~gait_assistance.config.PhaseConfig`.

    Raises:
        TypeError: if ``factory`` is not a :class:`PhaseDetector` subclass.
        ValueError: if ``name`` is already taken by a different class, which
            would otherwise silently change what every existing config selects.
    """
    if not (isinstance(factory, type) and issubclass(factory, PhaseDetector)):
        raise TypeError(f"{factory!r} is not a PhaseDetector subclass")
    existing = PHASE_DETECTORS.get(name)
    if existing is not None and existing is not factory:
        raise ValueError(
            f"phase detector {name!r} is already registered to {existing.__name__}"
        )
    PHASE_DETECTORS[name] = factory


def create_phase_detector(config: Optional[PhaseConfig] = None) -> PhaseDetector:
    """Instantiate the detector named by ``config.detector``.

    Args:
        config: phase configuration; defaults are used when omitted.

    Returns:
        A ready-to-use detector.  Check :meth:`PhaseDetector.describe` before
        reporting any phase-derived quantity: every detector registered here
        produces an estimated phase.

    Raises:
        ValueError: if the configured detector name is unknown.
    """
    cfg = config or PhaseConfig()
    try:
        factory = PHASE_DETECTORS[cfg.detector]
    except KeyError as exc:
        raise ValueError(
            f"unknown phase detector {cfg.detector!r}; "
            f"available: {sorted(PHASE_DETECTORS)}"
        ) from exc
    return factory(cfg)


__all__ = [
    "PHASE_DETECTORS",
    "BeltVelocityPhaseDetector",
    "ScheduledPhaseDetector",
    "register_phase_detector",
    "GaitEvent",
    "GaitPhase",
    "GyroPhaseDetector",
    "PhaseDetector",
    "PhaseResult",
    "create_phase_detector",
]
