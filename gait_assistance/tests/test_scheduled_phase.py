"""Tests for the scheduled phase placeholder and the swing-only assistance rule.

Two things are pinned here.  The scheduled detector must produce exactly the
cadence it was configured with, because that is its whole purpose while the
real swing/stance detection is developed elsewhere.  And the device must assist
in swing and nowhere else, whatever the high-level loop has published.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pytest

from gait_assistance.config import AssistConfig, Config, PhaseConfig, StrideConfig
from gait_assistance.control.target_generator import BeltTargetGenerator
from gait_assistance.gait.feature_extractor import compute_stride_metrics
from gait_assistance.gait.phase_detector import (
    BeltLengthPhaseDetector,
    PHASE_DETECTORS,
    GaitEvent,
    GaitPhase,
    PhaseDetector,
    PhaseResult,
    ScheduledPhaseDetector,
    create_phase_detector,
    register_phase_detector,
)
from gait_assistance.gait.stride_segmenter import StrideSegmenter
from gait_assistance.sensors.sensor_manager import SensorSample


def _run(
    config: PhaseConfig, duration_s: float = 6.0, rate_hz: float = 100.0
) -> Tuple[ScheduledPhaseDetector, List[Tuple[float, PhaseResult]]]:
    """Drive the scheduled detector over a stretch of timestamps."""
    detector = create_phase_detector(config)
    results: List[Tuple[float, PhaseResult]] = []
    for index in range(int(duration_s * rate_hz)):
        t = index / rate_hz
        results.append((t, detector.update(SensorSample.from_mapping({"timestamp": t}))))
    return detector, results


def _event_times(
    results: List[Tuple[float, PhaseResult]], event: GaitEvent
) -> List[float]:
    """Timestamps at which ``event`` was emitted."""
    return [t for t, r in results if r.event is event]


#: Sampling a continuous schedule quantises every event to the sample grid, so
#: an event is expected within one sample period of its scheduled instant.  The
#: extra half sample absorbs the float representation of the timestamps
#: themselves (``230 / 100.0`` is a hair below 2.3), which would otherwise make
#: the test about IEEE-754 rather than about the detector.
SAMPLE_S: float = 0.01
EVENT_TOL: float = 1.5 * SAMPLE_S


# --------------------------------------------------------------------------- #
# The scheduled placeholder
# --------------------------------------------------------------------------- #


def test_the_default_detector_reads_the_belt_not_a_clock() -> None:
    """Swing/stance comes from the belt length; the schedule is opt-in."""
    assert Config().phase.detector == "belt_cycle"
    assert isinstance(create_phase_detector(PhaseConfig()), BeltLengthPhaseDetector)


def test_the_scheduled_placeholder_is_still_selectable() -> None:
    """The fixed cadence stays available for pipeline runs without a patient."""
    assert isinstance(
        create_phase_detector(PhaseConfig(detector="scheduled")), ScheduledPhaseDetector
    )


def test_the_placeholder_declares_that_it_measures_nothing() -> None:
    """Reports must not present a configured cadence as an estimate."""
    assert ScheduledPhaseDetector.is_placeholder
    assert ScheduledPhaseDetector.phase_source == "scheduled"
    description = ScheduledPhaseDetector.describe()
    assert "PLACEHOLDER" in description
    assert "not measured" in description


def test_heel_strikes_land_on_the_configured_period() -> None:
    """One stride per ``scheduled_period_s``, on the clock."""
    config = PhaseConfig(detector="scheduled", scheduled_period_s=1.0)
    _, results = _run(config, duration_s=6.0)
    strikes = _event_times(results, GaitEvent.HEEL_STRIKE)
    assert strikes == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], abs=EVENT_TOL)


def test_toe_off_splits_the_cycle_at_the_configured_ratio() -> None:
    """Stance runs first, then swing, in the configured proportion."""
    config = PhaseConfig(detector="scheduled", scheduled_period_s=1.0, scheduled_swing_ratio=0.4)
    _, results = _run(config, duration_s=3.0)
    toe_offs = _event_times(results, GaitEvent.TOE_OFF)
    # stance = 0.6 s, so swing starts 0.6 s into every cycle
    assert toe_offs == pytest.approx([0.6, 1.6, 2.6], abs=EVENT_TOL)


def test_an_explicit_swing_duration_overrides_the_ratio() -> None:
    """The timing can be given as a duration instead of a fraction."""
    config = PhaseConfig(
        detector="scheduled",
        scheduled_period_s=1.0, scheduled_swing_ratio=0.4,
        scheduled_swing_duration_s=0.25,
    )
    assert config.swing_duration_s() == pytest.approx(0.25)
    assert config.stance_duration_s() == pytest.approx(0.75)
    _, results = _run(config, duration_s=3.0)
    assert _event_times(results, GaitEvent.TOE_OFF) == pytest.approx(
        [0.75, 1.75, 2.75], abs=EVENT_TOL
    )


def test_the_offset_shifts_the_whole_cycle() -> None:
    """``scheduled_offset_s`` moves the first heel strike."""
    config = PhaseConfig(detector="scheduled", scheduled_period_s=1.0, scheduled_offset_s=0.3)
    _, results = _run(config, duration_s=3.0)
    strikes = _event_times(results, GaitEvent.HEEL_STRIKE)
    # The run still opens a cycle on the first sample; the scheduled strikes
    # after it are shifted by the offset.
    assert strikes[1:] == pytest.approx([0.3, 1.3, 2.3], abs=EVENT_TOL)


def test_the_phase_fractions_match_the_configuration() -> None:
    """The time spent in swing is the configured fraction of the cycle."""
    config = PhaseConfig(detector="scheduled", scheduled_period_s=1.0, scheduled_swing_ratio=0.35)
    _, results = _run(config, duration_s=10.0)
    phases = [r.phase for _, r in results]
    swing_fraction = sum(p is GaitPhase.SWING for p in phases) / len(phases)
    assert swing_fraction == pytest.approx(0.35, abs=0.02)


def test_progress_sweeps_zero_to_one_within_each_phase() -> None:
    """Every phase reports a monotone progress the belt profile can use."""
    config = PhaseConfig(detector="scheduled", scheduled_period_s=1.0, scheduled_swing_ratio=0.4)
    _, results = _run(config, duration_s=2.0)
    swing = [r.phase_progress for _, r in results if r.phase is GaitPhase.SWING]
    assert swing and all(p is not None for p in swing)
    assert min(swing) == pytest.approx(0.0, abs=0.03)
    assert max(swing) == pytest.approx(1.0, abs=0.03)
    # Progress is always defined, so the belt target is never left guessing.
    assert all(r.phase_progress is not None for _, r in results)


def test_swing_never_swallows_the_whole_period() -> None:
    """A misconfigured swing longer than the cycle still leaves a stance."""
    config = PhaseConfig(detector="scheduled", scheduled_period_s=1.0, scheduled_swing_duration_s=5.0)
    assert config.swing_duration_s() < 1.0
    assert config.stance_duration_s() > 0.0
    _, results = _run(config, duration_s=3.0)
    assert any(r.phase is GaitPhase.STANCE for _, r in results)
    assert any(r.phase is GaitPhase.SWING for _, r in results)


def test_the_segmenter_cuts_one_stride_per_scheduled_period() -> None:
    """The scheduled heel strikes drive the stride segmentation."""
    config = PhaseConfig(detector="scheduled", scheduled_period_s=1.0)
    detector = create_phase_detector(config)
    segmenter = StrideSegmenter(StrideConfig())
    strides = []
    for index in range(600):
        t = index / 100.0
        sample = SensorSample.from_mapping({"timestamp": t, "belt_length": -130.0})
        completed = segmenter.update(sample, detector.update(sample))
        if completed is not None:
            strides.append(completed)
    assert len(strides) == 5
    for stride in strides:
        assert stride.duration_s == pytest.approx(1.0, abs=0.02)
        assert stride.phase_source == "scheduled"
        metrics = compute_stride_metrics(stride)
        assert metrics.phase_source == "scheduled"
        assert metrics.swing_ratio == pytest.approx(0.4, abs=0.03)


# --------------------------------------------------------------------------- #
# The seam for the real detector
# --------------------------------------------------------------------------- #


def test_a_new_detector_can_be_registered_and_selected() -> None:
    """The real detector plugs in without touching any existing module."""

    class _FsrPhaseDetector(PhaseDetector):
        """Stand-in for a future contact-driven detector."""

        phase_source = "fsr_contact"
        estimates_from = "foot-switch contact"
        ground_truth_validated = True
        validated_against = "FSR insole"

        def update(self, sample: SensorSample) -> PhaseResult:
            """Swing while the (fictional) contact channel is released."""
            contact = sample.belt_length < 0.0
            phase = GaitPhase.STANCE if contact else GaitPhase.SWING
            event = None
            if phase is not self._phase:
                event = (
                    GaitEvent.HEEL_STRIKE
                    if phase is GaitPhase.STANCE
                    else GaitEvent.TOE_OFF
                )
                if event is GaitEvent.HEEL_STRIKE:
                    self._register_heel_strike(sample.timestamp)
                self._enter(phase, sample.timestamp)
            return self._result(event, self._progress(sample.timestamp))

    name = "fsr_test_only"
    try:
        register_phase_detector(name, _FsrPhaseDetector)
        detector = create_phase_detector(PhaseConfig(detector=name))
        assert isinstance(detector, _FsrPhaseDetector)
        # A validated detector says so, and its provenance reaches the stride.
        assert "validated against FSR insole" in _FsrPhaseDetector.describe()
        result = detector.update(
            SensorSample.from_mapping({"timestamp": 0.0, "belt_length": 10.0})
        )
        assert result.phase is GaitPhase.SWING
        assert result.source == "fsr_contact"

        # Re-registering the same class is a no-op; a different one is refused
        # so an existing config cannot silently start selecting something else.
        register_phase_detector(name, _FsrPhaseDetector)
        with pytest.raises(ValueError):
            register_phase_detector(name, ScheduledPhaseDetector)
        with pytest.raises(TypeError):
            register_phase_detector("not_a_detector", int)  # type: ignore[arg-type]
    finally:
        PHASE_DETECTORS.pop(name, None)
        PHASE_DETECTORS.pop("not_a_detector", None)


def test_the_signal_driven_detectors_remain_available() -> None:
    """Swapping back to belt-derived detection is a configuration change."""
    for name in ("scheduled", "belt_velocity", "gyro"):
        assert name in PHASE_DETECTORS
    detector = create_phase_detector(PhaseConfig(detector="belt_velocity"))
    assert detector.phase_source == "belt_derived"
    assert not detector.is_placeholder


# --------------------------------------------------------------------------- #
# Assistance happens in swing only
# --------------------------------------------------------------------------- #


def test_no_assistance_is_produced_during_stance() -> None:
    """In stance the belt is held at the neutral length, whatever the gain."""
    generator = BeltTargetGenerator(AssistConfig())
    for progress in (0.0, 0.25, 0.5, 0.75, 1.0):
        target = generator.generate(
            baseline_belt_length=-130.0,
            assist_gain=0.8,
            phase=GaitPhase.STANCE,
            phase_progress=progress,
        )
        assert target.profile_value == 0.0
        assert target.target_belt_length == pytest.approx(-130.0)
        # The published gain is still reported: "held but not applied here" is
        # a different state from "no gain".
        assert target.assist_gain == pytest.approx(0.8)


def test_assistance_is_produced_during_swing() -> None:
    """In swing the belt retracts along the profile."""
    config = AssistConfig()
    generator = BeltTargetGenerator(config)
    peak = generator.generate(
        baseline_belt_length=-130.0, assist_gain=0.5,
        phase=GaitPhase.SWING, phase_progress=0.5,
    )
    assert peak.profile_value == pytest.approx(1.0)
    assert peak.target_belt_length == pytest.approx(
        -130.0 - 0.5 * config.max_retraction_mm
    )
    assert peak.target_belt_length < -130.0


def test_assistance_needs_a_known_swing_progress() -> None:
    """Without a progress value nothing is applied at a guessed phase."""
    generator = BeltTargetGenerator(AssistConfig())
    assert not BeltTargetGenerator.assists_in(GaitPhase.SWING, None)
    assert not BeltTargetGenerator.assists_in(GaitPhase.STANCE, 0.5)
    assert BeltTargetGenerator.assists_in(GaitPhase.SWING, 0.5)
    target = generator.generate(
        baseline_belt_length=-130.0, assist_gain=0.8,
        phase=GaitPhase.SWING, phase_progress=None,
    )
    assert target.profile_value == 0.0
    assert target.target_belt_length == pytest.approx(-130.0)


def test_over_a_scheduled_cycle_the_belt_only_moves_in_swing() -> None:
    """End to end: the retraction window lines up with the scheduled swing."""
    phase_config = PhaseConfig(detector="scheduled", scheduled_period_s=1.0, scheduled_swing_ratio=0.4)
    detector = create_phase_detector(phase_config)
    generator = BeltTargetGenerator(AssistConfig())

    retracted: List[float] = []
    neutral: List[float] = []
    for index in range(300):
        t = index / 100.0
        result = detector.update(SensorSample.from_mapping({"timestamp": t}))
        target = generator.generate(
            baseline_belt_length=-130.0, assist_gain=0.5,
            phase=result.phase, phase_progress=result.phase_progress, timestamp=t,
        )
        if result.phase is GaitPhase.SWING:
            retracted.append(target.target_belt_length)
        else:
            neutral.append(target.target_belt_length)

    assert neutral and all(v == pytest.approx(-130.0) for v in neutral)
    assert retracted and min(retracted) < -130.0
    # Stance never pulls: every stance sample sits exactly at the baseline.
    assert max(neutral) == pytest.approx(min(neutral))
