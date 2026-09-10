"""Tests for phase detection, segmentation and normalisation (spec 30)."""

from __future__ import annotations

import math

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pytest

from gait_assistance.config import Config, PhaseConfig, StrideConfig
from gait_assistance.gait.feature_extractor import (
    FeatureExtractor,
    belt_excursion,
    compute_stride_metrics,
    peak_belt_velocity,
    stance_time,
    swing_ratio,
    swing_time,
    trunk_acc_rms,
    trunk_gyro_rms,
)
from gait_assistance.gait.normalization import ZScoreNormalizer
from gait_assistance.gait.phase_detector import (
    BeltLengthPhaseDetector,
    BeltVelocityPhaseDetector,
    GaitEvent,
    GaitPhase,
    GyroPhaseDetector,
    PhaseDetector,
    create_phase_detector,
)
from gait_assistance.gait.stride_segmenter import (
    Stride,
    StrideSegmenter,
    resample_gait_cycle,
    resample_stride,
)
from gait_assistance.sensors.sensor_manager import SensorSample

from .conftest import simulate_strides


def _sample(t: float, **kwargs: float) -> SensorSample:
    """Build a sensor sample with the given channel overrides."""
    values = {"timestamp": t}
    values.update(kwargs)
    return SensorSample.from_mapping(values)


def test_detector_factory_returns_the_configured_class() -> None:
    """The detector really is pluggable through the configuration."""
    assert isinstance(
        create_phase_detector(PhaseConfig(detector="belt_velocity")),
        BeltVelocityPhaseDetector,
    )
    assert isinstance(create_phase_detector(PhaseConfig(detector="gyro")), GyroPhaseDetector)
    assert isinstance(create_phase_detector(), PhaseDetector)
    with pytest.raises(ValueError):
        create_phase_detector(PhaseConfig(detector="nope"))


def test_belt_velocity_detector_state_machine() -> None:
    """Stance -> swing on belt extension, swing -> stance at heel strike."""
    detector = BeltVelocityPhaseDetector(
        PhaseConfig(min_belt_mm=-130.0, refractory_s=0.3, loading_vel=10.0, strike_vel=-5.0)
    )
    assert detector.update(_sample(0.0, belt_length=-150.0, belt_velocity=0.0)).phase is GaitPhase.STANCE

    toe_off = detector.update(_sample(0.1, belt_length=-150.0, belt_velocity=40.0))
    assert toe_off.phase is GaitPhase.SWING
    assert toe_off.event is GaitEvent.TOE_OFF

    strike = detector.update(_sample(0.6, belt_length=-150.0, belt_velocity=-30.0))
    assert strike.phase is GaitPhase.STANCE
    assert strike.heel_strike


def test_heel_strike_needs_the_belt_condition() -> None:
    """A reversal above the belt threshold is not a heel strike."""
    detector = BeltVelocityPhaseDetector(PhaseConfig(min_belt_mm=-130.0))
    detector.update(_sample(0.0, belt_length=-50.0, belt_velocity=40.0))
    result = detector.update(_sample(0.5, belt_length=-50.0, belt_velocity=-30.0))
    assert not result.heel_strike


def test_refractory_period_suppresses_double_detection() -> None:
    """Two reversals inside the refractory window yield a single heel strike."""
    detector = BeltVelocityPhaseDetector(PhaseConfig(min_belt_mm=-130.0, refractory_s=0.5))
    detector.update(_sample(0.0, belt_length=-150.0, belt_velocity=40.0))
    assert detector.update(_sample(0.2, belt_length=-150.0, belt_velocity=-30.0)).heel_strike
    detector.update(_sample(0.3, belt_length=-150.0, belt_velocity=40.0))
    assert not detector.update(_sample(0.4, belt_length=-150.0, belt_velocity=-30.0)).heel_strike


def test_gyro_detector_uses_the_angular_rate() -> None:
    """The alternative detector reacts to the gyroscope, not to the belt."""
    detector = GyroPhaseDetector(PhaseConfig(gyro_swing_threshold=40.0, gyro_strike_threshold=-20.0))
    assert detector.update(_sample(0.0, gyro_z=100.0)).phase is GaitPhase.SWING
    result = detector.update(_sample(0.5, gyro_z=-50.0))
    assert result.heel_strike and result.phase is GaitPhase.STANCE


def test_detector_reset_clears_the_state() -> None:
    """After a reset the detector behaves like a fresh instance."""
    detector = BeltVelocityPhaseDetector(PhaseConfig(min_belt_mm=-130.0))
    detector.update(_sample(0.0, belt_length=-150.0, belt_velocity=40.0))
    detector.reset()
    assert detector.phase is GaitPhase.STANCE
    assert detector.stride_period_s is None


def test_segmenter_cuts_at_heel_strikes(config: Config) -> None:
    """Strides run from one heel strike to the next.

    Driven through the belt-velocity detector: the waveform below holds the
    belt length constant, which is exactly the case the default belt-length
    detector refuses to call walking.
    """
    detector = create_phase_detector(PhaseConfig(detector="belt_velocity"))
    segmenter = StrideSegmenter(config.stride)
    strides = []
    for i in range(600):
        t = i * 0.01
        # 1.2 s cycle: extend for 0.7 s, retract for 0.5 s
        phase_t = t % 1.2
        velocity = 60.0 if phase_t < 0.7 else -60.0
        completed = segmenter.update(
            _sample(t, belt_length=-150.0, belt_velocity=velocity),
            detector.update(_sample(t, belt_length=-150.0, belt_velocity=velocity)),
        )
        if completed is not None:
            strides.append(completed)
    assert len(strides) >= 3
    for stride in strides:
        assert stride.duration_s == pytest.approx(1.2, abs=0.05)
        assert stride.n_samples > config.stride.min_samples


def test_segmenter_rejects_out_of_range_strides() -> None:
    """Strides that are too short or too long never reach the model."""
    stride = Stride(stride_id=0)
    for i in range(5):
        stride.samples.append(_sample(i * 0.01))
        stride.phases.append(GaitPhase.STANCE)
    ok, reason = stride.is_valid(StrideConfig())
    assert not ok and "too few samples" in reason


def test_stride_resampling_output_length(strides: List[Stride]) -> None:
    """Every stride is interpolated onto exactly ``n_cycle_points`` samples."""
    config = Config()
    extractor = FeatureExtractor(config.feature, config.stride)
    for stride in strides[:5]:
        matrix = extractor.extract(stride)
        assert matrix.shape == (config.stride.n_cycle_points, len(config.feature.channels))
        assert np.all(np.isfinite(matrix))


@pytest.mark.parametrize("n_points", [20, 50, 100, 256])
def test_resample_gait_cycle_lengths(n_points: int) -> None:
    """Resampling honours the requested number of points."""
    data = np.linspace(0.0, 1.0, 37).reshape(-1, 1) * np.array([[1.0, 2.0]])
    out = resample_gait_cycle(data, n_points)
    assert out.shape == (n_points, 2)
    assert out[0, 0] == pytest.approx(0.0)
    assert out[-1, 0] == pytest.approx(1.0)


def test_resample_preserves_a_linear_ramp() -> None:
    """Interpolation of a straight line is that same straight line."""
    ramp = np.linspace(0.0, 10.0, 13).reshape(-1, 1)
    out = resample_gait_cycle(ramp, 100)
    assert np.allclose(out[:, 0], np.linspace(0.0, 10.0, 100), atol=1e-9)


def test_resample_rejects_degenerate_input() -> None:
    """A single sample or a 1-D array cannot be resampled."""
    with pytest.raises(ValueError):
        resample_gait_cycle(np.zeros((1, 3)), 100)
    with pytest.raises(ValueError):
        resample_gait_cycle(np.zeros(10), 100)
    with pytest.raises(ValueError):
        resample_gait_cycle(np.zeros((10, 2)), 1)


def test_feature_extractor_rejects_unknown_channels() -> None:
    """A typo in the feature list fails loudly at construction time."""
    from gait_assistance.config import FeatureConfig

    with pytest.raises(ValueError):
        FeatureExtractor(FeatureConfig(channels=("belt_length", "not_a_channel")))


def test_feature_configuration_changes_the_matrix_width(strides: List[Stride]) -> None:
    """The feature set is configurable (spec 5)."""
    from gait_assistance.config import FeatureConfig

    stride = strides[0]
    two = FeatureExtractor(FeatureConfig(channels=("belt_length", "gyro_z"))).extract(stride)
    three = FeatureExtractor(
        FeatureConfig(channels=("belt_length", "gyro_z", "accel_y"))
    ).extract(stride)
    assert two.shape[1] == 2 and three.shape[1] == 3


def test_biomechanical_metrics(strides: List[Stride]) -> None:
    """The stride metrics are consistent with each other and non-negative."""
    stride = strides[0]
    metrics = compute_stride_metrics(stride)
    assert metrics.swing_time == pytest.approx(swing_time(stride))
    assert metrics.stance_time == pytest.approx(stance_time(stride))
    assert metrics.swing_ratio == pytest.approx(swing_ratio(stride))
    assert metrics.belt_excursion == pytest.approx(belt_excursion(stride))
    assert metrics.peak_belt_velocity == pytest.approx(peak_belt_velocity(stride))
    assert metrics.trunk_acc_rms == pytest.approx(trunk_acc_rms(stride))
    assert 0.0 <= metrics.swing_ratio <= 1.0
    assert metrics.belt_excursion >= 0.0
    assert metrics.trunk_acc_rms >= 0.0
    assert metrics.swing_time + metrics.stance_time == pytest.approx(
        stride.duration_s, abs=0.05
    )


def test_metrics_of_an_empty_stride_are_zero() -> None:
    """The metric functions tolerate a stride with no samples."""
    empty = Stride(stride_id=0)
    assert belt_excursion(empty) == 0.0
    assert peak_belt_velocity(empty) == 0.0
    # A stride with no IMU samples has no trunk motion to report; None keeps
    # "not measured" distinct from a measured, perfectly steady trunk.
    assert trunk_acc_rms(empty) is None
    assert trunk_gyro_rms(empty) is None
    assert swing_ratio(empty) == 0.0


def test_zscore_statistics_are_frozen(strides: List[Stride]) -> None:
    """Online strides reuse the baseline statistics (spec 6)."""
    config = Config()
    extractor = FeatureExtractor(config.feature, config.stride)
    matrices = [extractor.extract(s) for s in strides[:20]]
    normalizer = ZScoreNormalizer()
    stats = normalizer.fit(matrices, extractor.channels)

    with pytest.raises(RuntimeError):
        normalizer.fit(matrices)
    normalizer.fit(matrices, refit=True)  # explicit override is allowed

    online = extractor.extract(strides[21])
    transformed = normalizer.transform(online)
    assert np.allclose(transformed, (online - stats.mean) / stats.std)
    # the baseline itself is standardised by construction
    baseline_z = np.vstack(normalizer.transform_many(matrices))
    assert np.allclose(baseline_z.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(baseline_z.std(axis=0), 1.0, atol=1e-6)


def test_normalizer_roundtrip_and_errors(strides: List[Stride]) -> None:
    """Serialisation, inversion and shape checks behave as documented."""
    config = Config()
    extractor = FeatureExtractor(config.feature, config.stride)
    matrices = [extractor.extract(s) for s in strides[:20]]
    normalizer = ZScoreNormalizer()
    with pytest.raises(RuntimeError):
        normalizer.transform(matrices[0])
    normalizer.fit(matrices, extractor.channels)

    restored = ZScoreNormalizer.from_dict(normalizer.to_dict())
    assert np.allclose(restored.stats.mean, normalizer.stats.mean)
    assert np.allclose(
        normalizer.inverse_transform(normalizer.transform(matrices[0])), matrices[0], atol=1e-9
    )
    with pytest.raises(ValueError):
        normalizer.transform(np.zeros((100, 2)))
    with pytest.raises(ValueError):
        ZScoreNormalizer().fit([])


# --------------------------------------------------------------------------- #
# The belt-length detector (default): belt_length is the only channel it reads
# --------------------------------------------------------------------------- #


def _belt_walk(
    n: int = 900,
    rate_hz: float = 100.0,
    period_s: float = 1.2,
    amplitude_mm: float = 60.0,
    offset_mm: float = -130.0,
) -> List[Tuple[float, float]]:
    """Synthesise ``(timestamp, belt_length)`` for a steady belt cycle.

    The shape mirrors the recordings in ``csv/``: the belt extends through the
    swing and retracts around heel strike.
    """
    return [
        (
            i / rate_hz,
            offset_mm - amplitude_mm * math.cos(2.0 * math.pi * (i / rate_hz) / period_s),
        )
        for i in range(n)
    ]


def _run_belt_length(
    walk: List[Tuple[float, float]], config: Optional[PhaseConfig] = None, **channels: float
) -> Tuple[BeltLengthPhaseDetector, List[float], List[float]]:
    """Drive the belt-length detector and collect its event times."""
    detector = BeltLengthPhaseDetector(config or PhaseConfig())
    strikes: List[float] = []
    toe_offs: List[float] = []
    for t, belt in walk:
        result = detector.update(_sample(t, belt_length=belt, **channels))
        if result.heel_strike:
            strikes.append(t)
        elif result.toe_off:
            toe_offs.append(t)
    return detector, strikes, toe_offs


def test_the_belt_is_the_only_signal_the_default_detector_reads() -> None:
    """The default takes the stride boundary from the belt and nothing else."""
    assert Config().phase.detector == "belt_cycle"
    detector = create_phase_detector()
    assert isinstance(detector, BeltLengthPhaseDetector)   # belt_cycle subclasses it
    assert detector.phase_source == "belt_cycle"
    assert BeltLengthPhaseDetector.phase_source == "belt_length_derived"
    assert not BeltLengthPhaseDetector.is_placeholder
    assert not BeltLengthPhaseDetector.ground_truth_validated


def test_belt_length_detector_finds_one_stride_per_cycle() -> None:
    """A 1.2 s belt cycle yields heel strikes 1.2 s apart."""
    _detector, strikes, toe_offs = _run_belt_length(_belt_walk(period_s=1.2))
    assert len(strikes) >= 5
    intervals = np.diff(strikes)
    # The amplitude envelope needs about one time constant to settle, so the
    # first strides trip the falling threshold slightly early; the cadence is
    # exact once it has.
    assert np.allclose(intervals, 1.2, atol=0.05)
    assert np.allclose(intervals[2:], 1.2, atol=0.02)
    # exactly one toe-off between consecutive heel strikes
    for start, end in zip(strikes, strikes[1:]):
        assert sum(1 for x in toe_offs if start < x < end) == 1


def test_belt_length_detector_reads_nothing_but_the_belt() -> None:
    """Velocity, IMU and motor channels cannot change a single event."""
    walk = _belt_walk()
    _d1, strikes_a, toe_a = _run_belt_length(walk)
    _d2, strikes_b, toe_b = _run_belt_length(
        walk,
        belt_velocity=-9999.0, gyro_z=5000.0, gyro_x=-5000.0,
        accel_x=9.0, accel_y=-9.0, motor_current=7.0, motor_position=1234.0,
    )
    assert strikes_a == strikes_b
    assert toe_a == toe_b
    assert strikes_a  # and it did detect something


def test_belt_length_detector_is_immune_to_a_resting_length_offset() -> None:
    """The same gait at a different harness setting gives the same events."""
    base = _belt_walk(offset_mm=-130.0)
    shifted = [(t, b + 500.0) for t, b in base]
    _d1, strikes_a, _ = _run_belt_length(base)
    _d2, strikes_b, _ = _run_belt_length(shifted)
    assert strikes_a == strikes_b


def test_belt_length_detector_is_immune_to_amplitude_scaling() -> None:
    """A larger stride triggers at the same instants, not more often."""
    _d1, strikes_a, _ = _run_belt_length(_belt_walk(amplitude_mm=40.0))
    _d2, strikes_b, _ = _run_belt_length(_belt_walk(amplitude_mm=160.0))
    assert strikes_a == strikes_b


def test_belt_length_detector_tracks_a_drifting_baseline() -> None:
    """A slowly settling harness does not add or drop a stride."""
    walk = [(t, b + 30.0 * t / 9.0) for t, b in _belt_walk()]  # +30 mm over the run
    _detector, strikes, _ = _run_belt_length(walk)
    assert len(strikes) >= 5
    assert np.allclose(np.diff(strikes), 1.2, atol=0.05)


def test_standing_still_produces_no_strides() -> None:
    """Below the amplitude floor there is no gait, so no events at all."""
    rng = np.random.default_rng(0)
    still = [(i / 100.0, -130.0 + float(rng.normal(0.0, 0.3))) for i in range(900)]
    detector, strikes, toe_offs = _run_belt_length(still)
    assert strikes == [] and toe_offs == []
    assert not detector.is_walking
    assert detector.amplitude_mm < PhaseConfig().belt_min_amplitude_mm


def test_the_amplitude_floor_is_what_gates_the_events() -> None:
    """A belt swing just under and just over the floor separates the two cases."""
    config = PhaseConfig(belt_min_amplitude_mm=20.0)
    _d1, quiet, _ = _run_belt_length(_belt_walk(amplitude_mm=10.0), config)
    _d2, walking, _ = _run_belt_length(_belt_walk(amplitude_mm=120.0), config)
    assert quiet == []
    assert len(walking) >= 5


def test_belt_length_refractory_suppresses_a_double_strike() -> None:
    """A cycle shorter than the refractory period cannot produce two strikes."""
    walk = _belt_walk(n=600, period_s=0.4)
    _detector, strikes, _ = _run_belt_length(walk, PhaseConfig(refractory_s=1.0))
    assert all(b - a >= 1.0 for a, b in zip(strikes, strikes[1:]))


def test_belt_length_detector_survives_a_non_finite_reading() -> None:
    """A NaN frame is skipped; the envelope and the phase stay usable."""
    detector = BeltLengthPhaseDetector(PhaseConfig())
    for t, belt in _belt_walk(n=300):
        detector.update(_sample(t, belt_length=belt))
    baseline, amplitude, phase = detector.baseline_mm, detector.amplitude_mm, detector.phase
    result = detector.update(_sample(3.0, belt_length=float("nan")))
    assert result.event is None
    assert result.phase is phase
    assert detector.baseline_mm == baseline
    assert detector.amplitude_mm == amplitude


def test_belt_length_detector_reset_clears_the_envelope() -> None:
    """After a reset the detector is indistinguishable from a fresh one."""
    detector = BeltLengthPhaseDetector(PhaseConfig())
    for t, belt in _belt_walk(n=300):
        detector.update(_sample(t, belt_length=belt))
    assert detector.baseline_mm is not None
    detector.reset()
    assert detector.baseline_mm is None
    assert detector.amplitude_mm == 0.0
    assert detector.phase is GaitPhase.STANCE
    assert detector.stride_period_s is None


def test_belt_length_detector_reports_phase_progress() -> None:
    """The belt target needs progress inside the swing, so it must be filled."""
    detector = BeltLengthPhaseDetector(PhaseConfig())
    progress: List[float] = []
    for t, belt in _belt_walk():
        result = detector.update(_sample(t, belt_length=belt))
        if result.phase is GaitPhase.SWING and result.phase_progress is not None:
            progress.append(result.phase_progress)
    assert progress
    assert min(progress) >= 0.0 and max(progress) <= 1.0
    assert max(progress) > 0.8  # the profile really does sweep the swing


def test_belt_length_detector_populates_the_stride_statistics() -> None:
    """Stride period and swing duration come out of the inherited helpers."""
    detector, strikes, _ = _run_belt_length(_belt_walk(period_s=1.2))
    assert detector.stride_period_s == pytest.approx(1.2, abs=0.05)
    assert detector.swing_duration_s is not None
    assert 0.0 < detector.swing_duration_s < 1.2


def test_segmenter_cuts_strides_from_the_belt_length_alone() -> None:
    """End to end: the default detector drives the stride segmentation."""
    config = Config()
    detector = create_phase_detector(config.phase)
    segmenter = StrideSegmenter(config.stride)
    strides = []
    for t, belt in _belt_walk(n=1200):
        sample = _sample(t, belt_length=belt)
        completed = segmenter.update(sample, detector.update(sample))
        if completed is not None:
            strides.append(completed)
    assert len(strides) >= 5
    for stride in strides:
        assert stride.duration_s == pytest.approx(1.2, abs=0.05)
        assert stride.phase_source == "belt_cycle"
