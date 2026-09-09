"""Tests for phase detection, segmentation and normalisation (spec 30)."""

from __future__ import annotations

from pathlib import Path
from typing import List

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
    """Strides run from one heel strike to the next."""
    detector = create_phase_detector(config.phase)
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
