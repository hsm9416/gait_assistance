"""Tests for the healthy region, the metric intervals and the assist policy.

The behaviours pinned here are the ones the algorithm is built around: a
reference is a *region* with a zero-error interior, an unmeasured quantity is
never a measured zero, and a Log-Euclidean deviation cannot command the motor
on its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest

from gait_assistance.config import (
    AssistConfig,
    Config,
    DeviationConfig,
    OodAssistPolicy,
    ReferenceConfig,
    TargetMode,
    ThresholdMethod,
)
from gait_assistance.control.assistance_policy import (
    SOURCE_HEALTHY,
    AssistancePolicy,
    BiomechanicalEvaluator,
    DecisionState,
    PersistenceGate,
    interval_error,
)
from gait_assistance.gait.feature_extractor import (
    GaitMetrics,
    collect_metric_samples,
    compute_stride_metrics,
    stance_ratio,
    swing_ratio,
    swing_stance_ratio,
)
from gait_assistance.gait.symmetry import SideTiming, compute_symmetry
from gait_assistance.manifold.log_euclidean import matrix_log
from gait_assistance.manifold.reference import (
    HealthyReference,
    MetricRange,
    build_healthy_reference,
    build_metric_ranges,
    distance_threshold,
)
from gait_assistance.patient.baseline import build_baseline_from_strides
from gait_assistance.patient.patient_model import PatientModel, StrideAnalysis

from .conftest import make_spd, simulate_strides


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _healthy_logs(n: int = 15, size: int = 4) -> List[np.ndarray]:
    """A small set of healthy log-SPD matrices.

    ``size`` must match the number of feature channels when the reference is
    handed to a :class:`PatientModel`; the default 4 is fine for the pure
    reference tests.
    """
    return [matrix_log(make_spd(size, seed=s)) for s in range(n)]


def _metrics(
    swing: float = 0.38,
    excursion: float = 100.0,
    *,
    symmetry: Optional[float] = None,
    trunk: Optional[float] = None,
) -> GaitMetrics:
    """Stride metrics with the device-actionable fields set."""
    stance = 1.0 - swing
    return GaitMetrics(
        stride_time=1.2,
        swing_time=1.2 * swing,
        stance_time=1.2 * stance,
        swing_ratio=swing,
        stance_ratio=stance,
        swing_stance_ratio=swing / stance,
        belt_excursion=excursion,
        peak_belt_velocity=400.0,
        swing_symmetry_ratio=symmetry,
        trunk_acc_rms=trunk,
    )


def _analysis(
    *, manifold_deviation: float = 0.0, ood: bool = False, valid: bool = True
) -> StrideAnalysis:
    """A stride analysis carrying a chosen manifold context score."""
    return StrideAnalysis(
        stride_id=0,
        timestamp=0.0,
        valid=valid,
        gait_state=0,
        state_label="S0",
        confidence=0.8,
        nearest_distance=0.5,
        runner_up_distance=1.0,
        d_patient=0.5,
        d_healthy=1.0,
        d_target=1.0,
        is_ood=ood,
        manifold_deviation=manifold_deviation,
        healthy_region_threshold=0.9,
    )


def _policy(
    ranges: Optional[dict] = None, config: Optional[Config] = None
) -> AssistancePolicy:
    """A policy whose reference intervals are fixed for the test."""
    cfg = config or Config()
    default = {
        "swing_ratio": MetricRange(lower=0.30, median=0.38, upper=0.45, scale=0.10),
        "belt_excursion": MetricRange(lower=90.0, median=110.0, upper=130.0, scale=20.0),
    }
    evaluator = BiomechanicalEvaluator(
        cfg.deviation, ranges if ranges is not None else default, SOURCE_HEALTHY
    )
    return AssistancePolicy(cfg, evaluator=evaluator)


# --------------------------------------------------------------------------- #
# Healthy region (spec 2)
# --------------------------------------------------------------------------- #


def test_distance_to_the_healthy_centroid_is_zero_at_the_centroid() -> None:
    """The centroid sits at distance 0 from itself and inside its own region."""
    reference = build_healthy_reference(_healthy_logs())
    assert reference.distance(reference.log_centroid) == pytest.approx(0.0)
    assert reference.is_healthy_like(reference.log_centroid)
    assert reference.region_excess(reference.log_centroid) == 0.0


def test_manifold_deviation_is_zero_inside_the_healthy_region() -> None:
    """Anywhere inside the region scores 0: the centroid is not a target."""
    reference = build_healthy_reference(_healthy_logs())
    threshold = reference.manifold_distance_threshold
    assert threshold > 0.0

    direction = np.eye(reference.log_centroid.shape[0])
    direction = direction / np.linalg.norm(direction)
    # Just inside the boundary, but far from the centroid.
    inside = reference.log_centroid + 0.9 * threshold * direction
    assert reference.distance(inside) > 0.0
    assert reference.manifold_deviation(inside) == 0.0


def test_manifold_deviation_is_positive_outside_the_healthy_region() -> None:
    """Outside the boundary the deviation grows with the excess distance."""
    reference = build_healthy_reference(_healthy_logs())
    threshold = reference.manifold_distance_threshold
    direction = np.eye(reference.log_centroid.shape[0])
    direction = direction / np.linalg.norm(direction)

    near = reference.log_centroid + 1.1 * threshold * direction
    far = reference.log_centroid + 4.0 * threshold * direction
    assert reference.manifold_deviation(near) > 0.0
    assert reference.manifold_deviation(far) > reference.manifold_deviation(near)
    assert reference.manifold_deviation(far) <= 1.0


@pytest.mark.parametrize(
    "method", [ThresholdMethod.PERCENTILE, ThresholdMethod.MEAN_STD, ThresholdMethod.MEDIAN_MAD]
)
def test_every_threshold_method_produces_a_usable_boundary(
    method: ThresholdMethod,
) -> None:
    """All three configured methods return a positive, finite boundary."""
    distances = np.array([0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.9])
    value = distance_threshold(distances, ReferenceConfig(threshold_method=method))
    assert np.isfinite(value) and value > 0.0


def test_percentile_is_the_default_and_resists_a_single_outlier() -> None:
    """The default 95th percentile is not dragged by one extreme stride."""
    clean = np.array([0.1, 0.2, 0.25, 0.3, 0.35, 0.4])
    contaminated = np.append(clean, 50.0)
    config = ReferenceConfig()
    assert ThresholdMethod(config.threshold_method) is ThresholdMethod.PERCENTILE
    mean_std = ReferenceConfig(threshold_method=ThresholdMethod.MEAN_STD)
    # The mean/std boundary moves far more than the percentile one.
    shift_percentile = distance_threshold(contaminated, config) - distance_threshold(
        clean, config
    )
    shift_mean_std = distance_threshold(contaminated, mean_std) - distance_threshold(
        clean, mean_std
    )
    assert shift_percentile < shift_mean_std


def test_healthy_reference_roundtrip_preserves_the_region(tmp_path: Path) -> None:
    """Region, intervals and metadata survive a save/load cycle."""
    reference = build_healthy_reference(
        _healthy_logs(),
        metric_samples={"swing_ratio": [0.30 + 0.01 * i for i in range(15)]},
        metrics={"swing_ratio": 0.37},
    )
    path = tmp_path / "healthy.npz"
    reference.save(path)
    loaded = HealthyReference.load(path)

    assert np.allclose(loaded.log_centroid, reference.log_centroid)
    assert loaded.manifold_distance_threshold == pytest.approx(
        reference.manifold_distance_threshold
    )
    assert loaded.manifold_distance_scale == pytest.approx(
        reference.manifold_distance_scale
    )
    assert loaded.num_strides == reference.num_strides
    original = reference.get_metric_range("swing_ratio")
    restored = loaded.get_metric_range("swing_ratio")
    assert original is not None and restored is not None
    assert restored.lower == pytest.approx(original.lower)
    assert restored.upper == pytest.approx(original.upper)


def test_metric_range_lookup_accepts_a_walking_speed() -> None:
    """The speed-conditioned interface exists and falls back to the pooled range."""
    reference = build_healthy_reference(
        _healthy_logs(), metric_samples={"swing_ratio": [0.3, 0.35, 0.4]}
    )
    assert not reference.speed_conditioned
    pooled = reference.get_metric_range("swing_ratio")
    assert pooled is not None
    assert reference.get_metric_range("swing_ratio", walking_speed=0.4) == pooled
    assert reference.get_metric_range("swing_ratio", walking_speed=1.2) == pooled
    assert reference.get_metric_range("not_measured") is None


# --------------------------------------------------------------------------- #
# Metric ranges (spec 6, 7)
# --------------------------------------------------------------------------- #


def test_metric_ranges_use_the_configured_percentiles() -> None:
    """The interval is the 5th-95th percentile band by default."""
    values = [float(v) for v in range(101)]
    ranges = build_metric_ranges({"swing_ratio": values})
    interval = ranges["swing_ratio"]
    assert interval.lower == pytest.approx(5.0)
    assert interval.upper == pytest.approx(95.0)
    assert interval.median == pytest.approx(50.0)

    wider = build_metric_ranges(
        {"swing_ratio": values},
        ReferenceConfig(metric_lower_percentile=1.0, metric_upper_percentile=99.0),
    )["swing_ratio"]
    assert wider.lower < interval.lower and wider.upper > interval.upper


def test_error_is_zero_inside_the_reference_interval() -> None:
    """Any value inside the healthy band is not something to correct."""
    interval = MetricRange(lower=0.30, median=0.38, upper=0.45, scale=0.10)
    for value in (0.30, 0.34, 0.38, 0.45):
        assert interval_error(value, interval, direction="two_sided") == 0.0
        assert interval_error(value, interval, direction="deficit") == 0.0


def test_error_below_the_lower_bound_is_a_deficit() -> None:
    """Falling short of the interval produces a positive deficit."""
    interval = MetricRange(lower=0.30, median=0.38, upper=0.45, scale=0.10)
    assert interval_error(0.25, interval, direction="deficit") == pytest.approx(0.5)
    assert interval_error(0.20, interval, direction="deficit") == pytest.approx(1.0)
    assert interval_error(0.10, interval, direction="deficit") == 1.0


def test_a_surplus_is_not_a_deficit() -> None:
    """A directional term ignores the side the device cannot act on."""
    interval = MetricRange(lower=90.0, median=110.0, upper=130.0, scale=20.0)
    assert interval_error(160.0, interval, direction="deficit") == 0.0
    assert interval_error(160.0, interval, direction="excess") == pytest.approx(1.0)
    assert interval_error(160.0, interval, direction="two_sided") == pytest.approx(1.0)
    assert interval_error(60.0, interval, direction="deficit") == pytest.approx(1.0)


def test_an_unmeasured_value_has_no_error() -> None:
    """A missing measurement or a missing interval leaves the error undefined."""
    interval = MetricRange(lower=0.30, median=0.38, upper=0.45, scale=0.10)
    assert interval_error(None, interval) is None
    assert interval_error(0.35, None) is None
    assert interval_error(float("nan"), interval) is None


# --------------------------------------------------------------------------- #
# Swing / stance metrics (spec 4)
# --------------------------------------------------------------------------- #


def test_swing_and_stance_partition_the_stride() -> None:
    """The two phase times add up to the stride and the ratios to 1."""
    config = Config()
    strides = simulate_strides(config, 40.0)
    assert strides
    for stride in strides[:8]:
        metrics = compute_stride_metrics(stride)
        assert metrics.swing_time + metrics.stance_time == pytest.approx(
            metrics.stride_time, rel=1e-6, abs=1e-6
        )
        assert metrics.swing_ratio + metrics.stance_ratio == pytest.approx(1.0)
        assert metrics.swing_ratio == pytest.approx(swing_ratio(stride))
        assert metrics.stance_ratio == pytest.approx(stance_ratio(stride))


def test_swing_stance_ratio_is_the_quotient_of_the_two_times() -> None:
    """``swing_stance_ratio`` divides the two durations, not the stride."""
    config = Config()
    stride = simulate_strides(config, 40.0)[0]
    metrics = compute_stride_metrics(stride)
    assert metrics.swing_stance_ratio == pytest.approx(
        metrics.swing_time / metrics.stance_time
    )
    assert metrics.swing_stance_ratio == pytest.approx(swing_stance_ratio(stride))
    # The metrics name the detector that actually labelled the phases, so a
    # later run with the real detector is distinguishable in the same log.
    assert metrics.phase_source == "scheduled"


def test_metric_samples_skip_what_was_not_measured() -> None:
    """A metric no stride carried is absent, not present as zeros."""
    measured = _metrics(symmetry=None, trunk=None)
    samples = collect_metric_samples([measured, measured])
    assert "swing_ratio" in samples and len(samples["swing_ratio"]) == 2
    assert "swing_symmetry_ratio" not in samples
    assert "trunk_acc_rms" not in samples
    assert "phase_source" not in samples


# --------------------------------------------------------------------------- #
# Symmetry (spec 5)
# --------------------------------------------------------------------------- #


def test_identical_sides_are_perfectly_symmetric() -> None:
    """Equal timings on both sides give ratios of exactly 1."""
    side = SideTiming(swing_time=0.4, stance_time=0.8)
    symmetry = compute_symmetry(side, side)
    assert symmetry is not None
    assert symmetry.swing_symmetry_ratio == pytest.approx(1.0)
    assert symmetry.stance_symmetry_ratio == pytest.approx(1.0)
    assert symmetry.swing_stance_symmetry_ratio == pytest.approx(1.0)
    assert symmetry.ss_ratio_paretic == pytest.approx(0.5)
    assert symmetry.asymmetry == pytest.approx(0.0)


def test_asymmetric_sides_report_the_paretic_over_nonparetic_ratio() -> None:
    """The ratio is paretic / non-paretic, in that order."""
    paretic = SideTiming(swing_time=0.3, stance_time=0.9)
    nonparetic = SideTiming(swing_time=0.6, stance_time=0.6)
    symmetry = compute_symmetry(paretic, nonparetic)
    assert symmetry is not None
    assert symmetry.swing_symmetry_ratio == pytest.approx(0.5)
    assert symmetry.stance_symmetry_ratio == pytest.approx(1.5)
    assert symmetry.swing_stance_symmetry_ratio == pytest.approx((0.3 / 0.9) / 1.0)


def test_a_missing_contralateral_side_yields_no_symmetry() -> None:
    """Without the other side nothing is computed and nothing is imputed."""
    paretic = SideTiming(swing_time=0.4, stance_time=0.8)
    assert compute_symmetry(paretic, None) is None
    assert compute_symmetry(paretic, SideTiming(swing_time=0.0, stance_time=0.8)) is None

    config = Config()
    stride = simulate_strides(config, 40.0)[0]
    metrics = compute_stride_metrics(stride)
    assert metrics.swing_symmetry_ratio is None
    assert metrics.stance_symmetry_ratio is None
    assert metrics.swing_stance_symmetry_ratio is None
    assert not metrics.has_symmetry


def test_symmetry_is_computed_when_the_other_side_is_supplied() -> None:
    """Given a contralateral timing the metrics appear, still un-imputed."""
    config = Config()
    stride = simulate_strides(config, 40.0)[0]
    other = SideTiming(swing_time=0.5, stance_time=0.7)
    metrics = compute_stride_metrics(stride, contralateral=other)
    assert metrics.has_symmetry
    assert metrics.swing_symmetry_ratio == pytest.approx(metrics.swing_time / 0.5)


# --------------------------------------------------------------------------- #
# Assistance policy (spec 9, 10, 11)
# --------------------------------------------------------------------------- #


def test_no_deviation_at_all_produces_no_assistance() -> None:
    """``E_R = 0`` and ``E_B = 0`` is IN_RANGE and a gain of zero."""
    policy = _policy()
    assessment = policy.assess(_analysis(manifold_deviation=0.0), _metrics())
    assert assessment.state is DecisionState.IN_RANGE
    assert assessment.biomechanical_deviation == 0.0
    assert assessment.manifold_deviation == 0.0
    assert assessment.raw_assist_gain == 0.0
    assert assessment.limited_assist_gain == 0.0


def test_a_manifold_deviation_alone_never_assists() -> None:
    """``E_R > 0`` with ``E_B = 0`` is watched, not assisted."""
    policy = _policy()
    for _ in range(6):
        assessment = policy.assess(_analysis(manifold_deviation=1.0), _metrics())
    assert assessment.state is DecisionState.MANIFOLD_DEVIATION_ONLY
    assert assessment.manifold_deviation == 1.0
    assert assessment.biomechanical_deviation == 0.0
    assert assessment.raw_assist_gain == 0.0
    assert assessment.limited_assist_gain == 0.0


def test_a_biomechanical_deficit_alone_produces_base_assistance() -> None:
    """``E_B > 0`` with ``E_R = 0`` assists at the base factor."""
    config = Config()
    policy = _policy(config=config)
    deficient = _metrics(swing=0.20)          # a full-scale swing deficit
    for _ in range(config.assist.required_consecutive_deficit_strides):
        assessment = policy.assess(_analysis(manifold_deviation=0.0), deficient)
    assert assessment.state is DecisionState.BIOMECH_DEFICIT_ONLY
    assert assessment.biomechanical_deviation > 0.0
    expected = assessment.biomechanical_deviation * config.deviation.base_factor
    assert assessment.assist_score == pytest.approx(expected)
    assert assessment.limited_assist_gain > 0.0


def test_a_manifold_deviation_amplifies_an_existing_deficit() -> None:
    """The same deficit assists harder when the whole pattern is deviant too."""
    config = Config()
    deficient = _metrics(swing=0.20)

    calm = _policy(config=config)
    severe = _policy(config=config)
    for _ in range(config.assist.required_consecutive_deficit_strides):
        calm_assessment = calm.assess(_analysis(manifold_deviation=0.0), deficient)
        severe_assessment = severe.assess(_analysis(manifold_deviation=1.0), deficient)

    assert calm_assessment.state is DecisionState.BIOMECH_DEFICIT_ONLY
    assert severe_assessment.state is DecisionState.COMBINED_DEVIATION
    assert severe_assessment.biomechanical_deviation == pytest.approx(
        calm_assessment.biomechanical_deviation
    )
    assert severe_assessment.raw_assist_gain > calm_assessment.raw_assist_gain
    expected = deficient_score = calm_assessment.biomechanical_deviation * (
        config.deviation.base_factor + config.deviation.manifold_factor * 1.0
    )
    assert severe_assessment.assist_score == pytest.approx(expected)
    assert deficient_score > 0.0


def test_an_out_of_distribution_stride_refuses_aggressive_assistance() -> None:
    """OOD holds the gain instead of acting on a model that does not apply."""
    config = Config()
    policy = _policy(config=config)
    deficient = _metrics(swing=0.20)
    for _ in range(config.assist.required_consecutive_deficit_strides):
        policy.assess(_analysis(manifold_deviation=1.0), deficient)
    established = policy.gain
    assert established > 0.0

    assessment = policy.assess(_analysis(manifold_deviation=1.0, ood=True), deficient)
    assert assessment.state is DecisionState.OOD
    assert assessment.ood
    # The raw score still says "assist harder"; the OOD rule refuses to.
    assert assessment.raw_assist_gain > 0.0
    assert assessment.limited_assist_gain <= established + 1e-12
    assert assessment.limited_assist_gain <= config.assist.ood_max_gain + 1e-12


def test_the_safe_minimum_ood_policy_falls_back_instead_of_holding() -> None:
    """``safe_minimum`` releases the gain towards the configured floor."""
    config = Config()
    config.assist.ood_policy = OodAssistPolicy.SAFE_MINIMUM
    config.assist.ood_safe_gain = 0.0
    policy = _policy(config=config)
    deficient = _metrics(swing=0.20)
    for _ in range(6):
        policy.assess(_analysis(manifold_deviation=1.0), deficient)
    established = policy.gain

    assessment = policy.assess(_analysis(manifold_deviation=1.0, ood=True), deficient)
    assert assessment.limited_assist_gain < established


def test_an_unmeasured_term_lowers_the_deficit_rather_than_inventing_one() -> None:
    """A term with no reference interval is undefined and contributes nothing."""
    policy = _policy(ranges={})
    assessment = policy.assess(_analysis(manifold_deviation=1.0), _metrics(swing=0.05))
    assert assessment.swing_ratio_error is None
    assert assessment.belt_excursion_error is None
    assert assessment.biomechanical_deviation == 0.0
    assert assessment.limited_assist_gain == 0.0


def test_weights_decide_which_metrics_can_command_the_device() -> None:
    """A zero-weighted metric is still measured and reported, never actioned."""
    config = Config()
    config.deviation.biomech_weights = {
        "swing_ratio": 0.0,
        "belt_excursion": 1.0,
        "temporal_symmetry": 0.0,
        "trunk_compensation": 0.0,
    }
    policy = _policy(config=config)
    assessment = policy.assess(
        _analysis(manifold_deviation=0.0), _metrics(swing=0.10, excursion=100.0)
    )
    assert assessment.swing_ratio_error == pytest.approx(1.0)   # measured
    assert assessment.biomechanical_deviation == 0.0            # not actioned
    assert assessment.limited_assist_gain == 0.0


def test_the_gain_never_leaves_its_bounds_or_moves_too_fast() -> None:
    """The rate limit still holds under the new score."""
    config = Config()
    policy = _policy(config=config)
    previous = 0.0
    for index in range(30):
        metrics = _metrics(swing=0.05 if index % 2 == 0 else 0.38)
        assessment = policy.assess(_analysis(manifold_deviation=1.0), metrics)
        gain = assessment.limited_assist_gain
        assert 0.0 <= gain <= config.assist.max_gain
        assert abs(gain - previous) <= config.assist.max_gain_delta + 1e-12
        previous = gain


# --------------------------------------------------------------------------- #
# Persistence (spec 13)
# --------------------------------------------------------------------------- #


def test_a_single_anomalous_stride_does_not_raise_the_gain() -> None:
    """One deficit stride is noise until it repeats."""
    config = Config()
    policy = _policy(config=config)
    policy.assess(_analysis(manifold_deviation=0.0), _metrics())          # in range
    assessment = policy.assess(_analysis(manifold_deviation=0.0), _metrics(swing=0.20))
    assert assessment.state is DecisionState.BIOMECH_DEFICIT_ONLY
    assert assessment.raw_assist_gain > 0.0
    assert assessment.limited_assist_gain == 0.0
    assert assessment.consecutive_deficit_strides == 1


def test_the_gain_rises_once_the_deficit_persists() -> None:
    """The configured number of consecutive deficit strides opens the throttle."""
    config = Config()
    required = config.assist.required_consecutive_deficit_strides
    policy = _policy(config=config)
    deficient = _metrics(swing=0.20)

    gains = []
    for _ in range(required):
        gains.append(policy.assess(_analysis(), deficient).limited_assist_gain)
    assert gains[:-1] == [0.0] * (required - 1)
    assert gains[-1] > 0.0


def test_the_gain_falls_only_after_a_sustained_recovery() -> None:
    """A single good stride does not release the assistance."""
    config = Config()
    policy = _policy(config=config)
    deficient = _metrics(swing=0.20)
    for _ in range(6):
        policy.assess(_analysis(), deficient)
    assisted = policy.gain
    assert assisted > 0.0

    recovered = _metrics(swing=0.38)
    first = policy.assess(_analysis(), recovered)
    assert first.limited_assist_gain == pytest.approx(assisted)
    assert first.consecutive_recovery_strides == 1

    for _ in range(config.assist.required_consecutive_recovery_strides - 1):
        last = policy.assess(_analysis(), recovered)
    assert last.limited_assist_gain < assisted


def test_persistence_gate_streaks_are_mutually_exclusive() -> None:
    """Observing one kind of stride clears the opposite streak."""
    gate = PersistenceGate(AssistConfig())
    gate.update(True)
    gate.update(True)
    assert gate.deficit_streak == 2 and gate.recovery_streak == 0
    gate.update(False)
    assert gate.deficit_streak == 0 and gate.recovery_streak == 1
    assert gate.allow(0.5, 0.0) == 0.0          # may not increase yet
    gate.update(True)
    gate.update(True)
    gate.update(True)
    assert gate.allow(0.5, 0.0) == 0.5          # streak long enough


# --------------------------------------------------------------------------- #
# Target modes (spec 3)
# --------------------------------------------------------------------------- #


def test_healthy_region_is_the_default_target_mode() -> None:
    """With a healthy reference loaded the region is what the model targets."""
    config = Config()
    assert TargetMode(config.deviation.target_mode) is TargetMode.HEALTHY_REGION
    strides = simulate_strides(config, 60.0)
    baseline = build_baseline_from_strides(
        strides[: config.patient.baseline_strides], config
    )
    healthy = build_healthy_reference(_healthy_logs(12, size=5))
    model = PatientModel.build(config, baseline, healthy)
    assert model.target_mode is TargetMode.HEALTHY_REGION
    assert np.allclose(model.z_target, healthy.log_centroid)
    assert model.manifold_deviation(healthy.log_centroid) == 0.0


def test_interpolated_target_remains_available() -> None:
    """The compatibility mode still interpolates and has no zero-error region."""
    config = Config()
    config.deviation.target_mode = TargetMode.INTERPOLATED_TARGET
    config.patient.alpha = 0.3
    strides = simulate_strides(config, 60.0)
    baseline = build_baseline_from_strides(
        strides[: config.patient.baseline_strides], config
    )
    healthy = build_healthy_reference(_healthy_logs(12, size=5))
    model = PatientModel.build(config, baseline, healthy)
    assert model.target_mode is TargetMode.INTERPOLATED_TARGET
    expected = 0.7 * model.z_patient + 0.3 * healthy.log_centroid
    assert np.allclose(model.z_target, expected)
    # No region: sitting on the healthy centroid is still a non-zero distance
    # from the interpolated point, so this mode has no zero-error interior.
    assert model.manifold_deviation(healthy.log_centroid) > 0.0


def test_without_a_healthy_reference_the_target_falls_back_to_the_baseline() -> None:
    """A region mode with nothing to be a region of degrades honestly."""
    config = Config()
    strides = simulate_strides(config, 60.0)
    baseline = build_baseline_from_strides(
        strides[: config.patient.baseline_strides], config
    )
    model = PatientModel.build(config, baseline)
    assert model.target_mode is TargetMode.PATIENT_BASELINE
    assert np.allclose(model.z_target, model.z_patient)
    assert np.isnan(model.healthy_region_threshold)
