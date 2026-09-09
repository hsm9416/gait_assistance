"""Tests for clustering, state assignment and OOD detection (spec 30)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gait_assistance.config import (
    ClusterConfig,
    Config,
    ReferenceConfig,
    TargetMode,
    ThresholdMethod,
)
from gait_assistance.manifold.clustering import (
    STATE_OOD,
    LogEuclideanClusterer,
    assign_state,
)
from gait_assistance.manifold.log_euclidean import matrix_log, vectorize_many
from gait_assistance.manifold.reference import build_healthy_reference
from gait_assistance.patient.baseline import build_baseline_from_strides
from gait_assistance.patient.patient_model import PatientModel

from .conftest import make_spd, simulate_strides


def _clustered_vectors(k: int, per_cluster: int = 12, spread: float = 8.0) -> np.ndarray:
    """Build ``k`` well-separated Gaussian blobs in vector space."""
    rng = np.random.default_rng(0)
    return np.vstack(
        [rng.normal(size=(per_cluster, 15)) + i * spread for i in range(k)]
    )


def test_nearest_centroid_assignment() -> None:
    """A stride is assigned to the centroid it is closest to."""
    centroids = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    assert assign_state(np.array([0.4, 0.1]), centroids).label == 0
    assert assign_state(np.array([9.0, 0.5]), centroids).label == 1
    assert assign_state(np.array([0.2, 9.7]), centroids).label == 2


def test_assignment_distances_and_confidence() -> None:
    """``d1``, ``d2`` and ``1 - d1/(d2+eps)`` are reported consistently."""
    centroids = np.array([[0.0, 0.0], [10.0, 0.0]])
    result = assign_state(np.array([1.0, 0.0]), centroids, epsilon=1e-6)
    assert result.label == 0
    assert result.distance == pytest.approx(1.0)
    assert result.runner_up_distance == pytest.approx(9.0)
    assert result.confidence == pytest.approx(1.0 - 1.0 / (9.0 + 1e-6))
    assert 0.0 <= result.confidence <= 1.0


def test_equidistant_point_has_zero_confidence() -> None:
    """A point halfway between two centroids carries no confidence."""
    centroids = np.array([[0.0, 0.0], [10.0, 0.0]])
    assert assign_state(np.array([5.0, 0.0]), centroids).confidence == pytest.approx(
        0.0, abs=1e-6
    )


def test_single_cluster_assignment_is_confident() -> None:
    """With one centroid there is no competitor, so confidence is 1."""
    result = assign_state(np.array([3.0, 4.0]), np.array([[0.0, 0.0]]))
    assert result.label == 0
    assert result.distance == pytest.approx(5.0)
    assert result.confidence == 1.0


def test_assign_state_requires_a_centroid() -> None:
    """An empty centroid set is a programming error."""
    with pytest.raises(ValueError):
        assign_state(np.zeros(3), np.zeros((0, 3)))


@pytest.mark.parametrize("k", [2, 3, 4])
def test_recovers_the_true_number_of_clusters(k: int) -> None:
    """Well-separated blobs are recovered with the right K."""
    model = LogEuclideanClusterer(ClusterConfig()).fit(_clustered_vectors(k))
    assert model.k == k
    assert model.silhouette > 0.5
    assert model.stability >= 0.6
    assert sum(model.cluster_sizes.values()) == 12 * k


def test_unstructured_data_yields_k_equal_one() -> None:
    """K is not forced to 3: featureless data must return a single cluster."""
    rng = np.random.default_rng(7)
    model = LogEuclideanClusterer(ClusterConfig()).fit(rng.normal(size=(30, 15)))
    assert model.k == 1
    assert model.centroid_vectors.shape[0] == 1
    assert np.all(model.labels == 0)
    assert [e.accepted for e in model.evaluations] == [False, False, False]


def test_stability_gate_rejects_unstable_partitions() -> None:
    """A high stability requirement rejects a partition that is not reproducible."""
    rng = np.random.default_rng(3)
    vectors = np.vstack([rng.normal(size=(15, 15)), rng.normal(size=(15, 15)) + 1.0])
    strict = LogEuclideanClusterer(
        ClusterConfig(min_silhouette=0.0, min_stability=0.999)
    ).fit(vectors)
    lenient = LogEuclideanClusterer(
        ClusterConfig(min_silhouette=0.0, min_stability=0.0)
    ).fit(vectors)
    assert strict.k == 1
    assert lenient.k >= 2


def test_centroids_live_in_the_log_domain() -> None:
    """Centroids are symmetric matrices, recoverable from their vectors."""
    logs = [matrix_log(make_spd(5, seed=s)) for s in range(12)]
    model = LogEuclideanClusterer(ClusterConfig()).fit_matrices(logs)
    assert model.centroids_log.shape[1:] == (5, 5)
    for matrix in model.centroids_log:
        assert np.allclose(matrix, matrix.T, atol=1e-12)


def test_fit_requires_two_strides() -> None:
    """Clustering a single stride is meaningless."""
    with pytest.raises(ValueError):
        LogEuclideanClusterer().fit(np.zeros((1, 15)))


def test_ood_detection_flags_a_distant_stride() -> None:
    """A stride far from every centroid is reported as OOD, not as a state."""
    config = Config()
    strides = simulate_strides(config, 40.0)
    assert len(strides) >= config.patient.baseline_strides
    baseline = build_baseline_from_strides(
        strides[: config.patient.baseline_strides], config
    )
    model = PatientModel.build(config, baseline)

    inlier = model.analyze_stride(strides[config.patient.baseline_strides])
    assert inlier.valid and not inlier.is_ood

    far = model.z_patient + 50.0 * np.eye(model.z_patient.shape[0])
    outlier = model.analyze_log_matrix(far, stride_id=999)
    assert outlier.is_ood
    assert outlier.gait_state == STATE_OOD
    assert outlier.state_label == "OOD"
    assert outlier.nearest_distance > model.ood_threshold


def test_ood_threshold_follows_the_baseline_spread() -> None:
    """A wider baseline spread yields a more permissive OOD threshold."""
    config = Config()
    strides = simulate_strides(config, 40.0)
    baseline = build_baseline_from_strides(
        strides[: config.patient.baseline_strides], config
    )
    tight = PatientModel.build(config, baseline)
    loose_config = Config()
    loose_config.patient.ood_sigma = 6.0
    loose = PatientModel.build(loose_config, baseline)
    assert loose.ood_threshold > tight.ood_threshold

    override = Config()
    override.patient.ood_threshold_override = 42.0
    assert PatientModel.build(override, baseline).ood_threshold == 42.0


def test_healthy_reference_threshold_is_mean_plus_two_sigma() -> None:
    """The ``mean_std`` method still follows ``mean + sigma * std``.

    It is no longer the default - the region boundary is a percentile now -
    so the method has to be requested explicitly.
    """
    logs = [matrix_log(make_spd(4, seed=s)) for s in range(15)]
    reference = build_healthy_reference(
        logs, config=ReferenceConfig(threshold_method=ThresholdMethod.MEAN_STD)
    )
    expected = reference.distance_mean + 2.0 * reference.distance_std
    assert reference.distance_threshold == pytest.approx(expected)
    assert reference.manifold_distance_threshold == pytest.approx(expected)
    assert reference.n_strides == 15
    assert reference.num_strides == 15
    assert np.allclose(reference.log_centroid, reference.log_centroid.T)


def test_healthy_reference_roundtrip(tmp_path: Path) -> None:
    """A healthy reference survives a save/load cycle."""
    logs = [matrix_log(make_spd(4, seed=s)) for s in range(10)]
    reference = build_healthy_reference(logs, metrics={"swing_ratio": 0.38})
    path = tmp_path / "healthy.npz"
    reference.save(path)
    from gait_assistance.manifold.reference import HealthyReference

    loaded = HealthyReference.load(path)
    assert np.allclose(loaded.log_centroid, reference.log_centroid)
    assert loaded.distance_threshold == pytest.approx(reference.distance_threshold)
    assert loaded.metrics["swing_ratio"] == pytest.approx(0.38)


def test_target_interpolates_between_patient_and_healthy() -> None:
    """``Z_target = (1-alpha) Z_patient + alpha Z_healthy`` (spec 16).

    This is the ``interpolated_target`` compatibility mode; the default mode
    targets the healthy *region* instead and is covered separately.
    """
    config = Config()
    config.deviation.target_mode = TargetMode.INTERPOLATED_TARGET
    config.patient.alpha = 0.3
    strides = simulate_strides(config, 40.0)
    baseline = build_baseline_from_strides(
        strides[: config.patient.baseline_strides], config
    )
    healthy = build_healthy_reference([matrix_log(make_spd(5, seed=s)) for s in range(12)])
    model = PatientModel.build(config, baseline, healthy)
    expected = 0.7 * model.z_patient + 0.3 * healthy.log_centroid
    assert np.allclose(model.z_target, expected, atol=1e-12)

    no_healthy = PatientModel.build(config, baseline)
    assert np.allclose(no_healthy.z_target, no_healthy.z_patient)


def test_minimum_cluster_size_is_relative_to_the_sample() -> None:
    """The occupancy requirement tightens as the baseline grows."""
    clusterer = LogEuclideanClusterer(ClusterConfig(min_cluster_size=3, min_cluster_fraction=0.15))
    assert clusterer.required_cluster_size(10) == 3    # absolute floor wins
    assert clusterer.required_cluster_size(20) == 3    # ceil(3.0) == 3
    assert clusterer.required_cluster_size(30) == 5    # ceil(4.5) == 5
    assert clusterer.required_cluster_size(50) == 8    # ceil(7.5) == 8


def test_a_tiny_cluster_cannot_carry_a_k() -> None:
    """A well-separated handful of outliers must not be accepted as a state."""
    rng = np.random.default_rng(11)
    # 28 strides in one blob, 2 far away: geometrically clean, statistically not
    # a gait state.
    vectors = np.vstack([rng.normal(size=(28, 15)), rng.normal(size=(2, 15)) + 40.0])

    strict = LogEuclideanClusterer(
        ClusterConfig(min_cluster_size=3, min_cluster_fraction=0.15)
    ).fit(vectors)
    assert strict.k == 1
    rejected = [e for e in strict.evaluations if e.k == 2]
    assert rejected and not rejected[0].accepted
    assert "required" in rejected[0].reason
    assert rejected[0].required_cluster_size == 5

    lenient = LogEuclideanClusterer(
        ClusterConfig(min_cluster_size=1, min_cluster_fraction=0.0, min_silhouette=0.0)
    ).fit(vectors)
    assert lenient.k >= 2


def test_unreachable_k_is_not_fitted() -> None:
    """A K that could not fill its clusters is rejected without running k-means."""
    rng = np.random.default_rng(5)
    model = LogEuclideanClusterer(
        ClusterConfig(k_candidates=(1, 4), min_cluster_size=6, min_cluster_fraction=0.0)
    ).fit(rng.normal(size=(20, 15)))
    assert model.k == 1
    evaluation = next(e for e in model.evaluations if e.k == 4)
    assert not evaluation.accepted
    assert "24 strides" in evaluation.reason


def test_an_under_powered_sample_is_flagged() -> None:
    """Below the recommended stride count the partition carries a warning."""
    rng = np.random.default_rng(2)
    small = LogEuclideanClusterer(ClusterConfig(recommended_strides=30)).fit(
        rng.normal(size=(20, 15))
    )
    assert small.n_strides == 20
    assert any("recommended 30" in w for w in small.warnings)

    ample = LogEuclideanClusterer(ClusterConfig(recommended_strides=30)).fit(
        rng.normal(size=(35, 15))
    )
    assert ample.warnings == []


def test_k_selection_is_reported_without_a_verdict() -> None:
    """Every candidate keeps its raw scores and its accept/reject reason."""
    model = LogEuclideanClusterer(ClusterConfig()).fit(_clustered_vectors(2))
    selection = [e.to_dict() for e in model.evaluations]
    assert selection and all(
        set(row) == {
            "k",
            "silhouette",
            "stability",
            "min_cluster_size",
            "required_cluster_size",
            "accepted",
            "reason",
        }
        for row in selection
    )
    # The report carries measurements and admission outcomes only: no field
    # grades the silhouette, because it rates the partition, not the gait.
    assert not any(
        key in row for row in selection for key in ("quality", "rating", "grade")
    )
