"""Tests for :mod:`gait_assistance.manifold.log_euclidean` (spec 30)."""

from __future__ import annotations

import numpy as np
import pytest

from gait_assistance.manifold.covariance import is_symmetric
from gait_assistance.manifold.log_euclidean import (
    frobenius_distance,
    interpolate_log,
    log_euclidean_distance,
    log_euclidean_mean,
    matrix_exp,
    matrix_log,
    unvectorize_spd,
    vector_dim,
    vectorize_spd,
)

from .conftest import make_spd


def test_matrix_log_is_symmetric() -> None:
    """The logarithm of an SPD matrix is symmetric."""
    C = make_spd(5, seed=1)
    L = matrix_log(C)
    assert is_symmetric(L)
    assert np.allclose(L, L.T, atol=1e-12)


def test_matrix_log_matches_eigen_definition() -> None:
    """``log`` acts on the eigenvalues of a diagonal matrix."""
    C = np.diag([1.0, np.e, np.e ** 2])
    assert np.allclose(matrix_log(C), np.diag([0.0, 1.0, 2.0]), atol=1e-12)


def test_matrix_log_of_identity_is_zero() -> None:
    """``log(I) = 0``."""
    assert np.allclose(matrix_log(np.eye(4)), np.zeros((4, 4)), atol=1e-12)


def test_exp_inverts_log() -> None:
    """``exp(log(C)) == C`` for an SPD matrix."""
    C = make_spd(4, seed=2)
    assert np.allclose(matrix_exp(matrix_log(C)), C, atol=1e-9)


def test_matrix_log_clips_small_eigenvalues() -> None:
    """Eigenvalues below epsilon are clipped, so the log stays finite."""
    C = np.diag([1.0, 1e-30, 4.0])
    L = matrix_log(C, epsilon=1e-6)
    assert np.all(np.isfinite(L))
    assert np.isclose(L[1, 1], np.log(1e-6))


def test_distance_of_identical_matrices_is_zero() -> None:
    """``d_LE(C, C) == 0``."""
    C = make_spd(5, seed=3)
    assert log_euclidean_distance(C, C) == pytest.approx(0.0, abs=1e-12)


def test_distance_is_symmetric() -> None:
    """``d_LE(C1, C2) == d_LE(C2, C1)``."""
    C1, C2 = make_spd(5, seed=4), make_spd(5, seed=5, scale=2.0)
    d12 = log_euclidean_distance(C1, C2)
    d21 = log_euclidean_distance(C2, C1)
    assert d12 == pytest.approx(d21, abs=1e-12)
    assert d12 > 0.0


def test_distance_satisfies_triangle_inequality() -> None:
    """The Log-Euclidean distance is a metric."""
    C1, C2, C3 = (make_spd(4, seed=s, scale=1.0 + s) for s in (6, 7, 8))
    d13 = log_euclidean_distance(C1, C3)
    d12 = log_euclidean_distance(C1, C2)
    d23 = log_euclidean_distance(C2, C3)
    assert d13 <= d12 + d23 + 1e-9


def test_precomputed_logs_give_the_same_distance() -> None:
    """Passing pre-transformed references does not change the result."""
    C1, C2 = make_spd(5, seed=9), make_spd(5, seed=10)
    direct = log_euclidean_distance(C1, C2)
    cached = log_euclidean_distance(matrix_log(C1), matrix_log(C2), already_log=True)
    assert direct == pytest.approx(cached, abs=1e-12)


def test_vectorization_is_an_isometry() -> None:
    """``||vec(A) - vec(B)||_2 == ||A - B||_F`` thanks to the sqrt(2) weights."""
    A, B = matrix_log(make_spd(5, seed=11)), matrix_log(make_spd(5, seed=12))
    euclidean = float(np.linalg.norm(vectorize_spd(A) - vectorize_spd(B)))
    assert euclidean == pytest.approx(frobenius_distance(A, B), abs=1e-12)


def test_vectorization_length_and_roundtrip() -> None:
    """Vectorisation has the triangular length and is invertible."""
    A = matrix_log(make_spd(6, seed=13))
    v = vectorize_spd(A)
    assert v.size == vector_dim(6) == 21
    assert np.allclose(unvectorize_spd(v, 6), A, atol=1e-12)


def test_unvectorize_rejects_wrong_length() -> None:
    """A length mismatch raises rather than reshaping silently."""
    with pytest.raises(ValueError):
        unvectorize_spd(np.zeros(5), 4)


def test_log_euclidean_mean_of_identical_matrices() -> None:
    """The mean of identical matrices is their common logarithm."""
    C = make_spd(4, seed=14)
    mean = log_euclidean_mean([C, C, C])
    assert np.allclose(mean, matrix_log(C), atol=1e-12)
    assert is_symmetric(mean)


def test_log_euclidean_mean_requires_input() -> None:
    """An empty sequence has no mean."""
    with pytest.raises(ValueError):
        log_euclidean_mean([])


def test_interpolate_log_endpoints() -> None:
    """Interpolation reproduces the endpoints and stays symmetric."""
    A, B = matrix_log(make_spd(4, seed=15)), matrix_log(make_spd(4, seed=16))
    assert np.allclose(interpolate_log(A, B, 0.0), A, atol=1e-12)
    assert np.allclose(interpolate_log(A, B, 1.0), B, atol=1e-12)
    assert is_symmetric(interpolate_log(A, B, 0.3))
