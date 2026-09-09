"""Tests for :mod:`gait_assistance.manifold.covariance` (spec 30)."""

from __future__ import annotations

import numpy as np
import pytest

from gait_assistance.manifold.covariance import (
    condition_number,
    covariance_matrix,
    is_positive_definite,
    is_symmetric,
    min_eigenvalue,
    nearest_spd,
    validate_spd,
)


def test_covariance_matrix_is_symmetric(rng: np.random.Generator) -> None:
    """The covariance of any feature matrix must be symmetric."""
    X = rng.normal(size=(100, 5))
    C = covariance_matrix(X)
    assert C.shape == (5, 5)
    assert is_symmetric(C)
    assert np.allclose(C, C.T, atol=1e-12)


def test_covariance_matrix_is_spd(rng: np.random.Generator) -> None:
    """The regularised covariance must be positive definite."""
    X = rng.normal(size=(100, 5))
    C = covariance_matrix(X, epsilon=1e-6)
    assert is_positive_definite(C)
    assert min_eigenvalue(C) > 0.0


def test_epsilon_makes_rank_deficient_data_spd() -> None:
    """A rank-deficient feature matrix stays SPD thanks to the ridge."""
    base = np.linspace(0.0, 1.0, 100)
    X = np.stack([base, 2.0 * base, 3.0 * base], axis=1)  # rank 1
    C = covariance_matrix(X, epsilon=1e-6)
    assert is_positive_definite(C)
    assert min_eigenvalue(C) >= 1e-7
    assert np.isfinite(condition_number(C))


def test_constant_channel_is_still_spd() -> None:
    """A channel with zero variance cannot break positive definiteness."""
    X = np.zeros((50, 4))
    X[:, 0] = np.linspace(0.0, 1.0, 50)
    C = covariance_matrix(X, epsilon=1e-6)
    assert is_positive_definite(C)


def test_covariance_rejects_bad_input() -> None:
    """Invalid inputs raise instead of producing a silent bad matrix."""
    with pytest.raises(ValueError):
        covariance_matrix(np.zeros((1, 3)))
    with pytest.raises(ValueError):
        covariance_matrix(np.full((10, 3), np.nan))
    with pytest.raises(ValueError):
        covariance_matrix(np.zeros((10, 3)), epsilon=0.0)
    with pytest.raises(ValueError):
        covariance_matrix(np.zeros(10))


def test_is_positive_definite_rejects_non_spd() -> None:
    """Indefinite, asymmetric and non-finite matrices are rejected."""
    assert not is_positive_definite(np.diag([1.0, -1.0]))
    assert not is_positive_definite(np.array([[1.0, 2.0], [0.0, 1.0]]))
    assert not is_positive_definite(np.array([[np.nan, 0.0], [0.0, 1.0]]))
    assert not is_symmetric(np.zeros((2, 3)))


def test_nearest_spd_projects_indefinite_matrix() -> None:
    """Eigenvalue clipping turns an indefinite matrix into an SPD one."""
    A = np.diag([2.0, -3.0, 0.0])
    projected = nearest_spd(A, epsilon=1e-6)
    assert is_positive_definite(projected)
    ok, reason = validate_spd(projected)
    assert ok and reason == ""


def test_validate_spd_explains_the_failure() -> None:
    """``validate_spd`` reports why a matrix was rejected."""
    ok, reason = validate_spd(np.diag([1.0, -1.0]))
    assert not ok and "positive definite" in reason
    ok, reason = validate_spd(np.zeros((2, 3)))
    assert not ok and "square" in reason
