"""SPD covariance construction and validity checks (spec 7)."""

from __future__ import annotations

from typing import Tuple

import numpy as np

DEFAULT_EPSILON: float = 1e-6


def covariance_matrix(
    X: np.ndarray, epsilon: float = DEFAULT_EPSILON, *, assume_centered: bool = False
) -> np.ndarray:
    """Compute the regularised feature covariance ``C = Cov(X) + eps*I``.

    Args:
        X: ``(n_samples, n_features)`` matrix, normally z-scored stride data.
        epsilon: ridge added to the diagonal to guarantee positive definiteness.
        assume_centered: skip mean removal when the data is already centred.

    Returns:
        A symmetric positive-definite ``(n_features, n_features)`` matrix.

    Raises:
        ValueError: if ``X`` is not 2-D, has fewer than two rows, contains
            non-finite values, or ``epsilon`` is not positive.
    """
    data = np.asarray(X, dtype=float)
    if data.ndim != 2:
        raise ValueError("X must be 2-D (n_samples, n_features)")
    if data.shape[0] < 2:
        raise ValueError("at least two samples are required")
    if not np.all(np.isfinite(data)):
        raise ValueError("X contains NaN/Inf")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    if assume_centered:
        centred = data
        denom = max(data.shape[0] - 1, 1)
        cov = centred.T @ centred / denom
    else:
        cov = np.cov(data, rowvar=False)
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    cov = symmetrize(cov)
    return cov + epsilon * np.eye(cov.shape[0])


def symmetrize(C: np.ndarray) -> np.ndarray:
    """Return the symmetric part ``(C + C.T) / 2``."""
    arr = np.asarray(C, dtype=float)
    return 0.5 * (arr + arr.T)


def is_symmetric(C: np.ndarray, tol: float = 1e-8) -> bool:
    """Return True when ``C`` is square and symmetric within ``tol``."""
    arr = np.asarray(C, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return False
    if not np.all(np.isfinite(arr)):
        return False
    return bool(np.allclose(arr, arr.T, atol=tol, rtol=0.0))


def is_positive_definite(C: np.ndarray, tol: float = 0.0) -> bool:
    """Return True when ``C`` is symmetric with all eigenvalues above ``tol``.

    Args:
        C: candidate matrix.
        tol: strict lower bound the smallest eigenvalue must exceed.
    """
    if not is_symmetric(C, tol=1e-8):
        return False
    try:
        eigenvalues = np.linalg.eigvalsh(np.asarray(C, dtype=float))
    except np.linalg.LinAlgError:  # pragma: no cover - numerical edge case
        return False
    return bool(np.min(eigenvalues) > tol)


def is_spd(C: np.ndarray, tol: float = 0.0) -> bool:
    """Alias of :func:`is_positive_definite` reading better at call sites."""
    return is_positive_definite(C, tol)


def min_eigenvalue(C: np.ndarray) -> float:
    """Smallest eigenvalue of the symmetric matrix ``C``."""
    return float(np.min(np.linalg.eigvalsh(symmetrize(C))))


def condition_number(C: np.ndarray) -> float:
    """Ratio of the largest to the smallest eigenvalue of ``C``."""
    eigenvalues = np.linalg.eigvalsh(symmetrize(C))
    smallest = float(np.min(eigenvalues))
    if smallest <= 0.0:
        return float("inf")
    return float(np.max(eigenvalues) / smallest)


def nearest_spd(C: np.ndarray, epsilon: float = DEFAULT_EPSILON) -> np.ndarray:
    """Project ``C`` onto the SPD cone by clipping its eigenvalues.

    Args:
        C: symmetric (possibly indefinite) matrix.
        epsilon: floor applied to the eigenvalues.

    Returns:
        The closest SPD matrix in the eigenvalue-clipping sense.
    """
    sym = symmetrize(C)
    eigenvalues, eigenvectors = np.linalg.eigh(sym)
    clipped = np.maximum(eigenvalues, epsilon)
    return symmetrize(eigenvectors @ np.diag(clipped) @ eigenvectors.T)


def validate_spd(C: np.ndarray, tol: float = 0.0) -> Tuple[bool, str]:
    """Validate an SPD candidate and explain the failure.

    Returns:
        ``(ok, reason)``; ``reason`` is empty when ``C`` is a valid SPD matrix.
    """
    arr = np.asarray(C, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return False, "matrix is not square"
    if not np.all(np.isfinite(arr)):
        return False, "matrix contains NaN/Inf"
    if not is_symmetric(arr):
        return False, "matrix is not symmetric"
    if not is_positive_definite(arr, tol):
        return False, f"matrix is not positive definite (min eig {min_eigenvalue(arr):.3e})"
    return True, ""


__all__ = [
    "DEFAULT_EPSILON",
    "condition_number",
    "covariance_matrix",
    "is_positive_definite",
    "is_spd",
    "is_symmetric",
    "min_eigenvalue",
    "nearest_spd",
    "symmetrize",
    "validate_spd",
]
