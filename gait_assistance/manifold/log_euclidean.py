"""Log-Euclidean geometry on the SPD manifold (spec 8, 9, 10).

The matrix logarithm is computed from the symmetric eigendecomposition
``C = V diag(lambda) V^T``  ->  ``log(C) = V diag(log(lambda)) V^T``
rather than with ``scipy.linalg.logm``: for symmetric matrices this is exact,
much faster, and it lets us clip the eigenvalues at ``epsilon`` so a
near-singular covariance can never produce ``-inf``.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from .covariance import DEFAULT_EPSILON, symmetrize


def spd_eigh(C: np.ndarray, epsilon: float = DEFAULT_EPSILON) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric eigendecomposition with eigenvalues clipped at ``epsilon``.

    Args:
        C: symmetric matrix.
        epsilon: lower bound applied to the eigenvalues.

    Returns:
        ``(eigenvalues, eigenvectors)`` with ascending, clipped eigenvalues.

    Raises:
        ValueError: if ``C`` is not a finite square matrix.
    """
    arr = np.asarray(C, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("C must be a square matrix")
    if not np.all(np.isfinite(arr)):
        raise ValueError("C contains NaN/Inf")
    eigenvalues, eigenvectors = np.linalg.eigh(symmetrize(arr))
    return np.maximum(eigenvalues, epsilon), eigenvectors


def matrix_log(C: np.ndarray, epsilon: float = DEFAULT_EPSILON) -> np.ndarray:
    """Matrix logarithm of an SPD matrix via ``numpy.linalg.eigh`` (spec 8).

    Args:
        C: symmetric positive-definite matrix.
        epsilon: eigenvalue floor; eigenvalues below it are clipped so the
            logarithm stays finite.

    Returns:
        The symmetric matrix ``log(C)``.
    """
    eigenvalues, eigenvectors = spd_eigh(C, epsilon)
    return symmetrize(eigenvectors @ np.diag(np.log(eigenvalues)) @ eigenvectors.T)


def matrix_exp(S: np.ndarray) -> np.ndarray:
    """Matrix exponential of a symmetric matrix, inverse of :func:`matrix_log`.

    Args:
        S: symmetric matrix (typically a point in the log domain).

    Returns:
        The SPD matrix ``exp(S)``.
    """
    arr = symmetrize(np.asarray(S, dtype=float))
    eigenvalues, eigenvectors = np.linalg.eigh(arr)
    return symmetrize(eigenvectors @ np.diag(np.exp(eigenvalues)) @ eigenvectors.T)


def frobenius_norm(A: np.ndarray) -> float:
    """Frobenius norm of ``A``."""
    return float(np.linalg.norm(np.asarray(A, dtype=float), ord="fro"))


def frobenius_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Frobenius distance ``||A - B||_F`` between two matrices."""
    return float(
        np.linalg.norm(np.asarray(A, dtype=float) - np.asarray(B, dtype=float), ord="fro")
    )


def log_euclidean_distance(
    C1: np.ndarray,
    C2: np.ndarray,
    epsilon: float = DEFAULT_EPSILON,
    *,
    already_log: bool = False,
) -> float:
    """Log-Euclidean distance ``||log(C1) - log(C2)||_F`` (spec 9).

    Args:
        C1: first SPD matrix, or its logarithm when ``already_log`` is True.
        C2: second SPD matrix, or its logarithm when ``already_log`` is True.
        epsilon: eigenvalue floor used by :func:`matrix_log`.
        already_log: skip the logarithms; reference matrices are stored
            pre-transformed so the online path never recomputes them.

    Returns:
        The non-negative distance.
    """
    if already_log:
        return frobenius_distance(C1, C2)
    return frobenius_distance(matrix_log(C1, epsilon), matrix_log(C2, epsilon))


def log_euclidean_mean(
    matrices: Sequence[np.ndarray], epsilon: float = DEFAULT_EPSILON, *, already_log: bool = False
) -> np.ndarray:
    """Log-Euclidean mean, returned in the log domain (spec 11).

    Args:
        matrices: SPD matrices, or their logarithms when ``already_log`` is True.
        epsilon: eigenvalue floor used by :func:`matrix_log`.
        already_log: treat the inputs as already log-transformed.

    Returns:
        ``mean(log(C_i))`` as a symmetric matrix.  Apply :func:`matrix_exp` to
        obtain the mean SPD matrix itself.

    Raises:
        ValueError: if ``matrices`` is empty.
    """
    if len(matrices) == 0:
        raise ValueError("at least one matrix is required")
    logs = [
        symmetrize(np.asarray(m, dtype=float)) if already_log else matrix_log(m, epsilon)
        for m in matrices
    ]
    return symmetrize(np.mean(np.stack(logs, axis=0), axis=0))


def vector_dim(n: int) -> int:
    """Length of the vectorisation of an ``n x n`` symmetric matrix."""
    return n * (n + 1) // 2


def vectorize_spd(S: np.ndarray) -> np.ndarray:
    """Vectorise a symmetric matrix, isometrically (spec 10).

    The upper triangle is flattened with the off-diagonal entries scaled by
    ``sqrt(2)`` so that the Euclidean norm of the vector equals the Frobenius
    norm of the matrix.

    Args:
        S: symmetric ``(n, n)`` matrix, typically ``log(C)``.

    Returns:
        A vector of length ``n (n + 1) / 2``.

    Raises:
        ValueError: if ``S`` is not square.
    """
    arr = np.asarray(S, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("S must be a square matrix")
    n = arr.shape[0]
    rows, cols = np.triu_indices(n)
    values = arr[rows, cols].astype(float).copy()
    values[rows != cols] *= np.sqrt(2.0)
    return values


def unvectorize_spd(v: np.ndarray, n: int) -> np.ndarray:
    """Inverse of :func:`vectorize_spd`.

    Args:
        v: vector of length ``n (n + 1) / 2``.
        n: size of the reconstructed matrix.

    Returns:
        The symmetric ``(n, n)`` matrix.

    Raises:
        ValueError: on a length mismatch.
    """
    values = np.asarray(v, dtype=float)
    if values.size != vector_dim(n):
        raise ValueError(f"expected {vector_dim(n)} entries, got {values.size}")
    out = np.zeros((n, n), dtype=float)
    rows, cols = np.triu_indices(n)
    scaled = values.copy()
    scaled[rows != cols] /= np.sqrt(2.0)
    out[rows, cols] = scaled
    return symmetrize(out + np.triu(out, 1).T)


def vectorize_many(matrices: Sequence[np.ndarray]) -> np.ndarray:
    """Stack :func:`vectorize_spd` over ``matrices`` into a 2-D array."""
    if len(matrices) == 0:
        return np.zeros((0, 0), dtype=float)
    return np.stack([vectorize_spd(m) for m in matrices], axis=0)


def interpolate_log(A: np.ndarray, B: np.ndarray, alpha: float) -> np.ndarray:
    """Geodesic interpolation in the log domain (spec 16).

    Args:
        A: first log-domain point (weight ``1 - alpha``).
        B: second log-domain point (weight ``alpha``).
        alpha: interpolation factor, usually in ``[0, 1]``.

    Returns:
        ``(1 - alpha) * A + alpha * B``, symmetrised.
    """
    return symmetrize(
        (1.0 - alpha) * np.asarray(A, dtype=float) + alpha * np.asarray(B, dtype=float)
    )


__all__ = [
    "frobenius_distance",
    "frobenius_norm",
    "interpolate_log",
    "log_euclidean_distance",
    "log_euclidean_mean",
    "matrix_exp",
    "matrix_log",
    "spd_eigh",
    "unvectorize_spd",
    "vector_dim",
    "vectorize_many",
    "vectorize_spd",
]
