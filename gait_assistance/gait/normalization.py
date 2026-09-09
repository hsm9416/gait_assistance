"""Z-score normalisation with frozen baseline statistics (spec 6).

The mean and standard deviation are estimated once, on the patient's baseline
strides, and are then applied unchanged to every online stride.  Recomputing
them per stride would cancel exactly the deviation the system is meant to
measure, so :class:`ZScoreNormalizer` refuses to be refitted unless explicitly
asked to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class NormalizationStats:
    """Frozen per-feature mean and standard deviation."""

    mean: np.ndarray
    std: np.ndarray
    n_samples: int
    feature_names: Sequence[str] = ()

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable representation."""
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "n_samples": int(self.n_samples),
            "feature_names": list(self.feature_names),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "NormalizationStats":
        """Rebuild statistics from :meth:`to_dict` output."""
        return cls(
            mean=np.asarray(data["mean"], dtype=float),
            std=np.asarray(data["std"], dtype=float),
            n_samples=int(data["n_samples"]),  # type: ignore[arg-type]
            feature_names=tuple(data.get("feature_names", ())),  # type: ignore[arg-type]
        )


class ZScoreNormalizer:
    """Fit-once / apply-forever z-score normaliser.

    Args:
        std_floor: lower bound applied to every standard deviation so that
            constant channels cannot blow the normalised values up.
    """

    def __init__(self, std_floor: float = 1e-8) -> None:
        self.std_floor = std_floor
        self._stats: Optional[NormalizationStats] = None

    @property
    def is_fitted(self) -> bool:
        """True once baseline statistics have been computed."""
        return self._stats is not None

    @property
    def stats(self) -> NormalizationStats:
        """The frozen statistics.

        Raises:
            RuntimeError: if the normaliser has not been fitted.
        """
        if self._stats is None:
            raise RuntimeError("ZScoreNormalizer has not been fitted")
        return self._stats

    def fit(
        self,
        matrices: Sequence[np.ndarray],
        feature_names: Sequence[str] = (),
        *,
        refit: bool = False,
    ) -> NormalizationStats:
        """Estimate mean and standard deviation from the baseline strides.

        Args:
            matrices: list of ``(n_points, n_features)`` stride matrices.
            feature_names: names of the feature columns, kept for traceability.
            refit: allow overwriting existing statistics.

        Returns:
            The frozen :class:`NormalizationStats`.

        Raises:
            RuntimeError: if already fitted and ``refit`` is False.
            ValueError: if ``matrices`` is empty or shapes are inconsistent.
        """
        if self._stats is not None and not refit:
            raise RuntimeError(
                "baseline statistics are frozen; pass refit=True to override"
            )
        if not matrices:
            raise ValueError("at least one stride matrix is required")
        stacked = np.vstack([np.asarray(m, dtype=float) for m in matrices])
        if stacked.ndim != 2:
            raise ValueError("stride matrices must be 2-D")
        if not np.all(np.isfinite(stacked)):
            raise ValueError("baseline data contains NaN/Inf")
        mean = stacked.mean(axis=0)
        std = np.maximum(stacked.std(axis=0), self.std_floor)
        self._stats = NormalizationStats(
            mean=mean, std=std, n_samples=int(stacked.shape[0]),
            feature_names=tuple(feature_names),
        )
        return self._stats

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        """Apply the frozen statistics to one stride matrix.

        Args:
            matrix: ``(n_points, n_features)`` raw feature matrix.

        Returns:
            The z-scored matrix.

        Raises:
            RuntimeError: if the normaliser has not been fitted.
            ValueError: on a feature-count mismatch.
        """
        stats = self.stats
        data = np.asarray(matrix, dtype=float)
        if data.ndim != 2 or data.shape[1] != stats.mean.size:
            raise ValueError(
                f"expected {stats.mean.size} features, got shape {data.shape}"
            )
        return (data - stats.mean) / stats.std

    def transform_many(self, matrices: Sequence[np.ndarray]) -> List[np.ndarray]:
        """Apply :meth:`transform` to every matrix in ``matrices``."""
        return [self.transform(m) for m in matrices]

    def inverse_transform(self, matrix: np.ndarray) -> np.ndarray:
        """Map a z-scored matrix back to physical units."""
        stats = self.stats
        return np.asarray(matrix, dtype=float) * stats.std + stats.mean

    def to_dict(self) -> Dict[str, object]:
        """Serialise the normaliser (statistics plus the std floor)."""
        return {"std_floor": self.std_floor, "stats": self.stats.to_dict()}

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ZScoreNormalizer":
        """Rebuild a normaliser from :meth:`to_dict` output."""
        obj = cls(float(data.get("std_floor", 1e-8)))  # type: ignore[arg-type]
        obj._stats = NormalizationStats.from_dict(data["stats"])  # type: ignore[index]
        return obj


__all__ = ["NormalizationStats", "ZScoreNormalizer"]
