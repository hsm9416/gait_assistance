"""Patient baseline collection (spec 11).

The first ``baseline_strides`` strides of a session define the patient's own
normal: they fix the z-score statistics, the Log-Euclidean centroid
``Z_patient``, the gait-state clusters and the reference biomechanical metrics.
Nothing here is trained offline and no patient labels are used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..config import Config
from ..gait.feature_extractor import (
    FeatureExtractor,
    StrideMetrics,
    aggregate_metrics,
    compute_stride_metrics,
)
from ..gait.normalization import ZScoreNormalizer
from ..gait.stride_segmenter import Stride
from ..manifold.covariance import covariance_matrix, validate_spd
from ..manifold.log_euclidean import log_euclidean_mean, matrix_log, vectorize_many


@dataclass
class BaselineResult:
    """Everything derived from the baseline strides."""

    normalizer: ZScoreNormalizer
    raw_matrices: List[np.ndarray]           #: resampled, physical units
    covariances: List[np.ndarray]            #: SPD matrices
    log_matrices: List[np.ndarray]           #: ``log(C_i)``
    vectors: np.ndarray                      #: ``(n, d)`` vectorised logs
    log_centroid: np.ndarray                 #: ``Z_patient``
    metrics: List[StrideMetrics]
    mean_metrics: Dict[str, float]
    stride_ids: List[int] = field(default_factory=list)

    @property
    def n_strides(self) -> int:
        """Number of strides that entered the baseline."""
        return len(self.log_matrices)

    @property
    def baseline_belt_length(self) -> float:
        """Mean belt length over the baseline strides (mm), used as target."""
        return float(self.mean_metrics.get("mean_belt_length", 0.0))


class BaselineCollector:
    """Accumulate strides until the baseline is complete, then build the model.

    Args:
        config: full system configuration.
        feature_extractor: extractor to reuse; one is created when omitted.
    """

    def __init__(
        self, config: Config, feature_extractor: Optional[FeatureExtractor] = None
    ) -> None:
        self.config = config
        self.extractor = feature_extractor or FeatureExtractor(
            config.feature, config.stride
        )
        self._strides: List[Stride] = []
        self._matrices: List[np.ndarray] = []
        self._metrics: List[StrideMetrics] = []

    @property
    def count(self) -> int:
        """Number of strides collected so far."""
        return len(self._matrices)

    @property
    def required(self) -> int:
        """Number of strides needed before the baseline can be built."""
        return self.config.patient.baseline_strides

    @property
    def is_ready(self) -> bool:
        """True once enough strides have been collected."""
        return self.count >= self.required

    def reset(self) -> None:
        """Discard everything collected so far."""
        self._strides.clear()
        self._matrices.clear()
        self._metrics.clear()

    def add(self, stride: Stride) -> bool:
        """Add one stride to the baseline.

        Args:
            stride: a validated stride.

        Returns:
            True when the baseline is complete after this stride.
        """
        matrix = self.extractor.extract(stride)
        if not np.all(np.isfinite(matrix)):
            return self.is_ready
        self._strides.append(stride)
        self._matrices.append(matrix)
        self._metrics.append(compute_stride_metrics(stride))
        return self.is_ready

    def build(self) -> BaselineResult:
        """Fit the normaliser and compute the SPD/log representation.

        Returns:
            The :class:`BaselineResult`.

        Raises:
            RuntimeError: if too few strides were collected, or if a baseline
                covariance is not a valid SPD matrix.
        """
        if self.count < 2:
            raise RuntimeError(
                f"baseline needs at least 2 strides, got {self.count}"
            )
        epsilon = self.config.manifold.epsilon
        normalizer = ZScoreNormalizer(self.config.feature.std_floor)
        normalizer.fit(self._matrices, self.extractor.channels)

        covariances: List[np.ndarray] = []
        logs: List[np.ndarray] = []
        for index, matrix in enumerate(self._matrices):
            cov = covariance_matrix(normalizer.transform(matrix), epsilon)
            ok, reason = validate_spd(cov, self.config.manifold.spd_check_tol)
            if not ok:
                raise RuntimeError(f"baseline stride {index}: invalid SPD matrix ({reason})")
            covariances.append(cov)
            logs.append(matrix_log(cov, epsilon))

        return BaselineResult(
            normalizer=normalizer,
            raw_matrices=list(self._matrices),
            covariances=covariances,
            log_matrices=logs,
            vectors=vectorize_many(logs),
            log_centroid=log_euclidean_mean(logs, epsilon, already_log=True),
            metrics=list(self._metrics),
            mean_metrics=aggregate_metrics(self._metrics),
            stride_ids=[s.stride_id for s in self._strides],
        )


def build_baseline_from_strides(
    strides: Sequence[Stride],
    config: Config,
    feature_extractor: Optional[FeatureExtractor] = None,
) -> BaselineResult:
    """Build a baseline directly from a list of strides (offline helper)."""
    collector = BaselineCollector(config, feature_extractor)
    for stride in strides:
        collector.add(stride)
    return collector.build()


__all__ = ["BaselineCollector", "BaselineResult", "build_baseline_from_strides"]
