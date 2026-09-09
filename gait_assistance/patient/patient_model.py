"""Patient model: baseline, gait states, deviations and OOD (spec 11-17).

``PatientModel`` is the whole high-level representation of one patient:

* ``Z_patient``  - Log-Euclidean centroid of the patient-specific baseline
* ``Z_healthy``  - centroid of the offline able-bodied reference (optional)
* ``Z_target``   - ``(1 - alpha) * Z_patient + alpha * Z_healthy``
* cluster centroids - patient-specific gait states, log domain
* the OOD threshold derived from the baseline distance distribution

Two distances that are easy to conflate are kept apart everywhere:

``d_patient`` (baseline deviation)
    How far the stride sits from *this patient's own* opening strides.  It
    measures within-session consistency.  A stride on top of the baseline is
    typical for this patient; it says nothing about whether the pattern is
    pathological, because the baseline is a description of the patient, not a
    norm.

``d_healthy`` (pathological deviation)
    How far the stride sits from the able-bodied reference.  This is the only
    distance here that speaks to pathology, and it exists solely in
    ``AssistMode.HEALTHY_DIRECTED``; without a healthy reference it is NaN and
    must stay NaN rather than fall back to ``d_patient``.

``d_target`` is the distance to whatever the configured
:class:`~gait_assistance.config.TargetMode` selected as the reference point.
It equals ``d_patient`` exactly when no healthy reference is loaded, which is
why the mode is reported next to it.

``manifold_deviation`` is the normalised context score ``E_R`` derived from
these distances.  In the default ``HEALTHY_REGION`` mode it is exactly 0 for
any stride inside the healthy region and rises only with the excess outside it,
so the patient is never driven towards the healthy centroid itself.  ``E_R``
never commands the motor on its own; it scales a biomechanical deficit that the
device can actually correct.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from ..config import AssistMode, Config, TargetMode
from ..gait.feature_extractor import FeatureExtractor, StrideMetrics, compute_stride_metrics
from ..gait.normalization import ZScoreNormalizer
from ..gait.stride_segmenter import Stride
from ..manifold.clustering import (
    STATE_OOD,
    STATE_UNKNOWN,
    Assignment,
    ClusterModel,
    LogEuclideanClusterer,
    assign_state,
)
from ..manifold.covariance import covariance_matrix, validate_spd
from ..manifold.log_euclidean import (
    frobenius_distance,
    interpolate_log,
    matrix_log,
    vectorize_spd,
)
from ..manifold.reference import HealthyReference
from .baseline import BaselineResult


def state_name(label: int) -> str:
    """Human-readable name of a gait-state label."""
    if label == STATE_OOD:
        return "OOD"
    if label == STATE_UNKNOWN:
        return "UNKNOWN"
    return f"S{label}"


@dataclass
class StrideAnalysis:
    """Result of the high-level analysis of one stride."""

    stride_id: int
    timestamp: float
    valid: bool
    gait_state: int
    state_label: str
    confidence: float
    nearest_distance: float
    runner_up_distance: float
    #: baseline deviation: distance to this patient's own baseline centroid
    d_patient: float
    #: pathological deviation: distance to the able-bodied centroid, NaN in
    #: ``AssistMode.BASELINE_STABILIZATION``
    d_healthy: float
    #: distance to the reference point selected by the target mode
    d_target: float
    is_ood: bool
    metrics: Optional[StrideMetrics] = None
    log_vector: Optional[np.ndarray] = None
    message: str = ""
    #: normalised manifold context score ``E_R`` in ``[0, 1]``; 0 inside the
    #: healthy region
    manifold_deviation: float = 0.0
    #: boundary of the healthy region, NaN without a healthy reference
    healthy_region_threshold: float = float("nan")
    #: whether the stride fell inside the healthy region; ``None`` when there
    #: is no healthy reference to be inside of
    inside_healthy_region: Optional[bool] = None

    @property
    def d_baseline(self) -> float:
        """Alias of :attr:`d_patient`, named after what it measures.

        Deviation from the patient-specific baseline: a consistency signal, not
        a severity signal.
        """
        return self.d_patient

    @property
    def d_pathological(self) -> float:
        """Alias of :attr:`d_healthy`, named after what it measures.

        Deviation from able-bodied gait; NaN unless a healthy reference is
        loaded, and deliberately never substituted with :attr:`d_patient`.
        """
        return self.d_healthy

    def to_dict(self) -> Dict[str, object]:
        """Flat dictionary of the scalar fields (used for logging)."""
        return {
            "stride_id": self.stride_id,
            "timestamp": self.timestamp,
            "gait_state": self.state_label,
            "confidence": self.confidence,
            "d_patient": self.d_patient,
            "d_healthy": self.d_healthy,
            "d_target": self.d_target,
            "manifold_deviation": self.manifold_deviation,
            "healthy_region_threshold": self.healthy_region_threshold,
            "ood": int(self.is_ood),
        }


class PatientModel:
    """Online gait-state classifier and deviation estimator.

    Args:
        config: full system configuration.
        baseline: result of the baseline collection.
        cluster_model: fitted gait-state clusters.
        healthy: optional healthy reference; when absent ``Z_target`` collapses
            onto ``Z_patient`` and ``d_healthy`` is reported as NaN.
        extractor: feature extractor used for online strides.
    """

    def __init__(
        self,
        config: Config,
        baseline: BaselineResult,
        cluster_model: ClusterModel,
        healthy: Optional[HealthyReference] = None,
        extractor: Optional[FeatureExtractor] = None,
    ) -> None:
        self.config = config
        self.baseline = baseline
        self.clusters = cluster_model
        self.healthy = healthy
        self.extractor = extractor or FeatureExtractor(config.feature, config.stride)
        self.normalizer: ZScoreNormalizer = baseline.normalizer

        self.z_patient: np.ndarray = baseline.log_centroid
        self.z_healthy: Optional[np.ndarray] = healthy.log_centroid if healthy else None
        self.target_mode: TargetMode = self._resolve_target_mode()
        self.z_target: np.ndarray = self._resolve_target()
        self.ood_threshold: float = self._compute_ood_threshold()
        self.baseline_distance_stats: Dict[str, float] = self._baseline_distance_stats()

        self._previous_state: int = STATE_UNKNOWN
        self.n_analyzed: int = 0

    def _resolve_target_mode(self) -> TargetMode:
        """Pick the target mode that the loaded references can support.

        A configured mode that needs a healthy reference falls back to
        ``PATIENT_BASELINE`` when there is none, rather than silently treating
        the patient's own baseline as if it were a healthy region.
        """
        requested = TargetMode(self.config.deviation.target_mode)
        if self.healthy is None:
            return TargetMode.PATIENT_BASELINE
        return requested

    def _resolve_target(self) -> np.ndarray:
        """Reference point the ``d_target`` distance is measured from."""
        if self.target_mode is TargetMode.INTERPOLATED_TARGET and self.z_healthy is not None:
            return interpolate_log(
                self.z_patient, self.z_healthy, self.config.patient.alpha
            )
        if self.target_mode is TargetMode.HEALTHY_REGION and self.z_healthy is not None:
            # The region's centre is the reference the distance is reported
            # against; the region boundary is what makes the deviation zero
            # inside it, applied in `manifold_deviation`.
            return self.z_healthy
        return self.z_patient

    @property
    def healthy_region_threshold(self) -> float:
        """Boundary of the healthy region; NaN without a healthy reference."""
        if self.healthy is None:
            return float("nan")
        return float(self.healthy.manifold_distance_threshold)

    def manifold_deviation(self, log_current: np.ndarray) -> float:
        """Normalised manifold context score ``E_R`` of one stride.

        ``HEALTHY_REGION`` returns 0 for every stride inside the region and
        grows with the excess outside it.  The other two modes have no region
        to be inside of, so they normalise the plain distance to their target
        against the baseline spread; they are compatibility paths and do not
        get a zero-error zone.

        Args:
            log_current: ``Z_t = log(C_t)``.

        Returns:
            A value in ``[0, 1]``.
        """
        override = float(self.config.deviation.manifold_scale)
        if self.target_mode is TargetMode.HEALTHY_REGION and self.healthy is not None:
            return self.healthy.manifold_deviation(
                log_current, scale=override if override > 0.0 else None
            )
        scale = override if override > 0.0 else self._fallback_manifold_scale()
        if scale <= 0.0:
            return 0.0
        distance = frobenius_distance(log_current, self.z_target)
        if not np.isfinite(distance):
            return 0.0
        return float(np.clip(distance / scale, 0.0, 1.0))

    def _fallback_manifold_scale(self) -> float:
        """Distance scale used by the non-region target modes."""
        stats = self.baseline_distance_stats
        return float(stats["d_target_mean"] + 2.0 * stats["d_target_std"])

    @property
    def assist_mode(self) -> AssistMode:
        """Which reference the assistance is steering towards.

        Decided by the presence of a healthy reference, never configured by
        hand, because the two modes support different claims about what the
        deviation score means.
        """
        return (
            AssistMode.HEALTHY_DIRECTED
            if self.healthy is not None
            else AssistMode.BASELINE_STABILIZATION
        )

    @property
    def warnings(self) -> List[str]:
        """Caveats that qualify how far this model can be trusted."""
        messages = list(self.clusters.warnings)
        n = self.baseline.n_strides
        recommended = int(self.config.cluster.recommended_strides)
        if n < recommended:
            messages.append(
                f"patient-specific baseline built from {n} strides; 30-50 is "
                f"recommended before the gait states are treated as settled"
            )
        requested = TargetMode(self.config.deviation.target_mode)
        if self.healthy is not None and requested is not self.target_mode:
            messages.append(
                f"target_mode {requested.value} requested but resolved to "
                f"{self.target_mode.value}"
            )
        if self.assist_mode is AssistMode.BASELINE_STABILIZATION:
            messages.append(
                "no healthy reference: running in BASELINE_STABILIZATION, so "
                "d_healthy is undefined and the deviation score measures "
                "departure from this patient's own baseline, not pathology"
            )
        return messages

    # -- construction ------------------------------------------------------- #

    @classmethod
    def build(
        cls,
        config: Config,
        baseline: BaselineResult,
        healthy: Optional[HealthyReference] = None,
        extractor: Optional[FeatureExtractor] = None,
    ) -> "PatientModel":
        """Cluster the baseline strides and assemble the model (spec 12)."""
        clusterer = LogEuclideanClusterer(config.cluster)
        cluster_model = clusterer.fit(baseline.vectors)
        return cls(config, baseline, cluster_model, healthy, extractor)

    # -- online path -------------------------------------------------------- #

    def analyze_stride(self, stride: Stride) -> StrideAnalysis:
        """Run the full high-level chain on one stride.

        Args:
            stride: a segmented stride.

        Returns:
            The :class:`StrideAnalysis`; ``valid`` is False when the stride
            could not be represented on the manifold, in which case the caller
            must not apply assistance.
        """
        metrics = compute_stride_metrics(stride)
        try:
            raw = self.extractor.extract(stride)
        except ValueError as exc:
            return self._invalid(stride, metrics, f"feature extraction failed: {exc}")
        if not np.all(np.isfinite(raw)):
            return self._invalid(stride, metrics, "non-finite features")

        try:
            normalized = self.normalizer.transform(raw)
            cov = covariance_matrix(normalized, self.config.manifold.epsilon)
        except (ValueError, RuntimeError) as exc:
            return self._invalid(stride, metrics, f"covariance failed: {exc}")

        ok, reason = validate_spd(cov, self.config.manifold.spd_check_tol)
        if not ok:
            return self._invalid(stride, metrics, f"invalid SPD: {reason}")

        log_current = matrix_log(cov, self.config.manifold.epsilon)
        return self.analyze_log_matrix(
            log_current, stride_id=stride.stride_id, timestamp=stride.t_end, metrics=metrics
        )

    def analyze_log_matrix(
        self,
        log_current: np.ndarray,
        *,
        stride_id: int = -1,
        timestamp: float = 0.0,
        metrics: Optional[StrideMetrics] = None,
    ) -> StrideAnalysis:
        """Classify one log-domain stride and compute its deviations.

        Args:
            log_current: ``Z_current = log(C_current)``.
            stride_id: identifier carried into the analysis.
            timestamp: stride end time.
            metrics: pre-computed biomechanical metrics, if available.

        Returns:
            The :class:`StrideAnalysis`.
        """
        epsilon = self.config.manifold.epsilon
        vector = vectorize_spd(log_current)
        assignment = assign_state(vector, self.clusters.centroid_vectors, epsilon)

        is_ood = assignment.distance > self.ood_threshold
        state = self._resolve_state(assignment, is_ood)

        d_patient = frobenius_distance(log_current, self.z_patient)
        d_target = frobenius_distance(log_current, self.z_target)
        d_healthy = (
            frobenius_distance(log_current, self.z_healthy)
            if self.z_healthy is not None
            else float("nan")
        )

        threshold = self.healthy_region_threshold
        inside = (
            None
            if self.healthy is None
            else bool(d_healthy <= threshold)
        )

        self.n_analyzed += 1
        return StrideAnalysis(
            stride_id=stride_id,
            timestamp=timestamp,
            valid=True,
            gait_state=state,
            state_label=state_name(state),
            confidence=assignment.confidence,
            nearest_distance=assignment.distance,
            runner_up_distance=assignment.runner_up_distance,
            d_patient=d_patient,
            d_healthy=d_healthy,
            d_target=d_target,
            is_ood=is_ood,
            metrics=metrics,
            log_vector=vector,
            manifold_deviation=self.manifold_deviation(log_current),
            healthy_region_threshold=threshold,
            inside_healthy_region=inside,
        )

    # -- helpers ------------------------------------------------------------ #

    def _resolve_state(self, assignment: Assignment, is_ood: bool) -> int:
        """Apply the OOD and confidence rules to a raw assignment (spec 13, 14)."""
        if is_ood:
            self._previous_state = STATE_OOD
            return STATE_OOD
        if assignment.confidence < self.config.patient.confidence_threshold:
            if (
                self.config.patient.hold_state_on_low_confidence
                and self._previous_state >= 0
            ):
                return self._previous_state
            return STATE_UNKNOWN
        self._previous_state = assignment.label
        return assignment.label

    def _invalid(
        self, stride: Stride, metrics: StrideMetrics, message: str
    ) -> StrideAnalysis:
        """Build the analysis returned when a stride cannot be processed."""
        return StrideAnalysis(
            stride_id=stride.stride_id,
            timestamp=stride.t_end,
            valid=False,
            gait_state=STATE_UNKNOWN,
            state_label=state_name(STATE_UNKNOWN),
            confidence=0.0,
            nearest_distance=float("nan"),
            runner_up_distance=float("nan"),
            d_patient=float("nan"),
            d_healthy=float("nan"),
            d_target=float("nan"),
            is_ood=False,
            metrics=metrics,
            message=message,
            manifold_deviation=0.0,
            healthy_region_threshold=self.healthy_region_threshold,
            inside_healthy_region=None,
        )

    def _compute_ood_threshold(self) -> float:
        """Derive the OOD threshold from the baseline distances (spec 14)."""
        override = self.config.patient.ood_threshold_override
        if override > 0.0:
            return float(override)
        distances = self.clusters.distances
        if distances.size == 0:
            return float("inf")
        mean = float(np.mean(distances))
        std = float(np.std(distances))
        return mean + self.config.patient.ood_sigma * std

    def _baseline_distance_stats(self) -> Dict[str, float]:
        """Distance statistics of the baseline strides to the three references."""
        logs = self.baseline.log_matrices
        d_patient = np.array([frobenius_distance(m, self.z_patient) for m in logs])
        d_target = np.array([frobenius_distance(m, self.z_target) for m in logs])
        stats = {
            "d_patient_mean": float(np.mean(d_patient)),
            "d_patient_std": float(np.std(d_patient)),
            "d_target_mean": float(np.mean(d_target)),
            "d_target_std": float(np.std(d_target)),
        }
        if self.z_healthy is not None:
            d_healthy = np.array([frobenius_distance(m, self.z_healthy) for m in logs])
            stats["d_healthy_mean"] = float(np.mean(d_healthy))
            stats["d_healthy_std"] = float(np.std(d_healthy))
        return stats

    # -- summary and persistence ------------------------------------------- #

    def summary(self) -> Dict[str, object]:
        """Compact description of the fitted model.

        Every clustering number is reported as measured, together with the
        accept/reject reason of each candidate K.  The silhouette in particular
        is deliberately left uninterpreted: it is an admission score for the
        partition, not a rating of the gait, so no quality label is derived
        from it here or anywhere downstream.
        """
        return {
            "assist_mode": self.assist_mode.value,
            "target_mode": self.target_mode.value,
            "healthy_region_threshold": self.healthy_region_threshold,
            "n_baseline_strides": self.baseline.n_strides,
            "features": list(self.extractor.channels),
            "k": self.clusters.k,
            "silhouette": self.clusters.silhouette,
            "stability": self.clusters.stability,
            "cluster_sizes": self.clusters.cluster_sizes,
            "k_selection": [e.to_dict() for e in self.clusters.evaluations],
            "ood_threshold": self.ood_threshold,
            "alpha": self.config.patient.alpha,
            "has_healthy_reference": self.healthy is not None,
            **self.baseline_distance_stats,
            "warnings": self.warnings,
        }

    def save(self, path: Union[str, Path]) -> None:
        """Persist the model parameters needed to resume a session."""
        payload = {
            "assist_mode": self.assist_mode.value,
            "target_mode": self.target_mode.value,
            "normalizer": self.normalizer.to_dict(),
            "clusters": self.clusters.to_dict(),
            "z_patient": self.z_patient.tolist(),
            "z_target": self.z_target.tolist(),
            "z_healthy": None if self.z_healthy is None else self.z_healthy.tolist(),
            "ood_threshold": self.ood_threshold,
            "features": list(self.extractor.channels),
            "mean_metrics": self.baseline.mean_metrics,
            "summary": {k: v for k, v in self.summary().items()},
        }
        Path(path).write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


__all__ = [
    "AssistMode",
    "TargetMode",
    "PatientModel",
    "StrideAnalysis",
    "state_name",
    "STATE_OOD",
    "STATE_UNKNOWN",
]
