"""Patient-specific gait-state clustering in Log-Euclidean space (spec 12, 13).

Because the ``sqrt(2)``-weighted vectorisation is an isometry between the
Frobenius metric on symmetric matrices and the Euclidean metric on vectors,
plain k-means on the vectorised logarithms *is* Log-Euclidean k-means, and the
centroids it returns are log-domain means.

The number of clusters is never assumed: every candidate K is scored with a
silhouette coefficient *and* a bootstrap stability index *and* a minimum
cluster-occupancy test, and K=1 is returned whenever no candidate shows a
genuinely reproducible cluster structure.

The three gates are admission criteria for a partition, nothing else.  In
particular the silhouette coefficient is **not** a measure of gait quality:
it describes the geometric separation of the clusters that were found, so a
high value on a pathological gait simply means the pathology is consistent.
:class:`KEvaluation` therefore records the raw score and the accept/reject
reason for every candidate K and never condenses them into a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from ..config import ClusterConfig
from .log_euclidean import unvectorize_spd, vectorize_many

#: Label returned when the confidence of the nearest-centroid assignment is
#: below the configured threshold and no previous state can be reused.
STATE_UNKNOWN: int = -1
#: Label returned when the stride lies outside the patient's model (spec 14).
STATE_OOD: int = -2


@dataclass
class KEvaluation:
    """Raw scores of one candidate number of clusters.

    Every field is evidence, not a judgement: ``silhouette`` and ``stability``
    are reported as measured, and ``accepted``/``reason`` say only whether the
    candidate cleared the configured admission thresholds.
    """

    k: int
    silhouette: float
    stability: float
    inertia: float
    min_cluster_size: int          #: smallest cluster this K produced
    accepted: bool
    reason: str = ""
    required_cluster_size: int = 0  #: occupancy this K had to reach

    def to_dict(self) -> Dict[str, object]:
        """Return the evaluation as a JSON-serialisable dictionary."""
        return {
            "k": int(self.k),
            "silhouette": float(self.silhouette),
            "stability": float(self.stability),
            "min_cluster_size": int(self.min_cluster_size),
            "required_cluster_size": int(self.required_cluster_size),
            "accepted": bool(self.accepted),
            "reason": self.reason,
        }


@dataclass
class ClusterModel:
    """Fitted gait-state model in the log domain."""

    k: int
    centroid_vectors: np.ndarray          #: ``(k, d)`` vectorised log centroids
    centroids_log: np.ndarray             #: ``(k, n, n)`` log-domain centroids
    labels: np.ndarray                    #: label of every training stride
    silhouette: float
    stability: float
    inertia: float
    distances: np.ndarray                 #: nearest-centroid distance per stride
    evaluations: List[KEvaluation] = field(default_factory=list)
    #: number of strides the partition was fitted on
    n_strides: int = 0
    #: non-fatal caveats about the partition (e.g. an under-powered sample);
    #: they qualify how far the gait states can be trusted, they do not
    #: invalidate the model
    warnings: List[str] = field(default_factory=list)

    @property
    def cluster_sizes(self) -> Dict[int, int]:
        """Number of training strides assigned to each cluster."""
        return {int(k): int(np.sum(self.labels == k)) for k in range(self.k)}

    def distance_stats(self) -> Tuple[float, float]:
        """Mean and standard deviation of the training distances."""
        if self.distances.size == 0:
            return 0.0, 0.0
        return float(np.mean(self.distances)), float(np.std(self.distances))

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable representation."""
        return {
            "k": int(self.k),
            "centroid_vectors": self.centroid_vectors.tolist(),
            "labels": self.labels.tolist(),
            "silhouette": float(self.silhouette),
            "stability": float(self.stability),
            "inertia": float(self.inertia),
            "distances": self.distances.tolist(),
            "n_strides": int(self.n_strides),
            "warnings": list(self.warnings),
            "k_selection": [e.to_dict() for e in self.evaluations],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object], n_features: int) -> "ClusterModel":
        """Rebuild a model from :meth:`to_dict` output."""
        vectors = np.asarray(data["centroid_vectors"], dtype=float)
        return cls(
            k=int(data["k"]),  # type: ignore[arg-type]
            centroid_vectors=vectors,
            centroids_log=np.stack(
                [unvectorize_spd(v, n_features) for v in vectors], axis=0
            ),
            labels=np.asarray(data["labels"], dtype=int),
            silhouette=float(data["silhouette"]),  # type: ignore[arg-type]
            stability=float(data["stability"]),  # type: ignore[arg-type]
            inertia=float(data["inertia"]),  # type: ignore[arg-type]
            distances=np.asarray(data["distances"], dtype=float),
            n_strides=int(data.get("n_strides", 0)),  # type: ignore[arg-type]
            warnings=list(data.get("warnings", [])),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class Assignment:
    """Nearest-centroid assignment of one stride (spec 13)."""

    label: int
    distance: float          #: distance to the nearest centroid, ``d1``
    runner_up_distance: float  #: distance to the second nearest centroid, ``d2``
    confidence: float        #: ``1 - d1 / (d2 + epsilon)``


class LogEuclideanClusterer:
    """Selects K and fits gait-state clusters on vectorised log matrices.

    Args:
        config: clustering configuration (candidates, thresholds, seeds).
    """

    def __init__(self, config: Optional[ClusterConfig] = None) -> None:
        self.config = config or ClusterConfig()

    def fit(self, vectors: np.ndarray) -> ClusterModel:
        """Fit the model on ``(n_strides, d)`` vectorised log matrices.

        Args:
            vectors: one vectorised ``log(C)`` per stride.

        Returns:
            The fitted :class:`ClusterModel`; ``k == 1`` when the data shows no
            reproducible cluster structure.

        Raises:
            ValueError: if fewer than two strides are supplied.
        """
        data = np.asarray(vectors, dtype=float)
        if data.ndim != 2 or data.shape[0] < 2:
            raise ValueError("at least two stride vectors are required")

        n_strides = int(data.shape[0])
        warnings = self._sample_warnings(n_strides)
        evaluations: List[KEvaluation] = []
        best: Optional[Tuple[KEvaluation, np.ndarray, np.ndarray, float]] = None

        for k in sorted(set(int(k) for k in self.config.k_candidates)):
            if k < 1:
                continue
            if k == 1:
                continue
            required = self.required_cluster_size(n_strides)
            if k >= n_strides:
                evaluations.append(
                    KEvaluation(k, float("nan"), 0.0, float("nan"), 0, False,
                                "not enough strides", required)
                )
                continue
            if k * required > n_strides:
                # Even a perfectly balanced partition could not fill every
                # cluster, so this K is unreachable for this sample size.
                evaluations.append(
                    KEvaluation(k, float("nan"), 0.0, float("nan"), 0, False,
                                f"needs {k * required} strides for {k} clusters "
                                f"of >= {required}", required)
                )
                continue
            labels, centers, inertia = self._kmeans(data, k)
            sizes = np.bincount(labels, minlength=k)
            silhouette = float(silhouette_score(data, labels))
            stability = self._stability(data, k)
            reason = ""
            accepted = True
            if int(sizes.min()) < required:
                accepted, reason = (
                    False,
                    f"smallest cluster {int(sizes.min())} < required {required}",
                )
            elif silhouette < self.config.min_silhouette:
                accepted, reason = False, "silhouette below threshold"
            elif stability < self.config.min_stability:
                accepted, reason = False, "clusters not stable"
            evaluation = KEvaluation(
                k, silhouette, stability, inertia, int(sizes.min()), accepted,
                reason, required,
            )
            evaluations.append(evaluation)
            if accepted and (best is None or silhouette > best[0].silhouette):
                best = (evaluation, labels, centers, inertia)

        if best is None:
            model = self._single_cluster(data)
            model.evaluations = evaluations
            model.warnings = warnings
            return model

        evaluation, labels, centers, inertia = best
        distances = _distance_matrix(data, centers)
        n = _matrix_size(data.shape[1])
        return ClusterModel(
            k=evaluation.k,
            centroid_vectors=centers,
            centroids_log=np.stack([unvectorize_spd(c, n) for c in centers], axis=0),
            labels=labels,
            silhouette=evaluation.silhouette,
            stability=evaluation.stability,
            inertia=inertia,
            distances=distances.min(axis=1),
            evaluations=evaluations,
            n_strides=n_strides,
            warnings=warnings,
        )

    def required_cluster_size(self, n_strides: int) -> int:
        """Minimum occupancy a cluster must reach for its K to be accepted.

        Combines the absolute floor with the relative one so that the test
        tightens as the baseline grows instead of staying at a fixed handful of
        strides.

        Args:
            n_strides: number of strides being clustered.

        Returns:
            ``max(min_cluster_size, ceil(min_cluster_fraction * n_strides))``.
        """
        fraction = max(float(self.config.min_cluster_fraction), 0.0)
        relative = int(np.ceil(fraction * max(int(n_strides), 0)))
        return max(int(self.config.min_cluster_size), relative, 1)

    def _sample_warnings(self, n_strides: int) -> List[str]:
        """Caveats about clustering this many strides."""
        recommended = int(self.config.recommended_strides)
        if n_strides >= recommended:
            return []
        return [
            f"clustered {n_strides} strides, below the recommended {recommended} "
            f"(30-50): the silhouette and stability gates are weakly powered at "
            f"this sample size, so treat the gait states as provisional"
        ]

    def fit_matrices(self, log_matrices: Sequence[np.ndarray]) -> ClusterModel:
        """Convenience wrapper fitting directly on log-domain matrices."""
        return self.fit(vectorize_many(log_matrices))

    # -- internals --------------------------------------------------------- #

    def _kmeans(self, data: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """Run k-means and return ``(labels, centers, inertia)``."""
        model = KMeans(
            n_clusters=k,
            n_init=self.config.n_init,
            random_state=self.config.random_state,
        )
        labels = model.fit_predict(data)
        return labels.astype(int), model.cluster_centers_.astype(float), float(model.inertia_)

    def _stability(self, data: np.ndarray, k: int) -> float:
        """Bootstrap cluster-stability index in ``[0, 1]``.

        Two independent subsamples are clustered and both models label the full
        data set; the mean adjusted Rand index between the two labellings
        measures how reproducible the partition is.
        """
        cfg = self.config
        n = data.shape[0]
        size = max(int(round(cfg.stability_subsample * n)), k + 1)
        if size >= n:
            size = n - 1
        if size <= k:
            return 0.0
        rng = np.random.default_rng(cfg.random_state)
        scores: List[float] = []
        for repeat in range(cfg.stability_repeats):
            idx_a = rng.choice(n, size=size, replace=False)
            idx_b = rng.choice(n, size=size, replace=False)
            try:
                model_a = KMeans(n_clusters=k, n_init=cfg.n_init,
                                 random_state=cfg.random_state + repeat).fit(data[idx_a])
                model_b = KMeans(n_clusters=k, n_init=cfg.n_init,
                                 random_state=cfg.random_state + repeat + 1000).fit(data[idx_b])
            except ValueError:  # pragma: no cover - degenerate subsample
                continue
            scores.append(
                float(adjusted_rand_score(model_a.predict(data), model_b.predict(data)))
            )
        if not scores:
            return 0.0
        return float(np.clip(np.mean(scores), 0.0, 1.0))

    def _single_cluster(self, data: np.ndarray) -> ClusterModel:
        """Build the degenerate K=1 model (spec 12: K=1 must be allowed)."""
        center = data.mean(axis=0, keepdims=True)
        distances = np.linalg.norm(data - center, axis=1)
        n = _matrix_size(data.shape[1])
        return ClusterModel(
            k=1,
            centroid_vectors=center,
            centroids_log=np.stack([unvectorize_spd(center[0], n)], axis=0),
            labels=np.zeros(data.shape[0], dtype=int),
            silhouette=float("nan"),
            stability=1.0,
            inertia=float(np.sum(distances ** 2)),
            distances=distances,
            n_strides=int(data.shape[0]),
        )


def assign_state(
    vector: np.ndarray, centroid_vectors: np.ndarray, epsilon: float = 1e-6
) -> Assignment:
    """Assign a stride to its nearest gait-state centroid (spec 13).

    Args:
        vector: vectorised ``log(C)`` of the current stride.
        centroid_vectors: ``(k, d)`` centroid matrix.
        epsilon: guard in the confidence denominator.

    Returns:
        The :class:`Assignment`.  With a single cluster the confidence is 1.0
        because there is no competing centroid.

    Raises:
        ValueError: if no centroid is supplied.
    """
    centers = np.atleast_2d(np.asarray(centroid_vectors, dtype=float))
    if centers.size == 0:
        raise ValueError("at least one centroid is required")
    distances = np.linalg.norm(centers - np.asarray(vector, dtype=float), axis=1)
    order = np.argsort(distances)
    d1 = float(distances[order[0]])
    if centers.shape[0] == 1:
        return Assignment(int(order[0]), d1, float("inf"), 1.0)
    d2 = float(distances[order[1]])
    confidence = float(np.clip(1.0 - d1 / (d2 + epsilon), 0.0, 1.0))
    return Assignment(int(order[0]), d1, d2, confidence)


def _distance_matrix(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances between ``data`` rows and ``centers``."""
    return np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)


def _matrix_size(dim: int) -> int:
    """Recover ``n`` from the vectorised dimension ``n (n + 1) / 2``.

    Raises:
        ValueError: if ``dim`` is not a triangular number.
    """
    n = int(round((np.sqrt(8.0 * dim + 1.0) - 1.0) / 2.0))
    if n * (n + 1) // 2 != dim:
        raise ValueError(f"{dim} is not a valid symmetric-vector dimension")
    return n


__all__ = [
    "STATE_OOD",
    "STATE_UNKNOWN",
    "Assignment",
    "ClusterModel",
    "KEvaluation",
    "LogEuclideanClusterer",
    "assign_state",
]
