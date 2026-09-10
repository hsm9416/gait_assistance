"""Healthy reference *region* built from an offline able-bodied data set (spec 15).

The reference is a region, not a point.  Two things are stored:

* the log-domain centroid ``Z_H`` together with the distribution of the healthy
  strides around it, which yields the boundary
  ``manifold_distance_threshold`` separating "inside the healthy region" from
  "outside" it;
* a reference interval per biomechanical metric, taken as a percentile range of
  the healthy distribution rather than a single mean.

Both exist for the same reason: able-bodied gait is a spread, and steering a
patient at the *centre* of that spread would demand a precision no healthy
walker maintains either.  A stride inside the region has a manifold deviation
of exactly zero and a metric inside its interval has an error of exactly zero;
only the excess beyond the boundary is treated as something to assist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np

from ..config import ReferenceConfig, ThresholdMethod
from .covariance import DEFAULT_EPSILON
from .log_euclidean import (
    frobenius_distance,
    log_euclidean_mean,
    vectorize_spd,
)


@dataclass(frozen=True)
class MetricRange:
    """Reference interval of one biomechanical metric.

    Attributes:
        lower: lower bound of the healthy interval.
        median: central value, reported but never used as a target.
        upper: upper bound of the healthy interval.
        scale: deviation beyond a bound that saturates the error; ``None``
            lets the caller supply one from configuration.
    """

    lower: float
    median: float
    upper: float
    scale: Optional[float] = None

    def contains(self, value: float) -> bool:
        """True when ``value`` lies inside the interval (bounds included)."""
        return bool(np.isfinite(value) and self.lower <= value <= self.upper)

    def excess(self, value: float) -> float:
        """Signed distance outside the interval.

        Returns:
            ``0`` inside the interval, a negative number below ``lower`` and a
            positive one above ``upper``.  The sign is what lets a caller apply
            a direction-specific rule, e.g. assisting a deficit but not a
            surplus.
        """
        if not np.isfinite(value):
            return 0.0
        if value < self.lower:
            return float(value - self.lower)
        if value > self.upper:
            return float(value - self.upper)
        return 0.0

    def deficit(self, value: float) -> float:
        """How far ``value`` falls *below* the interval; 0 otherwise."""
        return float(max(self.lower - value, 0.0)) if np.isfinite(value) else 0.0

    def width(self) -> float:
        """Width of the interval."""
        return float(self.upper - self.lower)

    def to_dict(self) -> Dict[str, Optional[float]]:
        """Return the range as a JSON-serialisable dictionary."""
        return {
            "lower": float(self.lower),
            "median": float(self.median),
            "upper": float(self.upper),
            "scale": None if self.scale is None else float(self.scale),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "MetricRange":
        """Rebuild a range from :meth:`to_dict` output."""
        scale = data.get("scale")
        return cls(
            lower=float(data["lower"]),        # type: ignore[arg-type]
            median=float(data["median"]),      # type: ignore[arg-type]
            upper=float(data["upper"]),        # type: ignore[arg-type]
            scale=None if scale is None else float(scale),  # type: ignore[arg-type]
        )


@dataclass
class HealthyReference:
    """Healthy centroid, its region boundary and the metric intervals."""

    log_centroid: np.ndarray                     #: ``Z_H`` (log domain)

    #: outer boundary of the healthy region in Frobenius distance
    manifold_distance_threshold: float = 0.0
    #: median distance of the healthy strides to their own centroid
    manifold_distance_median: float = 0.0
    #: distance beyond the threshold that maps to a manifold deviation of 1
    manifold_distance_scale: float = 0.0

    #: reference interval per biomechanical metric
    metric_ranges: Dict[str, MetricRange] = field(default_factory=dict)

    num_strides: int = 0
    #: whether the intervals are conditioned on walking speed; the interface
    #: accepts a speed already, this build always answers with the pooled
    #: population interval
    speed_conditioned: bool = False

    # -- distribution of the healthy distances, kept for reporting --------- #
    distance_mean: float = 0.0
    distance_std: float = 0.0
    feature_names: Sequence[str] = ()
    #: mean value of every healthy metric, kept for the compatibility path
    metrics: Dict[str, float] = field(default_factory=dict)

    # -- backward-compatible aliases --------------------------------------- #

    @property
    def distance_threshold(self) -> float:
        """Historical name of :attr:`manifold_distance_threshold`."""
        return self.manifold_distance_threshold

    @property
    def n_strides(self) -> int:
        """Historical name of :attr:`num_strides`."""
        return self.num_strides

    @property
    def centroid_vector(self) -> np.ndarray:
        """Vectorised healthy centroid."""
        return vectorize_spd(self.log_centroid)

    # -- the region ---------------------------------------------------------- #

    def distance(self, log_matrix: np.ndarray) -> float:
        """Distance from a log-domain stride to the healthy centroid."""
        return frobenius_distance(log_matrix, self.log_centroid)

    def is_healthy_like(self, log_matrix: np.ndarray) -> bool:
        """Return True when the stride lies inside the healthy region."""
        return self.distance(log_matrix) <= self.manifold_distance_threshold

    def region_excess(self, log_matrix: np.ndarray) -> float:
        """Distance by which the stride overshoots the region boundary.

        Returns:
            ``0`` for any stride inside the region, so the patient is never
            driven towards the healthy centroid itself.
        """
        return max(self.distance(log_matrix) - self.manifold_distance_threshold, 0.0)

    def manifold_deviation(
        self, log_matrix: np.ndarray, scale: Optional[float] = None
    ) -> float:
        """Normalised deviation outside the healthy region, clipped to [0, 1].

        Args:
            log_matrix: ``Z_t = log(C_t)`` of the current stride.
            scale: excess distance mapping to 1.0; defaults to
                :attr:`manifold_distance_scale`.

        Returns:
            ``clip((d_H - threshold) / scale, 0, 1)``; exactly 0 inside the
            region.
        """
        excess = self.region_excess(log_matrix)
        if excess <= 0.0:
            return 0.0
        span = float(scale if scale else self.manifold_distance_scale)
        if span <= 0.0:
            return 1.0
        return float(np.clip(excess / span, 0.0, 1.0))

    # -- the metric intervals ------------------------------------------------ #

    def get_metric_range(
        self, metric: str, walking_speed: Optional[float] = None
    ) -> Optional[MetricRange]:
        """Return the reference interval of ``metric``.

        Args:
            metric: metric name, e.g. ``"swing_ratio"``.
            walking_speed: speed of the current stride (m/s).  Accepted so
                call sites are already speed-aware; this build has no
                speed-conditioned reference and answers with the pooled
                population interval regardless.  A future implementation can
                select a speed bin, a nearest speed group or a regression-based
                expectation here without changing any caller.

        Returns:
            The :class:`MetricRange`, or ``None`` when the healthy data set did
            not measure this metric.  ``None`` must leave the corresponding
            error undefined rather than zero.
        """
        del walking_speed  # accepted for the interface; see the docstring
        return self.metric_ranges.get(metric)

    def has_metric(self, metric: str) -> bool:
        """True when a reference interval exists for ``metric``."""
        return metric in self.metric_ranges

    # -- persistence -------------------------------------------------------- #

    def to_metadata(self) -> Dict[str, object]:
        """JSON-serialisable description of everything but the centroid."""
        return {
            "manifold_distance_threshold": float(self.manifold_distance_threshold),
            "manifold_distance_median": float(self.manifold_distance_median),
            "manifold_distance_scale": float(self.manifold_distance_scale),
            "metric_ranges": {k: v.to_dict() for k, v in self.metric_ranges.items()},
            "num_strides": int(self.num_strides),
            "speed_conditioned": bool(self.speed_conditioned),
            "distance_mean": float(self.distance_mean),
            "distance_std": float(self.distance_std),
            "feature_names": list(self.feature_names),
            "metrics": {k: float(v) for k, v in self.metrics.items()},
        }

    @classmethod
    def from_metadata(
        cls,
        log_centroid: np.ndarray,
        metadata: Mapping[str, object],
        *,
        fallback_threshold: float = 0.0,
        fallback_mean: float = 0.0,
        fallback_std: float = 0.0,
        fallback_strides: int = 0,
        fallback_feature_names: Sequence[str] = (),
        fallback_metrics: Optional[Mapping[str, float]] = None,
    ) -> "HealthyReference":
        """Rebuild a reference from its centroid and :meth:`to_metadata` dict.

        Args:
            log_centroid: ``Z_H``.
            metadata: what :meth:`to_metadata` produced.
            fallback_threshold: threshold to use when the metadata omits it.
            fallback_mean: distance mean to use when the metadata omits it.
            fallback_std: distance spread to use when the metadata omits it.
            fallback_strides: stride count to use when the metadata omits it.
            fallback_feature_names: feature names to use when omitted.
            fallback_metrics: healthy metric means to use when omitted.

        Returns:
            The reference.  Shared by :meth:`load` and by
            :class:`ReferenceBank`, so both formats interpret the metadata the
            same way.
        """
        meta = dict(metadata)
        return cls(
            log_centroid=np.asarray(log_centroid, dtype=float),
            manifold_distance_threshold=float(
                meta.get("manifold_distance_threshold", fallback_threshold)
            ),
            manifold_distance_median=float(
                meta.get("manifold_distance_median", fallback_mean)
            ),
            manifold_distance_scale=float(
                meta.get("manifold_distance_scale", max(fallback_std, 1e-9))
            ),
            metric_ranges={
                k: MetricRange.from_dict(v)
                for k, v in meta.get("metric_ranges", {}).items()
            },
            num_strides=int(meta.get("num_strides", fallback_strides)),
            speed_conditioned=bool(meta.get("speed_conditioned", False)),
            distance_mean=float(meta.get("distance_mean", fallback_mean)),
            distance_std=float(meta.get("distance_std", fallback_std)),
            feature_names=tuple(meta.get("feature_names", fallback_feature_names)),
            metrics=dict(meta.get("metrics", fallback_metrics or {})),
        )

    def save(self, path: Union[str, Path]) -> None:
        """Store the reference as a compressed ``.npz`` archive.

        The scalar fields are written both individually and inside a JSON
        ``metadata`` entry.  The individual fields keep readers of the previous
        format working; the JSON entry carries the region and the metric
        ranges, which that format had no place for.
        """
        np.savez(
            Path(path),
            log_centroid=self.log_centroid,
            distance_mean=self.distance_mean,
            distance_std=self.distance_std,
            distance_threshold=self.manifold_distance_threshold,
            n_strides=self.num_strides,
            feature_names=np.array(list(self.feature_names), dtype=object),
            metrics=json.dumps(self.metrics),
            metadata=json.dumps(self.to_metadata()),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "HealthyReference":
        """Load a reference written by :meth:`save`.

        Archives written before the region existed carry no ``metadata`` entry;
        they are loaded with their stored threshold, an empty set of metric
        ranges and a scale derived from the stored spread, so an old reference
        still runs - with no metric intervals, which the policy reports as
        undefined rather than satisfied.
        """
        with np.load(Path(path), allow_pickle=True) as data:
            centroid = np.asarray(data["log_centroid"], dtype=float)
            mean = float(data["distance_mean"])
            std = float(data["distance_std"])
            threshold = float(data["distance_threshold"])
            legacy_metrics = json.loads(str(data["metrics"])) if "metrics" in data else {}
            feature_names = (
                tuple(str(x) for x in data["feature_names"].tolist())
                if "feature_names" in data
                else ()
            )
            if "metadata" in data:
                return cls.from_metadata(
                    centroid,
                    json.loads(str(data["metadata"])),
                    fallback_threshold=threshold,
                    fallback_mean=mean,
                    fallback_std=std,
                    fallback_strides=int(data["n_strides"]),
                    fallback_feature_names=feature_names,
                    fallback_metrics=legacy_metrics,
                )
            return cls(
                log_centroid=centroid,
                manifold_distance_threshold=threshold,
                manifold_distance_median=mean,
                manifold_distance_scale=max(std, 1e-9),
                metric_ranges={},
                num_strides=int(data["n_strides"]),
                distance_mean=mean,
                distance_std=std,
                feature_names=feature_names,
                metrics=legacy_metrics,
            )


@dataclass
class ReferenceBank:
    """Several healthy references, one per cadence, chosen per stride.

    One healthy reference is only comparable to gait at the speed it was
    recorded at.  The same person walking more slowly takes longer strides in
    time and moves the belt less, so measured against a faster reference that
    reads as a deficit that is not there: the 14:42 run (stride 1.76 s) scored
    ``E_B = 0.105`` against the 1.0 m/s reference and ``0.004`` against the
    0.4 m/s one, from the same recordings of the same person.

    The bank holds one reference per recorded cadence and selects the one whose
    ``stride_time`` interval the stride actually falls in.  Stride time is used
    rather than walking speed because the device measures it; a belt-mounted
    sensor has no ground speed.

    Args:
        references: label -> reference, one per recorded cadence.
        select_metric: metric whose interval decides the selection.
    """

    references: Dict[str, HealthyReference]
    select_metric: str = "stride_time"

    def __post_init__(self) -> None:
        if not self.references:
            raise ValueError("a reference bank needs at least one reference")

    @property
    def labels(self) -> Sequence[str]:
        """Labels ordered by their selection interval, slowest cadence last."""
        def key(label: str) -> float:
            interval = self.interval(label)
            return interval.lower if interval is not None else float("inf")
        return sorted(self.references, key=key)

    def interval(self, label: str) -> Optional[MetricRange]:
        """Selection interval of one reference, or ``None`` when it has none."""
        return self.references[label].metric_ranges.get(self.select_metric)

    def gap(self, label: str, value: float) -> float:
        """Distance from ``value`` to a reference's selection interval.

        Returns:
            ``0`` when the value lies inside the interval, otherwise how far
            outside it is.  A reference without the selection metric is
            infinitely far away, so it is only ever chosen as a last resort.
        """
        interval = self.interval(label)
        if interval is None:
            return float("inf")
        if value < interval.lower:
            return float(interval.lower - value)
        if value > interval.upper:
            return float(value - interval.upper)
        return 0.0

    def select(self, value: Optional[float]) -> str:
        """Label of the reference to measure a stride of this cadence against.

        Args:
            value: the stride's ``select_metric`` value, e.g. its stride time.

        Returns:
            The label whose interval contains the value, or failing that the
            nearest one.  A missing or non-finite value selects the slowest
            reference rather than guessing: over-estimating the cadence is what
            invents a deficit.
        """
        labels = list(self.labels)
        if value is None or not np.isfinite(value):
            return labels[-1]
        return min(labels, key=lambda label: (self.gap(label, float(value)), label))

    def reference(self, value: Optional[float]) -> HealthyReference:
        """The reference :meth:`select` picks for this cadence."""
        return self.references[self.select(value)]

    def describe(self) -> str:
        """One line per reference: its label, interval and stride count."""
        rows = []
        for label in self.labels:
            interval = self.interval(label)
            span = (
                f"{interval.lower:.2f}..{interval.upper:.2f}"
                if interval is not None
                else "no interval"
            )
            rows.append(f"{label}: {self.select_metric} {span} "
                        f"({self.references[label].num_strides} strides)")
        return "\n".join(rows)

    # -- persistence -------------------------------------------------------- #

    def save(self, path: Union[str, Path]) -> None:
        """Store the whole bank as one ``.npz`` archive.

        The archive carries a ``bank`` entry, which is what
        :func:`load_reference` sniffs for to tell a bank from a single
        reference.
        """
        arrays: Dict[str, object] = {
            "bank": json.dumps(
                {"labels": list(self.labels), "select_metric": self.select_metric}
            )
        }
        for label, ref in self.references.items():
            arrays[f"{label}__log_centroid"] = ref.log_centroid
            arrays[f"{label}__metadata"] = json.dumps(ref.to_metadata())
        np.savez(Path(path), **arrays)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ReferenceBank":
        """Load a bank written by :meth:`save`.

        Raises:
            ValueError: if the archive is not a bank.
        """
        with np.load(Path(path), allow_pickle=True) as data:
            if "bank" not in data:
                raise ValueError(f"{path} is a single reference, not a bank")
            header = json.loads(str(data["bank"]))
            references = {
                label: HealthyReference.from_metadata(
                    np.asarray(data[f"{label}__log_centroid"], dtype=float),
                    json.loads(str(data[f"{label}__metadata"])),
                )
                for label in header["labels"]
            }
        return cls(references, str(header.get("select_metric", "stride_time")))

    @classmethod
    def from_files(
        cls, paths: Mapping[str, Union[str, Path]], select_metric: str = "stride_time"
    ) -> "ReferenceBank":
        """Build a bank from single-reference archives.

        Args:
            paths: label -> path of a reference written by
                :meth:`HealthyReference.save`.
            select_metric: metric whose interval decides the selection.

        Returns:
            The bank.  Every reference is marked ``speed_conditioned`` so a
            reader of one member knows it is not a pooled population interval.
        """
        references = {}
        for label, path in paths.items():
            ref = HealthyReference.load(path)
            ref.speed_conditioned = True
            references[label] = ref
        return cls(references, select_metric)


def load_reference(
    path: Union[str, Path]
) -> Union[HealthyReference, "ReferenceBank"]:
    """Load whichever reference format sits at ``path``.

    Args:
        path: ``.npz`` archive holding either one reference or a bank.

    Returns:
        A :class:`ReferenceBank` when the archive carries one, otherwise a
        single :class:`HealthyReference`.  Call sites take both, so pointing
        ``reference.path`` at a bank is the only change a cadence-conditioned
        run needs.
    """
    with np.load(Path(path), allow_pickle=True) as data:
        is_bank = "bank" in data
    return ReferenceBank.load(path) if is_bank else HealthyReference.load(path)


def distance_threshold(
    distances: np.ndarray, config: Optional[ReferenceConfig] = None
) -> float:
    """Compute the healthy region boundary from the healthy distances.

    Args:
        distances: distance of every healthy stride to the healthy centroid.
        config: selects the method and its parameter.

    Returns:
        The boundary distance.  ``percentile`` (default, 95th) makes no
        distributional assumption; ``mean_std`` reproduces the earlier
        ``mean + sigma*std``; ``median_mad`` is the robust variant, which
        matters because a single mistracked healthy stride inflates both the
        mean and the standard deviation.

    Raises:
        ValueError: if the method is unknown.
    """
    cfg = config or ReferenceConfig()
    values = np.asarray(distances, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    method = ThresholdMethod(cfg.threshold_method)
    if method is ThresholdMethod.PERCENTILE:
        return float(np.percentile(values, cfg.threshold_percentile))
    if method is ThresholdMethod.MEAN_STD:
        return float(np.mean(values) + cfg.threshold_sigma * np.std(values))
    if method is ThresholdMethod.MEDIAN_MAD:
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return median + cfg.threshold_mad_k * mad
    raise ValueError(f"unknown threshold method: {cfg.threshold_method!r}")


def build_metric_ranges(
    samples: Mapping[str, Sequence[float]],
    config: Optional[ReferenceConfig] = None,
) -> Dict[str, MetricRange]:
    """Build a reference interval for every metric measured on healthy strides.

    Args:
        samples: metric name -> the healthy values observed for it.  A metric
            that no healthy stride carried must be absent, not empty-listed.
        config: supplies the two percentiles bounding the interval.

    Returns:
        Mapping from metric name to its :class:`MetricRange`.  The scale of
        each range defaults to half the interval width, so a deviation as large
        as the healthy spread itself saturates the error.
    """
    cfg = config or ReferenceConfig()
    ranges: Dict[str, MetricRange] = {}
    for name, values in samples.items():
        data = np.asarray(list(values), dtype=float)
        data = data[np.isfinite(data)]
        if data.size == 0:
            continue
        lower = float(np.percentile(data, cfg.metric_lower_percentile))
        upper = float(np.percentile(data, cfg.metric_upper_percentile))
        median = float(np.median(data))
        half_width = max((upper - lower) / 2.0, 0.0)
        ranges[name] = MetricRange(
            lower=lower,
            median=median,
            upper=upper,
            scale=half_width if half_width > 0.0 else None,
        )
    return ranges


def build_healthy_reference(
    log_matrices: Sequence[np.ndarray],
    *,
    config: Optional[ReferenceConfig] = None,
    feature_names: Sequence[str] = (),
    metrics: Optional[Mapping[str, float]] = None,
    metric_samples: Optional[Mapping[str, Sequence[float]]] = None,
    epsilon: float = DEFAULT_EPSILON,
) -> HealthyReference:
    """Build the healthy reference region from log-transformed healthy strides.

    Args:
        log_matrices: ``log(C_i)`` of every healthy stride.
        config: reference configuration (threshold method, metric percentiles).
        feature_names: feature channels used, kept for traceability.
        metrics: mean biomechanical metrics of the healthy data set; kept for
            the compatibility path and for reporting.
        metric_samples: the per-stride healthy values of each metric, from
            which the reference intervals are built.  Without them the
            reference carries no metric ranges and the biomechanical errors
            stay undefined.
        epsilon: eigenvalue floor (unused here but kept for symmetry).

    Returns:
        The :class:`HealthyReference`.

    Raises:
        ValueError: if no stride is supplied.
    """
    if len(log_matrices) == 0:
        raise ValueError("at least one healthy stride is required")
    cfg = config or ReferenceConfig()
    centroid = log_euclidean_mean(log_matrices, epsilon, already_log=True)
    distances = np.array(
        [frobenius_distance(m, centroid) for m in log_matrices], dtype=float
    )
    threshold = distance_threshold(distances, cfg)
    median = float(np.median(distances))
    # The excess that saturates the manifold deviation is the spread of the
    # healthy population itself: being a whole healthy spread outside the
    # region is as deviant as this reference can describe.
    scale = max(threshold - median, float(np.std(distances)), 1e-9)
    return HealthyReference(
        log_centroid=centroid,
        manifold_distance_threshold=threshold,
        manifold_distance_median=median,
        manifold_distance_scale=scale,
        metric_ranges=build_metric_ranges(metric_samples or {}, cfg),
        num_strides=len(log_matrices),
        speed_conditioned=False,
        distance_mean=float(np.mean(distances)),
        distance_std=float(np.std(distances)),
        feature_names=tuple(feature_names),
        metrics=dict(metrics or {}),
    )


__all__ = [
    "HealthyReference",
    "MetricRange",
    "ReferenceBank",
    "build_healthy_reference",
    "build_metric_ranges",
    "distance_threshold",
    "load_reference",
]
