"""Deviation score and assist gain (spec 19, 20).

Riemannian deviation alone never decides the assistance: the manifold term is
combined with interpretable biomechanical terms, and the resulting score is
clipped to ``[0, 1]`` before it is turned into a gain.  The gain itself is rate
limited between strides and hard-capped while the stride is out of
distribution.

Two manifold deviations are computed and reported side by side instead of one
lumped number, because they answer different questions and are normalised
against different scales:

``e_baseline``
    Departure from the patient-specific baseline, scaled by the spread of the
    baseline strides around their own centroid.  1.0 means "as unlike this
    patient's opening strides as their most atypical opening stride was".  It
    is a consistency measure and carries no claim about pathology.

``e_healthy``
    Departure from the able-bodied reference, scaled by the healthy threshold.
    This is the pathological deviation.  It is NaN in
    ``AssistMode.BASELINE_STABILIZATION`` and is never silently replaced by
    ``e_baseline`` there.

``e_manifold`` stays the term that actually drives the gain: the distance to
``Z_target``.  In ``BASELINE_STABILIZATION`` that target *is* ``Z_patient``, so
``e_manifold`` and ``e_baseline`` coincide by construction - reporting all
three makes that coincidence visible rather than hiding it behind one column.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ..config import (
    AssistConfig,
    AssistMode,
    Config,
    DeviationConfig,
    OodAssistPolicy,
    ReferenceConfig,
)
from ..gait.feature_extractor import (
    GaitMetrics,
    StrideMetrics,
    collect_metric_samples,
)
from ..manifold.reference import (
    HealthyReference,
    MetricRange,
    ReferenceBank,
    build_metric_ranges,
)
from ..patient.patient_model import PatientModel, StrideAnalysis


@dataclass(frozen=True)
class DeviationComponents:
    """Individual error terms and the combined score.

    ``e_manifold`` is the term the gain is computed from; ``e_baseline`` and
    ``e_healthy`` are the two deviations it sits between, kept separate so a
    consistency signal is never read as a severity signal.
    """

    e_manifold: float
    e_swing: float
    e_excursion: float
    score: float
    #: departure from the patient-specific baseline (consistency)
    e_baseline: float = 0.0
    #: departure from able-bodied gait (pathology); NaN without a reference
    e_healthy: float = float("nan")
    #: which reference ``e_manifold`` was measured against
    mode: AssistMode = AssistMode.BASELINE_STABILIZATION

    def to_dict(self) -> Dict[str, object]:
        """Return the components as a plain dictionary."""
        return {
            "e_manifold": self.e_manifold,
            "e_baseline": self.e_baseline,
            "e_healthy": self.e_healthy,
            "e_swing": self.e_swing,
            "e_excursion": self.e_excursion,
            "deviation_score": self.score,
            "assist_mode": self.mode.value,
        }


class DeviationScorer:
    """Combine manifold and biomechanical deviation into one score.

    Args:
        config: deviation weights and tolerances.
        manifold_scale: distance to ``Z_target`` mapping to a manifold error of
            1.0; when ``0`` the value from ``config`` is used, and when that is
            also 0 the scale must be provided through :meth:`calibrate`.
        target_swing_ratio: swing-ratio target the estimated ratio is compared
            against.
        reference_excursion_mm: belt excursion used as the reference.
        baseline_scale: distance to ``Z_patient`` mapping to a baseline
            deviation of 1.0; falls back to ``manifold_scale``.
        healthy_scale: distance to ``Z_healthy`` mapping to a pathological
            deviation of 1.0; ``0`` leaves ``e_healthy`` undefined.
        mode: which reference the target represents.
    """

    def __init__(
        self,
        config: Optional[DeviationConfig] = None,
        *,
        manifold_scale: float = 0.0,
        target_swing_ratio: Optional[float] = None,
        reference_excursion_mm: Optional[float] = None,
        baseline_scale: float = 0.0,
        healthy_scale: float = 0.0,
        mode: AssistMode = AssistMode.BASELINE_STABILIZATION,
    ) -> None:
        self.config = config or DeviationConfig()
        self.manifold_scale = manifold_scale or self.config.manifold_scale
        self.baseline_scale = (
            baseline_scale or self.config.baseline_scale or self.manifold_scale
        )
        self.healthy_scale = healthy_scale or self.config.healthy_scale
        self.mode = mode
        self.target_swing_ratio = (
            self.config.target_swing_ratio if target_swing_ratio is None else target_swing_ratio
        )
        self.reference_excursion_mm = (
            self.config.reference_excursion_mm
            if reference_excursion_mm is None
            else reference_excursion_mm
        )

    @classmethod
    def from_model(
        cls, model: PatientModel, config: Optional[DeviationConfig] = None
    ) -> "DeviationScorer":
        """Calibrate a scorer against a fitted patient model.

        The manifold scale defaults to ``mean + 2*std`` of the baseline
        distances to ``Z_target`` (or the healthy threshold when a healthy
        reference is available), so a score of 1 means "as far from target as
        the worst baseline stride".

        The baseline and healthy scales are derived independently, from the
        baseline spread and from the healthy threshold respectively, so that
        ``e_baseline`` and ``e_healthy`` remain separately interpretable
        instead of being two readings of the same normalisation.
        """
        cfg = config or DeviationConfig()
        stats = model.baseline_distance_stats
        scale = cfg.manifold_scale
        if scale <= 0.0:
            if model.healthy is not None and model.healthy.distance_threshold > 0.0:
                scale = model.healthy.distance_threshold
            else:
                scale = stats["d_target_mean"] + 2.0 * stats["d_target_std"]
        baseline_scale = cfg.baseline_scale
        if baseline_scale <= 0.0:
            baseline_scale = stats["d_patient_mean"] + 2.0 * stats["d_patient_std"]
        healthy_scale = cfg.healthy_scale
        if healthy_scale <= 0.0 and model.healthy is not None:
            healthy_scale = model.healthy.distance_threshold
            if healthy_scale <= 0.0:
                healthy_scale = stats.get("d_healthy_mean", 0.0) + 2.0 * stats.get(
                    "d_healthy_std", 0.0
                )
        # Biomechanical targets come from the healthy data set when one exists.
        # Without it the patient's own baseline supplies the reference, so the
        # score measures departure from this patient's habitual gait rather
        # than a constant offset against a population value that may not apply
        # to this sensor mounting.  That is a stabilisation target, not a
        # normative one, which is what AssistMode records.
        healthy_metrics = model.healthy.metrics if model.healthy is not None else {}
        baseline_metrics = model.baseline.mean_metrics
        target_ratio = float(
            healthy_metrics.get(
                "swing_ratio",
                baseline_metrics.get("swing_ratio", cfg.target_swing_ratio),
            )
        )
        excursion = float(
            healthy_metrics.get(
                "belt_excursion",
                cfg.reference_excursion_mm
                or baseline_metrics.get("belt_excursion", 0.0),
            )
        )
        return cls(
            cfg,
            manifold_scale=scale,
            target_swing_ratio=target_ratio,
            reference_excursion_mm=excursion,
            baseline_scale=baseline_scale,
            healthy_scale=healthy_scale,
            mode=model.assist_mode,
        )

    def calibrate(self, manifold_scale: float) -> None:
        """Set the manifold scale explicitly."""
        self.manifold_scale = float(manifold_scale)

    def score(
        self, analysis: StrideAnalysis, metrics: Optional[StrideMetrics] = None
    ) -> DeviationComponents:
        """Compute the deviation components of one stride.

        Args:
            analysis: high-level stride analysis.
            metrics: biomechanical metrics; taken from ``analysis`` if omitted.

        Returns:
            The :class:`DeviationComponents`.  An invalid analysis yields an
            all-zero score so that no assistance is produced.  Only
            ``e_manifold`` enters the score; ``e_baseline`` and ``e_healthy``
            are reported for interpretation and never summed into it, so the
            gain cannot be driven twice by the same distance.
        """
        cfg = self.config
        if not analysis.valid:
            return DeviationComponents(
                e_manifold=0.0,
                e_swing=0.0,
                e_excursion=0.0,
                score=0.0,
                e_baseline=0.0,
                e_healthy=float("nan"),
                mode=self.mode,
            )

        e_manifold = self._manifold_error(analysis.d_target)
        e_baseline = self._normalised(analysis.d_patient, self.baseline_scale)
        e_healthy = self._normalised(analysis.d_healthy, self.healthy_scale)
        stride_metrics = metrics or analysis.metrics
        e_swing = self._swing_error(stride_metrics)
        e_excursion = self._excursion_error(stride_metrics)

        raw = (
            cfg.w_manifold * e_manifold
            + cfg.w_swing * e_swing
            + cfg.w_excursion * e_excursion
        )
        return DeviationComponents(
            e_manifold=e_manifold,
            e_swing=e_swing,
            e_excursion=e_excursion,
            score=float(np.clip(raw, 0.0, 1.0)),
            e_baseline=e_baseline,
            e_healthy=e_healthy,
            mode=self.mode,
        )

    # -- individual error terms -------------------------------------------- #

    def _manifold_error(self, distance: float) -> float:
        """Normalised Log-Euclidean distance to ``Z_target``, clipped to [0, 1]."""
        if not np.isfinite(distance) or self.manifold_scale <= 0.0:
            return 0.0
        return float(np.clip(distance / self.manifold_scale, 0.0, 1.0))

    @staticmethod
    def _normalised(distance: float, scale: float) -> float:
        """Scale a reported distance to ``[0, 1]``, or NaN when undefined.

        Unlike :meth:`_manifold_error` a missing scale or a NaN distance yields
        NaN rather than 0, because these terms are reported rather than acted
        on: a pathological deviation that could not be measured must not read
        as a pathological deviation of zero.
        """
        if not np.isfinite(distance) or scale <= 0.0:
            return float("nan")
        return float(np.clip(distance / scale, 0.0, 1.0))

    def _swing_error(self, metrics: Optional[StrideMetrics]) -> float:
        """Normalised deviation of the swing ratio from its target."""
        if metrics is None or self.config.swing_ratio_tolerance <= 0.0:
            return 0.0
        deviation = abs(metrics.swing_ratio - self.target_swing_ratio)
        return float(np.clip(deviation / self.config.swing_ratio_tolerance, 0.0, 1.0))

    def _excursion_error(self, metrics: Optional[StrideMetrics]) -> float:
        """Normalised *deficit* of belt excursion; an excess is not a deficit."""
        if metrics is None or self.reference_excursion_mm <= 0.0:
            return 0.0
        deficit = (self.reference_excursion_mm - metrics.belt_excursion) / self.reference_excursion_mm
        tolerance = max(self.config.excursion_tolerance, 1e-6)
        return float(np.clip(deficit / tolerance, 0.0, 1.0))


class AssistanceGainPolicy:
    """Turn a deviation score into a rate-limited assist gain (spec 20).

    Args:
        config: assist configuration (``max_gain``, ``max_gain_delta``,
            ``ood_max_gain``).
    """

    def __init__(self, config: Optional[AssistConfig] = None) -> None:
        self.config = config or AssistConfig()
        self._gain = 0.0
        self.previous_gain = 0.0

    @property
    def gain(self) -> float:
        """Current assist gain."""
        return self._gain

    def reset(self, gain: float = 0.0) -> None:
        """Force the gain without rate limiting (used on safe stop)."""
        self._gain = float(np.clip(gain, 0.0, self.config.max_gain))
        self.previous_gain = self._gain

    def update(
        self, score: float, *, is_ood: bool = False, assist_allowed: bool = True
    ) -> float:
        """Compute the gain for the next stride.

        Args:
            score: deviation score in ``[0, 1]``.
            is_ood: whether the stride was flagged out of distribution; the
                gain is then capped at ``ood_max_gain`` (spec 14, 25).
            assist_allowed: safety veto; when False the target gain is 0.

        Returns:
            The new gain, which differs from the previous one by at most
            ``max_gain_delta``.
        """
        cfg = self.config
        if not np.isfinite(score):
            score = 0.0
        if cfg.fixed_gain > 0.0:
            # Bring-up override: hold the gain so the force is deterministic.
            # The safety veto below still applies, and the current limit still
            # caps what reaches the motor.
            target = float(np.clip(cfg.fixed_gain, 0.0, cfg.max_gain))
        else:
            target = float(np.clip(score, 0.0, 1.0)) * cfg.max_gain
            if is_ood:
                target = min(target, cfg.ood_max_gain)
        if not assist_allowed:
            target = 0.0
        delta = float(np.clip(target - self._gain, -cfg.max_gain_delta, cfg.max_gain_delta))
        self.previous_gain = self._gain
        self._gain = float(np.clip(self._gain + delta, 0.0, cfg.max_gain))
        return self._gain

    def emergency_zero(self) -> float:
        """Drop the gain to zero immediately, bypassing the rate limit."""
        self.previous_gain = self._gain
        self._gain = 0.0
        return self._gain




# --------------------------------------------------------------------------- #
# Device-actionable biomechanical deficit (spec 7, 8)
# --------------------------------------------------------------------------- #


class DecisionState(str, Enum):
    """What the policy concluded about one stride.

    The state is the readable form of the two-factor rule: a manifold
    deviation describes *how atypical* the whole gait pattern is, a
    biomechanical deficit describes *what this device can correct*, and only
    the second one is allowed to open the throttle.
    """

    IN_RANGE = "IN_RANGE"
    BIOMECH_DEFICIT_ONLY = "BIOMECH_DEFICIT_ONLY"
    MANIFOLD_DEVIATION_ONLY = "MANIFOLD_DEVIATION_ONLY"
    COMBINED_DEVIATION = "COMBINED_DEVIATION"
    OOD = "OOD"


#: Reference-interval source: an able-bodied data set.
SOURCE_HEALTHY: str = "healthy_reference"
#: Reference-interval source: the patient's own opening strides, used when no
#: healthy data set was supplied.  Errors then mean "worse than this patient's
#: own baseline", never "abnormal".
SOURCE_BASELINE: str = "patient_baseline"

#: Direction of the error each device-actionable term measures.
#: ``deficit`` - only values below the interval count; ``excess`` - only values
#: above it; ``two_sided`` - either side counts.
TERM_DIRECTIONS: Dict[str, str] = {
    "swing_ratio": "deficit",
    "belt_excursion": "deficit",
    "temporal_symmetry": "two_sided",
    "trunk_compensation": "excess",
}

#: Metric fields whose value comes from the gait phase.  While a placeholder
#: phase detector is in use these are functions of the configured cadence, not
#: of the patient, so a term reading one of them cannot show a real deficit.
PHASE_DERIVED_METRICS: frozenset = frozenset(
    {
        "swing_time",
        "stance_time",
        "swing_ratio",
        "stance_ratio",
        "swing_stance_ratio",
        "swing_symmetry_ratio",
        "stance_symmetry_ratio",
        "swing_stance_symmetry_ratio",
    }
)

#: Which metric field each device-actionable term reads.
TERM_METRICS: Dict[str, str] = {
    "swing_ratio": "swing_ratio",
    "belt_excursion": "belt_excursion",
    "temporal_symmetry": "swing_symmetry_ratio",
    "trunk_compensation": "trunk_acc_rms",
}


def interval_error(
    value: Optional[float],
    metric_range: Optional[MetricRange],
    *,
    direction: str = "two_sided",
    scale: float = 0.0,
) -> Optional[float]:
    """Normalised departure of ``value`` from its reference interval.

    Inside the interval the error is exactly 0: the reference is a range, so a
    value anywhere in it is already as normal as the reference can attest, and
    nudging it towards the median would be assisting a difference that healthy
    walkers show among themselves.

    Args:
        value: measured metric, or ``None`` when it was not measured.
        metric_range: reference interval, or ``None`` when the reference has
            none for this metric.
        direction: ``"deficit"`` counts only values below the interval,
            ``"excess"`` only values above it, ``"two_sided"`` counts both.
            A directional term exists because the device acts one way: a belt
            excursion *larger* than healthy is not something a swing-assist
            device should push harder against.
        scale: departure that saturates the error; ``0`` falls back to the
            range's own scale, and failing that to half its width.

    Returns:
        A value in ``[0, 1]``, or ``None`` when either the measurement or the
        reference interval is missing.  ``None`` is not 0: an unmeasured
        deficit must not read as an absent deficit.
    """
    if value is None or metric_range is None:
        return None
    if not np.isfinite(value):
        return None
    excess = metric_range.excess(float(value))
    if direction == "deficit":
        departure = max(-excess, 0.0)
    elif direction == "excess":
        departure = max(excess, 0.0)
    else:
        departure = abs(excess)
    if departure <= 0.0:
        return 0.0
    span = scale if scale > 0.0 else (metric_range.scale or metric_range.width() / 2.0)
    if not span or span <= 0.0:
        return 1.0
    return float(np.clip(departure / span, 0.0, 1.0))


@dataclass(frozen=True)
class BiomechanicalDeficit:
    """The device-actionable error terms and their weighted combination."""

    swing_ratio_error: Optional[float]
    belt_excursion_error: Optional[float]
    temporal_symmetry_error: Optional[float]
    trunk_error: Optional[float]
    #: ``E_B``: weighted sum of the terms that were measurable
    score: float
    #: where the reference intervals came from
    reference_source: str
    #: terms that could not be evaluated, for the log
    undefined_terms: Sequence[str] = ()

    def to_dict(self) -> Dict[str, object]:
        """Return the deficit as a plain dictionary."""
        return {
            "swing_ratio_error": self.swing_ratio_error,
            "belt_excursion_error": self.belt_excursion_error,
            "temporal_symmetry_error": self.temporal_symmetry_error,
            "trunk_error": self.trunk_error,
            "biomechanical_deviation": self.score,
            "reference_source": self.reference_source,
        }


class BiomechanicalEvaluator:
    """Turn one stride's metrics into a device-actionable deficit ``E_B``.

    Only the terms carrying a non-zero weight reach the score.  Everything
    else - stance ratio, swing/stance ratio, trunk motion, the manifold
    distances - is still computed and logged, but as *evaluation* material:
    keeping the two sets apart is what stops a research metric from quietly
    becoming a control input.

    Args:
        config: weights, scales and the healthy-interval settings.
        metric_ranges: reference interval per metric.
        reference_source: whether those intervals came from an able-bodied data
            set or from the patient's own baseline.
    """

    def __init__(
        self,
        config: Optional[DeviationConfig] = None,
        metric_ranges: Optional[Mapping[str, MetricRange]] = None,
        reference_source: str = SOURCE_BASELINE,
        bank: Optional[ReferenceBank] = None,
    ) -> None:
        self.config = config or DeviationConfig()
        self.metric_ranges: Dict[str, MetricRange] = dict(metric_ranges or {})
        self.reference_source = reference_source
        #: when set, the stride's own cadence picks the intervals it is scored
        #: against, instead of one recording's speed standing in for all of them
        self.bank = bank

    @classmethod
    def from_reference(
        cls,
        config: Optional[DeviationConfig],
        healthy: Optional[Union[HealthyReference, ReferenceBank]] = None,
        baseline_metrics: Optional[Sequence[GaitMetrics]] = None,
        reference_config: Optional[ReferenceConfig] = None,
    ) -> "BiomechanicalEvaluator":
        """Build an evaluator from whichever reference is available.

        The healthy data set is preferred.  Without one the patient's own
        baseline strides supply the intervals, computed with the same
        percentile rule, and the source is labelled accordingly so no reader
        mistakes "outside this patient's own opening spread" for "abnormal".

        Args:
            config: deviation configuration.
            healthy: healthy reference, when one was loaded.
            baseline_metrics: per-stride metrics of the baseline strides.
            reference_config: percentiles used to build baseline intervals.

        Returns:
            The evaluator.
        """
        if isinstance(healthy, ReferenceBank):
            # the slowest reference is the opening choice; evaluate() reselects
            # per stride, and a missing cadence must not score against a fast
            # reference because that is what invents a deficit
            opening = healthy.reference(None)
            return cls(config, opening.metric_ranges, SOURCE_HEALTHY, bank=healthy)
        if healthy is not None and healthy.metric_ranges:
            return cls(config, healthy.metric_ranges, SOURCE_HEALTHY)
        if baseline_metrics:
            samples = collect_metric_samples(baseline_metrics)
            ranges = build_metric_ranges(samples, reference_config)
            return cls(config, ranges, SOURCE_BASELINE)
        return cls(config, {}, SOURCE_BASELINE)

    def weight(self, term: str) -> float:
        """Configured weight of one device-actionable term."""
        return float(self.config.biomech_weights.get(term, 0.0))

    def ranges_for(
        self, metrics: Optional[GaitMetrics]
    ) -> Tuple[Mapping[str, MetricRange], str]:
        """Reference intervals this stride must be compared against.

        Args:
            metrics: the stride's metrics; its ``stride_time`` selects the
                cadence when a bank is loaded.

        Returns:
            ``(intervals, source)``.  The source carries the selected label, so
            the log says which cadence scored the stride rather than leaving
            the reader to guess.
        """
        if self.bank is None or metrics is None:
            return self.metric_ranges, self.reference_source
        label = self.bank.select(getattr(metrics, "stride_time", None))
        return self.bank.references[label].metric_ranges, f"{self.reference_source}:{label}"

    def term_error(
        self,
        term: str,
        metrics: Optional[GaitMetrics],
        walking_speed: Optional[float] = None,
        ranges: Optional[Mapping[str, MetricRange]] = None,
    ) -> Optional[float]:
        """Normalised error of one device-actionable term.

        Args:
            term: key of :data:`TERM_DIRECTIONS`.
            metrics: the stride's metrics.
            walking_speed: forwarded to the reference so a future
                speed-conditioned interval needs no change here.

        Returns:
            The error in ``[0, 1]``, or ``None`` when it is not measurable.
        """
        if metrics is None:
            return None
        field_name = TERM_METRICS.get(term)
        if field_name is None:
            return None
        value = getattr(metrics, field_name, None)
        metric_range = self._range_for(field_name, walking_speed, ranges)
        return interval_error(
            value,
            metric_range,
            direction=TERM_DIRECTIONS.get(term, "two_sided"),
            scale=self._scale_for(term, metric_range),
        )

    def evaluate(
        self, metrics: Optional[GaitMetrics], walking_speed: Optional[float] = None
    ) -> BiomechanicalDeficit:
        """Compute every term and the combined deficit ``E_B``.

        ``E_B`` is the weighted sum of the terms that could be measured.  A
        term that could not be measured contributes nothing, which lowers
        ``E_B`` and therefore the assistance - the safe direction when a signal
        is missing.

        Args:
            metrics: the stride's metrics.
            walking_speed: current walking speed, when known.

        Returns:
            The :class:`BiomechanicalDeficit`.
        """
        ranges, source = self.ranges_for(metrics)
        errors: Dict[str, Optional[float]] = {
            term: self.term_error(term, metrics, walking_speed, ranges)
            for term in TERM_DIRECTIONS
        }
        total = 0.0
        undefined: List[str] = []
        for term, error in errors.items():
            weight = self.weight(term)
            if error is None:
                if weight > 0.0:
                    undefined.append(term)
                continue
            total += weight * error
        return BiomechanicalDeficit(
            swing_ratio_error=errors["swing_ratio"],
            belt_excursion_error=errors["belt_excursion"],
            temporal_symmetry_error=errors["temporal_symmetry"],
            trunk_error=errors["trunk_compensation"],
            score=float(np.clip(total, 0.0, 1.0)),
            reference_source=source,
            undefined_terms=tuple(undefined),
        )

    # -- internals ---------------------------------------------------------- #

    def _range_for(
        self,
        field_name: str,
        walking_speed: Optional[float],
        ranges: Optional[Mapping[str, MetricRange]] = None,
    ) -> Optional[MetricRange]:
        """Reference interval of one metric field.

        The cadence already selected the interval set in :meth:`ranges_for`;
        ``walking_speed`` stays in the signature because a ground-speed sensor
        would condition on speed directly, which the belt cannot measure.
        """
        del walking_speed
        return (self.metric_ranges if ranges is None else ranges).get(field_name)

    def _scale_for(self, term: str, metric_range: Optional[MetricRange]) -> float:
        """Saturation scale of one term, from config or from the interval."""
        cfg = self.config
        configured = {
            "swing_ratio": cfg.swing_ratio_scale,
            "belt_excursion": cfg.belt_excursion_scale,
            "temporal_symmetry": cfg.symmetry_scale,
            "trunk_compensation": cfg.trunk_scale,
        }.get(term, 0.0)
        if configured and configured > 0.0:
            return float(configured)
        if term == "belt_excursion" and metric_range is not None:
            # A total loss of excursion is a full-scale deficit: the lower
            # bound of the healthy interval is the natural denominator.
            return float(max(metric_range.lower, 1e-9))
        return 0.0


# --------------------------------------------------------------------------- #
# Assistance decision (spec 9, 10, 11, 13)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AssistanceAssessment:
    """Everything the high-level loop concluded about one stride."""

    gait_state: Optional[int]
    gait_state_confidence: float
    #: this stride's raw out-of-distribution flag
    ood: bool

    #: baseline deviation: distance to the patient's own opening strides
    d_patient: float
    #: distance to the healthy centroid; NaN without a healthy reference
    d_healthy: float
    #: boundary of the healthy region; NaN without a healthy reference
    healthy_region_threshold: float

    #: ``E_R``: context/severity score, 0 inside the healthy region
    manifold_deviation: float

    swing_ratio_error: Optional[float]
    belt_excursion_error: Optional[float]
    temporal_symmetry_error: Optional[float]
    trunk_error: Optional[float]

    #: ``E_B``: the deficit this device can actually correct
    biomechanical_deviation: float

    decision_state: str

    raw_assist_gain: float
    limited_assist_gain: float

    #: ``E_assist = E_B * (base_factor + manifold_factor * E_R)``
    assist_score: float = 0.0
    reference_source: str = SOURCE_BASELINE
    #: baseline_healthy_distance - current: positive means the stride moved
    #: closer to the healthy region than the session started
    delta_healthy: float = float("nan")
    consecutive_deficit_strides: int = 0
    consecutive_recovery_strides: int = 0
    #: whether the OOD rule actually acted: ``ood`` repeated for
    #: ``required_consecutive_ood_strides`` strides in a row
    ood_engaged: bool = False
    consecutive_ood_strides: int = 0
    #: whether this stride recomputed the gain or held the previous one
    gain_updated: bool = True
    metrics: Optional[GaitMetrics] = None
    valid: bool = True
    message: str = ""

    @property
    def state(self) -> DecisionState:
        """Decision state as an enum."""
        return DecisionState(self.decision_state)

    def to_dict(self) -> Dict[str, object]:
        """Flat dictionary of the scalar fields, for logging."""
        return {
            "gait_state": self.gait_state,
            "cluster_confidence": self.gait_state_confidence,
            "ood": int(self.ood),
            "ood_engaged": int(self.ood_engaged),
            "consecutive_ood_strides": self.consecutive_ood_strides,
            "gain_updated": int(self.gain_updated),
            "d_patient": self.d_patient,
            "d_healthy": self.d_healthy,
            "healthy_region_threshold": self.healthy_region_threshold,
            "manifold_deviation": self.manifold_deviation,
            "swing_ratio_error": self.swing_ratio_error,
            "belt_excursion_error": self.belt_excursion_error,
            "temporal_symmetry_error": self.temporal_symmetry_error,
            "trunk_error": self.trunk_error,
            "biomechanical_deviation": self.biomechanical_deviation,
            "decision_state": self.decision_state,
            "raw_assist_gain": self.raw_assist_gain,
            "assist_gain": self.limited_assist_gain,
            "assist_score": self.assist_score,
            "reference_source": self.reference_source,
            "delta_healthy": self.delta_healthy,
        }


class PersistenceGate:
    """Require a deviation to repeat before it may change the assistance.

    One noisy stride must not move the gain.  The gate counts consecutive
    strides that show a deficit and consecutive strides that are back inside
    every reference interval, and only lets the gain rise or fall once the
    matching streak is long enough.  It sits *before* the rate limiter and
    solves a different problem: the rate limiter bounds how fast the gain may
    move, this decides whether it may move at all.

    Args:
        config: supplies the two streak lengths.
    """

    def __init__(self, config: Optional[AssistConfig] = None) -> None:
        self.config = config or AssistConfig()
        self.deficit_streak = 0
        self.recovery_streak = 0

    def reset(self) -> None:
        """Forget both streaks."""
        self.deficit_streak = 0
        self.recovery_streak = 0

    def update(self, has_deficit: bool) -> None:
        """Record one stride's verdict.

        Args:
            has_deficit: whether this stride showed a device-actionable
                deficit.  The two streaks are mutually exclusive, so observing
                a deficit clears the recovery streak and vice versa.
        """
        if has_deficit:
            self.deficit_streak += 1
            self.recovery_streak = 0
        else:
            self.recovery_streak += 1
            self.deficit_streak = 0

    @property
    def may_increase(self) -> bool:
        """True once a deficit has persisted long enough to act on."""
        return self.deficit_streak >= max(
            int(self.config.required_consecutive_deficit_strides), 1
        )

    @property
    def may_decrease(self) -> bool:
        """True once the gait has been back in range long enough."""
        return self.recovery_streak >= max(
            int(self.config.required_consecutive_recovery_strides), 1
        )

    def allow(self, target: float, current: float) -> float:
        """Filter a target gain through the persistence rule.

        Args:
            target: gain the score alone would ask for.
            current: gain currently applied.

        Returns:
            ``target`` when the corresponding streak is long enough, otherwise
            ``current`` - the gain is held rather than nudged.
        """
        if target > current and not self.may_increase:
            return current
        if target < current and not self.may_decrease:
            return current
        return target


class AssistancePolicy:
    """Full stride-rate decision: deficit, context, state, gain.

    The rule the whole class exists to enforce is that a Log-Euclidean
    deviation never reaches the motor by itself.  ``E_R`` only scales an
    ``E_B`` that is already non-zero::

        E_assist = E_B * (base_factor + manifold_factor * E_R)

    so a gait that is statistically unusual but shows no correctable deficit
    produces no assistance at all - it is logged and watched instead.

    Args:
        config: full system configuration.
        evaluator: biomechanical evaluator; built from the model when omitted.
        gain_policy: rate-limited gain policy.
        persistence: consecutive-stride gate.
        baseline_healthy_distance: mean distance from the baseline strides to
            the healthy centroid, used for the ``delta_healthy`` trend.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        evaluator: Optional[BiomechanicalEvaluator] = None,
        gain_policy: Optional[AssistanceGainPolicy] = None,
        persistence: Optional[PersistenceGate] = None,
        baseline_healthy_distance: float = float("nan"),
    ) -> None:
        self.config = config or Config()
        self.evaluator = evaluator or BiomechanicalEvaluator(self.config.deviation)
        self.gain_policy = gain_policy or AssistanceGainPolicy(self.config.assist)
        self.persistence = persistence or PersistenceGate(self.config.assist)
        self.baseline_healthy_distance = float(baseline_healthy_distance)
        self.ood_streak = 0
        #: strides since the gain was last recomputed; see :meth:`_hold_elapsed`
        self.strides_since_gain_update = 0

    @classmethod
    def from_model(
        cls, config: Config, model: PatientModel
    ) -> "AssistancePolicy":
        """Build a policy calibrated against a fitted patient model."""
        evaluator = BiomechanicalEvaluator.from_reference(
            config.deviation,
            healthy=model.bank or model.healthy,
            baseline_metrics=model.baseline.metrics,
            reference_config=config.reference,
        )
        return cls(
            config,
            evaluator=evaluator,
            baseline_healthy_distance=model.baseline_distance_stats.get(
                "d_healthy_mean", float("nan")
            ),
        )

    @property
    def gain(self) -> float:
        """Currently applied assist gain."""
        return self.gain_policy.gain

    def reset(self) -> None:
        """Return the policy to its initial state."""
        self.gain_policy.reset(0.0)
        self.persistence.reset()
        self.ood_streak = 0
        self.strides_since_gain_update = 0

    def assess(
        self,
        analysis: StrideAnalysis,
        metrics: Optional[GaitMetrics] = None,
        *,
        assist_allowed: bool = True,
        walking_speed: Optional[float] = None,
    ) -> AssistanceAssessment:
        """Evaluate one stride and produce the gain to apply next.

        Args:
            analysis: high-level manifold analysis of the stride.
            metrics: the stride's biomechanical metrics.
            assist_allowed: safety veto from the low-level loop.
            walking_speed: current walking speed, when known.

        Returns:
            The :class:`AssistanceAssessment`; ``limited_assist_gain`` is what
            the low-level loop should publish.
        """
        stride_metrics = metrics if metrics is not None else analysis.metrics
        if not analysis.valid:
            return self._invalid(analysis, stride_metrics, assist_allowed)

        deficit = self.evaluator.evaluate(stride_metrics, walking_speed)
        e_b = deficit.score
        e_r = float(np.clip(analysis.manifold_deviation, 0.0, 1.0))
        ood_engaged = self._update_ood_streak(analysis.is_ood)
        state = self._decision_state(e_r, e_b, ood_engaged)

        cfg = self.config.deviation
        assist_score = float(
            np.clip(e_b * (cfg.base_factor + cfg.manifold_factor * e_r), 0.0, 1.0)
        )
        raw_gain = float(self.config.assist.max_gain * assist_score)

        self.persistence.update(e_b > 0.0)
        gain = self._apply_gain(
            raw_gain, state=state, assist_allowed=assist_allowed, is_ood=ood_engaged
        )
        return AssistanceAssessment(
            gait_state=None if analysis.gait_state < 0 else int(analysis.gait_state),
            gait_state_confidence=float(analysis.confidence),
            ood=bool(analysis.is_ood),
            d_patient=float(analysis.d_patient),
            d_healthy=float(analysis.d_healthy),
            healthy_region_threshold=float(analysis.healthy_region_threshold),
            manifold_deviation=e_r,
            swing_ratio_error=deficit.swing_ratio_error,
            belt_excursion_error=deficit.belt_excursion_error,
            temporal_symmetry_error=deficit.temporal_symmetry_error,
            trunk_error=deficit.trunk_error,
            biomechanical_deviation=e_b,
            decision_state=state.value,
            raw_assist_gain=raw_gain,
            limited_assist_gain=gain,
            assist_score=assist_score,
            reference_source=deficit.reference_source,
            delta_healthy=self._delta_healthy(analysis.d_healthy),
            consecutive_deficit_strides=self.persistence.deficit_streak,
            consecutive_recovery_strides=self.persistence.recovery_streak,
            ood_engaged=ood_engaged,
            consecutive_ood_strides=self.ood_streak,
            gain_updated=self.strides_since_gain_update == 0,
            metrics=stride_metrics,
            valid=True,
        )

    # -- internals ---------------------------------------------------------- #

    def _update_ood_streak(self, is_ood: bool) -> bool:
        """Count consecutive OOD strides and report whether the cap engages.

        The cap exists for gait the patient model has stopped recognising, not
        for the single odd stride a mis-segmented step produces, so it waits
        for a streak.  Until the streak is long enough the stride is still
        logged as OOD but the gain follows the normal deficit path.

        Args:
            is_ood: this stride's raw out-of-distribution flag.

        Returns:
            True once the streak has reached
            ``assist.required_consecutive_ood_strides``.
        """
        self.ood_streak = self.ood_streak + 1 if is_ood else 0
        required = max(int(self.config.assist.required_consecutive_ood_strides), 1)
        return self.ood_streak >= required

    @staticmethod
    def _decision_state(e_r: float, e_b: float, is_ood: bool) -> DecisionState:
        """Classify the stride from the two deviation scores."""
        if is_ood:
            return DecisionState.OOD
        if e_b > 0.0:
            return (
                DecisionState.COMBINED_DEVIATION
                if e_r > 0.0
                else DecisionState.BIOMECH_DEFICIT_ONLY
            )
        return (
            DecisionState.MANIFOLD_DEVIATION_ONLY
            if e_r > 0.0
            else DecisionState.IN_RANGE
        )

    def _apply_gain(
        self,
        raw_gain: float,
        *,
        state: DecisionState,
        assist_allowed: bool,
        is_ood: bool,
    ) -> float:
        """Send the raw gain through the OOD rule, the gate and the rate limit."""
        cfg = self.config.assist
        if not assist_allowed:
            # a safety veto is never held: it de-energises on this stride
            self.strides_since_gain_update = 0
            return self.gain_policy.update(0.0, assist_allowed=False)

        if is_ood:
            # Aggressive assistance is refused while the stride is out of the
            # model: either hold what is already applied or fall back to the
            # configured safe minimum, both capped by ood_max_gain.
            policy = OodAssistPolicy(cfg.ood_policy)
            held = (
                self.gain_policy.gain
                if policy is OodAssistPolicy.HOLD
                else float(cfg.ood_safe_gain)
            )
            target = min(held, cfg.ood_max_gain)
            score = target / cfg.max_gain if cfg.max_gain > 0.0 else 0.0
            # the cap is a refusal to assist into gait the model lost, so it
            # acts on the stride it engages on rather than waiting out the hold
            self.strides_since_gain_update = 0
            return self.gain_policy.update(score, is_ood=True, assist_allowed=True)

        if state is DecisionState.MANIFOLD_DEVIATION_ONLY:
            # An unusual pattern with nothing correctable behind it is watched,
            # not assisted: E_B is 0, so the raw gain is already 0 and the gate
            # decides only how quickly the previous gain is released.
            raw_gain = 0.0

        if not self._hold_elapsed():
            return self.gain_policy.gain

        target = self.persistence.allow(raw_gain, self.gain_policy.gain)
        score = target / cfg.max_gain if cfg.max_gain > 0.0 else 0.0
        return self.gain_policy.update(score, is_ood=False, assist_allowed=True)

    def _hold_elapsed(self) -> bool:
        """Whether this stride may move the gain, or has to hold it.

        One gain is kept for ``gain_update_interval_strides`` strides.  The
        assistance changes the gait the deficit is measured from, so a gain
        recomputed every stride closes that loop at stride rate with nothing
        settled in between, and the wearer feels a different pull on every
        step.  The deficit is still computed and logged on every stride - only
        the applied gain waits.

        Returns:
            True when the interval has elapsed; the counter then restarts.
        """
        interval = max(int(self.config.assist.gain_update_interval_strides), 1)
        self.strides_since_gain_update += 1
        if self.strides_since_gain_update < interval:
            return False
        self.strides_since_gain_update = 0
        return True

    def _delta_healthy(self, d_healthy: float) -> float:
        """Improvement in healthy-region distance since the session baseline."""
        if not np.isfinite(self.baseline_healthy_distance) or not np.isfinite(d_healthy):
            return float("nan")
        return float(self.baseline_healthy_distance - d_healthy)

    def _invalid(
        self,
        analysis: StrideAnalysis,
        metrics: Optional[GaitMetrics],
        assist_allowed: bool,
    ) -> AssistanceAssessment:
        """Assessment of a stride that could not be represented at all."""
        gain = self.gain_policy.update(0.0, assist_allowed=False)
        return AssistanceAssessment(
            gait_state=None,
            gait_state_confidence=0.0,
            ood=False,
            d_patient=float(analysis.d_patient),
            d_healthy=float(analysis.d_healthy),
            healthy_region_threshold=float(analysis.healthy_region_threshold),
            manifold_deviation=0.0,
            swing_ratio_error=None,
            belt_excursion_error=None,
            temporal_symmetry_error=None,
            trunk_error=None,
            biomechanical_deviation=0.0,
            decision_state=DecisionState.IN_RANGE.value,
            raw_assist_gain=0.0,
            limited_assist_gain=gain,
            assist_score=0.0,
            reference_source=self.evaluator.reference_source,
            delta_healthy=float("nan"),
            consecutive_deficit_strides=self.persistence.deficit_streak,
            consecutive_recovery_strides=self.persistence.recovery_streak,
            metrics=metrics,
            valid=False,
            message=analysis.message,
        )


__all__ = [
    "PHASE_DERIVED_METRICS",
    "SOURCE_BASELINE",
    "SOURCE_HEALTHY",
    "AssistanceAssessment",
    "AssistanceGainPolicy",
    "AssistancePolicy",
    "BiomechanicalDeficit",
    "BiomechanicalEvaluator",
    "DecisionState",
    "DeviationComponents",
    "DeviationScorer",
    "PersistenceGate",
    "interval_error",
]
