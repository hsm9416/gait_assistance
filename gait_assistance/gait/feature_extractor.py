"""Stride feature extraction and biomechanical metrics (spec 5, 18).

Two independent things happen here:

* :class:`FeatureExtractor` turns a stride into the configurable
  ``(n_cycle_points, n_features)`` matrix that feeds the SPD covariance.
* :func:`compute_stride_metrics` and the single-purpose functions around it
  compute the interpretable clinical metrics used by the deviation score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..config import CHANNELS, FeatureConfig, StrideConfig
from .phase_detector import GaitPhase
from .stride_segmenter import Stride, resample_gait_cycle
from .symmetry import SideTiming, SymmetryMetrics, compute_symmetry


class FeatureExtractor:
    """Extract the configured feature matrix from a stride.

    Args:
        feature_config: which channels form the feature vector.
        stride_config: resampling settings.

    Raises:
        ValueError: if a configured channel is not a known sensor channel.
    """

    def __init__(
        self,
        feature_config: Optional[FeatureConfig] = None,
        stride_config: Optional[StrideConfig] = None,
    ) -> None:
        self.feature_config = feature_config or FeatureConfig()
        self.stride_config = stride_config or StrideConfig()
        unknown = [c for c in self.feature_config.channels if c not in CHANNELS]
        if unknown:
            raise ValueError(f"unknown feature channels: {unknown}")

    @property
    def channels(self) -> Sequence[str]:
        """Feature channel names in matrix-column order."""
        return self.feature_config.channels

    @property
    def n_features(self) -> int:
        """Number of feature channels."""
        return len(self.feature_config.channels)

    def extract(self, stride: Stride) -> np.ndarray:
        """Return the resampled raw feature matrix of ``stride``.

        Args:
            stride: segmented stride.

        Returns:
            A ``(n_cycle_points, n_features)`` array in physical units; the
            z-score normalisation is applied separately (spec 6).
        """
        raw = stride.to_array(self.channels)
        return resample_gait_cycle(raw, self.stride_config.n_cycle_points)

    def extract_many(self, strides: Sequence[Stride]) -> List[np.ndarray]:
        """Apply :meth:`extract` to every stride in ``strides``."""
        return [self.extract(s) for s in strides]


# --------------------------------------------------------------------------- #
# Biomechanical metrics - one independent function per quantity (spec 18)
# --------------------------------------------------------------------------- #


def swing_time(stride: Stride) -> float:
    """Time spent in the *estimated* swing phase during ``stride`` (s).

    Derived from the phase detector's labels, which infer swing from belt or
    shank motion rather than foot contact, so this is an estimate offset from
    biomechanical swing by an unmeasured amount.
    """
    return _phase_time(stride, GaitPhase.SWING)


def stance_time(stride: Stride) -> float:
    """Time spent in the *estimated* stance phase during ``stride`` (s).

    Carries the same estimation caveat as :func:`swing_time`.
    """
    return _phase_time(stride, GaitPhase.STANCE)


def swing_ratio(stride: Stride) -> float:
    """Estimated swing time divided by stride time, in ``[0, 1]``.

    Because both terms come from the estimated phase, this ratio sits on an
    uncalibrated scale and must not be compared against normative values such
    as 0.38; compare it only against a healthy interval measured through the
    same detector, or against the same patient's other strides.
    """
    total = swing_time(stride) + stance_time(stride)
    if total <= 0.0:
        return 0.0
    return float(swing_time(stride) / total)


def stance_ratio(stride: Stride) -> float:
    """Estimated stance time divided by stride time, in ``[0, 1]``.

    Complements :func:`swing_ratio`: the two sum to 1 whenever the phase
    labels cover the whole stride.
    """
    total = swing_time(stride) + stance_time(stride)
    if total <= 0.0:
        return 0.0
    return float(stance_time(stride) / total)


def swing_stance_ratio(stride: Stride) -> float:
    """Estimated ``swing_time / stance_time``.

    Unbounded above, unlike the two ratios: it separates strides that share a
    swing ratio but differ in how the stance is spent.  Returns 0 when there
    is no stance to divide by.
    """
    stance = stance_time(stride)
    if stance <= 0.0:
        return 0.0
    return float(swing_time(stride) / stance)


def belt_excursion(stride: Stride) -> float:
    """Peak-to-peak belt length travelled during the stride (mm)."""
    belt = stride.to_array(("belt_length",))[:, 0]
    if belt.size == 0:
        return 0.0
    return float(np.max(belt) - np.min(belt))


def peak_belt_velocity(stride: Stride) -> float:
    """Largest absolute belt velocity in the stride (mm/s)."""
    vel = stride.to_array(("belt_velocity",))[:, 0]
    if vel.size == 0:
        return 0.0
    return float(np.max(np.abs(vel)))


def trunk_acc_rms(stride: Stride) -> Optional[float]:
    """RMS of the mean-removed acceleration magnitude (g), or ``None``.

    Gravity is removed by subtracting the stride mean of the magnitude, so the
    result reflects the dynamic acceleration content only.

    Returns:
        ``None`` when the accelerometer channels are absent or non-finite for
        this stride.  A recording without an IMU must not report a trunk
        motion of zero, which would read as "perfectly steady trunk".
    """
    return _axis_rms(stride, ("accel_x", "accel_y", "accel_z"))


def trunk_gyro_rms(stride: Stride) -> Optional[float]:
    """RMS of the mean-removed angular-rate magnitude (deg/s), or ``None``.

    The rotational counterpart of :func:`trunk_acc_rms`, carrying the same
    missing-channel rule.
    """
    return _axis_rms(stride, ("gyro_x", "gyro_y", "gyro_z"))


def _axis_rms(stride: Stride, channels: Sequence[str]) -> Optional[float]:
    """Mean-removed RMS of a three-axis magnitude, or ``None`` if unusable."""
    data = stride.to_array(tuple(channels))
    if data.shape[0] == 0 or not np.all(np.isfinite(data)):
        return None
    magnitude = np.linalg.norm(data, axis=1)
    centred = magnitude - float(np.mean(magnitude))
    return float(np.sqrt(np.mean(centred ** 2)))


def stride_time(stride: Stride) -> float:
    """Total stride duration (s)."""
    return float(stride.duration_s)


def mean_belt_length(stride: Stride) -> float:
    """Mean belt length over the stride (mm)."""
    belt = stride.to_array(("belt_length",))[:, 0]
    return float(np.mean(belt)) if belt.size else 0.0


#: Fallback provenance when a stride carries none (e.g. one built by hand in
#: a test).  Real strides carry the token of the detector that labelled them.
PHASE_SOURCE_BELT: str = "belt_derived"


@dataclass(frozen=True)
class GaitMetrics:
    """Biomechanical descriptors of a single stride (spec 18).

    Field naming follows the provenance of each quantity.  ``swing_time``,
    ``stance_time`` and the three ratios are derived from the **estimated**
    gait phase, and ``phase_source`` records what produced it, so a consumer
    can tell a belt-derived timing from a contact-validated one without
    guessing.  The stride log names these columns ``estimated_*`` for the same
    reason.

    Optional fields are ``None`` when the underlying signal was not measured -
    no IMU for the trunk terms, no contralateral source for the symmetry
    terms.  ``None`` means "not measured" and never becomes 0.
    """

    stride_time: float

    swing_time: float
    stance_time: float

    swing_ratio: float
    stance_ratio: float
    swing_stance_ratio: float

    belt_excursion: float
    peak_belt_velocity: Optional[float] = None

    mean_belt_length: float = 0.0

    swing_symmetry_ratio: Optional[float] = None
    stance_symmetry_ratio: Optional[float] = None
    swing_stance_symmetry_ratio: Optional[float] = None

    trunk_acc_rms: Optional[float] = None
    trunk_gyro_rms: Optional[float] = None

    #: what produced the phase labels these timings came from
    phase_source: str = PHASE_SOURCE_BELT

    def to_dict(self) -> Dict[str, Any]:
        """Return the metrics as a plain dictionary, ``None`` values included."""
        return asdict(self)

    @property
    def has_symmetry(self) -> bool:
        """True when the contralateral side was measured for this stride."""
        return self.swing_symmetry_ratio is not None

    @property
    def has_trunk(self) -> bool:
        """True when the trunk IMU channels were usable for this stride."""
        return self.trunk_acc_rms is not None


#: Historical name of :class:`GaitMetrics`, kept so existing call sites and
#: type annotations continue to work unchanged.
StrideMetrics = GaitMetrics


def compute_stride_metrics(
    stride: Stride,
    *,
    contralateral: Optional[SideTiming] = None,
    phase_source: Optional[str] = None,
) -> GaitMetrics:
    """Compute every biomechanical metric of ``stride``.

    Args:
        stride: the segmented stride.
        contralateral: timing measured on the other side for this stride.
            ``None`` - the default, and the only possibility on a
            single-sided device - leaves every symmetry metric ``None``
            rather than imputing the missing side.
        phase_source: provenance to record; taken from the stride itself when
            omitted, so the metrics always name the detector that actually
            produced their phase labels.

    Returns:
        The :class:`GaitMetrics`.
    """
    source = phase_source or stride.phase_source or PHASE_SOURCE_BELT
    paretic = SideTiming(
        swing_time=swing_time(stride), stance_time=stance_time(stride)
    )
    symmetry: Optional[SymmetryMetrics] = compute_symmetry(paretic, contralateral)
    return GaitMetrics(
        stride_time=stride_time(stride),
        swing_time=paretic.swing_time,
        stance_time=paretic.stance_time,
        swing_ratio=swing_ratio(stride),
        stance_ratio=stance_ratio(stride),
        swing_stance_ratio=swing_stance_ratio(stride),
        belt_excursion=belt_excursion(stride),
        peak_belt_velocity=peak_belt_velocity(stride),
        mean_belt_length=mean_belt_length(stride),
        swing_symmetry_ratio=(
            symmetry.swing_symmetry_ratio if symmetry is not None else None
        ),
        stance_symmetry_ratio=(
            symmetry.stance_symmetry_ratio if symmetry is not None else None
        ),
        swing_stance_symmetry_ratio=(
            symmetry.swing_stance_symmetry_ratio if symmetry is not None else None
        ),
        trunk_acc_rms=trunk_acc_rms(stride),
        trunk_gyro_rms=trunk_gyro_rms(stride),
        phase_source=source,
    )


def aggregate_metrics(metrics: Sequence[GaitMetrics]) -> Dict[str, float]:
    """Average a sequence of :class:`GaitMetrics` field by field.

    Fields that were not measured are skipped rather than counted as zero, and
    a field measured on no stride at all is left out of the result entirely.

    Args:
        metrics: metrics of several strides.

    Returns:
        Mapping from metric name to its mean over the strides that carried it;
        empty when ``metrics`` is empty.
    """
    if not metrics:
        return {}
    return {
        key: float(np.mean(values))
        for key, values in collect_metric_samples(metrics).items()
    }


def collect_metric_samples(
    metrics: Sequence[GaitMetrics],
) -> Dict[str, List[float]]:
    """Group the numeric metric values by name, dropping the missing ones.

    This is what the healthy reference builds its per-metric intervals from,
    so it must not fabricate values: a stride that lacks a metric contributes
    nothing to that metric's distribution instead of contributing a zero.

    Args:
        metrics: metrics of several strides.

    Returns:
        Mapping from metric name to the finite values observed for it.
    """
    samples: Dict[str, List[float]] = {}
    for item in metrics:
        for key, value in item.to_dict().items():
            if value is None or isinstance(value, str):
                continue
            number = float(value)
            if not np.isfinite(number):
                continue
            samples.setdefault(key, []).append(number)
    return samples


def _phase_time(stride: Stride, phase: GaitPhase) -> float:
    """Integrate the sample intervals labelled with ``phase``."""
    if stride.n_samples < 2:
        return 0.0
    times = stride.timestamps()
    dt = np.diff(times, append=times[-1])
    mask = stride.phase_mask(phase)
    if mask.size != dt.size:
        return 0.0
    return float(np.sum(dt[mask]))


__all__ = [
    "PHASE_SOURCE_BELT",
    "FeatureExtractor",
    "GaitMetrics",
    "StrideMetrics",
    "aggregate_metrics",
    "belt_excursion",
    "collect_metric_samples",
    "compute_stride_metrics",
    "mean_belt_length",
    "peak_belt_velocity",
    "stance_ratio",
    "stance_time",
    "stride_time",
    "swing_ratio",
    "swing_stance_ratio",
    "swing_time",
    "trunk_acc_rms",
    "trunk_gyro_rms",
]
