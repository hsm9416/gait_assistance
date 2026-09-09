"""Temporal symmetry between the paretic and non-paretic sides (spec 5).

The device instruments one leg.  Symmetry therefore exists only when a second
source supplies the contralateral timing - a second encoder, a foot switch on
the other side, or a data set that recorded both.  When that source is absent
every symmetry metric is ``None``.

The one rule this module enforces is that a missing contralateral side is
never imputed.  There is no population default, no mirroring of the paretic
side, no carrying forward of the last known value: a symmetry ratio computed
from a guess would look exactly like a measurement and would silently drive the
assistance, so it is not computed at all.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Protocol, runtime_checkable

from .stride_segmenter import Stride


@dataclass(frozen=True)
class SideTiming:
    """Estimated stance/swing durations of one side, in seconds."""

    swing_time: float
    stance_time: float

    @property
    def swing_stance_ratio(self) -> Optional[float]:
        """``swing_time / stance_time``; ``None`` when stance is not positive."""
        if self.stance_time <= 0.0:
            return None
        return float(self.swing_time / self.stance_time)

    def is_usable(self) -> bool:
        """True when both durations are positive and finite."""
        return (
            self.swing_time > 0.0
            and self.stance_time > 0.0
            and self.swing_time == self.swing_time      # not NaN
            and self.stance_time == self.stance_time
        )


@dataclass(frozen=True)
class SymmetryMetrics:
    """Paretic-to-non-paretic timing ratios.

    Every ratio is ``paretic / non-paretic``, so 1.0 is perfect symmetry and
    values away from 1.0 in either direction are asymmetry.
    """

    swing_symmetry_ratio: float
    stance_symmetry_ratio: float
    ss_ratio_paretic: float
    ss_ratio_nonparetic: float
    swing_stance_symmetry_ratio: float

    def to_dict(self) -> Dict[str, float]:
        """Return the ratios as a plain dictionary."""
        return asdict(self)

    @property
    def asymmetry(self) -> float:
        """Absolute departure of the swing ratio from perfect symmetry."""
        return abs(self.swing_symmetry_ratio - 1.0)


@runtime_checkable
class ContralateralTimingSource(Protocol):
    """Supplies the non-paretic timing that pairs with a paretic stride.

    Implement this to feed a second encoder, a contralateral foot switch or a
    two-sided recording into the metrics.  Returning ``None`` for a stride is
    the correct answer whenever the other side was not measured for it.
    """

    def timing_for(self, stride: Stride) -> Optional[SideTiming]:
        """Return the contralateral timing of ``stride``, or ``None``."""


def compute_symmetry(
    paretic: SideTiming, nonparetic: Optional[SideTiming]
) -> Optional[SymmetryMetrics]:
    """Compute the timing symmetry of one stride.

    Args:
        paretic: timing measured on the instrumented (paretic) side.
        nonparetic: timing of the other side, or ``None`` when it was not
            measured.

    Returns:
        The :class:`SymmetryMetrics`, or ``None`` when either side is missing
        or degenerate.  ``None`` means "not measured" and must be logged as an
        empty cell rather than replaced by a default.
    """
    if nonparetic is None:
        return None
    if not paretic.is_usable() or not nonparetic.is_usable():
        return None
    ss_paretic = paretic.swing_stance_ratio
    ss_nonparetic = nonparetic.swing_stance_ratio
    if not ss_paretic or not ss_nonparetic:
        return None
    return SymmetryMetrics(
        swing_symmetry_ratio=float(paretic.swing_time / nonparetic.swing_time),
        stance_symmetry_ratio=float(paretic.stance_time / nonparetic.stance_time),
        ss_ratio_paretic=float(ss_paretic),
        ss_ratio_nonparetic=float(ss_nonparetic),
        swing_stance_symmetry_ratio=float(ss_paretic / ss_nonparetic),
    )


__all__ = [
    "ContralateralTimingSource",
    "SideTiming",
    "SymmetryMetrics",
    "compute_symmetry",
]
