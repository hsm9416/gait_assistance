"""Stride-level CSV logging (spec 27)."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

#: Column order of the stride log (spec 19/27).
#:
#: Phase-derived timings carry an ``estimated_`` prefix because they come from
#: a belt- or shank-inferred phase rather than an observed foot contact; the
#: ``phase_source`` column records which detector produced them, so a later
#: contact-validated run is distinguishable from this one in the same file.
#: Columns that were not measured are written empty, never as 0.
STRIDE_LOG_COLUMNS: List[str] = [
    "stride_id",
    "timestamp",

    "gait_cluster",
    "cluster_confidence",
    "ood",
    "ood_engaged",

    "d_patient",
    "d_healthy",
    "d_target",
    "healthy_region_threshold",
    "manifold_deviation",
    "delta_healthy",

    "estimated_swing_time",
    "estimated_stance_time",
    "estimated_stride_time",
    "estimated_swing_ratio",
    "estimated_stance_ratio",
    "estimated_swing_stance_ratio",
    "phase_source",

    "swing_symmetry_ratio",
    "stance_symmetry_ratio",
    "swing_stance_symmetry_ratio",

    "belt_excursion",
    "peak_belt_velocity",
    "trunk_acc_rms",
    "trunk_gyro_rms",

    "swing_ratio_error",
    "belt_excursion_error",
    "temporal_symmetry_error",
    "trunk_error",

    "biomechanical_deviation",
    "reference_source",
    "decision_state",

    "raw_assist_gain",
    "assist_gain",
    "target_belt_length",
    "motor_position",
    "motor_velocity",
    "motor_current",
]


@dataclass
class StrideLogRecord:
    """One row of the stride log.

    Optional fields default to ``None`` and are written as empty cells.  That
    is the point of them being optional: a run without a contralateral sensor
    must produce blank symmetry columns, not zeros that a later analysis would
    read as perfect symmetry.
    """

    stride_id: int
    timestamp: float

    gait_cluster: Optional[str] = None
    cluster_confidence: float = float("nan")
    ood: int = 0
    ood_engaged: int = 0

    d_patient: float = float("nan")
    d_healthy: float = float("nan")
    d_target: float = float("nan")
    healthy_region_threshold: float = float("nan")
    manifold_deviation: float = float("nan")
    delta_healthy: float = float("nan")

    estimated_swing_time: float = float("nan")
    estimated_stance_time: float = float("nan")
    estimated_stride_time: float = float("nan")
    estimated_swing_ratio: float = float("nan")
    estimated_stance_ratio: float = float("nan")
    estimated_swing_stance_ratio: float = float("nan")
    phase_source: str = ""

    swing_symmetry_ratio: Optional[float] = None
    stance_symmetry_ratio: Optional[float] = None
    swing_stance_symmetry_ratio: Optional[float] = None

    belt_excursion: float = float("nan")
    peak_belt_velocity: Optional[float] = None
    trunk_acc_rms: Optional[float] = None
    trunk_gyro_rms: Optional[float] = None

    swing_ratio_error: Optional[float] = None
    belt_excursion_error: Optional[float] = None
    temporal_symmetry_error: Optional[float] = None
    trunk_error: Optional[float] = None

    biomechanical_deviation: float = float("nan")
    reference_source: str = ""
    decision_state: str = ""

    raw_assist_gain: float = float("nan")
    assist_gain: float = float("nan")
    target_belt_length: float = float("nan")
    motor_position: float = float("nan")
    motor_velocity: float = float("nan")
    motor_current: float = float("nan")

    def to_row(self) -> Dict[str, Any]:
        """Return the record as a dictionary keyed by column name."""
        return {k: _clean(v) for k, v in asdict(self).items()}


class StrideLogger:
    """Append-only CSV writer for :class:`StrideLogRecord`.

    The file is opened lazily, so constructing a logger with an empty path
    keeps every record in memory only (useful for tests and offline runs).

    Args:
        path: destination CSV file; ``None`` or ``""`` disables file output.
    """

    def __init__(self, path: Optional[Union[str, Path]] = None) -> None:
        self.path = Path(path) if path else None
        self._handle = None
        self._writer: Optional[csv.DictWriter] = None
        self.records: List[StrideLogRecord] = []

    def __enter__(self) -> "StrideLogger":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def open(self) -> None:
        """Create the file and write the header row."""
        if self.path is None or self._handle is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=STRIDE_LOG_COLUMNS)
        self._writer.writeheader()

    def write(self, record: StrideLogRecord) -> None:
        """Append one record, opening the file on first use."""
        self.records.append(record)
        if self.path is not None and self._writer is None:
            self.open()
        if self._writer is not None:
            self._writer.writerow(record.to_row())
            if self._handle is not None:
                self._handle.flush()

    def write_many(self, records: Iterable[StrideLogRecord]) -> None:
        """Append several records."""
        for record in records:
            self.write(record)

    def close(self) -> None:
        """Close the underlying file, if any."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._writer = None

    def to_rows(self) -> List[Dict[str, Any]]:
        """Return every buffered record as a list of dictionaries."""
        return [r.to_row() for r in self.records]


def _clean(value: Any) -> Any:
    """Render a value for CSV, blanking anything that was not measured.

    ``None`` and non-finite floats both become an empty cell.  Both mean "not
    measured" and must stay distinguishable from a measured zero.
    """
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


__all__ = ["STRIDE_LOG_COLUMNS", "StrideLogRecord", "StrideLogger"]
