"""Stride segmentation (spec 3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..config import StrideConfig
from ..sensors.sensor_manager import SensorSample
from .phase_detector import GaitPhase, PhaseResult


@dataclass
class Stride:
    """One heel-strike-to-heel-strike stride with its raw samples."""

    stride_id: int
    samples: List[SensorSample] = field(default_factory=list)
    phases: List[GaitPhase] = field(default_factory=list)
    #: provenance of the phase labels, copied from the detector that produced
    #: them, so every metric derived from this stride can say where its timing
    #: came from without asking the detector again
    phase_source: str = ""

    @property
    def t_start(self) -> float:
        """Timestamp of the first sample."""
        return self.samples[0].timestamp if self.samples else 0.0

    @property
    def t_end(self) -> float:
        """Timestamp of the last sample."""
        return self.samples[-1].timestamp if self.samples else 0.0

    @property
    def duration_s(self) -> float:
        """Stride duration in seconds."""
        return self.t_end - self.t_start

    @property
    def n_samples(self) -> int:
        """Number of raw samples in the stride."""
        return len(self.samples)

    def timestamps(self) -> np.ndarray:
        """Timestamps of all samples as a 1-D array."""
        return np.array([s.timestamp for s in self.samples], dtype=float)

    def to_array(self, channels: Sequence[str]) -> np.ndarray:
        """Return the selected channels as a ``(n_samples, n_channels)`` array."""
        if not self.samples:
            return np.zeros((0, len(channels)), dtype=float)
        return np.array([s.to_vector(channels) for s in self.samples], dtype=float)

    def phase_mask(self, phase: GaitPhase) -> np.ndarray:
        """Boolean mask selecting the samples labelled with ``phase``."""
        return np.array([p is phase for p in self.phases], dtype=bool)

    def is_valid(self, config: Optional[StrideConfig] = None) -> Tuple[bool, str]:
        """Check the stride against the duration and length limits.

        Returns:
            ``(ok, reason)``; ``reason`` is empty when the stride is usable.
        """
        cfg = config or StrideConfig()
        if self.n_samples < cfg.min_samples:
            return False, f"too few samples ({self.n_samples} < {cfg.min_samples})"
        if self.duration_s < cfg.min_duration_s:
            return False, f"too short ({self.duration_s:.3f}s)"
        if self.duration_s > cfg.max_duration_s:
            return False, f"too long ({self.duration_s:.3f}s)"
        if not np.all(np.isfinite(self.to_array(("belt_length", "belt_velocity")))):
            return False, "non-finite samples"
        return True, ""


class StrideSegmenter:
    """Accumulate samples between consecutive heel strikes.

    Args:
        config: stride configuration (validity limits).
    """

    def __init__(self, config: Optional[StrideConfig] = None) -> None:
        self.config = config or StrideConfig()
        self._current: Optional[Stride] = None
        self._next_id = 0
        self.rejected: List[Tuple[int, str]] = []

    def reset(self) -> None:
        """Drop the partial stride and the rejection log."""
        self._current = None
        self.rejected.clear()

    @property
    def current(self) -> Optional[Stride]:
        """Stride currently being accumulated, if any."""
        return self._current

    def update(self, sample: SensorSample, phase_result: PhaseResult) -> Optional[Stride]:
        """Feed one sample.

        Args:
            sample: current sensor frame.
            phase_result: output of the phase detector for that frame.

        Returns:
            The completed stride when this sample closed one, else ``None``.
        """
        completed: Optional[Stride] = None
        if phase_result.heel_strike:
            if self._current is not None:
                self._current.samples.append(sample)
                self._current.phases.append(phase_result.phase)
                ok, reason = self._current.is_valid(self.config)
                if ok:
                    completed = self._current
                else:
                    self.rejected.append((self._current.stride_id, reason))
            self._current = Stride(stride_id=self._next_id)
            self._next_id += 1
        if self._current is not None:
            self._current.samples.append(sample)
            self._current.phases.append(phase_result.phase)
            if phase_result.source:
                self._current.phase_source = phase_result.source
        return completed


def resample_gait_cycle(matrix: np.ndarray, n_points: int = 100) -> np.ndarray:
    """Interpolate a stride onto a 0-100 % gait cycle grid (spec 4).

    Args:
        matrix: ``(n_samples, n_channels)`` array of raw stride data.
        n_points: number of output samples (default 100).

    Returns:
        A ``(n_points, n_channels)`` array.

    Raises:
        ValueError: if the input has fewer than two samples or ``n_points < 2``.
    """
    data = np.asarray(matrix, dtype=float)
    if data.ndim != 2:
        raise ValueError("matrix must be 2-D (n_samples, n_channels)")
    if data.shape[0] < 2:
        raise ValueError("at least two samples are required for resampling")
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    src = np.linspace(0.0, 1.0, data.shape[0])
    dst = np.linspace(0.0, 1.0, n_points)
    out = np.empty((n_points, data.shape[1]), dtype=float)
    for j in range(data.shape[1]):
        out[:, j] = np.interp(dst, src, data[:, j])
    return out


def resample_stride(
    stride: Stride, channels: Sequence[str], n_points: int = 100
) -> np.ndarray:
    """Resample the selected channels of ``stride`` onto the gait-cycle grid."""
    return resample_gait_cycle(stride.to_array(channels), n_points)


__all__ = ["Stride", "StrideSegmenter", "resample_gait_cycle", "resample_stride"]
