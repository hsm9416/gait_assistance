"""Light-weight signal filters used by the low-level loop."""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Sequence

import numpy as np
from scipy import signal


class EMAFilter:
    """First-order exponential moving average.

    Args:
        alpha: smoothing factor in ``(0, 1]``; 1.0 disables filtering.
    """

    def __init__(self, alpha: float) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must lie in (0, 1]")
        self.alpha = alpha
        self._value: Optional[float] = None

    @property
    def value(self) -> Optional[float]:
        """Current filter output, or ``None`` before the first update."""
        return self._value

    def update(self, x: float) -> float:
        """Feed one sample and return the filtered value."""
        if self._value is None or not np.isfinite(self._value):
            self._value = float(x)
        else:
            self._value = self.alpha * float(x) + (1.0 - self.alpha) * self._value
        return self._value

    def reset(self) -> None:
        """Forget the filter state."""
        self._value = None

    @classmethod
    def from_cutoff(cls, cutoff_hz: float, sample_rate_hz: float) -> "EMAFilter":
        """Build a filter approximating a first-order low pass.

        Args:
            cutoff_hz: -3 dB cut-off frequency.
            sample_rate_hz: sampling rate of the incoming signal.
        """
        if cutoff_hz <= 0.0 or sample_rate_hz <= 0.0:
            return cls(1.0)
        dt = 1.0 / sample_rate_hz
        tau = 1.0 / (2.0 * np.pi * cutoff_hz)
        return cls(float(np.clip(dt / (tau + dt), 1e-6, 1.0)))


class MovingAverageFilter:
    """Sliding-window mean filter."""

    def __init__(self, window: int) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self._buf: Deque[float] = deque(maxlen=window)

    def update(self, x: float) -> float:
        """Feed one sample and return the windowed mean."""
        self._buf.append(float(x))
        return float(np.mean(self._buf))

    def reset(self) -> None:
        """Empty the window."""
        self._buf.clear()


class MedianFilter:
    """Sliding-window median filter, robust to isolated spikes."""

    def __init__(self, window: int = 5) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self._buf: Deque[float] = deque(maxlen=window)

    def update(self, x: float) -> float:
        """Feed one sample and return the windowed median."""
        self._buf.append(float(x))
        return float(np.median(self._buf))

    def reset(self) -> None:
        """Empty the window."""
        self._buf.clear()


class RateLimiter:
    """Limit the absolute change of a signal between two updates."""

    def __init__(self, max_delta: float, initial: float = 0.0) -> None:
        if max_delta < 0.0:
            raise ValueError("max_delta must be non-negative")
        self.max_delta = max_delta
        self._value = float(initial)

    @property
    def value(self) -> float:
        """Last emitted value."""
        return self._value

    def update(self, target: float) -> float:
        """Move at most ``max_delta`` towards ``target`` and return the result."""
        delta = float(target) - self._value
        if delta > self.max_delta:
            delta = self.max_delta
        elif delta < -self.max_delta:
            delta = -self.max_delta
        self._value += delta
        return self._value

    def reset(self, value: float = 0.0) -> None:
        """Force the internal value without rate limiting."""
        self._value = float(value)


def butter_lowpass(
    x: Sequence[float], cutoff_hz: float, sample_rate_hz: float, order: int = 2
) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter for offline post-processing.

    Args:
        x: input signal.
        cutoff_hz: cut-off frequency.
        sample_rate_hz: sampling rate.
        order: filter order.

    Returns:
        The filtered signal; the input is returned unchanged when it is too
        short or the cut-off is not below Nyquist.
    """
    arr = np.asarray(x, dtype=float)
    nyquist = 0.5 * sample_rate_hz
    if arr.size < 3 * (order + 1) or not 0.0 < cutoff_hz < nyquist:
        return arr
    b, a = signal.butter(order, cutoff_hz / nyquist, btype="low")
    return signal.filtfilt(b, a, arr)


__all__ = [
    "EMAFilter",
    "MedianFilter",
    "MovingAverageFilter",
    "RateLimiter",
    "butter_lowpass",
]
