"""Timing helpers shared by both control loops (spec 24)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional


def now() -> float:
    """Return a monotonic timestamp in seconds."""
    return time.monotonic()


class LoopRate:
    """Fixed-rate pacer for the low-level loop.

    Args:
        hz: target loop frequency.
        sleep: when False, :meth:`sleep` returns immediately (offline replay).
    """

    def __init__(self, hz: float, *, sleep: bool = True) -> None:
        if hz <= 0.0:
            raise ValueError("hz must be positive")
        self.period_s = 1.0 / hz
        self._sleep = sleep
        self._next_t = now()
        self.overruns = 0
        self.iterations = 0

    def reset(self) -> None:
        """Restart the schedule at the current time."""
        self._next_t = now()
        self.overruns = 0
        self.iterations = 0

    def sleep(self) -> float:
        """Block until the next period boundary.

        Returns:
            The slack in seconds; negative values indicate a loop overrun.
        """
        self.iterations += 1
        self._next_t += self.period_s
        slack = self._next_t - now()
        if slack < 0.0:
            self.overruns += 1
            self._next_t = now()
        elif self._sleep:
            time.sleep(slack)
        return slack


class Stopwatch:
    """Context manager measuring wall-clock duration of a block."""

    def __init__(self) -> None:
        self.elapsed_s: float = 0.0
        self._t0: float = 0.0

    def __enter__(self) -> "Stopwatch":
        self._t0 = now()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed_s = now() - self._t0


@dataclass
class RateMonitor:
    """Sliding-window estimator of an event rate (Hz)."""

    window: int = 100
    _times: List[float] = field(default_factory=list, repr=False)

    def tick(self, t: Optional[float] = None) -> None:
        """Record one event occurrence."""
        self._times.append(now() if t is None else t)
        if len(self._times) > self.window:
            del self._times[0 : len(self._times) - self.window]

    @property
    def hz(self) -> float:
        """Estimated rate; ``0.0`` until at least two events were recorded."""
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        if span <= 0.0:
            return 0.0
        return (len(self._times) - 1) / span

    def reset(self) -> None:
        """Discard all recorded events."""
        self._times.clear()


__all__ = ["LoopRate", "RateMonitor", "Stopwatch", "now"]
