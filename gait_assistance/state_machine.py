"""System-level state machine (spec 26)."""

from __future__ import annotations

from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple


class SystemState(Enum):
    """High-level operating state of the gait-assistance system."""

    INIT = "INIT"
    CALIBRATION = "CALIBRATION"
    BASELINE_COLLECTION = "BASELINE_COLLECTION"
    MODEL_BUILDING = "MODEL_BUILDING"
    READY = "READY"
    WALKING = "WALKING"
    OOD = "OOD"
    SAFE_STOP = "SAFE_STOP"


#: Allowed transitions.  ``SAFE_STOP`` is reachable from every state.
_TRANSITIONS: Dict[SystemState, Set[SystemState]] = {
    SystemState.INIT: {SystemState.CALIBRATION, SystemState.SAFE_STOP},
    SystemState.CALIBRATION: {SystemState.BASELINE_COLLECTION, SystemState.SAFE_STOP},
    SystemState.BASELINE_COLLECTION: {SystemState.MODEL_BUILDING, SystemState.SAFE_STOP},
    SystemState.MODEL_BUILDING: {SystemState.READY, SystemState.SAFE_STOP},
    SystemState.READY: {SystemState.WALKING, SystemState.OOD, SystemState.SAFE_STOP},
    SystemState.WALKING: {SystemState.READY, SystemState.OOD, SystemState.SAFE_STOP},
    SystemState.OOD: {SystemState.WALKING, SystemState.READY, SystemState.SAFE_STOP},
    SystemState.SAFE_STOP: {SystemState.INIT},
}


class InvalidTransitionError(RuntimeError):
    """Raised when a forbidden state transition is requested."""


class StateMachine:
    """Guarded state machine with a transition history.

    Args:
        initial: state the machine starts in.
        on_change: optional callback ``(old, new) -> None``.
    """

    def __init__(
        self,
        initial: SystemState = SystemState.INIT,
        on_change: Optional[Callable[[SystemState, SystemState], None]] = None,
    ) -> None:
        self._state = initial
        self._on_change = on_change
        self._history: List[Tuple[SystemState, SystemState]] = []

    @property
    def state(self) -> SystemState:
        """Current state."""
        return self._state

    @property
    def history(self) -> List[Tuple[SystemState, SystemState]]:
        """List of ``(from, to)`` transitions performed so far."""
        return list(self._history)

    def can_transition(self, target: SystemState) -> bool:
        """Return whether a transition to ``target`` is allowed right now."""
        if target is self._state:
            return True
        return target in _TRANSITIONS.get(self._state, set())

    def transition_to(self, target: SystemState, *, force: bool = False) -> SystemState:
        """Move to ``target``.

        Args:
            target: requested state.
            force: bypass the transition table (used by emergency paths).

        Returns:
            The new state.

        Raises:
            InvalidTransitionError: if the transition is not allowed.
        """
        if target is self._state:
            return self._state
        if not force and not self.can_transition(target):
            raise InvalidTransitionError(
                f"illegal transition {self._state.value} -> {target.value}"
            )
        old, self._state = self._state, target
        self._history.append((old, target))
        if self._on_change is not None:
            self._on_change(old, target)
        return self._state

    def safe_stop(self) -> SystemState:
        """Force the machine into :attr:`SystemState.SAFE_STOP`."""
        return self.transition_to(SystemState.SAFE_STOP, force=True)

    def reset(self) -> SystemState:
        """Return to :attr:`SystemState.INIT` (allowed only from SAFE_STOP)."""
        return self.transition_to(SystemState.INIT, force=True)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"StateMachine(state={self._state.value})"


__all__ = ["InvalidTransitionError", "StateMachine", "SystemState"]
