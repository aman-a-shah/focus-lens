"""Distraction state label space (plan.md §3.2)."""

from __future__ import annotations

from enum import StrEnum


class DistractionState(StrEnum):
    """The four attention states the system reasons about."""

    FOCUSED = "FOCUSED"  # gaze on screen, low velocity, normal blink rate
    DRIFTING = "DRIFTING"  # gaze wandering but returning — early warning
    DISTRACTED = "DISTRACTED"  # gaze/head off-screen or unfocused for a sustained stretch
    FATIGUED = "FATIGUED"  # high blink rate, slow gaze, EAR dropping


# Canonical ordering — the class-index space PersonalFocusNet's head predicts (Phase 7).
STATES: tuple[DistractionState, ...] = (
    DistractionState.FOCUSED,
    DistractionState.DRIFTING,
    DistractionState.DISTRACTED,
    DistractionState.FATIGUED,
)
NUM_STATES = len(STATES)
_INDEX = {s: i for i, s in enumerate(STATES)}


def state_index(state: DistractionState) -> int:
    """Class index for a state (FOCUSED=0 … FATIGUED=3)."""
    return _INDEX[DistractionState(state)]


def state_from_index(index: int) -> DistractionState:
    """Inverse of ``state_index``."""
    return STATES[index]
