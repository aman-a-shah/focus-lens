"""Distraction state label space (plan.md §3.2)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .context.activity import ActivityCategory


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


@dataclass(frozen=True)
class FusedDecision:
    """The classifier's full per-window call: the attention state plus *why*.

    ``state`` is the debounce-able attention label; ``activity`` is what you were doing (from
    app context); ``reason`` is a short human-readable explanation for notifications/UI.
    """

    state: DistractionState
    activity: ActivityCategory = ActivityCategory.UNKNOWN
    reason: str = ""
