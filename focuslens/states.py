"""Distraction state label space (plan.md §3.2)."""

from __future__ import annotations

from enum import StrEnum


class DistractionState(StrEnum):
    """The four attention states the system reasons about."""

    FOCUSED = "FOCUSED"  # gaze on screen, low velocity, normal blink rate
    DRIFTING = "DRIFTING"  # gaze wandering but returning — early warning
    DISTRACTED = "DISTRACTED"  # gaze/head off-screen or unfocused for a sustained stretch
    FATIGUED = "FATIGUED"  # high blink rate, slow gaze, EAR dropping
