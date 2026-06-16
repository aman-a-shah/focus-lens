"""Notification layer (roadmap Phase 3).

Fires a desktop notification when the attention state changes to a concerning one. Kept
dependency-free: uses ``osascript`` on macOS and ``notify-send`` on Linux when available, and
always logs. A debounce prevents notification spam — the same state won't re-fire within a
cooldown, and only "worse" states (DRIFTING/DISTRACTED/FATIGUED) notify.
"""

from __future__ import annotations

import platform
import shutil
import subprocess

from .logging import get_logger
from .states import DistractionState

log = get_logger(__name__)

_NOTIFYING_STATES = {
    DistractionState.DRIFTING,
    DistractionState.DISTRACTED,
    DistractionState.FATIGUED,
}

_MESSAGES = {
    DistractionState.DRIFTING: "Your attention is drifting — eyes back on it.",
    DistractionState.DISTRACTED: "You've been distracted for a bit. Refocus?",
    DistractionState.FATIGUED: "You look fatigued — consider a short break.",
}

# Lead-in per state when a specific reason is available, e.g. "Distracted — on your phone."
_REASON_LEAD = {
    DistractionState.DRIFTING: "Heads-up",
    DistractionState.DISTRACTED: "Distracted",
    DistractionState.FATIGUED: "Heads-up",
}


def _message_for(state: DistractionState, reason: str) -> str:
    """Reason-aware notification text, falling back to the generic per-state message."""
    if reason:
        return f"{_REASON_LEAD.get(state, 'Heads-up')} — {reason}. Refocus?"
    return _MESSAGES[state]


class Notifier:
    """Debounced desktop notifier."""

    def __init__(self, cooldown_s: float = 30.0, enabled: bool = True) -> None:
        self.cooldown_s = cooldown_s
        self.enabled = enabled
        self._last_state: DistractionState | None = None
        self._last_fire_t: float | None = None

    def on_state(self, state: DistractionState, timestamp: float, reason: str = "") -> bool:
        """Maybe notify on a state transition. Returns True if a notification fired.

        ``reason`` is the classifier's short explanation (e.g. "on your phone"); when present it
        replaces the generic message so the nudge says *why* it fired.
        """
        changed = state != self._last_state
        self._last_state = state
        if not changed or state not in _NOTIFYING_STATES:
            return False
        if self._last_fire_t is not None and timestamp - self._last_fire_t < self.cooldown_s:
            return False
        self._last_fire_t = timestamp
        self._dispatch("FocusLens", _message_for(state, reason))
        return True

    def on_intervention(self, timestamp: float) -> bool:
        """Fire a pre-emptive "refocus" nudge from the hazard timer (roadmap Phase 9).

        Bypasses the state-transition logic (it's risk-driven, not state-driven) but still honours
        the cooldown so a rising hazard doesn't spam.
        """
        if self._last_fire_t is not None and timestamp - self._last_fire_t < self.cooldown_s:
            return False
        self._last_fire_t = timestamp
        self._dispatch("FocusLens", "Heads-up: you're about to drift — refocus now.")
        return True

    def _dispatch(self, title: str, message: str) -> None:
        log.info("NOTIFY: %s — %s", title, message)
        if not self.enabled:
            return
        try:
            system = platform.system()
            if system == "Darwin":
                subprocess.run(
                    ["osascript", "-e", f'display notification "{message}" with title "{title}"'],
                    check=False,
                    capture_output=True,
                )
            elif system == "Linux" and shutil.which("notify-send"):
                subprocess.run(["notify-send", title, message], check=False, capture_output=True)
        except Exception as exc:  # never let a notification failure crash the session
            log.debug("Notification dispatch failed: %s", exc)
