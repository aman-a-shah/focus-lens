"""Rule-based distraction classifier (roadmap Phase 3).

A deterministic, threshold-based stand-in for PersonalFocusNet (Phase 7). It maps the recent
window stream to a ``DistractionState`` so the end-to-end skeleton — capture → features →
windows → state → notification → SQLite — runs with zero trained models.

Design notes:
- Leans on signals robust to the *uncalibrated* gaze proxy: head-pose deviation, gaze
  velocity (relative), blink rate, EAR. Absolute gaze offset is used only with a wide margin.
- ``DISTRACTED`` requires the off-screen condition to persist (``distract_hold_s``) so a
  single glance away doesn't trip it — "off-screen for >3s" from plan.md §3.2.
"""

from __future__ import annotations

from dataclasses import dataclass

from .states import DistractionState
from .window import WindowFeatures


@dataclass(frozen=True)
class RuleThresholds:
    # |gaze| beyond this (in proxy units) reads as off-screen.
    gaze_offscreen: float = 0.85
    # How long the off-screen condition must hold to count as DISTRACTED.
    distract_hold_s: float = 3.0
    # |gaze| in [drift_gaze, gaze_offscreen) = hovering near the screen edge -> drifting.
    drift_gaze: float = 0.45
    # Gaze saccade velocity (proxy units / s) signalling wandering.
    drift_velocity: float = 1.5
    # Head motion rate (deg/s) signalling wandering — transient, not sustained distraction.
    drift_head_rate: float = 60.0
    # Fatigue: high blink rate (per min) and/or low eye openness.
    fatigue_blink_rate: float = 26.0
    fatigue_ear: float = 0.18


def _is_offscreen(w: WindowFeatures, t: RuleThresholds) -> bool:
    """Sustained off-screen: gaze away from centre, or no face. (Head *motion* is drift.)"""
    return (
        w.face_fraction < 0.5
        or abs(w.gaze_x) >= t.gaze_offscreen
        or abs(w.gaze_y) >= t.gaze_offscreen
    )


class RuleClassifier:
    """Stateful classifier — tracks how long the off-screen condition has held."""

    def __init__(self, thresholds: RuleThresholds | None = None) -> None:
        self.t = thresholds or RuleThresholds()
        self._offscreen_since: float | None = None

    def classify(self, window: WindowFeatures) -> DistractionState:
        t = self.t

        offscreen = _is_offscreen(window, t)
        if offscreen:
            if self._offscreen_since is None:
                self._offscreen_since = window.t_start
            held = window.t_end - self._offscreen_since
        else:
            self._offscreen_since = None
            held = 0.0

        # Fatigue takes priority — it's about *how* you're looking, not where.
        if window.face_fraction >= 0.5 and (
            window.blink_rate >= t.fatigue_blink_rate or (0.0 < window.ear <= t.fatigue_ear)
        ):
            return DistractionState.FATIGUED

        if offscreen and held >= t.distract_hold_s:
            return DistractionState.DISTRACTED

        gaze_edge = max(abs(window.gaze_x), abs(window.gaze_y)) >= t.drift_gaze
        if (
            offscreen
            or gaze_edge
            or window.gaze_velocity >= t.drift_velocity
            or abs(window.head_pose_change_rate) >= t.drift_head_rate
        ):
            return DistractionState.DRIFTING

        return DistractionState.FOCUSED
