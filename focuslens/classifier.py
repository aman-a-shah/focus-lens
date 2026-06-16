"""Rule-based attention classifier — fused body + gaze + activity (roadmap overhaul).

The original version was, in practice, a drowsiness detector: low eye-openness (EAR) → FATIGUED,
otherwise gaze offset decided the rest. So anyone with their eyes open read as FOCUSED — even
on their phone, gaming, or doom-scrolling. This version demotes eye-openness to a *fatigue-only*
signal and decides focus by **fusing three independent lines of evidence**:

- **Activity** (what app is in front of you) — the single most reliable cue for distinguishing
  working from social media / video / games. Sustained non-work activity → DISTRACTED.
- **Body language** (from PoseLandmarker) — a hand up at the face plus a downward gaze/head is
  the classic phone-in-hand posture, independent of how open your eyes are.
- **Gaze / head direction** — sustained looking away or off-screen.

Any one of these, held long enough, escalates to DISTRACTED regardless of eye-openness. The
transient versions surface as DRIFTING (early warning). When app context is unavailable
(disabled / off-platform), the classifier degrades gracefully to body + gaze only — still far
better than eyes-open-means-focused. It returns a ``FusedDecision`` (state + activity + a short
human reason) and also exposes ``last_reason`` so callers expecting a bare state still work.
"""

from __future__ import annotations

from dataclasses import dataclass

from .context.activity import ActivityCategory, is_distracting_activity
from .states import DistractionState, FusedDecision
from .window import WindowFeatures


@dataclass(frozen=True)
class RuleThresholds:
    # ---- gaze / off-screen (where you're looking) ----
    # |gaze| beyond this (in proxy units) reads as off-screen.
    gaze_offscreen: float = 0.85
    # How long off-screen / non-work must hold to count as DISTRACTED.
    distract_hold_s: float = 3.0
    # |gaze| in [drift_gaze, gaze_offscreen) = hovering near the screen edge -> drifting.
    drift_gaze: float = 0.45
    # Gaze saccade velocity (proxy units / s) signalling wandering.
    drift_velocity: float = 1.5
    # Head motion rate (deg/s) signalling wandering — transient, not sustained distraction.
    drift_head_rate: float = 60.0

    # ---- activity (what you're doing) ----
    # How long a distracting foreground app must hold to count as DISTRACTED.
    activity_hold_s: float = 3.0

    # ---- body language (phone in hand / hunched over) ----
    # Wrist-at-face score above this is "a hand is up by your head".
    phone_hands: float = 0.55
    # Downward-gaze component above this corroborates looking-down-at-a-phone.
    phone_looking_down: float = 0.3
    # head_drop above this (baseline ~ -0.9 upright) = head hunched down toward the phone/lap.
    phone_head_drop: float = -0.4
    # How long the phone posture must hold to count as DISTRACTED.
    phone_hold_s: float = 2.0

    # ---- fatigue (the demoted eye-openness signal) ----
    fatigue_blink_rate: float = 26.0
    fatigue_ear: float = 0.18


class _Sustain:
    """Tracks how long a condition has held continuously, in window time."""

    def __init__(self) -> None:
        self._since: float | None = None

    def update(self, active: bool, t_start: float, t_end: float) -> float:
        if not active:
            self._since = None
            return 0.0
        if self._since is None:
            self._since = t_start
        return t_end - self._since


# Human phrasing for the distracting activities, used in notifications/UI.
_ACTIVITY_PHRASE = {
    ActivityCategory.SOCIAL: "scrolling social media",
    ActivityCategory.ENTERTAINMENT: "watching videos",
    ActivityCategory.GAMING: "gaming",
}


class RuleClassifier:
    """Stateful fused classifier — tracks how long each distracting condition has held."""

    def __init__(self, thresholds: RuleThresholds | None = None) -> None:
        self.t = thresholds or RuleThresholds()
        self._off = _Sustain()
        self._activity = _Sustain()
        self._phone = _Sustain()
        self.last_reason = ""
        self.last_activity = ActivityCategory.UNKNOWN

    def decide(
        self, window: WindowFeatures, activity: ActivityCategory | None = None
    ) -> FusedDecision:
        """Full call: attention state + activity + reason for one window."""
        t = self.t
        activity = activity or ActivityCategory.UNKNOWN
        self.last_activity = activity
        has_face = window.face_fraction >= 0.5
        has_body = window.body_fraction >= 0.5

        # --- evidence ---
        offscreen = (
            window.face_fraction < 0.5
            or abs(window.gaze_x) >= t.gaze_offscreen
            or abs(window.gaze_y) >= t.gaze_offscreen
        )
        distracting_activity = is_distracting_activity(activity)
        phone_posture = (
            has_body
            and window.hands_near_face >= t.phone_hands
            and (
                window.looking_down >= t.phone_looking_down
                or window.head_drop >= t.phone_head_drop
            )
        )

        held_off = self._off.update(offscreen, window.t_start, window.t_end)
        held_act = self._activity.update(distracting_activity, window.t_start, window.t_end)
        held_phone = self._phone.update(phone_posture, window.t_start, window.t_end)

        # --- sustained distractions (any one wins, regardless of eye-openness) ---
        if held_act >= t.activity_hold_s:
            return self._distracted(activity, self._activity_reason(activity))
        if held_phone >= t.phone_hold_s:
            return self._distracted(activity, "on your phone")
        if held_off >= t.distract_hold_s:
            away = not (has_face or has_body)
            reason = "away from your desk" if away else "looking away from the screen"
            return self._distracted(activity, reason)

        # --- fatigue: the only remaining use of eye-openness/blink ---
        eye_fatigue = window.blink_rate >= t.fatigue_blink_rate or 0.0 < window.ear <= t.fatigue_ear
        if has_face and eye_fatigue:
            return self._decision(DistractionState.FATIGUED, activity, "you look fatigued")

        # --- transient versions surface as an early-warning DRIFTING ---
        gaze_edge = max(abs(window.gaze_x), abs(window.gaze_y)) >= t.drift_gaze
        if distracting_activity:
            return self._decision(
                DistractionState.DRIFTING, activity, self._activity_reason(activity)
            )
        if phone_posture:
            return self._decision(DistractionState.DRIFTING, activity, "reaching for your phone")
        if (
            offscreen
            or gaze_edge
            or window.gaze_velocity >= t.drift_velocity
            or abs(window.head_pose_change_rate) >= t.drift_head_rate
        ):
            return self._decision(DistractionState.DRIFTING, activity, "your attention is drifting")

        return self._decision(DistractionState.FOCUSED, activity, "")

    def classify(
        self, window: WindowFeatures, activity: ActivityCategory | None = None
    ) -> DistractionState:
        """Backward-compatible entry point: returns just the attention state."""
        return self.decide(window, activity).state

    def _decision(
        self, state: DistractionState, activity: ActivityCategory, reason: str
    ) -> FusedDecision:
        self.last_reason = reason
        return FusedDecision(state=state, activity=activity, reason=reason)

    def _distracted(self, activity: ActivityCategory, reason: str) -> FusedDecision:
        return self._decision(DistractionState.DISTRACTED, activity, reason)

    @staticmethod
    def _activity_reason(activity: ActivityCategory) -> str:
        return _ACTIVITY_PHRASE.get(activity, f"on {str(activity).lower()}")
