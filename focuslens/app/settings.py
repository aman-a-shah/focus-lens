"""App settings + the sensitivity → thresholds mapping (roadmap Phase 10).

The user-facing sensitivity slider is a single 0–1 knob; ``0.5`` reproduces the tuned
``RuleThresholds`` defaults. Turning it up makes every detector trip sooner (lower gaze/velocity/
blink thresholds, shorter sustained-distraction hold, higher eye-closed cutoff); turning it down
makes the app more forgiving. One knob, monotonic in "how eager is the nudge".
"""

from __future__ import annotations

from dataclasses import dataclass

from ..classifier import RuleThresholds

_DEFAULTS = RuleThresholds()
_SPAN = 0.8  # how far ±0.5 of slider travel scales a threshold


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def sensitivity_to_thresholds(sensitivity: float) -> RuleThresholds:
    """Map a 0–1 sensitivity to ``RuleThresholds`` (0.5 = defaults; higher = more eager)."""
    s = _clamp01(sensitivity)
    lower = 1.0 + (0.5 - s) * _SPAN  # <1 when sensitive: easier to trip
    raise_ = 1.0 + (s - 0.5) * _SPAN  # >1 when sensitive: for the "eye-closed" cutoff
    d = _DEFAULTS
    return RuleThresholds(
        gaze_offscreen=d.gaze_offscreen * lower,
        distract_hold_s=d.distract_hold_s * lower,
        drift_gaze=d.drift_gaze * lower,
        drift_velocity=d.drift_velocity * lower,
        drift_head_rate=d.drift_head_rate * lower,
        fatigue_blink_rate=d.fatigue_blink_rate * lower,
        fatigue_ear=d.fatigue_ear * raise_,
    )


@dataclass
class AppSettings:
    """Mutable user settings the tray/control panel edits live."""

    sensitivity: float = 0.5
    paused: bool = False
    notify: bool = True

    def set_sensitivity(self, value: float) -> float:
        self.sensitivity = _clamp01(value)
        return self.sensitivity

    def thresholds(self) -> RuleThresholds:
        return sensitivity_to_thresholds(self.sensitivity)
