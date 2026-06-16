"""Sensitivity → thresholds mapping + settings (roadmap Phase 10)."""

from focuslens.app.settings import AppSettings, sensitivity_to_thresholds
from focuslens.classifier import RuleThresholds


def test_midpoint_reproduces_defaults():
    th = sensitivity_to_thresholds(0.5)
    d = RuleThresholds()
    assert abs(th.gaze_offscreen - d.gaze_offscreen) < 1e-9
    assert abs(th.distract_hold_s - d.distract_hold_s) < 1e-9
    assert abs(th.fatigue_ear - d.fatigue_ear) < 1e-9


def test_higher_sensitivity_trips_sooner():
    lax = sensitivity_to_thresholds(0.1)
    eager = sensitivity_to_thresholds(0.9)
    # Lower gaze/hold/velocity thresholds => more eager to flag distraction.
    assert eager.gaze_offscreen < lax.gaze_offscreen
    assert eager.distract_hold_s < lax.distract_hold_s
    assert eager.drift_velocity < lax.drift_velocity
    # The eye-closed cutoff moves the other way (higher = easier to call fatigue).
    assert eager.fatigue_ear > lax.fatigue_ear


def test_sensitivity_is_clamped():
    s = AppSettings()
    assert s.set_sensitivity(5.0) == 1.0
    assert s.set_sensitivity(-2.0) == 0.0
