from focuslens.classifier import RuleClassifier
from focuslens.context.activity import ActivityCategory
from focuslens.states import DistractionState
from focuslens.window import WindowFeatures


def _w(t_start=0.0, t_end=0.2, **kw):
    base = dict(
        face_fraction=1.0,
        gaze_x=0.0,
        gaze_y=0.0,
        gaze_velocity=0.0,
        gaze_accel=0.0,
        blink_rate=12.0,
        blink_duration=0.15,
        head_pose_change_rate=0.0,
        ear=0.27,
    )
    base.update(kw)
    return WindowFeatures(t_start=t_start, t_end=t_end, **base)


def test_focused_default():
    assert RuleClassifier().classify(_w()) == DistractionState.FOCUSED


def test_edge_gaze_is_drifting():
    assert RuleClassifier().classify(_w(gaze_x=0.6)) == DistractionState.DRIFTING


def test_high_gaze_velocity_is_drifting():
    assert RuleClassifier().classify(_w(gaze_velocity=2.0)) == DistractionState.DRIFTING


def test_low_ear_is_fatigued():
    assert RuleClassifier().classify(_w(ear=0.16)) == DistractionState.FATIGUED


def test_high_blink_rate_is_fatigued():
    assert RuleClassifier().classify(_w(blink_rate=30.0)) == DistractionState.FATIGUED


def test_no_face_window_is_offscreen_then_distracted():
    clf = RuleClassifier()
    # face missing -> offscreen; before the hold elapses it's DRIFTING, after it's DISTRACTED
    early = clf.classify(_w(t_start=0.0, t_end=0.2, face_fraction=0.0))
    assert early == DistractionState.DRIFTING
    state = early
    for k in range(1, 20):  # advance well past distract_hold_s (3s)
        state = clf.classify(_w(t_start=k * 0.2, t_end=(k + 1) * 0.2, face_fraction=0.0))
    assert state == DistractionState.DISTRACTED


def test_returning_to_screen_resets_distraction_hold():
    clf = RuleClassifier()
    for k in range(10):  # offscreen for ~2s (not yet distracted)
        clf.classify(_w(t_start=k * 0.2, t_end=(k + 1) * 0.2, gaze_x=1.0))
    # glance back resets the timer
    assert clf.classify(_w(t_start=2.0, t_end=2.2, gaze_x=0.0)) == DistractionState.FOCUSED
    assert clf.classify(_w(t_start=2.2, t_end=2.4, gaze_x=1.0)) == DistractionState.DRIFTING


def _hold(clf, activity=None, seconds=4.0, **kw):
    """Feed identical windows for ``seconds`` and return the final state."""
    state = None
    n = int(seconds / 0.2) + 1
    for k in range(n):
        state = clf.classify(_w(t_start=k * 0.2, t_end=(k + 1) * 0.2, **kw), activity)
    return state


# --- the core regression: eyes wide open but clearly not working ---------------------------


def test_eyes_open_but_on_social_media_is_distracted():
    # Perfectly focused-looking face (gaze centred, eyes open) — but a social app is in front.
    # The old EAR-driven classifier called this FOCUSED; the fused one must not.
    clf = RuleClassifier()
    assert _hold(clf, ActivityCategory.SOCIAL) == DistractionState.DISTRACTED


def test_eyes_open_but_gaming_is_distracted():
    clf = RuleClassifier()
    decision = None
    n = 25
    for k in range(n):
        decision = clf.decide(
            _w(t_start=k * 0.2, t_end=(k + 1) * 0.2), ActivityCategory.GAMING
        )
    assert decision.state == DistractionState.DISTRACTED
    assert decision.activity == ActivityCategory.GAMING
    assert "gaming" in decision.reason


def test_phone_posture_is_distracted_even_with_open_eyes():
    # Hand up at the face + looking down = phone-in-hand, regardless of eye-openness or app.
    clf = RuleClassifier()
    state = _hold(
        clf,
        activity=ActivityCategory.UNKNOWN,  # webcam-only: no app context at all
        seconds=3.0,
        body_fraction=1.0,
        hands_near_face=0.8,
        looking_down=0.5,
    )
    assert state == DistractionState.DISTRACTED
    assert clf.last_reason == "on your phone"


def test_phone_in_lap_is_distracted_without_hand_at_face():
    # Head hunched down + eyes down, hands too low to register near the face = lap phone.
    clf = RuleClassifier()
    state = _hold(
        clf,
        seconds=2.5,
        body_fraction=1.0,
        hands_near_face=0.0,
        head_drop=-0.2,  # dropped from the ~-0.9 upright baseline
        looking_down=0.45,
    )
    assert state == DistractionState.DISTRACTED
    assert clf.last_reason == "on your phone"


def test_chin_on_hand_at_screen_is_not_a_phone():
    # Hand at the face but eyes level and head upright = resting your chin, still working.
    clf = RuleClassifier()
    state = _hold(
        clf,
        seconds=3.0,
        body_fraction=1.0,
        hands_near_face=0.8,
        head_drop=-0.9,
        looking_down=0.0,
    )
    assert state == DistractionState.FOCUSED


def test_work_app_on_screen_stays_focused():
    clf = RuleClassifier()
    assert _hold(clf, ActivityCategory.WORK) == DistractionState.FOCUSED


def test_wandering_eyes_escalate_to_distracted():
    # Eyes parked off-centre (on-screen but wandering): brief = drifting, sustained = distracted.
    clf = RuleClassifier()
    assert clf.classify(_w(t_start=0.0, t_end=0.2, gaze_x=0.6)) == DistractionState.DRIFTING
    state = None
    for k in range(1, 20):  # hold the wander past distract_hold_s
        state = clf.classify(_w(t_start=k * 0.2, t_end=(k + 1) * 0.2, gaze_x=0.6))
    assert state == DistractionState.DISTRACTED


def test_transient_distracting_activity_is_only_drifting():
    # A single glance at a distracting app shouldn't immediately escalate to DISTRACTED.
    clf = RuleClassifier()
    assert clf.classify(_w(), ActivityCategory.SOCIAL) == DistractionState.DRIFTING


def test_unknown_activity_falls_back_to_gaze_focused():
    # With no app context and an attentive face/body, we stay FOCUSED (graceful degradation).
    clf = RuleClassifier()
    assert clf.classify(_w(), ActivityCategory.UNKNOWN) == DistractionState.FOCUSED


def test_decide_reports_activity_and_reason():
    clf = RuleClassifier()
    decision = clf.decide(_w(), ActivityCategory.WORK)
    assert decision.state == DistractionState.FOCUSED
    assert decision.activity == ActivityCategory.WORK
