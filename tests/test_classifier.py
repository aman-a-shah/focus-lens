from focuslens.classifier import RuleClassifier
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
