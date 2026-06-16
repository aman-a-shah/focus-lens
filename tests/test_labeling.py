"""Self-supervised label propagation (roadmap Phase 6)."""

from focuslens.labeling import LabelConfig, label_session, propagate_labels
from focuslens.session import LoggedWindow, SessionStore, TimeInterval
from focuslens.states import DistractionState
from focuslens.window import WindowFeatures


def _w(t: float) -> WindowFeatures:
    return WindowFeatures(
        t_start=t,
        t_end=t + 1.0,
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


def _logged(idx: int, t: float, state: DistractionState) -> LoggedWindow:
    return LoggedWindow(window_id=idx, session_id=1, features=_w(t), state=state)


def test_mark_propagates_distraction_onset_backwards():
    # Heuristic said FOCUSED everywhere; a "noticed drift" mark at t=10 should back-label onset.
    windows = [_logged(i, float(i), DistractionState.FOCUSED) for i in range(12)]
    marks = [(10.0, "noticed_drift")]
    labels = propagate_labels(windows, marks, [], LabelConfig(onset_lookback_s=5.0))

    by_id = {lb.window_id: lb for lb in labels}
    # Windows overlapping [5, 10] become DISTRACTED via mark onset...
    assert by_id[7].label == DistractionState.DISTRACTED and by_id[7].source == "mark_onset"
    # ...while earlier windows keep the heuristic label.
    assert by_id[2].label == DistractionState.FOCUSED and by_id[2].source == "heuristic"


def test_weak_event_overrides_heuristic():
    windows = [_logged(i, float(i), DistractionState.FOCUSED) for i in range(6)]
    events = [TimeInterval(2.0, 3.5, "idle")]
    labels = propagate_labels(windows, [], events, LabelConfig())
    by_id = {lb.window_id: lb for lb in labels}
    assert by_id[2].label == DistractionState.DISTRACTED and by_id[2].source == "weak_idle"
    assert by_id[5].source == "heuristic"


def test_mark_takes_priority_over_weak_event():
    windows = [_logged(0, 0.0, DistractionState.FOCUSED)]
    labels = propagate_labels(
        windows,
        marks=[(0.5, "noticed_drift")],
        events=[TimeInterval(0.0, 1.0, "idle")],
        config=LabelConfig(onset_lookback_s=2.0),
    )
    assert labels[0].source == "mark_onset"  # most-specific source wins


def test_label_session_persists_to_sqlite():
    store = SessionStore(":memory:")
    sid = store.start_session(0.0)
    for i in range(8):
        store.log_window(sid, _w(float(i)), DistractionState.FOCUSED)
    store.add_mark(sid, t=6.0)
    store.add_event(sid, 0.0, 1.5, "app_switch")

    n = label_session(store, sid, LabelConfig(onset_lookback_s=3.0))
    assert n == 8
    assert store.labeled_window_count() == 8
    hist = store.label_histogram()
    assert hist.get("DISTRACTED", 0) >= 1  # both the mark onset and the app-switch contributed
    store.close()
