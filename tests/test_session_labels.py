"""SQLite marks / events / window-labels storage (roadmap Phase 6 inputs)."""

from focuslens.session import SessionStore
from focuslens.states import DistractionState
from focuslens.window import WindowFeatures


def _w(t: float) -> WindowFeatures:
    return WindowFeatures(
        t_start=t,
        t_end=t + 0.2,
        face_fraction=1.0,
        gaze_x=0.1,
        gaze_y=-0.2,
        gaze_velocity=0.3,
        gaze_accel=0.0,
        blink_rate=12.0,
        blink_duration=0.15,
        head_pose_change_rate=4.0,
        ear=0.27,
    )


def test_marks_and_events_roundtrip():
    store = SessionStore(":memory:")
    sid = store.start_session(0.0)
    store.add_mark(sid, 5.0)
    store.add_mark(sid, 9.0, kind="noticed_drift")
    store.add_event(sid, 1.0, 2.0, "idle")
    assert store.get_marks(sid) == [(5.0, "noticed_drift"), (9.0, "noticed_drift")]
    events = store.get_events(sid)
    assert len(events) == 1 and events[0].kind == "idle"
    store.close()


def test_get_windows_reconstructs_features_and_ids():
    store = SessionStore(":memory:")
    sid = store.start_session(0.0)
    store.log_window(sid, _w(0.0), DistractionState.FOCUSED)
    store.log_window(sid, _w(0.2), DistractionState.DRIFTING)
    rows = store.get_windows(sid)
    assert [r.window_id for r in rows] == [1, 2]
    assert rows[1].state == DistractionState.DRIFTING
    assert abs(rows[0].features.gaze_x - 0.1) < 1e-6
    store.close()


def test_window_labels_upsert_and_session_label_map():
    store = SessionStore(":memory:")
    sid = store.start_session(0.0)
    store.log_window(sid, _w(0.0), DistractionState.FOCUSED)
    store.write_window_labels([(1, "FOCUSED", "heuristic")])
    store.write_window_labels([(1, "DISTRACTED", "mark_onset")])  # upsert wins
    assert store.get_window_label(1) == ("DISTRACTED", "mark_onset")
    assert store.get_session_labels(sid) == {1: "DISTRACTED"}
    store.close()
