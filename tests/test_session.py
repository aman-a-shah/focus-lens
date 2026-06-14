from focuslens.session import SessionStore
from focuslens.states import DistractionState
from focuslens.window import WindowFeatures


def _w(state_t=0.0):
    return WindowFeatures(
        t_start=state_t,
        t_end=state_t + 0.2,
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


def test_session_lifecycle_and_counts():
    store = SessionStore(":memory:")
    sid = store.start_session(started_at=100.0)
    store.log_window(sid, _w(0.0), DistractionState.FOCUSED)
    store.log_window(sid, _w(0.2), DistractionState.FOCUSED)
    store.log_window(sid, _w(0.4), DistractionState.DRIFTING)
    store.end_session(sid, ended_at=200.0)

    assert store.window_count(sid) == 3
    assert store.state_histogram(sid) == {"FOCUSED": 2, "DRIFTING": 1}
    store.close()


def test_sessions_are_isolated():
    store = SessionStore(":memory:")
    a = store.start_session(0.0)
    b = store.start_session(0.0)
    store.log_window(a, _w(), DistractionState.FOCUSED)
    assert store.window_count(a) == 1
    assert store.window_count(b) == 0
    store.close()
