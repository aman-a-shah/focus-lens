import sqlite3

from focuslens.context.activity import ActivityCategory
from focuslens.session import SessionStore
from focuslens.states import DistractionState
from focuslens.window import WindowFeatures


def _w(state_t=0.0, **kw):
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
    return WindowFeatures(t_start=state_t, t_end=state_t + 0.2, **base)


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


def test_body_features_and_activity_round_trip():
    store = SessionStore(":memory:")
    sid = store.start_session(0.0)
    w = _w(hands_near_face=0.8, head_drop=-0.3, looking_down=0.5, body_fraction=1.0)
    store.log_window(sid, w, DistractionState.DISTRACTED, ActivityCategory.SOCIAL)

    logged = store.get_windows(sid)[0]
    assert logged.activity == ActivityCategory.SOCIAL
    assert logged.features.hands_near_face == 0.8
    assert logged.features.looking_down == 0.5
    assert store.activity_histogram(sid) == {"SOCIAL": 1}
    store.close()


def test_activity_defaults_to_unknown():
    store = SessionStore(":memory:")
    sid = store.start_session(0.0)
    store.log_window(sid, _w(), DistractionState.FOCUSED)  # no activity passed
    assert store.get_windows(sid)[0].activity == ActivityCategory.UNKNOWN
    store.close()


def test_migrates_pre_overhaul_windows_table(tmp_path):
    """A DB created before the overhaul (no body/activity columns) is back-filled on open."""
    db = tmp_path / "old.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, started_at REAL, ended_at REAL
        );
        CREATE TABLE windows (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER, t_start REAL, t_end REAL,
            face_fraction REAL, gaze_x REAL, gaze_y REAL, gaze_velocity REAL, gaze_accel REAL,
            blink_rate REAL, blink_duration REAL, head_pose_change_rate REAL, ear REAL, state TEXT
        );
        INSERT INTO sessions(started_at) VALUES (0.0);
        INSERT INTO windows(
            session_id, t_start, t_end, face_fraction, gaze_x, gaze_y,
            gaze_velocity, gaze_accel, blink_rate, blink_duration,
            head_pose_change_rate, ear, state
        ) VALUES (1, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0, 0.0, 12.0, 0.15, 0.0, 0.27, 'FOCUSED');
        """
    )
    conn.commit()
    conn.close()

    # Opening through SessionStore should add the new columns and read the old row back cleanly.
    store = SessionStore(str(db))
    logged = store.get_windows(1)
    assert len(logged) == 1
    assert logged[0].state == DistractionState.FOCUSED
    assert logged[0].activity == ActivityCategory.UNKNOWN  # back-filled default
    assert logged[0].features.hands_near_face == 0.0
    # New writes work against the migrated table.
    store.log_window(1, _w(0.2, hands_near_face=0.5), DistractionState.DISTRACTED,
                     ActivityCategory.GAMING)
    assert store.activity_histogram(1) == {"UNKNOWN": 1, "GAMING": 1}
    store.close()
