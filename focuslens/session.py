"""SQLite session log (roadmap Phase 3, plan.md §4).

Persists every session and its per-window features + classified state. This is the same
local store that feeds self-supervised labelling (Phase 6 — the ``marks``, ``events`` and
``window_labels`` tables) and the experience-replay buffer (Phase 8). Data never leaves the
device.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .states import DistractionState
from .window import WindowFeatures

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  REAL NOT NULL,
    ended_at    REAL
);
CREATE TABLE IF NOT EXISTS windows (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER NOT NULL REFERENCES sessions(id),
    t_start       REAL NOT NULL,
    t_end         REAL NOT NULL,
    face_fraction REAL NOT NULL,
    gaze_x        REAL NOT NULL,
    gaze_y        REAL NOT NULL,
    gaze_velocity REAL NOT NULL,
    gaze_accel    REAL NOT NULL,
    blink_rate    REAL NOT NULL,
    blink_duration REAL NOT NULL,
    head_pose_change_rate REAL NOT NULL,
    ear           REAL NOT NULL,
    state         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_windows_session ON windows(session_id);

-- Phase 6: retrospective "I just noticed I drifted" keypresses.
CREATE TABLE IF NOT EXISTS marks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id),
    t           REAL NOT NULL,
    kind        TEXT NOT NULL
);
-- Phase 6: weak proxy signals (idle periods, app switches) as time intervals.
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id),
    t_start     REAL NOT NULL,
    t_end       REAL NOT NULL,
    kind        TEXT NOT NULL
);
-- Phase 6: propagated self-supervised label per window (one row per labelled window).
CREATE TABLE IF NOT EXISTS window_labels (
    window_id   INTEGER PRIMARY KEY REFERENCES windows(id),
    label       TEXT NOT NULL,
    source      TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class LoggedWindow:
    """A window row read back from the store: its id, features and heuristic state."""

    window_id: int
    session_id: int
    features: WindowFeatures
    state: DistractionState


@dataclass(frozen=True)
class TimeInterval:
    t_start: float
    t_end: float
    kind: str


class SessionStore:
    """Thin wrapper over a SQLite database file (use ':memory:' for tests)."""

    def __init__(self, db_path: str | Path = "focuslens.sqlite") -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def start_session(self, started_at: float) -> int:
        cur = self._conn.execute("INSERT INTO sessions(started_at) VALUES (?)", (started_at,))
        self._conn.commit()
        return int(cur.lastrowid)

    def log_window(self, session_id: int, window: WindowFeatures, state: DistractionState) -> None:
        self._conn.execute(
            """
            INSERT INTO windows(
                session_id, t_start, t_end, face_fraction,
                gaze_x, gaze_y, gaze_velocity, gaze_accel,
                blink_rate, blink_duration, head_pose_change_rate, ear, state
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                session_id,
                window.t_start,
                window.t_end,
                window.face_fraction,
                window.gaze_x,
                window.gaze_y,
                window.gaze_velocity,
                window.gaze_accel,
                window.blink_rate,
                window.blink_duration,
                window.head_pose_change_rate,
                window.ear,
                str(state),
            ),
        )

    def end_session(self, session_id: int, ended_at: float) -> None:
        self._conn.execute("UPDATE sessions SET ended_at = ? WHERE id = ?", (ended_at, session_id))
        self._conn.commit()

    # ---- Phase 6: self-supervised labelling inputs -------------------------------------------

    def add_mark(self, session_id: int, t: float, kind: str = "noticed_drift") -> None:
        """Record a retrospective "I just noticed I drifted" mark at time ``t``."""
        self._conn.execute(
            "INSERT INTO marks(session_id, t, kind) VALUES (?,?,?)", (session_id, t, kind)
        )
        self._conn.commit()

    def add_event(self, session_id: int, t_start: float, t_end: float, kind: str) -> None:
        """Record a weak-signal interval (kind='idle' or 'app_switch')."""
        self._conn.execute(
            "INSERT INTO events(session_id, t_start, t_end, kind) VALUES (?,?,?,?)",
            (session_id, t_start, t_end, kind),
        )
        self._conn.commit()

    def get_marks(self, session_id: int) -> list[tuple[float, str]]:
        cur = self._conn.execute(
            "SELECT t, kind FROM marks WHERE session_id = ? ORDER BY t", (session_id,)
        )
        return [(float(t), str(kind)) for t, kind in cur.fetchall()]

    def get_events(self, session_id: int) -> list[TimeInterval]:
        cur = self._conn.execute(
            "SELECT t_start, t_end, kind FROM events WHERE session_id = ? ORDER BY t_start",
            (session_id,),
        )
        return [TimeInterval(float(a), float(b), str(k)) for a, b, k in cur.fetchall()]

    _WINDOW_COLS = (
        "id, session_id, t_start, t_end, face_fraction, gaze_x, gaze_y, gaze_velocity, "
        "gaze_accel, blink_rate, blink_duration, head_pose_change_rate, ear, state"
    )

    @staticmethod
    def _row_to_logged(row: tuple) -> LoggedWindow:
        wid, sid, t0, t1, ff, gx, gy, gv, ga, br, bd, hpr, ear, state = row
        return LoggedWindow(
            window_id=int(wid),
            session_id=int(sid),
            features=WindowFeatures(
                t_start=t0,
                t_end=t1,
                face_fraction=ff,
                gaze_x=gx,
                gaze_y=gy,
                gaze_velocity=gv,
                gaze_accel=ga,
                blink_rate=br,
                blink_duration=bd,
                head_pose_change_rate=hpr,
                ear=ear,
            ),
            state=DistractionState(state),
        )

    def get_windows(self, session_id: int) -> list[LoggedWindow]:
        """All windows for a session in time order, with their row ids and heuristic state."""
        cur = self._conn.execute(
            f"SELECT {self._WINDOW_COLS} FROM windows WHERE session_id = ? ORDER BY t_start",
            (session_id,),
        )
        return [self._row_to_logged(r) for r in cur.fetchall()]

    def session_ids(self) -> list[int]:
        cur = self._conn.execute("SELECT id FROM sessions ORDER BY id")
        return [int(r[0]) for r in cur.fetchall()]

    def write_window_labels(self, labels: list[tuple[int, str, str]]) -> None:
        """Upsert (window_id, label, source) rows produced by label propagation."""
        self._conn.executemany(
            "INSERT INTO window_labels(window_id, label, source) VALUES (?,?,?) "
            "ON CONFLICT(window_id) DO UPDATE SET label=excluded.label, source=excluded.source",
            labels,
        )
        self._conn.commit()

    def labeled_window_count(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM window_labels")
        return int(cur.fetchone()[0])

    def label_histogram(self) -> dict[str, int]:
        cur = self._conn.execute("SELECT label, COUNT(*) FROM window_labels GROUP BY label")
        return {str(label): int(count) for label, count in cur.fetchall()}

    def get_session_labels(self, session_id: int) -> dict[int, str]:
        """{window_id: label} for every labelled window in a session (Phase 7 dataset build)."""
        cur = self._conn.execute(
            "SELECT w.id, wl.label FROM windows w "
            "JOIN window_labels wl ON w.id = wl.window_id WHERE w.session_id = ?",
            (session_id,),
        )
        return {int(wid): str(label) for wid, label in cur.fetchall()}

    def get_window_label(self, window_id: int) -> tuple[str, str] | None:
        cur = self._conn.execute(
            "SELECT label, source FROM window_labels WHERE window_id = ?", (window_id,)
        )
        row = cur.fetchone()
        return (str(row[0]), str(row[1])) if row else None

    def window_count(self, session_id: int) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM windows WHERE session_id = ?", (session_id,))
        return int(cur.fetchone()[0])

    def state_histogram(self, session_id: int) -> dict[str, int]:
        cur = self._conn.execute(
            "SELECT state, COUNT(*) FROM windows WHERE session_id = ? GROUP BY state",
            (session_id,),
        )
        return {state: int(count) for state, count in cur.fetchall()}

    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        self._conn.commit()
        self._conn.close()

    def __enter__(self) -> SessionStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
