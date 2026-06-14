"""SQLite session log (roadmap Phase 3, plan.md §4).

Persists every session and its per-window features + classified state. This is the same
local store that later feeds self-supervised labelling (Phase 6) and the experience-replay
buffer (Phase 8). Data never leaves the device.
"""

from __future__ import annotations

import sqlite3
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
"""


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
