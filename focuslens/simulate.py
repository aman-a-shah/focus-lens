"""Offline session simulator (roadmap Phase 3 verification).

Synthesizes per-frame ``FrameFeatures`` for scripted behaviour segments and runs them through
the real ``AttentionPipeline`` — no webcam, no MediaPipe, no models. Used to validate that
windowing → classification → debounce → notification → SQLite logging all work end to end and
that each behaviour lands in the intended state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import FrameFeatures
from .notify import Notifier
from .pipeline import AttentionPipeline
from .session import SessionStore
from .states import DistractionState

# Default scripted session: (behaviour, seconds).
DEFAULT_SCENARIO: list[tuple[str, float]] = [
    ("focused", 4.0),
    ("drifting", 4.0),
    ("distracted", 6.0),
    ("focused", 3.0),
    ("fatigued", 5.0),
]


def _random_walk(n: int, step: float, bound: float, rng: np.random.RandomState) -> np.ndarray:
    """Bounded random walk — sustained, non-periodic motion (no derivative nulls)."""
    walk = np.cumsum(rng.normal(0, step, size=n))
    return np.clip(walk, -bound, bound)


def _segment_signals(label: str, n: int, rng: np.random.RandomState) -> dict[str, np.ndarray]:
    """Per-frame signal arrays for one behaviour segment (length ``n``)."""
    z = lambda s: rng.normal(0, s, size=n)  # noqa: E731
    ones = np.ones(n)
    if label == "focused":
        return dict(
            gaze_x=z(0.05),
            gaze_y=z(0.05),
            yaw=z(2),
            pitch=z(2),
            roll=z(1),
            ear=0.27 + z(0.01),
            blinks=12.0 * ones,
            blink_dur=0.15 * ones,
            face=ones,
        )
    if label == "drifting":
        # Gaze hovering near the screen edge (~0.6, well inside the drift band) and head sway
        # -> on-screen but unfocused.
        return dict(
            gaze_x=0.6 + z(0.04),
            gaze_y=z(0.06),
            yaw=_random_walk(n, 1.2, 18, rng),
            pitch=z(3),
            roll=z(2),
            ear=0.26 + z(0.01),
            blinks=14.0 * ones,
            blink_dur=0.15 * ones,
            face=ones,
        )
    if label == "distracted":
        # Gaze parked off-screen -> DRIFTING for the first ~3s, then DISTRACTED.
        return dict(
            gaze_x=1.1 + z(0.05),
            gaze_y=z(0.05),
            yaw=28 + z(3),
            pitch=z(3),
            roll=z(2),
            ear=0.26 + z(0.01),
            blinks=12.0 * ones,
            blink_dur=0.15 * ones,
            face=ones,
        )
    if label == "fatigued":
        return dict(
            gaze_x=z(0.05),
            gaze_y=z(0.05),
            yaw=z(2),
            pitch=z(2),
            roll=z(1),
            ear=0.16 + z(0.01),
            blinks=30.0 * ones,
            blink_dur=0.30 * ones,
            face=ones,
        )
    raise ValueError(f"unknown behaviour {label!r}")


def generate_frames(
    scenario: list[tuple[str, float]], fps: int = 30, seed: int = 0
) -> list[FrameFeatures]:
    """Render a scenario into a flat list of per-frame ``FrameFeatures``."""
    rng = np.random.RandomState(seed)
    dt = 1.0 / fps
    frames: list[FrameFeatures] = []
    t = 0.0
    for label, duration in scenario:
        n_frames = int(round(duration * fps))
        s = _segment_signals(label, n_frames, rng)
        for i in range(n_frames):
            ear = float(s["ear"][i])
            frames.append(
                FrameFeatures(
                    timestamp=t,
                    face_present=bool(s["face"][i]),
                    ear_right=ear,
                    ear_left=ear,
                    ear_mean=ear,
                    eye_closed=ear < 0.21,
                    blinks_per_min=float(s["blinks"][i]),
                    last_blink_duration_s=float(s["blink_dur"][i]),
                    yaw=float(s["yaw"][i]),
                    pitch=float(s["pitch"][i]),
                    roll=float(s["roll"][i]),
                    gaze_x=float(s["gaze_x"][i]),
                    gaze_y=float(s["gaze_y"][i]),
                )
            )
            t += dt
    return frames


@dataclass
class SimulationSummary:
    frames: int
    windows: int
    transitions: list[tuple[float, DistractionState]]
    notifications: int
    db_window_count: int
    state_histogram: dict[str, int]

    def report(self) -> str:
        lines = [
            f"frames={self.frames} windows={self.windows} "
            f"notifications={self.notifications} db_rows={self.db_window_count}",
            "transitions:",
        ]
        for t, state in self.transitions:
            lines.append(f"  {t:6.2f}s -> {state}")
        hist = " ".join(f"{k}={v}" for k, v in sorted(self.state_histogram.items()))
        lines.append(f"state windows: {hist}")
        return "\n".join(lines)


def run_simulation(
    scenario: list[tuple[str, float]] | None = None,
    fps: int = 30,
    seed: int = 0,
    db_path: str = ":memory:",
    notify: bool = False,
) -> SimulationSummary:
    scenario = scenario or DEFAULT_SCENARIO
    frames = generate_frames(scenario, fps=fps, seed=seed)

    store = SessionStore(db_path)
    session_id = store.start_session(frames[0].timestamp if frames else 0.0)
    pipeline = AttentionPipeline(
        store=store,
        session_id=session_id,
        notifier=Notifier(cooldown_s=5.0, enabled=notify),
    )

    transitions: list[tuple[float, DistractionState]] = []
    notifications = 0
    windows = 0
    hist: dict[str, int] = {}

    def record(out) -> None:
        nonlocal windows, notifications
        windows += 1
        hist[str(out.state)] = hist.get(str(out.state), 0) + 1
        if out.transitioned:
            transitions.append((out.window.t_end, out.state))
        if out.notified:
            notifications += 1

    for f in frames:
        out = pipeline.process_frame(f)
        if out is not None:
            record(out)
    tail = pipeline.finish()
    if tail is not None:
        record(tail)

    store.end_session(session_id, frames[-1].timestamp if frames else 0.0)
    summary = SimulationSummary(
        frames=len(frames),
        windows=windows,
        transitions=transitions,
        notifications=notifications,
        db_window_count=store.window_count(session_id),
        state_histogram=store.state_histogram(session_id),
    )
    store.close()
    return summary


def parse_scenario(text: str) -> list[tuple[str, float]]:
    """Parse 'focused:4,drifting:4,distracted:6' into scenario segments."""
    segments: list[tuple[str, float]] = []
    for part in text.split(","):
        label, _, dur = part.strip().partition(":")
        segments.append((label.strip().lower(), float(dur)))
    return segments
