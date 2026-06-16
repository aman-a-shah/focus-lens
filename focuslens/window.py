"""200ms feature-window aggregation (roadmap Phase 3).

Per-frame ``FrameFeatures`` are noisy and arrive at the camera rate. The model (and the
rule classifier) consume fixed 200ms windows of 13 features (plan.md §3.3, extended with
body-language + looking-down signals):

    0 gaze_x            5 blink_duration (s)        10 proximity
    1 gaze_y            6 head_pose_change_rate     11 hands_near_face
    2 gaze_velocity     7 ear                       12 looking_down
    3 gaze_accel        8 torso_lean
    4 blink_rate        9 head_drop

Velocity/acceleration/change-rate are temporal, so the aggregator is stateful: it remembers
the previous window's gaze and head pose to form first/second derivatives. ``SequenceBuffer``
stacks the most recent T windows into the [T, 13] tensor PersonalFocusNet will take later.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, fields

import numpy as np

from .features import FrameFeatures

NUM_FEATURES = 13


@dataclass(frozen=True)
class WindowFeatures:
    """Aggregated features over one 200ms window."""

    t_start: float
    t_end: float
    face_fraction: float  # fraction of frames in the window that had a face
    gaze_x: float
    gaze_y: float
    gaze_velocity: float
    gaze_accel: float
    blink_rate: float
    blink_duration: float
    head_pose_change_rate: float
    ear: float
    # Body-language + looking-down aggregates (roadmap overhaul). Default to 0.0 (= "no body /
    # level gaze") so windows built from face-only data still construct.
    torso_lean: float = 0.0
    head_drop: float = 0.0
    proximity: float = 0.0
    hands_near_face: float = 0.0
    looking_down: float = 0.0
    body_fraction: float = 0.0  # fraction of frames in the window that had a tracked body

    def to_vector(self) -> np.ndarray:
        """The 13-feature model input vector, in canonical order."""
        return np.array(
            [
                self.gaze_x,
                self.gaze_y,
                self.gaze_velocity,
                self.gaze_accel,
                self.blink_rate,
                self.blink_duration,
                self.head_pose_change_rate,
                self.ear,
                self.torso_lean,
                self.head_drop,
                self.proximity,
                self.hands_near_face,
                self.looking_down,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def header() -> list[str]:
        return [f.name for f in fields(WindowFeatures)]

    def to_row(self) -> list[object]:
        return [getattr(self, name) for name in self.header()]


def _nanmean(values: list[float]) -> float:
    arr = np.array(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else float("nan")


class WindowAggregator:
    """Accumulates frames into 200ms windows and emits ``WindowFeatures``.

    Feed every frame via ``add``; it returns a ``WindowFeatures`` on the frame that closes a
    window, else None.
    """

    def __init__(self, window_s: float = 0.2) -> None:
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        self.window_s = window_s
        self._win_start: float | None = None
        self._frames: list[FrameFeatures] = []
        # Previous-window state for temporal derivatives.
        self._prev_gaze: tuple[float, float] | None = None
        self._prev_velocity: float | None = None
        self._prev_pose: tuple[float, float, float] | None = None
        self._prev_end: float | None = None

    def add(self, features: FrameFeatures) -> WindowFeatures | None:
        if self._win_start is None:
            self._win_start = features.timestamp
        self._frames.append(features)
        if features.timestamp - self._win_start >= self.window_s:
            return self._flush(features.timestamp)
        return None

    def flush(self) -> WindowFeatures | None:
        """Emit a (possibly partial) final window for any buffered frames."""
        if not self._frames:
            return None
        return self._flush(self._frames[-1].timestamp)

    def _flush(self, t_end: float) -> WindowFeatures:
        frames = self._frames
        t_start = self._win_start if self._win_start is not None else t_end
        present = [f for f in frames if f.face_present]
        face_fraction = len(present) / len(frames) if frames else 0.0
        body_fraction = (
            sum(1 for f in frames if f.body_present) / len(frames) if frames else 0.0
        )

        gaze_x = _nanmean([f.gaze_x for f in frames])
        gaze_y = _nanmean([f.gaze_y for f in frames])
        ear = _nanmean([f.ear_mean for f in frames])
        blink_rate = _nanmean([f.blinks_per_min for f in frames])
        blink_duration = _nanmean([f.last_blink_duration_s for f in frames])
        yaw = _nanmean([f.yaw for f in frames])
        pitch = _nanmean([f.pitch for f in frames])
        roll = _nanmean([f.roll for f in frames])

        torso_lean = _nanmean([f.torso_lean for f in frames])
        head_drop = _nanmean([f.head_drop for f in frames])
        proximity = _nanmean([f.proximity for f in frames])
        hands_near_face = _nanmean([f.hands_near_face for f in frames])
        looking_down = _nanmean([f.looking_down for f in frames])

        dt = (t_end - self._prev_end) if self._prev_end is not None else self.window_s
        dt = dt if dt > 1e-6 else self.window_s

        gaze_velocity = self._gaze_velocity(gaze_x, gaze_y, dt)
        gaze_accel = self._gaze_accel(gaze_velocity, dt)
        pose_change = self._pose_change_rate(yaw, pitch, roll, dt)

        # Roll forward state for next window.
        if not (math.isnan(gaze_x) or math.isnan(gaze_y)):
            self._prev_gaze = (gaze_x, gaze_y)
        self._prev_velocity = gaze_velocity
        if not (math.isnan(yaw) or math.isnan(pitch) or math.isnan(roll)):
            self._prev_pose = (yaw, pitch, roll)
        self._prev_end = t_end
        self._frames = []
        self._win_start = None

        return WindowFeatures(
            t_start=t_start,
            t_end=t_end,
            face_fraction=face_fraction,
            gaze_x=_z(gaze_x),
            gaze_y=_z(gaze_y),
            gaze_velocity=gaze_velocity,
            gaze_accel=gaze_accel,
            blink_rate=_z(blink_rate),
            blink_duration=_z(blink_duration),
            head_pose_change_rate=pose_change,
            ear=_z(ear),
            torso_lean=_z(torso_lean),
            head_drop=_z(head_drop),
            proximity=_z(proximity),
            hands_near_face=_z(hands_near_face),
            looking_down=_z(looking_down),
            body_fraction=body_fraction,
        )

    def _gaze_velocity(self, gaze_x: float, gaze_y: float, dt: float) -> float:
        if self._prev_gaze is None or math.isnan(gaze_x) or math.isnan(gaze_y):
            return 0.0
        dx = gaze_x - self._prev_gaze[0]
        dy = gaze_y - self._prev_gaze[1]
        return math.hypot(dx, dy) / dt

    def _gaze_accel(self, velocity: float, dt: float) -> float:
        if self._prev_velocity is None:
            return 0.0
        return (velocity - self._prev_velocity) / dt

    def _pose_change_rate(self, yaw: float, pitch: float, roll: float, dt: float) -> float:
        if self._prev_pose is None or math.isnan(yaw) or math.isnan(pitch) or math.isnan(roll):
            return 0.0
        dyaw = yaw - self._prev_pose[0]
        dpitch = pitch - self._prev_pose[1]
        droll = roll - self._prev_pose[2]
        return math.sqrt(dyaw**2 + dpitch**2 + droll**2) / dt


def _z(x: float) -> float:
    """Replace NaN with 0.0 — windows with no face still need a numeric vector."""
    return 0.0 if math.isnan(x) else x


class SequenceBuffer:
    """Ring buffer of the most recent ``length`` windows -> [length, 8] tensor."""

    def __init__(self, length: int = 30) -> None:
        if length <= 0:
            raise ValueError("length must be positive")
        self.length = length
        self._windows: deque[WindowFeatures] = deque(maxlen=length)

    def append(self, window: WindowFeatures) -> None:
        self._windows.append(window)

    def __len__(self) -> int:
        return len(self._windows)

    @property
    def is_full(self) -> bool:
        return len(self._windows) == self.length

    def latest(self) -> WindowFeatures | None:
        return self._windows[-1] if self._windows else None

    def recent(self, n: int) -> list[WindowFeatures]:
        return list(self._windows)[-n:]

    def to_array(self) -> np.ndarray:
        """[len, 8] array of the buffered windows (oldest first)."""
        if not self._windows:
            return np.zeros((0, NUM_FEATURES), dtype=np.float32)
        return np.stack([w.to_vector() for w in self._windows])
