"""Per-frame feature extraction.

Combines EAR, blink detection, head pose and the naive gaze proxy into one ``FrameFeatures``
record per frame. These are the *instantaneous* signals; the temporal derivatives and
per-window aggregates the model consumes (gaze velocity/acceleration, head-pose change rate,
blink rate — plan.md §3.2) are built on top of this stream in roadmap Phase 3.

The extractor is stateful only through its ``BlinkDetector`` (one instance per session).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np

from ..face_mesh import FaceMeshResult
from .blink import BlinkDetector
from .ear import eye_aspect_ratio
from .gaze import naive_gaze
from .head_pose import HeadPoseEstimator

_NAN = float("nan")


@dataclass(frozen=True)
class FrameFeatures:
    """Instantaneous signals for one frame. NaN numeric fields mean "no face this frame"."""

    timestamp: float
    face_present: bool
    ear_right: float
    ear_left: float
    ear_mean: float
    eye_closed: bool
    blinks_per_min: float
    last_blink_duration_s: float
    yaw: float
    pitch: float
    roll: float
    gaze_x: float
    gaze_y: float

    @staticmethod
    def header() -> list[str]:
        return [f.name for f in fields(FrameFeatures)]

    def to_row(self) -> list[object]:
        return [getattr(self, name) for name in self.header()]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def console_line(self) -> str:
        if not self.face_present:
            return f"[{self.timestamp:7.2f}s] no face"
        return (
            f"[{self.timestamp:7.2f}s] EAR={self.ear_mean:.3f} "
            f"{'CLOSED' if self.eye_closed else 'open  '} "
            f"bpm={self.blinks_per_min:4.1f} "
            f"yaw={self.yaw:+6.1f} pitch={self.pitch:+6.1f} roll={self.roll:+6.1f} "
            f"gaze=({self.gaze_x:+.2f},{self.gaze_y:+.2f})"
        )


class FeatureExtractor:
    def __init__(self, blink_detector: BlinkDetector | None = None) -> None:
        self.blink = blink_detector or BlinkDetector()
        self.head_pose = HeadPoseEstimator()

    def extract(
        self,
        result: FaceMeshResult | None,
        image_shape: tuple[int, int],
        timestamp: float,
    ) -> FrameFeatures:
        """Produce features for one frame. ``image_shape`` is (height, width)."""
        height, width = image_shape[0], image_shape[1]

        if result is None:
            return FrameFeatures(
                timestamp=timestamp,
                face_present=False,
                ear_right=_NAN,
                ear_left=_NAN,
                ear_mean=_NAN,
                eye_closed=self.blink.is_closed,
                blinks_per_min=self.blink.blinks_per_min(timestamp),
                last_blink_duration_s=self.blink.last_blink_duration_s,
                yaw=_NAN,
                pitch=_NAN,
                roll=_NAN,
                gaze_x=_NAN,
                gaze_y=_NAN,
            )

        points_px = result.landmarks[:, :2] * np.array([width, height], dtype=np.float32)

        ear_right, ear_left = eye_aspect_ratio(points_px)
        ear_mean = (ear_right + ear_left) / 2.0
        blink = self.blink.update(ear_mean, timestamp)

        pose = self.head_pose.estimate(points_px, width, height)
        yaw, pitch, roll = (pose.yaw, pose.pitch, pose.roll) if pose else (_NAN, _NAN, _NAN)

        gaze = naive_gaze(points_px, result.has_iris)
        gaze_x, gaze_y = (gaze.x, gaze.y) if gaze else (_NAN, _NAN)

        return FrameFeatures(
            timestamp=timestamp,
            face_present=True,
            ear_right=ear_right,
            ear_left=ear_left,
            ear_mean=ear_mean,
            eye_closed=blink.eye_closed,
            blinks_per_min=blink.blinks_per_min,
            last_blink_duration_s=blink.last_blink_duration_s,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            gaze_x=gaze_x,
            gaze_y=gaze_y,
        )
