"""Per-frame feature extraction.

Combines EAR, blink detection, head pose and the naive gaze proxy into one ``FrameFeatures``
record per frame. These are the *instantaneous* signals; the temporal derivatives and
per-window aggregates the model consumes (gaze velocity/acceleration, head-pose change rate,
blink rate — plan.md §3.2) are built on top of this stream in roadmap Phase 3.

The extractor is stateful only through its ``BlinkDetector`` (one instance per session).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING

import numpy as np

from ..face_mesh import FaceMeshResult
from .blink import BlinkDetector
from .body import body_features
from .ear import eye_aspect_ratio
from .gaze import naive_gaze
from .head_pose import HeadPoseEstimator

if TYPE_CHECKING:
    from ..gaze.predictor import GazePredictor
    from ..pose import PoseResult

_NAN = float("nan")


@dataclass(frozen=True)
class FrameFeatures:
    """Instantaneous signals for one frame.

    NaN face fields mean "no face this frame"; NaN body fields mean "no body this frame". Face
    and body are tracked independently, so one can be present without the other.
    """

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
    # How far down the eyes are pointing (downward gaze component, 0 = level). Rises when
    # looking at a phone/lap; combined with body cues to tell that apart from reading. The body
    # fields below are NaN/absent by default so callers that only have face data still construct.
    looking_down: float = _NAN
    # Body-language signals (NaN when no body is tracked). See features/body.py.
    body_present: bool = False
    torso_lean: float = _NAN
    head_drop: float = _NAN
    proximity: float = _NAN
    hands_near_face: float = _NAN

    @staticmethod
    def header() -> list[str]:
        return [f.name for f in fields(FrameFeatures)]

    def to_row(self) -> list[object]:
        return [getattr(self, name) for name in self.header()]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def console_line(self) -> str:
        if not self.face_present and not self.body_present:
            return f"[{self.timestamp:7.2f}s] no face/body"
        face = "no face"
        if self.face_present:
            face = (
                f"EAR={self.ear_mean:.3f} {'CLOSED' if self.eye_closed else 'open '} "
                f"bpm={self.blinks_per_min:4.1f} "
                f"yaw={self.yaw:+6.1f} pitch={self.pitch:+6.1f} "
                f"gaze=({self.gaze_x:+.2f},{self.gaze_y:+.2f}) down={self.looking_down:.2f}"
            )
        body = "no body"
        if self.body_present:
            body = f"hands@face={self.hands_near_face:.2f} head_drop={self.head_drop:+.2f}"
        return f"[{self.timestamp:7.2f}s] {face} | {body}"


class FeatureExtractor:
    def __init__(
        self,
        blink_detector: BlinkDetector | None = None,
        gaze_predictor: GazePredictor | None = None,
        body_min_visibility: float = 0.5,
    ) -> None:
        self.blink = blink_detector or BlinkDetector()
        self.head_pose = HeadPoseEstimator()
        # None -> emit the naive iris-offset proxy unchanged (Phase 2). A calibrated predictor
        # (Phase 5) remaps it to on-screen gaze in the same [-1, 1] convention.
        self.gaze_predictor = gaze_predictor
        self.body_min_visibility = body_min_visibility

    def extract(
        self,
        result: FaceMeshResult | None,
        image_shape: tuple[int, int],
        timestamp: float,
        image: np.ndarray | None = None,
        pose: PoseResult | None = None,
    ) -> FrameFeatures:
        """Produce features for one frame. ``image_shape`` is (height, width).

        ``image`` is the optional source frame (BGR or grayscale); when given it enables
        reflection-masked iris centres (roadmap Phase 5). ``pose`` is the optional
        ``PoseResult`` for body-language features; None leaves them as "no body".
        """
        height, width = image_shape[0], image_shape[1]
        body = body_features(pose, self.body_min_visibility)

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
                looking_down=_NAN,
                body_present=body.body_present,
                torso_lean=body.torso_lean,
                head_drop=body.head_drop,
                proximity=body.proximity,
                hands_near_face=body.hands_near_face,
            )

        points_px = result.landmarks[:, :2] * np.array([width, height], dtype=np.float32)

        ear_right, ear_left = eye_aspect_ratio(points_px)
        ear_mean = (ear_right + ear_left) / 2.0
        blink = self.blink.update(ear_mean, timestamp)

        head = self.head_pose.estimate(points_px, width, height)
        yaw, pitch, roll = (head.yaw, head.pitch, head.roll) if head else (_NAN, _NAN, _NAN)

        gray = (
            None if image is None else (image if image.ndim == 2 else image[..., :3].mean(axis=2))
        )
        gaze = naive_gaze(points_px, result.has_iris, gray)
        if self.gaze_predictor is not None and gaze is not None:
            gaze = self.gaze_predictor.predict(gaze, head)
        gaze_x, gaze_y = (gaze.x, gaze.y) if gaze else (_NAN, _NAN)
        # Downward gaze component only (+y is down per the gaze proxy convention).
        looking_down = max(0.0, gaze_y) if gaze else _NAN

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
            looking_down=looking_down,
            body_present=body.body_present,
            torso_lean=body.torso_lean,
            head_drop=body.head_drop,
            proximity=body.proximity,
            hands_near_face=body.hands_near_face,
        )
