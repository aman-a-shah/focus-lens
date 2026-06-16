"""MediaPipe Pose (body) wrapper — Tasks API.

Face Mesh tells us where the eyes look; it says nothing about the *body*. To tell "hunched
over a phone" or "leaned back relaxing" from "working", we also track the upper body with the
modern ``mediapipe.tasks`` ``PoseLandmarker`` (33 body keypoints). As with ``FaceMeshTracker``
the rest of the codebase only sees a small typed result (``PoseResult``), and MediaPipe is
imported lazily so tests / ``--version`` neither pay the import cost nor require the dependency.

Run in VIDEO mode: ``process(image, timestamp_ms)`` must be called with monotonically
increasing timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import PoseConfig
from .logging import get_logger
from .models import ensure_pose_landmarker_model

log = get_logger(__name__)

# Canonical PoseLandmarker indices (33-point topology) for the upper-body points we use.
NOSE = 0
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24


@dataclass(frozen=True)
class PoseResult:
    """Normalized body landmarks for one detected person.

    ``landmarks`` is an (33, 4) array of x, y, z, visibility. x/y are in [0, 1] image-relative
    coordinates; ``visibility`` in [0, 1] is MediaPipe's confidence the point is actually seen
    (low for occluded wrists/hips), used to gate unreliable body features.
    """

    landmarks: np.ndarray  # (33, 4): x, y, z, visibility

    @property
    def num_landmarks(self) -> int:
        return int(self.landmarks.shape[0])

    def point(self, index: int) -> np.ndarray:
        """(x, y) of landmark ``index`` in normalized coords."""
        return self.landmarks[index, :2]

    def visibility(self, index: int) -> float:
        return float(self.landmarks[index, 3])


class PoseTracker:
    """Stateful PoseLandmarker tracker. Call ``process(bgr_image, timestamp_ms)`` per frame."""

    def __init__(self, config: PoseConfig | None = None) -> None:
        self.config = config or PoseConfig()
        self._landmarker: Any | None = None
        self._last_ts_ms = -1

    def _ensure_landmarker(self) -> Any:
        if self._landmarker is None:
            import mediapipe as mp
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions

            model_path = ensure_pose_landmarker_model()
            options = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=self.config.min_pose_detection_confidence,
                min_pose_presence_confidence=self.config.min_pose_presence_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
            self._mp = mp
            log.info("Initialized MediaPipe PoseLandmarker (Tasks API, VIDEO mode)")
        return self._landmarker

    def process(self, bgr_image: np.ndarray, timestamp_ms: int | None = None) -> PoseResult | None:
        """Run PoseLandmarker on a BGR frame. Returns None when no body is found.

        ``timestamp_ms`` must increase across calls; when omitted an internal counter is used.
        """
        import cv2

        landmarker = self._ensure_landmarker()

        ts = self._last_ts_ms + 1 if timestamp_ms is None else int(timestamp_ms)
        if ts <= self._last_ts_ms:
            ts = self._last_ts_ms + 1  # enforce strict monotonicity for VIDEO mode
        self._last_ts_ms = ts

        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, ts)

        if not result.pose_landmarks:
            return None
        lm = result.pose_landmarks[0]
        arr = np.array([(p.x, p.y, p.z, p.visibility) for p in lm], dtype=np.float32)
        return PoseResult(landmarks=arr)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> PoseTracker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
