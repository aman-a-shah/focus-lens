"""MediaPipe Face Mesh wrapper (Tasks API).

Wraps the modern ``mediapipe.tasks`` ``FaceLandmarker`` so the rest of the codebase deals
with a small typed result (``FaceMeshResult``) instead of MediaPipe types. The landmarker
emits 478 landmarks including the iris (indices 468-477) — the signal gaze needs later
(plan.md §3.1).

Run in VIDEO mode: ``process(image, timestamp_ms)`` must be called with monotonically
increasing timestamps. MediaPipe is imported lazily so unit tests and ``--version`` neither
pay the import cost nor require the dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import FaceMeshConfig
from .logging import get_logger
from .models import ensure_face_landmarker_model

log = get_logger(__name__)

# Iris landmark index ranges (present in the 478-landmark model output).
LEFT_IRIS = list(range(468, 473))
RIGHT_IRIS = list(range(473, 478))


@dataclass(frozen=True)
class FaceMeshResult:
    """Normalized landmarks for one detected face.

    ``landmarks`` is an (N, 3) array of x, y, z in [0, 1] image-relative coordinates.
    ``has_iris`` is True when the iris landmarks (478 total) are present.
    """

    landmarks: np.ndarray  # (N, 3)

    @property
    def num_landmarks(self) -> int:
        return int(self.landmarks.shape[0])

    @property
    def has_iris(self) -> bool:
        return self.num_landmarks >= 478

    def iris_centers(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (left, right) iris center (x, y, z) in normalized coords, or None."""
        if not self.has_iris:
            return None
        left = self.landmarks[LEFT_IRIS].mean(axis=0)
        right = self.landmarks[RIGHT_IRIS].mean(axis=0)
        return left, right


class FaceMeshTracker:
    """Stateful FaceLandmarker tracker. Call ``process(bgr_image, timestamp_ms)`` per frame."""

    def __init__(self, config: FaceMeshConfig | None = None) -> None:
        self.config = config or FaceMeshConfig()
        self._landmarker: Any | None = None
        self._last_ts_ms = -1

    def _ensure_landmarker(self) -> Any:
        if self._landmarker is None:
            import mediapipe as mp
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions

            model_path = ensure_face_landmarker_model()
            options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=self.config.max_num_faces,
                min_face_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
            self._mp = mp
            log.info("Initialized MediaPipe FaceLandmarker (Tasks API, VIDEO mode)")
        return self._landmarker

    def process(
        self, bgr_image: np.ndarray, timestamp_ms: int | None = None
    ) -> FaceMeshResult | None:
        """Run FaceLandmarker on a BGR frame. Returns None when no face is found.

        ``timestamp_ms`` must increase across calls; when omitted an internal counter is
        used (fine for tests / single-shot use).
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

        if not result.face_landmarks:
            return None
        lm = result.face_landmarks[0]
        arr = np.array([(p.x, p.y, p.z) for p in lm], dtype=np.float32)
        return FaceMeshResult(landmarks=arr)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> FaceMeshTracker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
