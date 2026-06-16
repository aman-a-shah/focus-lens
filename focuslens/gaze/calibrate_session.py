"""Interactive on-screen calibration routine (roadmap Phase 5).

Walks the user through the 5- (or 9-) point routine: a dot appears at each known on-screen
target, the user looks at it, and we collect the per-frame geometric features (iris offset +
head pose) labelled with that target. The collected ``CalibrationData`` is then fed to
``calibrate_user`` to fine-tune a per-user gaze head (~2 min, ~1000 samples).

The camera/UI loop lives here; the learning core (``calibration.py``) and the feature layout
(``predictor.features_from``) are camera-free and unit-tested. ``cv2`` is imported lazily so the
package imports without a display.
"""

from __future__ import annotations

import time

import numpy as np

from ..capture import WebcamCapture
from ..config import Config
from ..face_mesh import FaceMeshTracker
from ..features.gaze import naive_gaze
from ..features.head_pose import HeadPoseEstimator
from ..logging import get_logger
from .calibration import (
    CALIB_POINTS_5,
    CALIB_POINTS_9,
    CalibrationData,
    CalibrationResult,
    calibrate_user,
)
from .predictor import features_from

log = get_logger(__name__)

_SETTLE_S = 1.0  # let the user fixate before recording
_DWELL_S = 2.0  # recording window per target


def _target_pixel(target: tuple[float, float], w: int, h: int) -> tuple[int, int]:
    """Map a normalized [-1, 1] target to a pixel position with a margin."""
    margin = 0.08
    nx = (target[0] + 1.0) / 2.0
    ny = (target[1] + 1.0) / 2.0
    px = int((margin + (1 - 2 * margin) * nx) * w)
    py = int((margin + (1 - 2 * margin) * ny) * h)
    return px, py


def collect_calibration_data(
    config: Config,
    points: list[tuple[float, float]],
    *,
    source: int | str | None = None,
    settle_s: float = _SETTLE_S,
    dwell_s: float = _DWELL_S,
) -> CalibrationData:
    """Run the on-screen routine and return the labelled samples (needs a camera + display)."""
    import cv2

    data = CalibrationData()
    extractor = HeadPoseEstimator()
    src = source if source is not None else config.capture.camera_index
    win = "FocusLens — calibration (look at the dot, Esc to abort)"

    with (
        WebcamCapture(
            source=src,
            width=config.capture.width,
            height=config.capture.height,
            target_fps=config.capture.target_fps,
        ) as cap,
        FaceMeshTracker(config.face_mesh) as tracker,
    ):
        frames = cap.frames()
        for target in points:
            t0 = None
            for frame in frames:
                h, w = frame.image.shape[:2]
                if t0 is None:
                    t0 = frame.timestamp
                elapsed = frame.timestamp - t0
                recording = settle_s <= elapsed < settle_s + dwell_s

                if recording:
                    result = tracker.process(frame.image, timestamp_ms=int(frame.timestamp * 1000))
                    if result is not None:
                        points_px = result.landmarks[:, :2] * np.array([w, h], dtype=np.float32)
                        gray = frame.image[..., :3].mean(axis=2)
                        offset = naive_gaze(points_px, result.has_iris, gray)
                        if offset is not None:
                            pose = extractor.estimate(points_px, w, h)
                            data.add(features_from(offset, pose), target)

                canvas = np.zeros_like(frame.image)
                cx, cy = _target_pixel(target, w, h)
                color = (0, 255, 0) if recording else (0, 165, 255)
                cv2.circle(canvas, (cx, cy), 18, color, -1)
                cv2.circle(canvas, (cx, cy), 6, (255, 255, 255), -1)
                cv2.imshow(win, canvas)
                if cv2.waitKey(1) & 0xFF == 27:
                    cv2.destroyAllWindows()
                    raise KeyboardInterrupt("calibration aborted")
                if elapsed >= settle_s + dwell_s:
                    break
        cv2.destroyAllWindows()

    log.info("collected %d calibration samples over %d points", len(data), len(points))
    return data


def run_calibration(
    config: Config,
    *,
    user_id: str = "user",
    nine_point: bool = False,
    source: int | str | None = None,
    base_checkpoint: str | None = None,
) -> CalibrationResult:
    """Full routine: collect on-screen samples, then fine-tune + save the per-user checkpoint."""
    points = CALIB_POINTS_9 if nine_point else CALIB_POINTS_5
    start = time.time()
    data = collect_calibration_data(config, points, source=source)
    if len(data) < 2:
        raise RuntimeError("calibration collected too few samples (no face detected?)")
    result = calibrate_user(data, user_id=user_id, base_checkpoint=base_checkpoint)
    log.info("calibration finished in %.0fs", time.time() - start)
    return result
