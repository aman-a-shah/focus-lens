"""Phase 1 live loop: capture -> Face Mesh -> overlay window.

This is the walking-skeleton front-end. Later phases hang feature extraction and the
classifier off the same capture/track loop (see roadmap.md Phase 3).
"""

from __future__ import annotations

from collections import deque

from .capture import FrameBuffer, WebcamCapture
from .config import Config
from .face_mesh import FaceMeshTracker
from .logging import get_logger
from .viz import draw_overlay

log = get_logger(__name__)

_QUIT_KEYS = {ord("q"), 27}  # 'q' or Esc


class _FpsMeter:
    """Rolling FPS estimate over a sliding window of frame timestamps."""

    def __init__(self, window: int = 30) -> None:
        self._ts: deque[float] = deque(maxlen=window)

    def tick(self, t: float) -> float:
        self._ts.append(t)
        if len(self._ts) < 2:
            return 0.0
        span = self._ts[-1] - self._ts[0]
        return (len(self._ts) - 1) / span if span > 0 else 0.0


def run_live(config: Config, show_window: bool = True) -> None:
    """Open the camera, track faces, and render the debug overlay until the user quits."""
    import cv2

    buffer = FrameBuffer(config.capture.buffer_seconds, config.capture.target_fps)
    fps_meter = _FpsMeter()
    window = "FocusLens — Phase 1 (q/Esc to quit)"

    with (
        WebcamCapture(
            camera_index=config.capture.camera_index,
            width=config.capture.width,
            height=config.capture.height,
            target_fps=config.capture.target_fps,
        ) as cap,
        FaceMeshTracker(config.face_mesh) as tracker,
    ):
        for frame in cap.frames():
            buffer.append(frame)
            result = tracker.process(frame.image, timestamp_ms=int(frame.timestamp * 1000))
            fps = fps_meter.tick(frame.timestamp)

            if show_window:
                annotated = draw_overlay(frame.image, result, fps, config.viz)
                cv2.imshow(window, annotated)
                if cv2.waitKey(1) & 0xFF in _QUIT_KEYS:
                    break

    if show_window:
        cv2.destroyAllWindows()
    log.info("Live session ended (buffer held %d/%d frames)", len(buffer), buffer.capacity)
