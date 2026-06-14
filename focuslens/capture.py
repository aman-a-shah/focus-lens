"""Webcam capture and the circular frame buffer.

``FrameBuffer`` is a fixed-length ring of recent (timestamp, frame) pairs — the 5s buffer
from the runtime diagram in plan.md §4. It's deliberately framework-free so it can be unit
tested without a camera. ``WebcamCapture`` is a thin context manager over ``cv2.VideoCapture``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from .logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Frame:
    """A captured frame with the monotonic timestamp (seconds) it was grabbed."""

    timestamp: float
    image: Any  # np.ndarray (BGR); typed loosely to avoid a hard numpy import here


class FrameBuffer:
    """Fixed-length ring buffer of recent frames.

    Capacity is derived from ``seconds * fps`` so the buffer always holds roughly the last
    ``seconds`` of video regardless of frame rate.
    """

    def __init__(self, seconds: float, fps: int) -> None:
        if seconds <= 0 or fps <= 0:
            raise ValueError("seconds and fps must be positive")
        self.capacity = max(1, int(round(seconds * fps)))
        self._frames: deque[Frame] = deque(maxlen=self.capacity)

    def append(self, frame: Frame) -> None:
        self._frames.append(frame)

    def latest(self) -> Frame | None:
        return self._frames[-1] if self._frames else None

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[Frame]:
        return iter(self._frames)

    @property
    def is_full(self) -> bool:
        return len(self._frames) == self.capacity


class WebcamCapture:
    """Context manager around an OpenCV capture.

    ``source`` is a camera index (``int``) or a path to a video file (``str``). Width/height/
    fps hints are only applied to live cameras. Usage::

        with WebcamCapture(source=0) as cap:
            for frame in cap.frames():
                ...
    """

    def __init__(
        self,
        source: int | str = 0,
        width: int = 1280,
        height: int = 720,
        target_fps: int = 30,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self._cap = None

    @property
    def is_camera(self) -> bool:
        return isinstance(self.source, int)

    def __enter__(self) -> WebcamCapture:
        import cv2

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            if self.is_camera:
                raise RuntimeError(
                    f"Could not open camera index {self.source}. "
                    "Is another app using the webcam, or is the index wrong?"
                )
            raise RuntimeError(f"Could not open video source {self.source!r} (missing file?)")
        if self.is_camera:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self._cap = cap
        log.info("Opened source %r", self.source)
        return self

    def frames(self) -> Iterator[Any]:
        """Yield BGR frames until the stream ends or the context exits."""
        import time

        assert self._cap is not None, "WebcamCapture must be used as a context manager"
        while True:
            ok, image = self._cap.read()
            if not ok:
                # Expected at end-of-file; only noteworthy for a live camera.
                if self.is_camera:
                    log.warning("Camera read failed; stopping capture")
                break
            yield Frame(timestamp=time.monotonic(), image=image)

    def __exit__(self, *exc: object) -> None:
        if self._cap is not None:
            self._cap.release()
            log.info("Released source %r", self.source)
            self._cap = None
