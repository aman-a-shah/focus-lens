"""Live session loop and benchmark: capture -> Face Mesh -> features -> state -> notify/store.

The walking-skeleton runtime (roadmap Phase 3). Capture and tracking feed the same
``AttentionPipeline`` the offline simulator uses, so live and simulated runs share one code
path for windowing, classification, notification and SQLite logging.
"""

from __future__ import annotations

import csv
import time
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass

from .capture import FrameBuffer, WebcamCapture
from .config import Config
from .face_mesh import FaceMeshTracker
from .features import FeatureExtractor, FrameFeatures
from .logging import get_logger
from .notify import Notifier
from .pipeline import AttentionPipeline
from .session import SessionStore
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


_CONSOLE_PRINT_INTERVAL_S = 0.5


def run_live(
    config: Config,
    source: int | str | None = None,
    show_window: bool = True,
    max_frames: int | None = None,
    snapshot_path: str | None = None,
    features_csv: str | None = None,
    print_features: bool = False,
    db_path: str | None = "focuslens.sqlite",
    notify: bool = True,
) -> None:
    """Capture, track, extract features, classify state, notify, and log the session.

    ``source`` overrides the configured camera index — pass a video file path to replay a
    clip headlessly. ``max_frames`` stops after N frames; ``snapshot_path`` saves the first
    annotated frame with a face. ``features_csv`` streams per-frame features; ``print_features``
    echoes a throttled line. ``db_path`` (None to disable) is the SQLite session log; ``notify``
    toggles desktop notifications.
    """
    import cv2

    src = source if source is not None else config.capture.camera_index
    buffer = FrameBuffer(config.capture.buffer_seconds, config.capture.target_fps)
    fps_meter = _FpsMeter()
    extractor = FeatureExtractor()
    window = "FocusLens — live (q/Esc to quit)"
    snapped = False
    last_print = -1.0
    current_state: str | None = None
    n = 0

    with ExitStack() as stack:
        cap = stack.enter_context(
            WebcamCapture(
                source=src,
                width=config.capture.width,
                height=config.capture.height,
                target_fps=config.capture.target_fps,
            )
        )
        tracker = stack.enter_context(FaceMeshTracker(config.face_mesh))

        store = None
        session_id = None
        if db_path is not None:
            store = stack.enter_context(SessionStore(db_path))
            session_id = store.start_session(time.time())
        pipeline = AttentionPipeline(
            store=store, session_id=session_id, notifier=Notifier(enabled=notify)
        )

        csv_writer = None
        if features_csv is not None:
            fh = stack.enter_context(open(features_csv, "w", newline=""))
            csv_writer = csv.writer(fh)
            csv_writer.writerow(FrameFeatures.header())

        for frame in cap.frames():
            buffer.append(frame)
            result = tracker.process(frame.image, timestamp_ms=int(frame.timestamp * 1000))
            fps = fps_meter.tick(frame.timestamp)
            features = extractor.extract(result, frame.image.shape, frame.timestamp)
            n += 1

            out = pipeline.process_frame(features)
            if out is not None:
                current_state = str(out.state)
                if out.transitioned:
                    log.info("State -> %s @ %.1fs", current_state, out.window.t_end)

            if csv_writer is not None:
                csv_writer.writerow(features.to_row())
            if print_features and frame.timestamp - last_print >= _CONSOLE_PRINT_INTERVAL_S:
                print(features.console_line())
                last_print = frame.timestamp

            if show_window or (snapshot_path and not snapped):
                annotated = draw_overlay(frame.image, result, fps, config.viz, state=current_state)
                if snapshot_path and not snapped and result is not None:
                    cv2.imwrite(snapshot_path, annotated)
                    snapped = True
                    log.info("Saved snapshot -> %s", snapshot_path)
                if show_window:
                    cv2.imshow(window, annotated)
                    if cv2.waitKey(1) & 0xFF in _QUIT_KEYS:
                        break

            if max_frames is not None and n >= max_frames:
                break

        pipeline.finish()
        if store is not None and session_id is not None:
            store.end_session(session_id, time.time())

    if show_window:
        cv2.destroyAllWindows()
    log.info("Session ended: %d frames, buffer held %d/%d", n, len(buffer), buffer.capacity)


@dataclass(frozen=True)
class BenchResult:
    """Throughput/latency stats for the perception pipeline."""

    frames: int
    faces_found: int
    mean_fps: float
    p50_latency_ms: float
    p95_latency_ms: float

    def summary(self) -> str:
        return (
            f"{self.frames} frames | {self.faces_found} with a face | "
            f"{self.mean_fps:.1f} FPS mean | "
            f"latency p50 {self.p50_latency_ms:.1f}ms / p95 {self.p95_latency_ms:.1f}ms"
        )


def run_benchmark(config: Config, frames: int = 120, image_path: str | None = None) -> BenchResult:
    """Measure FaceLandmarker throughput on a fixed frame (no camera needed).

    Validates Phase 1's ≥20 FPS bar offline. Without ``image_path`` a sample portrait is
    downloaded and cached. Latencies cover ``tracker.process`` only.
    """
    import time

    import cv2

    from .models import ensure_sample_portrait

    path = image_path or str(ensure_sample_portrait())
    image = cv2.imread(path)
    if image is None:
        raise RuntimeError(f"Could not read benchmark image {path!r}")

    latencies_ms: list[float] = []
    faces_found = 0
    with FaceMeshTracker(config.face_mesh) as tracker:
        # Warm up: first call builds the graph and downloads the model.
        tracker.process(image, timestamp_ms=0)
        for i in range(frames):
            start = time.perf_counter()
            result = tracker.process(image, timestamp_ms=(i + 1) * 33)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            faces_found += result is not None

    ordered = sorted(latencies_ms)
    mean_latency = sum(latencies_ms) / len(latencies_ms)
    return BenchResult(
        frames=frames,
        faces_found=faces_found,
        mean_fps=1000.0 / mean_latency if mean_latency > 0 else 0.0,
        p50_latency_ms=ordered[len(ordered) // 2],
        p95_latency_ms=ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
    )
