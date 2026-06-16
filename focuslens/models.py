"""Model asset management.

MediaPipe's modern Tasks API (the legacy ``solutions.face_mesh`` module was dropped in
recent builds) needs a downloaded ``.task`` bundle. This module fetches it on first use and
caches it under the repo's gitignored ``checkpoints/`` dir so it's downloaded exactly once.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

from .logging import get_logger
from .paths import CHECKPOINTS_DIR as _CHECKPOINTS_DIR

log = get_logger(__name__)

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FACE_LANDMARKER_FILENAME = "face_landmarker.task"

# Pose (body) landmarker — 33 body keypoints. The *lite* bundle is the cheapest variant;
# we only need upper-body landmarks (shoulders, wrists, hips) at desk distance, so accuracy
# beyond lite buys little and costs FPS (roadmap: body-language sensing).
POSE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
POSE_LANDMARKER_FILENAME = "pose_landmarker_lite.task"

# A single face image used only for the offline `bench` command (not training data).
SAMPLE_PORTRAIT_URL = "https://storage.googleapis.com/mediapipe-assets/portrait.jpg"
SAMPLE_PORTRAIT_FILENAME = "sample_portrait.jpg"


def _ensure_download(url: str, filename: str, dest_dir: Path | None) -> Path:
    directory = dest_dir or _CHECKPOINTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    if path.exists() and path.stat().st_size > 0:
        return path
    log.info("Downloading %s -> %s", url, path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 (trusted Google CDN)
    tmp.replace(path)
    log.info("Downloaded %s (%.1f MB)", filename, path.stat().st_size / 1e6)
    return path


def ensure_sample_portrait(dest_dir: Path | None = None) -> Path:
    """Return a local face image for benchmarking, downloading it if absent."""
    return _ensure_download(SAMPLE_PORTRAIT_URL, SAMPLE_PORTRAIT_FILENAME, dest_dir)


def ensure_face_landmarker_model(dest_dir: Path | None = None) -> Path:
    """Return the local path to the FaceLandmarker bundle, downloading it if absent."""
    return _ensure_download(FACE_LANDMARKER_URL, FACE_LANDMARKER_FILENAME, dest_dir)


def ensure_pose_landmarker_model(dest_dir: Path | None = None) -> Path:
    """Return the local path to the PoseLandmarker bundle, downloading it if absent."""
    return _ensure_download(POSE_LANDMARKER_URL, POSE_LANDMARKER_FILENAME, dest_dir)
