"""Model asset management.

MediaPipe's modern Tasks API (the legacy ``solutions.face_mesh`` module was dropped in
recent builds) needs a downloaded ``.task`` bundle. This module fetches it on first use and
caches it under the repo's gitignored ``checkpoints/`` dir so it's downloaded exactly once.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

from .logging import get_logger

log = get_logger(__name__)

_CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FACE_LANDMARKER_FILENAME = "face_landmarker.task"


def ensure_face_landmarker_model(dest_dir: Path | None = None) -> Path:
    """Return the local path to the FaceLandmarker bundle, downloading it if absent."""
    directory = dest_dir or _CHECKPOINTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / FACE_LANDMARKER_FILENAME
    if path.exists() and path.stat().st_size > 0:
        return path

    log.info("Downloading FaceLandmarker model -> %s", path)
    tmp = path.with_suffix(".task.tmp")
    urllib.request.urlretrieve(FACE_LANDMARKER_URL, tmp)  # noqa: S310 (trusted Google CDN)
    tmp.replace(path)
    log.info("Downloaded FaceLandmarker model (%.1f MB)", path.stat().st_size / 1e6)
    return path
