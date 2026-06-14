"""Debug overlay rendering for the Phase 1 spike.

Draws landmarks, iris centers, an FPS readout and a "no face" banner onto a copy of the
frame. Pure drawing — no capture or tracking logic lives here.
"""

from __future__ import annotations

import numpy as np

from .config import VizConfig
from .face_mesh import LEFT_IRIS, RIGHT_IRIS, FaceMeshResult

_GREEN = (0, 255, 0)
_CYAN = (255, 255, 0)
_RED = (0, 0, 255)
_WHITE = (255, 255, 255)


def draw_overlay(
    bgr_image: np.ndarray,
    result: FaceMeshResult | None,
    fps: float | None = None,
    config: VizConfig | None = None,
) -> np.ndarray:
    """Return an annotated copy of ``bgr_image``."""
    import cv2

    cfg = config or VizConfig()
    canvas = bgr_image.copy()
    h, w = canvas.shape[:2]

    if result is None:
        cv2.putText(
            canvas,
            "NO FACE",
            (16, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            _RED,
            2,
            cv2.LINE_AA,
        )
    else:
        pts = result.landmarks[:, :2].copy()
        pts[:, 0] *= w
        pts[:, 1] *= h
        pts = pts.astype(int)

        iris_idx = set(LEFT_IRIS + RIGHT_IRIS) if result.has_iris else set()
        for i, (x, y) in enumerate(pts):
            if i in iris_idx:
                continue
            cv2.circle(canvas, (x, y), 1, _GREEN, -1, cv2.LINE_AA)

        if cfg.draw_iris and result.has_iris:
            centers = result.iris_centers()
            if centers is not None:
                for c in centers:
                    cx, cy = int(c[0] * w), int(c[1] * h)
                    cv2.circle(canvas, (cx, cy), 4, _CYAN, 2, cv2.LINE_AA)
            for i in iris_idx:
                x, y = pts[i]
                cv2.circle(canvas, (x, y), 1, _CYAN, -1, cv2.LINE_AA)

        cv2.putText(
            canvas,
            f"{result.num_landmarks} landmarks",
            (16, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            _WHITE,
            1,
            cv2.LINE_AA,
        )

    if cfg.show_fps and fps is not None:
        cv2.putText(
            canvas,
            f"{fps:5.1f} FPS",
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            _WHITE,
            2,
            cv2.LINE_AA,
        )

    return canvas
