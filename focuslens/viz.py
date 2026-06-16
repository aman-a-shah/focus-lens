"""Overlay rendering for the live runtime and the app preview.

Draws the face landmarks + iris, the upper-body pose skeleton, a hand-at-face marker, an FPS
readout, and a color-coded status bar (state + reason). Pure drawing — no capture/tracking
logic — so the same annotated frame feeds the OpenCV ``live`` window and the Tk dashboard.
"""

from __future__ import annotations

import numpy as np

from .config import VizConfig
from .face_mesh import LEFT_IRIS, RIGHT_IRIS, FaceMeshResult

# Pose landmark indices we draw (kept local so this module doesn't hard-depend on pose.py
# internals beyond the shoulder/arm/nose set).
_NOSE = 0
_L_SHOULDER, _R_SHOULDER = 11, 12
_L_ELBOW, _R_ELBOW = 13, 14
_L_WRIST, _R_WRIST = 15, 16
_SKELETON = [
    (_L_SHOULDER, _R_SHOULDER),
    (_L_SHOULDER, _L_ELBOW),
    (_L_ELBOW, _L_WRIST),
    (_R_SHOULDER, _R_ELBOW),
    (_R_ELBOW, _R_WRIST),
]

# BGR palette.
_GREEN = (90, 220, 120)
_CYAN = (255, 255, 0)
_RED = (60, 60, 235)
_WHITE = (255, 255, 255)
_AMBER = (40, 180, 250)
_GREY = (170, 170, 170)
_DARK = (35, 30, 28)

_STATE_COLORS = {
    "FOCUSED": _GREEN,
    "DRIFTING": _AMBER,
    "DISTRACTED": _RED,
    "FATIGUED": _AMBER,
}


def _alpha_rect(canvas, p0, p1, color, alpha) -> None:
    """Blend a filled rectangle onto ``canvas`` in place."""
    import cv2

    x0, y0 = p0
    x1, y1 = p1
    roi = canvas[y0:y1, x0:x1]
    if roi.size == 0:
        return
    overlay = np.full_like(roi, color)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, roi)


def _draw_pose(canvas, pose, w, h, min_visibility) -> None:
    import cv2

    lm = pose.landmarks
    visible = lm[:, 3] >= min_visibility

    def px(i):
        return int(lm[i, 0] * w), int(lm[i, 1] * h)

    for a, b in _SKELETON:
        if visible[a] and visible[b]:
            cv2.line(canvas, px(a), px(b), _GREY, 2, cv2.LINE_AA)
    for i in (_L_SHOULDER, _R_SHOULDER, _NOSE):
        if visible[i]:
            cv2.circle(canvas, px(i), 4, _GREY, -1, cv2.LINE_AA)
    for i in (_L_WRIST, _R_WRIST):
        if visible[i]:
            cv2.circle(canvas, px(i), 7, _CYAN, 2, cv2.LINE_AA)


def draw_overlay(
    bgr_image: np.ndarray,
    result: FaceMeshResult | None,
    fps: float | None = None,
    config: VizConfig | None = None,
    state: str | None = None,
    pose: object | None = None,
    reason: str | None = None,
    pose_min_visibility: float = 0.5,
) -> np.ndarray:
    """Return an annotated copy of ``bgr_image``."""
    import cv2

    cfg = config or VizConfig()
    canvas = bgr_image.copy()
    h, w = canvas.shape[:2]

    if pose is not None:
        _draw_pose(canvas, pose, w, h, pose_min_visibility)

    if result is None:
        _alpha_rect(canvas, (0, h - 40), (w, h), _DARK, 0.55)
        cv2.putText(
            canvas, "NO FACE", (16, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _RED, 2, cv2.LINE_AA
        )
    else:
        pts = result.landmarks[:, :2].copy()
        pts[:, 0] *= w
        pts[:, 1] *= h
        pts = pts.astype(int)

        iris_idx = set(LEFT_IRIS + RIGHT_IRIS) if result.has_iris else set()
        for i, (x, y) in enumerate(pts):
            if i not in iris_idx:
                cv2.circle(canvas, (x, y), 1, _GREEN, -1, cv2.LINE_AA)

        if cfg.draw_iris and result.has_iris:
            centers = result.iris_centers()
            if centers is not None:
                for c in centers:
                    cv2.circle(canvas, (int(c[0] * w), int(c[1] * h)), 4, _CYAN, 2, cv2.LINE_AA)

    if cfg.show_fps and fps is not None:
        cv2.putText(
            canvas, f"{fps:4.0f} FPS", (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _WHITE, 2,
            cv2.LINE_AA,
        )

    if state is not None:
        _draw_status_bar(canvas, w, state, reason)

    return canvas


def _draw_status_bar(canvas, w, state, reason) -> None:
    """A translucent top bar: a state pill + the human reason."""
    import cv2

    color = _STATE_COLORS.get(state, _WHITE)
    _alpha_rect(canvas, (0, 0), (w, 44), _DARK, 0.6)
    cv2.rectangle(canvas, (12, 10), (26, 34), color, -1)  # status swatch
    cv2.putText(canvas, state, (36, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    if reason:
        cv2.putText(
            canvas, reason, (190, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _WHITE, 1, cv2.LINE_AA
        )
