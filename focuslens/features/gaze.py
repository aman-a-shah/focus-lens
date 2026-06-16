"""Naive gaze proxy — placeholder for the trained gaze head (roadmap Phase 4/5).

Estimates where the iris sits inside each eye socket and returns a normalized offset:
(0, 0) ≈ looking straight ahead, +x ≈ toward the outer corner, +y ≈ downward. This is a
crude geometric stand-in so the downstream pipeline has a gaze signal to consume now; it is
replaced by the calibrated regression head later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .landmarks import (
    LEFT_EYE_CORNERS,
    LEFT_EYE_TOP_BOTTOM,
    LEFT_IRIS,
    RIGHT_EYE_CORNERS,
    RIGHT_EYE_TOP_BOTTOM,
    RIGHT_IRIS,
)
from .robust import robust_iris_center

_EPS = 1e-6


@dataclass(frozen=True)
class GazeProxy:
    x: float  # horizontal offset, ~[-1, 1]
    y: float  # vertical offset, ~[-1, 1]


def _eye_offset(
    points_px: np.ndarray,
    corners: tuple[int, int],
    top_bottom: tuple[int, int],
    iris: list[int],
    gray: np.ndarray | None,
) -> tuple[float, float]:
    outer = points_px[corners[0]]
    inner = points_px[corners[1]]
    top = points_px[top_bottom[0]]
    bottom = points_px[top_bottom[1]]
    # Specular-robust centre (Phase 5): drop iris points sitting on glasses glare when a
    # grayscale frame is available; otherwise fall back to the plain landmark centroid.
    iris_center = robust_iris_center(points_px, iris, gray)

    eye_center = (outer + inner) / 2.0
    half_width = np.linalg.norm(inner - outer) / 2.0
    half_height = np.linalg.norm(top - bottom) / 2.0
    gx = (iris_center[0] - eye_center[0]) / (half_width + _EPS)
    gy = (iris_center[1] - eye_center[1]) / (half_height + _EPS)
    return float(gx), float(gy)


def naive_gaze(
    points_px: np.ndarray, has_iris: bool, gray: np.ndarray | None = None
) -> GazeProxy | None:
    """Average iris-in-socket offset across both eyes; None when iris isn't available.

    ``gray`` is the optional full-frame grayscale image; when given, iris centres are computed
    with reflection masking (roadmap Phase 5).
    """
    if not has_iris:
        return None
    rx, ry = _eye_offset(points_px, RIGHT_EYE_CORNERS, RIGHT_EYE_TOP_BOTTOM, RIGHT_IRIS, gray)
    lx, ly = _eye_offset(points_px, LEFT_EYE_CORNERS, LEFT_EYE_TOP_BOTTOM, LEFT_IRIS, gray)
    return GazeProxy(x=(rx + lx) / 2.0, y=(ry + ly) / 2.0)
