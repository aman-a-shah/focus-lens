"""Eye Aspect Ratio (EAR).

EAR is the ratio of vertical eye opening to horizontal eye width (Soukupová & Čech, 2016).
It drops toward zero when the eye closes, so it doubles as the drowsiness / eye-openness
signal and the input to blink detection. Compute on **pixel** coordinates — the ratio is
scale-invariant but not invariant to the non-uniform x/y scaling of normalized landmarks.
"""

from __future__ import annotations

import numpy as np

from .landmarks import LEFT_EYE_EAR, RIGHT_EYE_EAR

_EPS = 1e-6


def _single_ear(points: np.ndarray) -> float:
    """EAR for one eye given its 6 points (p1..p6) as an array of shape (6, 2)."""
    p1, p2, p3, p4, p5, p6 = points
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal < _EPS:
        return 0.0
    return float(vertical / (2.0 * horizontal))


def eye_aspect_ratio(points_px: np.ndarray) -> tuple[float, float]:
    """Return (right_ear, left_ear) from an (N, 2) array of pixel-space landmarks."""
    right = _single_ear(points_px[RIGHT_EYE_EAR])
    left = _single_ear(points_px[LEFT_EYE_EAR])
    return right, left
