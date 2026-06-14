import numpy as np

from focuslens.features.ear import eye_aspect_ratio
from focuslens.features.landmarks import LEFT_EYE_EAR, RIGHT_EYE_EAR


def _set(points, indices, coords):
    for idx, xy in zip(indices, coords, strict=True):
        points[idx] = xy


def test_open_eye_has_expected_ratio():
    # p1..p6: horizontal width 4, each vertical pair spans 2 -> EAR = (2+2)/(2*4) = 0.5
    pts = np.zeros((478, 2), dtype=np.float32)
    _set(pts, RIGHT_EYE_EAR, [(0, 0), (1, 1), (3, 1), (4, 0), (3, -1), (1, -1)])
    right, _ = eye_aspect_ratio(pts)
    assert right == 0.5


def test_closed_eye_ratio_near_zero():
    pts = np.zeros((478, 2), dtype=np.float32)
    # vertical pairs collapse onto the horizontal axis -> EAR ~ 0
    _set(pts, RIGHT_EYE_EAR, [(0, 0), (1, 0), (3, 0), (4, 0), (3, 0), (1, 0)])
    right, _ = eye_aspect_ratio(pts)
    assert right == 0.0


def test_left_and_right_are_independent():
    pts = np.zeros((478, 2), dtype=np.float32)
    _set(pts, RIGHT_EYE_EAR, [(0, 0), (1, 1), (3, 1), (4, 0), (3, -1), (1, -1)])  # EAR 0.5
    _set(pts, LEFT_EYE_EAR, [(0, 0), (1, 0.5), (3, 0.5), (4, 0), (3, -0.5), (1, -0.5)])  # 0.25
    right, left = eye_aspect_ratio(pts)
    assert right == 0.5
    assert left == 0.25


def test_degenerate_horizontal_returns_zero():
    pts = np.zeros((478, 2), dtype=np.float32)  # all points coincide -> width 0
    right, left = eye_aspect_ratio(pts)
    assert right == 0.0 and left == 0.0
