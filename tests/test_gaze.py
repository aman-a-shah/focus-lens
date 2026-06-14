import numpy as np
import pytest

from focuslens.features.gaze import naive_gaze
from focuslens.features.landmarks import (
    LEFT_EYE_CORNERS,
    LEFT_EYE_TOP_BOTTOM,
    LEFT_IRIS,
    RIGHT_EYE_CORNERS,
    RIGHT_EYE_TOP_BOTTOM,
    RIGHT_IRIS,
)


def _build_eye(pts, corners, top_bottom, iris, iris_center):
    pts[corners[0]] = (0, 0)  # outer
    pts[corners[1]] = (10, 0)  # inner  -> eye center (5, 0), half-width 5
    pts[top_bottom[0]] = (5, 2)  # top
    pts[top_bottom[1]] = (5, -2)  # bottom -> half-height 2
    for idx in iris:
        pts[idx] = iris_center


def _frame(iris_center):
    pts = np.zeros((478, 2), dtype=np.float32)
    _build_eye(pts, RIGHT_EYE_CORNERS, RIGHT_EYE_TOP_BOTTOM, RIGHT_IRIS, iris_center)
    _build_eye(pts, LEFT_EYE_CORNERS, LEFT_EYE_TOP_BOTTOM, LEFT_IRIS, iris_center)
    return pts


def test_centered_iris_is_neutral_gaze():
    gaze = naive_gaze(_frame((5, 0)), has_iris=True)
    assert gaze is not None
    assert gaze.x == 0.0
    assert gaze.y == 0.0


def test_iris_shifted_outward_gives_positive_x():
    gaze = naive_gaze(_frame((7.5, 0)), has_iris=True)
    assert gaze.x == pytest.approx(0.5)  # (7.5 - 5) / half-width 5


def test_iris_shifted_down_gives_positive_y():
    gaze = naive_gaze(_frame((5, 1)), has_iris=True)
    assert gaze.y == pytest.approx(0.5)  # (1 - 0) / half-height 2


def test_no_iris_returns_none():
    assert naive_gaze(_frame((5, 0)), has_iris=False) is None
