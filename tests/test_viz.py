import numpy as np

from focuslens.face_mesh import FaceMeshResult
from focuslens.pose import PoseResult
from focuslens.viz import draw_overlay


def _frame():
    return np.zeros((240, 320, 3), dtype=np.uint8)


def _face():
    # 478 landmarks (with iris) at frame centre.
    lm = np.full((478, 3), 0.5, dtype=np.float32)
    return FaceMeshResult(landmarks=lm)


def _pose():
    arr = np.zeros((33, 4), dtype=np.float32)
    for idx in (0, 11, 12, 13, 14, 15, 16):
        arr[idx] = (0.5, 0.5, 0.0, 1.0)
    return PoseResult(landmarks=arr)


def test_overlay_with_state_and_reason_runs():
    out = draw_overlay(_frame(), _face(), fps=30.0, state="DISTRACTED", reason="on your phone")
    assert out.shape == (240, 320, 3)


def test_overlay_with_pose_runs():
    out = draw_overlay(_frame(), _face(), fps=30.0, state="FOCUSED", pose=_pose())
    assert out.shape == (240, 320, 3)


def test_overlay_no_face_runs():
    out = draw_overlay(_frame(), None, fps=None, state="DISTRACTED", reason="away from your desk")
    assert out.shape == (240, 320, 3)
