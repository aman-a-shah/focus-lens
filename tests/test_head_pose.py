"""Head-pose round-trip test.

Project the canonical 3D model through a *known* rotation, feed the resulting 2D points to
the estimator, and assert it recovers the same Euler angles. This validates the full
solvePnP + Euler-extraction path against ground truth without needing a real face.
"""

import cv2
import numpy as np
import pytest

from focuslens.features.head_pose import HeadPoseEstimator
from focuslens.features.landmarks import MODEL_POINTS_3D, POSE_LANDMARK_INDICES

W = H = 640


def _euler_to_matrix(pitch, yaw, roll):
    """Build R = Rz(roll) @ Ry(yaw) @ Rx(pitch) — the convention the estimator inverts."""
    px, ya, ro = np.radians([pitch, yaw, roll])
    rx = np.array([[1, 0, 0], [0, np.cos(px), -np.sin(px)], [0, np.sin(px), np.cos(px)]])
    ry = np.array([[np.cos(ya), 0, np.sin(ya)], [0, 1, 0], [-np.sin(ya), 0, np.cos(ya)]])
    rz = np.array([[np.cos(ro), -np.sin(ro), 0], [np.sin(ro), np.cos(ro), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _project(pitch, yaw, roll):
    r = _euler_to_matrix(pitch, yaw, roll)
    rvec, _ = cv2.Rodrigues(r)
    tvec = np.array([[0.0], [0.0], [1000.0]])
    cam = np.array([[W, 0, W / 2], [0, W, H / 2], [0, 0, 1]], dtype=np.float64)
    pts2d, _ = cv2.projectPoints(MODEL_POINTS_3D, rvec, tvec, cam, np.zeros((4, 1)))
    pts2d = pts2d.reshape(-1, 2)
    landmarks = np.zeros((478, 2), dtype=np.float64)
    for idx, xy in zip(POSE_LANDMARK_INDICES, pts2d, strict=True):
        landmarks[idx] = xy
    return landmarks


@pytest.mark.parametrize(
    "pitch,yaw,roll",
    [(0, 0, 0), (10, 0, 0), (0, 15, 0), (0, 0, 8), (12, -18, 5)],
)
def test_recovers_known_rotation(pitch, yaw, roll):
    landmarks = _project(pitch, yaw, roll)
    pose = HeadPoseEstimator().estimate(landmarks, W, H)
    assert pose is not None
    assert pose.pitch == pytest.approx(pitch, abs=1.0)
    assert pose.yaw == pytest.approx(yaw, abs=1.0)
    assert pose.roll == pytest.approx(roll, abs=1.0)
