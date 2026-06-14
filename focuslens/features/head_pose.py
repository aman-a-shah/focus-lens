"""Head pose estimation via solvePnP.

Solves the perspective-n-point problem between a canonical 3D face model and the detected
2D landmarks to recover head rotation, returned as (yaw, pitch, roll) in degrees. Separating
head pose from gaze is deliberate (plan.md §3.1) — the classifier should not conflate
"looking away by turning the head" with "eyes off screen".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .landmarks import MODEL_POINTS_3D, POSE_LANDMARK_INDICES


@dataclass(frozen=True)
class HeadPose:
    yaw: float
    pitch: float
    roll: float


def _wrap90(deg: float) -> float:
    """Fold an angle into [-90, 90], undoing the ~180° solvePnP frame offset."""
    deg = (deg + 180.0) % 360.0 - 180.0  # first into (-180, 180]
    if deg > 90.0:
        deg -= 180.0
    elif deg < -90.0:
        deg += 180.0
    return deg


def _rotation_to_euler(r: np.ndarray) -> tuple[float, float, float]:
    """Convert a rotation matrix to (pitch_x, yaw_y, roll_z) Euler angles in degrees."""
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy < 1e-6:  # gimbal lock
        pitch = math.atan2(-r[1, 2], r[1, 1])
        yaw = math.atan2(-r[2, 0], sy)
        roll = 0.0
    else:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])
    return _wrap90(math.degrees(pitch)), _wrap90(math.degrees(yaw)), _wrap90(math.degrees(roll))


class HeadPoseEstimator:
    """Stateless head-pose solver. Call ``estimate`` per frame."""

    def __init__(self) -> None:
        self._model = MODEL_POINTS_3D

    def estimate(self, points_px: np.ndarray, width: int, height: int) -> HeadPose | None:
        """Estimate head pose from pixel-space landmarks; returns None if solvePnP fails."""
        import cv2

        image_points = points_px[POSE_LANDMARK_INDICES].astype(np.float64)
        focal = float(width)
        camera_matrix = np.array(
            [[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1))
        ok, rvec, _ = cv2.solvePnP(
            self._model, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return None
        rmat, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = _rotation_to_euler(rmat)
        return HeadPose(yaw=yaw, pitch=pitch, roll=roll)
