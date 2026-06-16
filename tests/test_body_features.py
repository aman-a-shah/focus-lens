import numpy as np

from focuslens.features.body import body_features
from focuslens.pose import (
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    PoseResult,
)


def _pose(points: dict[int, tuple[float, float]], visibility: float = 1.0) -> PoseResult:
    """Build a 33-landmark PoseResult; only the given indices are placed (others off-screen)."""
    arr = np.zeros((33, 4), dtype=np.float32)
    arr[:, 3] = 0.0  # default: not visible
    for idx, (x, y) in points.items():
        arr[idx] = (x, y, 0.0, visibility)
    return PoseResult(landmarks=arr)


def _upright(overrides: dict[int, tuple[float, float]] | None = None):
    """An upright seated pose: shoulders apart, nose above them, hands resting low."""
    pts = {
        NOSE: (0.5, 0.25),
        LEFT_SHOULDER: (0.6, 0.5),
        RIGHT_SHOULDER: (0.4, 0.5),
        LEFT_WRIST: (0.65, 0.9),
        RIGHT_WRIST: (0.35, 0.9),
    }
    if overrides:
        pts.update(overrides)
    return _pose(pts)


def test_no_pose_is_absent():
    b = body_features(None)
    assert b.body_present is False
    assert np.isnan(b.head_drop)


def test_missing_shoulders_is_absent():
    # Only a nose visible -> no frame of reference -> not present.
    b = body_features(_pose({NOSE: (0.5, 0.3)}))
    assert b.body_present is False


def test_upright_pose_has_negative_head_drop_and_low_hands():
    b = body_features(_upright())
    assert b.body_present is True
    assert b.head_drop < -0.5  # nose well above the shoulder line
    assert b.hands_near_face < 0.3  # hands resting low, not at the face


def test_hand_at_face_scores_high():
    # A wrist right next to the nose = phone-to-face / hand-on-chin.
    b = body_features(_upright({LEFT_WRIST: (0.5, 0.27)}))
    assert b.hands_near_face > 0.7


def test_head_dropped_raises_head_drop():
    upright = body_features(_upright())
    # Drop the head toward the shoulders (looking down at a lap/phone).
    dropped = body_features(_upright({NOSE: (0.5, 0.46)}))
    assert dropped.head_drop > upright.head_drop


def test_low_visibility_wrist_is_ignored():
    pose = _upright()
    pose.landmarks[LEFT_WRIST] = (0.5, 0.27, 0.0, 0.1)  # at the face but barely visible
    pose.landmarks[RIGHT_WRIST] = (0.35, 0.9, 0.0, 0.1)
    b = body_features(pose, min_visibility=0.5)
    assert b.hands_near_face == 0.0  # both wrists below the visibility floor


def test_torso_lean_needs_hips():
    no_hips = body_features(_upright())
    assert np.isnan(no_hips.torso_lean)
    with_hips = body_features(_upright({LEFT_HIP: (0.58, 0.95), RIGHT_HIP: (0.42, 0.95)}))
    assert not np.isnan(with_hips.torso_lean)
