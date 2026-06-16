"""Body-language features from PoseLandmarker output.

Turns the 33 raw body keypoints (``PoseResult``) into a handful of interpretable signals the
classifier fuses with gaze. The point is to catch the cases eye-openness misses: a phone held
up to the face, a head hunched down over a lap, a body leaned back and disengaged.

Everything is normalized by shoulder width so it's roughly invariant to how close you sit to
the camera. Landmarks with low ``visibility`` (occluded wrists/hips) are treated as unseen so
we never invent a "hand at face" from a noisy off-screen wrist guess.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..pose import (
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    PoseResult,
)

_NAN = float("nan")
_EPS = 1e-6


@dataclass(frozen=True)
class BodyFeatures:
    """Instantaneous body-language signals for one frame (NaN when not measurable)."""

    body_present: bool
    # Lateral torso lean: shoulder-centre offset from hip-centre / shoulder width. NaN if hips
    # aren't visible (common at a desk). Small magnitude = upright.
    torso_lean: float
    # How far the head has dropped toward the shoulders (nose-to-shoulder gap / shoulder width).
    # Rises as you hunch down over a phone/lap; near a negative baseline when sitting upright.
    head_drop: float
    # Apparent shoulder width as a fraction of frame width — a proximity proxy. Larger = leaned
    # in / closer; smaller = leaned back / further.
    proximity: float
    # Soft 0..1 indicator that a (visible) wrist is up near the head — phone-to-face, holding a
    # phone up, hand-on-chin. Hands resting on the desk/keyboard score ~0.
    hands_near_face: float


def _visible(pose: PoseResult, index: int, min_visibility: float) -> bool:
    return pose.visibility(index) >= min_visibility


def body_features(pose: PoseResult | None, min_visibility: float = 0.5) -> BodyFeatures:
    """Derive ``BodyFeatures`` from a pose result. Returns an all-absent record for None."""
    if pose is None:
        return BodyFeatures(False, _NAN, _NAN, _NAN, _NAN)

    if not (
        _visible(pose, LEFT_SHOULDER, min_visibility)
        and _visible(pose, RIGHT_SHOULDER, min_visibility)
    ):
        # Without both shoulders we have no reliable frame of reference.
        return BodyFeatures(False, _NAN, _NAN, _NAN, _NAN)

    l_sh = pose.point(LEFT_SHOULDER)
    r_sh = pose.point(RIGHT_SHOULDER)
    shoulder_mid = (l_sh + r_sh) / 2.0
    shoulder_width = float(np.linalg.norm(l_sh - r_sh))
    scale = shoulder_width + _EPS

    nose = pose.point(NOSE)
    # Nose sits above the shoulders (smaller y) when upright -> negative; rises toward 0/positive
    # as the head drops down.
    head_drop = float((nose[1] - shoulder_mid[1]) / scale)

    torso_lean = _NAN
    if _visible(pose, LEFT_HIP, min_visibility) and _visible(pose, RIGHT_HIP, min_visibility):
        hip_mid = (pose.point(LEFT_HIP) + pose.point(RIGHT_HIP)) / 2.0
        torso_lean = float((shoulder_mid[0] - hip_mid[0]) / scale)

    hands_near_face = _hands_near_face(pose, nose, shoulder_width, min_visibility)

    return BodyFeatures(
        body_present=True,
        torso_lean=torso_lean,
        head_drop=head_drop,
        proximity=shoulder_width,
        hands_near_face=hands_near_face,
    )


def _hands_near_face(
    pose: PoseResult, nose: np.ndarray, shoulder_width: float, min_visibility: float
) -> float:
    """Closeness of the nearest visible wrist to the head, as a soft 0..1 score."""
    radius = 1.2 * shoulder_width + _EPS  # "near the head" out to ~1.2 shoulder-widths from nose
    score = 0.0
    for wrist in (LEFT_WRIST, RIGHT_WRIST):
        if not _visible(pose, wrist, min_visibility):
            continue
        dist = float(np.linalg.norm(pose.point(wrist) - nose))
        score = max(score, max(0.0, 1.0 - dist / radius))
    return score
