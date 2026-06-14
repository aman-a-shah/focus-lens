"""MediaPipe Face Mesh landmark indices used by the feature extractors.

Index sets are the widely-used canonical ones for the 468/478-point Face Mesh topology.
Eye corner / lid points drive EAR and the naive gaze proxy; a 6-point subset drives the
solvePnP head-pose estimate.
"""

from __future__ import annotations

import numpy as np

# 6-point EAR sets, ordered (p1, p2, p3, p4, p5, p6):
# p1,p4 = horizontal corners; (p2,p6) and (p3,p5) = the two vertical lid pairs.
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]
LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]

# Eye corners (outer, inner) and lid midpoints (top, bottom) for the gaze proxy.
RIGHT_EYE_CORNERS = (33, 133)
LEFT_EYE_CORNERS = (362, 263)
RIGHT_EYE_TOP_BOTTOM = (159, 145)
LEFT_EYE_TOP_BOTTOM = (386, 374)

# Iris landmark groups (present only in the 478-point refined output).
RIGHT_IRIS = [468, 469, 470, 471, 472]
LEFT_IRIS = [473, 474, 475, 476, 477]

# Head-pose correspondences: landmark index per reference point, kept in the same order as
# MODEL_POINTS_3D below.
POSE_LANDMARK_INDICES = [
    1,  # nose tip
    152,  # chin
    33,  # right eye outer corner
    263,  # left eye outer corner
    61,  # right mouth corner
    291,  # left mouth corner
]

# Canonical 3D face model (millimetres, nose tip at origin) — the classic dlib/MediaPipe
# head-pose reference geometry. solvePnP against a y-down image leaves a ~180° offset on
# pitch; HeadPoseEstimator wraps Euler angles into [-90, 90] to undo it (a visible face never
# exceeds ~90° on any axis).
MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),  # nose tip
        (0.0, -330.0, -65.0),  # chin
        (-225.0, 170.0, -135.0),  # right eye outer corner
        (225.0, 170.0, -135.0),  # left eye outer corner
        (-150.0, -150.0, -125.0),  # right mouth corner
        (150.0, -150.0, -125.0),  # left mouth corner
    ],
    dtype=np.float64,
)
