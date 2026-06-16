"""Screen/app context — what application you're actually engaged with.

The webcam sees your face and body; it cannot see whether the window in front of you is an
editor or a doom-scroll feed. This package reads the frontmost app/window title locally and
maps it to an activity category, the single most reliable signal for separating *working* from
*doom-scrolling / gaming / socializing*. Nothing here leaves the device.
"""

from .active_app import ActiveAppReader, AppContext
from .activity import (
    DISTRACTING_ACTIVITIES,
    ActivityCategory,
    ActivityClassifier,
    is_distracting_activity,
)

__all__ = [
    "ActiveAppReader",
    "AppContext",
    "ActivityCategory",
    "ActivityClassifier",
    "DISTRACTING_ACTIVITIES",
    "is_distracting_activity",
]
