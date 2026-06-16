"""Single-image inference for the public demo (roadmap Phase 11).

``Analyzer`` wraps the live perception stack (Face Mesh → ``FeatureExtractor`` → rule classifier)
behind a one-call ``analyze`` that takes a BGR image and returns a ``DemoResult``: the annotated
overlay plus the per-frame signals and a snapshot attention state. It holds the (expensive)
FaceLandmarker open across calls, so a Gradio app constructs one Analyzer and reuses it.

A single image has no temporal context, so velocities/derivatives are zero and the state is a
*snapshot* read of the rule classifier — enough to make the perception legible in a demo.
"""

from __future__ import annotations

import math
from contextlib import ExitStack
from dataclasses import dataclass

import numpy as np

from ..classifier import RuleClassifier
from ..config import Config
from ..face_mesh import FaceMeshTracker
from ..features import FeatureExtractor, FrameFeatures
from ..states import DistractionState
from ..viz import draw_overlay
from ..window import WindowFeatures


def _z(x: float) -> float:
    return 0.0 if math.isnan(x) else x


def _frame_to_window(ff: FrameFeatures) -> WindowFeatures:
    """A single-frame ``WindowFeatures`` (no temporal derivatives) for a snapshot classification."""
    return WindowFeatures(
        t_start=ff.timestamp,
        t_end=ff.timestamp,
        face_fraction=1.0 if ff.face_present else 0.0,
        gaze_x=_z(ff.gaze_x),
        gaze_y=_z(ff.gaze_y),
        gaze_velocity=0.0,
        gaze_accel=0.0,
        blink_rate=_z(ff.blinks_per_min),
        blink_duration=_z(ff.last_blink_duration_s),
        head_pose_change_rate=0.0,
        ear=_z(ff.ear_mean),
    )


@dataclass
class DemoResult:
    face_present: bool
    state: DistractionState
    features: FrameFeatures
    annotated: np.ndarray  # BGR overlay image

    def caption(self) -> str:
        if not self.face_present:
            return "No face detected."
        f = self.features
        return (
            f"State: {self.state} | gaze=({f.gaze_x:+.2f}, {f.gaze_y:+.2f}) | "
            f"yaw={f.yaw:+.0f}° pitch={f.pitch:+.0f}° | EAR={f.ear_mean:.2f}"
        )


class Analyzer:
    """Holds the perception stack open for repeated single-image inference."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self._stack = ExitStack()
        self.tracker = self._stack.enter_context(FaceMeshTracker(self.config.face_mesh))
        self.extractor = FeatureExtractor()
        self.classifier = RuleClassifier()

    def analyze(self, image: np.ndarray, timestamp: float = 0.0) -> DemoResult:
        """Run perception on one BGR image and return the annotated result."""
        result = self.tracker.process(image, timestamp_ms=int(timestamp * 1000))
        features = self.extractor.extract(result, image.shape, timestamp, image=image)
        state = self.classifier.classify(_frame_to_window(features))
        annotated = draw_overlay(image, result, None, self.config.viz, state=str(state))
        return DemoResult(
            face_present=features.face_present,
            state=state,
            features=features,
            annotated=annotated,
        )

    def close(self) -> None:
        self._stack.close()

    def __enter__(self) -> Analyzer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
