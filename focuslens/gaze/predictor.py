"""Gaze predictors — the swappable box the runtime plugs into (roadmap Phase 5).

The Phase-2 extractor hard-coded the naive iris-offset proxy. Phase 5 turns gaze into a strategy:

- ``NaiveGazePredictor`` — the geometric proxy, unchanged (default; no checkpoint needed).
- ``CalibratedGazePredictor`` — wraps a fine-tuned ``MLPGazeNet`` and maps the same per-frame
  geometric features to *calibrated* on-screen gaze.

Both take the already-computed iris offset + head pose (so the extractor never recomputes) and
return a ``GazeProxy`` in the shared normalized ``[-1, 1]`` convention. The extractor holds one
predictor and calls ``predict`` per frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch

from ..features.gaze import GazeProxy
from ..features.head_pose import HeadPose
from .model import build_model

_POSE_SCALE = 90.0  # head pose is in degrees in [-90, 90]; scale to ~[-1, 1] for the MLP


def features_from(offset: GazeProxy, pose: HeadPose | None) -> list[float]:
    """Build the 5-d MLP input ``[iris_dx, iris_dy, yaw, pitch, roll]`` (pose scaled to ~[-1, 1]).

    Shared by calibration-sample collection and inference so the two never disagree on layout.
    """
    if pose is None:
        yaw = pitch = roll = 0.0
    else:
        yaw, pitch, roll = pose.yaw / _POSE_SCALE, pose.pitch / _POSE_SCALE, pose.roll / _POSE_SCALE
    return [offset.x, offset.y, yaw, pitch, roll]


class GazePredictor(Protocol):
    def predict(self, offset: GazeProxy | None, pose: HeadPose | None) -> GazeProxy | None: ...


class NaiveGazePredictor:
    """Identity passthrough of the geometric proxy (the Phase-2 default behaviour)."""

    def predict(self, offset: GazeProxy | None, pose: HeadPose | None) -> GazeProxy | None:
        return offset


class CalibratedGazePredictor:
    """Maps geometric features through a fine-tuned per-user gaze head."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path) -> CalibratedGazePredictor:
        state = torch.load(checkpoint, map_location="cpu")
        model = build_model(state.get("arch", "mlp"))
        model.load_state_dict(state["state_dict"])
        return cls(model)

    @torch.no_grad()
    def predict(self, offset: GazeProxy | None, pose: HeadPose | None) -> GazeProxy | None:
        if offset is None:
            return None
        x = torch.tensor([features_from(offset, pose)], dtype=torch.float32)
        out = self.model(x)[0]
        return GazeProxy(x=float(out[0]), y=float(out[1]))
