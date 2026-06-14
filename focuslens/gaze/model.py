"""Gaze regression architectures (roadmap Phase 4 bake-off).

Two heads, both predicting (yaw, pitch) in radians:

- ``MLPGazeNet``  — operates on cheap geometric features (iris offsets + head pose). Fast,
  no image needed; the natural fit for the on-device pipeline where landmarks are already
  computed.
- ``SmallGazeCNN`` — operates on an eye-crop image. Stands in for the ResNet-18-on-eye-crop
  branch (swap in ``torchvision.models.resnet18`` for the real MPIIFaceGaze run); kept compact
  so it trains on CPU.
"""

from __future__ import annotations

import torch
from torch import nn


class MLPGazeNet(nn.Module):
    def __init__(self, in_dim: int = 5, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallGazeCNN(nn.Module):
    # Gaze depends on *where* the pupil is, so the head must preserve spatial location — pool
    # to a small grid (not 1×1 global average, which would discard position) before the FC.
    _GRID = 4

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((self._GRID, self._GRID)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * self._GRID * self._GRID, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_model(arch: str) -> nn.Module:
    """Construct a model by name: 'mlp' or 'cnn'."""
    arch = arch.lower()
    if arch == "mlp":
        return MLPGazeNet()
    if arch == "cnn":
        return SmallGazeCNN()
    raise ValueError(f"unknown arch {arch!r} (expected 'mlp' or 'cnn')")


def model_input_key(arch: str) -> str:
    """Which dataset field a given architecture consumes."""
    return "features" if arch.lower() == "mlp" else "image"
