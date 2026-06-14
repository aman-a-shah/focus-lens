"""Gaze evaluation metrics.

Gaze is represented as (yaw, pitch) angles in radians. The headline metric is **mean angular
error** — the angle between the predicted and ground-truth 3D gaze directions (plan.md §6) —
which is convention-independent as long as both sides use the same angles→vector mapping.
"""

from __future__ import annotations

import torch


def angles_to_unit_vector(yaw: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
    """Map (yaw, pitch) radians to a unit 3D direction. Input [..], output [.., 3]."""
    x = torch.cos(pitch) * torch.sin(yaw)
    y = torch.sin(pitch)
    z = torch.cos(pitch) * torch.cos(yaw)
    return torch.stack([x, y, z], dim=-1)


def angular_error_deg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample angular error in degrees. ``pred``/``target`` are [N, 2] (yaw, pitch)."""
    pv = angles_to_unit_vector(pred[..., 0], pred[..., 1])
    tv = angles_to_unit_vector(target[..., 0], target[..., 1])
    cos = (pv * tv).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos))


def mean_angular_error_deg(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean angular error (degrees) over a batch."""
    return float(angular_error_deg(pred, target).mean().item())
