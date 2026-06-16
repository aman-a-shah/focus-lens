"""Personal gaze calibration (roadmap Phase 5).

Pretraining (Phase 4) gives a *population* gaze head. Each user's eye geometry — interocular
distance, iris size, camera offset, how they hold their head — shifts the iris-offset → on-screen
mapping, so the population head reads a frontal face as off-centre (the DRIFTING bias noted in
Phase 3). Calibration fixes this: show targets at known on-screen positions, collect the
geometric features while the user looks at each, and **fine-tune** the head on those labelled
pairs into a per-user checkpoint.

Convention: targets and predictions live in normalized screen coordinates ``[-1, 1]`` —
``(0, 0)`` = looking at screen centre, ``+x`` = right, ``+y`` = down — the same units the
heuristic classifier thresholds (``classifier.py``), so a calibrated head is a drop-in for the
naive proxy. The feature vector is the 5-d MLP input ``[iris_dx, iris_dy, yaw, pitch, roll]``
(pose scaled to ~[-1, 1]); see ``predictor.features_from``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from ..logging import get_logger
from .metrics import mean_angular_error_deg
from .model import MLPGazeNet, build_model
from .train import _CHECKPOINTS, _angular_loss

log = get_logger(__name__)

# Normalized on-screen calibration targets in [-1, 1] (x right, y down).
CALIB_POINTS_5: list[tuple[float, float]] = [
    (0.0, 0.0),  # centre
    (-0.9, -0.9),  # top-left
    (0.9, -0.9),  # top-right
    (-0.9, 0.9),  # bottom-left
    (0.9, 0.9),  # bottom-right
]
# 3×3 grid — the denser routine the plan upgrades to once 5-point works.
CALIB_POINTS_9: list[tuple[float, float]] = [
    (x, y) for y in (-0.9, 0.0, 0.9) for x in (-0.9, 0.0, 0.9)
]


@dataclass
class CalibrationSample:
    """One labelled frame: 5-d geometric features paired with the on-screen target it looked at."""

    features: tuple[float, float, float, float, float]
    target: tuple[float, float]


@dataclass
class CalibrationData:
    """Accumulator for calibration samples; converts to tensors for fine-tuning."""

    samples: list[CalibrationSample] = field(default_factory=list)

    def add(self, features: tuple[float, ...], target: tuple[float, float]) -> None:
        self.samples.append(CalibrationSample(tuple(features), target))  # type: ignore[arg-type]

    def __len__(self) -> int:
        return len(self.samples)

    def to_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor([s.features for s in self.samples], dtype=torch.float32)
        y = torch.tensor([s.target for s in self.samples], dtype=torch.float32)
        return x, y


@dataclass
class CalibrationResult:
    user_id: str
    n_samples: int
    mae_before_deg: float
    mae_after_deg: float
    checkpoint: str

    @property
    def improvement_pct(self) -> float:
        if self.mae_before_deg <= 0.0:
            return 0.0
        return 100.0 * (self.mae_before_deg - self.mae_after_deg) / self.mae_before_deg

    def summary(self) -> str:
        return (
            f"[{self.user_id}] calibrated on {self.n_samples} samples | "
            f"MAE {self.mae_before_deg:.2f}° -> {self.mae_after_deg:.2f}° "
            f"({self.improvement_pct:+.0f}%) | checkpoint {self.checkpoint}"
        )


def load_base_model(checkpoint: str | Path) -> MLPGazeNet:
    """Load a pretrained MLP gaze head from a Phase-4 checkpoint."""
    state = torch.load(checkpoint, map_location="cpu")
    model = build_model(state.get("arch", "mlp"))
    model.load_state_dict(state["state_dict"])
    if not isinstance(model, MLPGazeNet):
        raise ValueError("calibration fine-tunes the geometric MLP head, not the image CNN")
    return model


def fine_tune(
    base_model: MLPGazeNet,
    data: CalibrationData,
    *,
    epochs: int = 60,
    lr: float = 5e-3,
    lam: float = 1.0,
    val_frac: float = 0.2,
    batch_size: int = 64,
    seed: int = 0,
) -> tuple[MLPGazeNet, float, float]:
    """Fine-tune a copy of ``base_model`` on calibration data.

    Returns ``(tuned_model, mae_before_deg, mae_after_deg)`` where the *before* error is the
    population head's error on the held-out calibration split — the uncalibrated baseline.
    """
    if len(data) < 2:
        raise ValueError("need at least 2 calibration samples")
    torch.manual_seed(seed)
    model = copy.deepcopy(base_model)

    x, y = data.to_tensors()
    n_val = max(1, int(len(data) * val_frac))
    perm = torch.randperm(len(data), generator=torch.Generator().manual_seed(seed))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    xv, yv = x[val_idx], y[val_idx]
    train_loader = DataLoader(
        TensorDataset(x[train_idx], y[train_idx]), batch_size=batch_size, shuffle=True
    )

    @torch.no_grad()
    def val_mae() -> float:
        model.eval()
        return mean_angular_error_deg(model(xv), yv)

    mae_before = val_mae()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = _angular_loss(model(xb), yb, lam)
            loss.backward()
            optimizer.step()
    mae_after = val_mae()
    log.info("fine-tune: MAE %.2f° -> %.2f° over %d epochs", mae_before, mae_after, epochs)
    return model, mae_before, mae_after


def calibrate_user(
    data: CalibrationData,
    *,
    user_id: str = "user",
    base_checkpoint: str | Path | None = None,
    checkpoint_dir: Path | None = None,
    **fit_kwargs: object,
) -> CalibrationResult:
    """Fine-tune the base gaze head on a user's calibration data and save a per-user checkpoint."""
    ckpt_dir = checkpoint_dir or _CHECKPOINTS
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Base head is the canonical Phase-4 pretrain; output goes to ckpt_dir (which may differ).
    base_path = base_checkpoint or (_CHECKPOINTS / "gaze_mlp.pt")
    base = load_base_model(base_path)

    model, before, after = fine_tune(base, data, **fit_kwargs)  # type: ignore[arg-type]

    out_path = ckpt_dir / f"gaze_user_{user_id}.pt"
    torch.save({"arch": "mlp", "state_dict": model.state_dict(), "user_id": user_id}, out_path)
    return CalibrationResult(
        user_id=user_id,
        n_samples=len(data),
        mae_before_deg=before,
        mae_after_deg=after,
        checkpoint=str(out_path),
    )
