"""Gaze regression training pipeline (roadmap Phase 4).

Trains MLPGazeNet or SmallGazeCNN on a gaze dataset, evaluating mean angular error (the
plan.md §6 headline metric) on a held-out split each epoch, saving the best checkpoint, and
measuring single-frame CPU inference latency. Metrics stream to the console and a JSON history
file; a ``--wandb`` hook is left for the real MPIIFaceGaze runs.

Loss = SmoothL1 on the angle vector + λ · mean angular error (a differentiable geometric term).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from ..logging import get_logger
from .dataset import SyntheticGazeDataset
from .metrics import angles_to_unit_vector, mean_angular_error_deg
from .model import build_model, model_input_key

log = get_logger(__name__)

_CHECKPOINTS = Path(__file__).resolve().parent.parent.parent / "checkpoints"


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_mae_deg: float


@dataclass
class TrainResult:
    arch: str
    best_val_mae_deg: float
    latency_ms_p50: float
    epochs: int
    checkpoint: str
    history: list[EpochMetrics] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"[{self.arch}] best val MAE = {self.best_val_mae_deg:.2f}° "
            f"over {self.epochs} epochs | inference {self.latency_ms_p50:.2f} ms/frame (CPU) | "
            f"checkpoint {self.checkpoint}"
        )


def _angular_loss(pred: torch.Tensor, target: torch.Tensor, lam: float) -> torch.Tensor:
    """SmoothL1 on the angle vector + a smooth geometric term (1 - cosine similarity).

    The cosine term is used instead of arccos: arccos has unbounded gradient as predictions
    approach the target (cos -> 1), which blows up to NaN. ``1 - cos`` is smooth everywhere.
    """
    reg = nn.functional.smooth_l1_loss(pred, target)
    pv = angles_to_unit_vector(pred[..., 0], pred[..., 1])
    tv = angles_to_unit_vector(target[..., 0], target[..., 1])
    geometric = (1.0 - (pv * tv).sum(dim=-1)).mean()
    return reg + lam * geometric


def _measure_latency(model: nn.Module, sample: torch.Tensor, iters: int = 100) -> float:
    """Median single-frame forward latency in milliseconds (CPU)."""
    model.eval()
    one = sample[:1]
    with torch.no_grad():
        for _ in range(5):  # warmup
            model(one)
        times = []
        for _ in range(iters):
            start = time.perf_counter()
            model(one)
            times.append((time.perf_counter() - start) * 1000.0)
    times.sort()
    return times[len(times) // 2]


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, input_key: str) -> float:
    model.eval()
    preds, targets = [], []
    for batch in loader:
        preds.append(model(batch[input_key]))
        targets.append(batch["gaze"])
    return mean_angular_error_deg(torch.cat(preds), torch.cat(targets))


def train_gaze(
    arch: str = "mlp",
    dataset: Dataset | None = None,
    n_samples: int = 4000,
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 1e-3,
    lam: float = 1.0,
    val_frac: float = 0.2,
    seed: int = 0,
    checkpoint_dir: Path | None = None,
) -> TrainResult:
    torch.manual_seed(seed)
    input_key = model_input_key(arch)
    dataset = dataset or SyntheticGazeDataset(n_samples=n_samples, seed=seed)

    n_val = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    model = build_model(arch)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ckpt_dir = checkpoint_dir or _CHECKPOINTS
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"gaze_{arch}.pt"

    history: list[EpochMetrics] = []
    best_mae = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch[input_key])
            loss = _angular_loss(pred, batch["gaze"], lam)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * batch["gaze"].shape[0]
        train_loss = total / n_train
        val_mae = evaluate(model, val_loader, input_key)
        history.append(EpochMetrics(epoch, train_loss, val_mae))
        log.info("epoch %02d | train_loss %.4f | val MAE %.2f°", epoch, train_loss, val_mae)
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save({"arch": arch, "state_dict": model.state_dict()}, ckpt_path)

    sample = next(iter(val_loader))[input_key]
    latency = _measure_latency(model, sample)

    history_path = ckpt_dir / f"gaze_{arch}_history.json"
    history_path.write_text(json.dumps([asdict(m) for m in history], indent=2))

    return TrainResult(
        arch=arch,
        best_val_mae_deg=best_mae,
        latency_ms_p50=latency,
        epochs=epochs,
        checkpoint=str(ckpt_path),
        history=history,
    )
