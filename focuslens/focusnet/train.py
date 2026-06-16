"""PersonalFocusNet training (roadmap Phase 7).

Trains on Phase-6 self-supervised labels (or the synthetic stand-in), reporting per-class
precision/recall — the headline being the DISTRACTED class the "Done when" targets
(>70% P @ >60% R). Loss is a **precision-attenuated cross-entropy**: the uncertainty head scales
the logits, so the model can widen the distribution (raise uncertainty) on sequences it would
otherwise get wrong, while a penalty keeps it confident where it can be. The best checkpoint
bundles the feature normalizer and a tuned uncertainty gate for the runtime adapter.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from ..logging import get_logger
from ..states import NUM_STATES, DistractionState, state_index
from .dataset import SyntheticFocusDataset
from .model import PersonalFocusNet, uncertainty_from

log = get_logger(__name__)

_CHECKPOINTS = Path(__file__).resolve().parent.parent.parent / "checkpoints"
_DISTRACTED = state_index(DistractionState.DISTRACTED)
_EPS = 1e-6


@dataclass
class ClassPR:
    precision: float
    recall: float


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_accuracy: float
    distracted_precision: float
    distracted_recall: float


@dataclass
class TrainResult:
    best_val_accuracy: float
    distracted: ClassPR
    macro_f1: float
    epochs: int
    checkpoint: str
    uncertainty_threshold: float
    history: list[EpochMetrics] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"PersonalFocusNet: val acc {self.best_val_accuracy:.2f} | "
            f"DISTRACTED P/R {self.distracted.precision:.2f}/{self.distracted.recall:.2f} | "
            f"macro-F1 {self.macro_f1:.2f} | gate u>{self.uncertainty_threshold:.2f} | "
            f"checkpoint {self.checkpoint}"
        )


def _attenuated_loss(logits: torch.Tensor, log_prec: torch.Tensor, y: torch.Tensor, beta: float):
    """Precision-attenuated CE: high precision sharpens logits; a 1/precision penalty rewards it."""
    precision = nn.functional.softplus(log_prec) + _EPS
    scaled = logits * precision.unsqueeze(-1)
    ce = nn.functional.cross_entropy(scaled, y)
    return ce + beta * (1.0 / precision).mean()


def _per_class_pr(preds: torch.Tensor, targets: torch.Tensor, cls: int) -> ClassPR:
    tp = int(((preds == cls) & (targets == cls)).sum())
    fp = int(((preds == cls) & (targets != cls)).sum())
    fn = int(((preds != cls) & (targets == cls)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return ClassPR(precision, recall)


def _macro_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    f1s = []
    for c in range(NUM_STATES):
        pr = _per_class_pr(preds, targets, c)
        denom = pr.precision + pr.recall
        f1s.append(2 * pr.precision * pr.recall / denom if denom else 0.0)
    return sum(f1s) / len(f1s)


@torch.no_grad()
def _evaluate(model: PersonalFocusNet, loader: DataLoader):
    model.eval()
    preds, targets, uncs = [], [], []
    for batch in loader:
        logits, log_prec = model(batch["seq"])
        preds.append(logits.argmax(dim=-1))
        targets.append(batch["label"])
        uncs.append(uncertainty_from(log_prec))
    return torch.cat(preds), torch.cat(targets), torch.cat(uncs)


def train_focusnet(
    dataset: Dataset | None = None,
    *,
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 2e-3,
    beta: float = 0.05,
    val_frac: float = 0.2,
    seed: int = 0,
    checkpoint_dir: Path | None = None,
) -> TrainResult:
    torch.manual_seed(seed)
    dataset = dataset if dataset is not None else SyntheticFocusDataset(seed=seed)
    normalizer = getattr(dataset, "normalizer", None)

    n_val = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    model = PersonalFocusNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ckpt_dir = checkpoint_dir or _CHECKPOINTS
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "focusnet.pt"

    history: list[EpochMetrics] = []
    best_acc = -1.0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits, log_prec = model(batch["seq"])
            loss = _attenuated_loss(logits, log_prec, batch["label"], beta)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * batch["label"].shape[0]
        preds, targets, _ = _evaluate(model, val_loader)
        acc = float((preds == targets).float().mean())
        dpr = _per_class_pr(preds, targets, _DISTRACTED)
        history.append(EpochMetrics(epoch, total / n_train, acc, dpr.precision, dpr.recall))
        log.info(
            "epoch %02d | loss %.4f | val acc %.3f | DISTRACTED P/R %.2f/%.2f",
            epoch,
            total / n_train,
            acc,
            dpr.precision,
            dpr.recall,
        )
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Tune the uncertainty gate: suppress the most-uncertain quartile seen on validation.
    preds, targets, uncs = _evaluate(model, val_loader)
    threshold = float(torch.quantile(uncs, 0.75)) if uncs.numel() else 0.5
    distracted = _per_class_pr(preds, targets, _DISTRACTED)

    ckpt = {
        "state_dict": model.state_dict(),
        "uncertainty_threshold": threshold,
        "seq_len": dataset.raw.shape[1] if hasattr(dataset, "raw") else 30,
    }
    if normalizer is not None:
        ckpt["normalizer"] = normalizer.state_dict()
    torch.save(ckpt, ckpt_path)
    (ckpt_dir / "focusnet_history.json").write_text(
        json.dumps([asdict(m) for m in history], indent=2)
    )

    return TrainResult(
        best_val_accuracy=best_acc,
        distracted=distracted,
        macro_f1=_macro_f1(preds, targets),
        epochs=epochs,
        checkpoint=str(ckpt_path),
        uncertainty_threshold=threshold,
        history=history,
    )
