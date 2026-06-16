"""Continual learning for PersonalFocusNet (roadmap Phase 8).

A model that fine-tunes on each new session would *catastrophically forget* earlier ones. Two
standard defences, combined here:

- **EWC** (Elastic Weight Consolidation) — after each session, estimate the diagonal Fisher
  information (which weights mattered) and anchor them with a quadratic penalty on future
  updates. ``EWC.consolidate`` stores one (snapshot, Fisher) per task; ``EWC.penalty`` sums the
  anchors.
- **Experience replay** — a fixed-capacity reservoir-sampled buffer of past window-sequences,
  mixed into each session's batches (``replay_ratio``) so old distributions keep being seen.

``ContinualTrainer`` ties them together for the post-session background fine-tune job; EWC and
the buffer are serializable so Fisher/replay survive across process restarts (checkpoint
rotation). Operates on raw ``[N, T, F]`` tensors so it's decoupled from the dataset classes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..logging import get_logger
from .dataset import FeatureNormalizer
from .model import PersonalFocusNet

log = get_logger(__name__)


class EWC:
    """Diagonal-Fisher Elastic Weight Consolidation across a growing list of tasks."""

    def __init__(self, lam: float = 100.0) -> None:
        self.lam = lam
        # One (param snapshot, Fisher diagonal) pair per consolidated task.
        self.tasks: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]] = []

    @torch.enable_grad()
    def consolidate(
        self, model: PersonalFocusNet, x: torch.Tensor, y: torch.Tensor, max_samples: int = 512
    ) -> None:
        """Estimate the diagonal empirical Fisher on (x, y) and snapshot the current weights.

        Fisher is the mean of *per-sample* squared log-likelihood gradients — computed one sample
        at a time, since a batch's mean gradient would under-estimate Σ_k g_k². Capped at
        ``max_samples`` so a long session doesn't make consolidation expensive.
        """
        model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        n_used = min(x.shape[0], max_samples)
        for i in range(n_used):
            model.zero_grad()
            logits, _ = model(x[i : i + 1])
            loss = nn.functional.nll_loss(torch.log_softmax(logits, dim=-1), y[i : i + 1])
            loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
        fisher = {n: f / max(1, n_used) for n, f in fisher.items()}
        snapshot = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.tasks.append((snapshot, fisher))
        model.zero_grad()

    def penalty(self, model: PersonalFocusNet) -> torch.Tensor:
        """Quadratic anchor: λ · Σ_tasks Σ_i F_i (θ_i − θ*_i)²."""
        if not self.tasks:
            return torch.zeros((), dtype=torch.float32)
        params = dict(model.named_parameters())
        total = torch.zeros((), dtype=torch.float32)
        for snapshot, fisher in self.tasks:
            for n, f in fisher.items():
                total = total + (f * (params[n] - snapshot[n]) ** 2).sum()
        return self.lam * total

    def state_dict(self) -> list[dict]:
        return [
            {
                "snapshot": {n: t.cpu() for n, t in snap.items()},
                "fisher": {n: t.cpu() for n, t in fish.items()},
            }
            for snap, fish in self.tasks
        ]

    def load_state_dict(self, state: list[dict]) -> None:
        self.tasks = [(d["snapshot"], d["fisher"]) for d in state]


class ReplayBuffer:
    """Reservoir-sampled buffer of past (sequence, label) pairs (Phase 8: ~500 samples)."""

    def __init__(self, capacity: int = 500, seed: int = 0) -> None:
        self.capacity = capacity
        self._x: list[torch.Tensor] = []
        self._y: list[int] = []
        self._seen = 0
        self._rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self._x)

    def add_many(self, x: torch.Tensor, y: torch.Tensor) -> None:
        for i in range(x.shape[0]):
            self._seen += 1
            if len(self._x) < self.capacity:
                self._x.append(x[i].clone())
                self._y.append(int(y[i]))
            else:
                j = self._rng.randint(0, self._seen)  # reservoir: keep with prob capacity/seen
                if j < self.capacity:
                    self._x[j] = x[i].clone()
                    self._y[j] = int(y[i])

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self._x:
            return None
        n = min(n, len(self._x))
        idx = self._rng.choice(len(self._x), size=n, replace=False)
        xs = torch.stack([self._x[i] for i in idx])
        ys = torch.tensor([self._y[i] for i in idx], dtype=torch.long)
        return xs, ys

    def state_dict(self) -> dict:
        return {
            "x": torch.stack(self._x) if self._x else torch.empty(0),
            "y": torch.tensor(self._y, dtype=torch.long),
            "seen": self._seen,
            "capacity": self.capacity,
        }

    def load_state_dict(self, state: dict) -> None:
        self.capacity = int(state["capacity"])
        self._seen = int(state["seen"])
        x = state["x"]
        self._x = [x[i].clone() for i in range(x.shape[0])] if x.numel() else []
        self._y = [int(v) for v in state["y"].tolist()]


class ContinualTrainer:
    """Fine-tune PersonalFocusNet on a new session with EWC anchoring + replay mix-in."""

    def __init__(
        self,
        model: PersonalFocusNet,
        normalizer: FeatureNormalizer | None = None,
        ewc: EWC | None = None,
        replay: ReplayBuffer | None = None,
        replay_ratio: float = 0.3,
        lr: float = 2e-3,
    ) -> None:
        self.model = model
        self.normalizer = normalizer
        self.ewc = ewc
        self.replay = replay
        self.replay_ratio = replay_ratio
        self.lr = lr

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalizer.apply(x) if self.normalizer is not None else x

    def fit_task(
        self, x: torch.Tensor, y: torch.Tensor, epochs: int = 8, batch_size: int = 64, seed: int = 0
    ) -> None:
        """Train on a new task's raw tensors, then consolidate EWC + grow the replay buffer."""
        g = torch.Generator().manual_seed(seed)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        n = x.shape[0]
        n_replay = int(self.replay_ratio * batch_size)
        for _ in range(epochs):
            self.model.train()
            perm = torch.randperm(n, generator=g)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                xb, yb = self._norm(x[idx]), y[idx]
                if self.replay is not None and n_replay > 0:
                    drawn = self.replay.sample(n_replay)
                    if drawn is not None:
                        rx, ry = drawn
                        xb = torch.cat([xb, self._norm(rx)], dim=0)
                        yb = torch.cat([yb, ry], dim=0)
                optimizer.zero_grad()
                logits, _ = self.model(xb)
                loss = nn.functional.cross_entropy(logits, yb)
                if self.ewc is not None:
                    loss = loss + self.ewc.penalty(self.model)
                loss.backward()
                optimizer.step()
        if self.ewc is not None:
            self.ewc.consolidate(self.model, self._norm(x), y)
        if self.replay is not None:
            self.replay.add_many(x, y)


@torch.no_grad()
def evaluate_accuracy(
    model: PersonalFocusNet,
    x: torch.Tensor,
    y: torch.Tensor,
    normalizer: FeatureNormalizer | None = None,
) -> float:
    model.eval()
    xb = normalizer.apply(x) if normalizer is not None else x
    preds = model(xb)[0].argmax(dim=-1)
    return float((preds == y).float().mean())


def save_session_checkpoint(
    model: PersonalFocusNet,
    session_idx: int,
    checkpoint_dir: Path,
    ewc: EWC | None = None,
    replay: ReplayBuffer | None = None,
    normalizer: FeatureNormalizer | None = None,
    keep: int = 5,
) -> Path:
    """Save a per-session checkpoint (model + EWC/replay state); rotate to the newest ``keep``."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"focusnet_session_{session_idx:04d}.pt"
    ckpt: dict = {"state_dict": model.state_dict(), "session_idx": session_idx}
    if ewc is not None:
        ckpt["ewc"] = ewc.state_dict()
    if replay is not None:
        ckpt["replay"] = replay.state_dict()
    if normalizer is not None:
        ckpt["normalizer"] = normalizer.state_dict()
    torch.save(ckpt, path)
    rotate_checkpoints(checkpoint_dir, keep=keep)
    return path


def rotate_checkpoints(checkpoint_dir: Path, keep: int = 5) -> list[Path]:
    """Delete all but the newest ``keep`` session checkpoints. Returns the removed paths."""
    ckpts = sorted(checkpoint_dir.glob("focusnet_session_*.pt"))
    removed = ckpts[:-keep] if keep > 0 else ckpts
    for p in removed:
        p.unlink()
    return removed


def continual_update_from_store(
    store,
    base_checkpoint: str | Path,
    checkpoint_dir: Path,
    *,
    ewc_lambda: float = 100.0,
    replay_capacity: int = 500,
    epochs: int = 8,
    seq_len: int = 30,
) -> tuple[Path, int]:
    """Post-session background job: fine-tune a checkpoint on newly logged labelled data.

    Loads the model plus any persisted EWC/replay state from ``base_checkpoint``, runs one
    continual update over the store's labelled window-sequences, and saves a rotated per-session
    checkpoint that carries the updated Fisher + replay buffer forward. Returns (path, n_samples).
    """
    from .dataset import WindowSequenceDataset

    ds = WindowSequenceDataset.from_store(store, seq_len=seq_len)
    ckpt = torch.load(base_checkpoint, map_location="cpu")
    model = PersonalFocusNet()
    model.load_state_dict(ckpt["state_dict"])
    normalizer = (
        FeatureNormalizer.from_state_dict(ckpt["normalizer"])
        if "normalizer" in ckpt
        else FeatureNormalizer.fit(ds.raw)
    )
    ewc = EWC(lam=ewc_lambda)
    replay = ReplayBuffer(capacity=replay_capacity)
    if "ewc" in ckpt:
        ewc.load_state_dict(ckpt["ewc"])
    if "replay" in ckpt:
        replay.load_state_dict(ckpt["replay"])

    trainer = ContinualTrainer(model, normalizer, ewc=ewc, replay=replay)
    trainer.fit_task(ds.raw, ds.labels, epochs=epochs)

    session_idx = int(ckpt.get("session_idx", -1)) + 1
    path = save_session_checkpoint(
        model, session_idx, checkpoint_dir, ewc=ewc, replay=replay, normalizer=normalizer
    )
    return path, len(ds)
