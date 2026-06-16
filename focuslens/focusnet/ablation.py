"""Backward-transfer ablation for continual learning (roadmap Phase 8).

Builds a sequence of synthetic "sessions", each a random rotation of the feature space so the
class→region mapping differs per task (forcing the decision boundary to move — the condition
under which naive fine-tuning forgets). Trains a fresh model sequentially under each strategy and
records, after every task, the accuracy on *all* tasks seen so far.

From the resulting accuracy matrix we compute **backward transfer** (mean accuracy drop on
earlier tasks after later learning — the forgetting metric). The headline comparison:

    naive          — fine-tune on each task, no protection (catastrophic forgetting)
    ewc_replay     — EWC anchoring + experience replay (the roadmap defence)
    frozen         — train task 0 only, then freeze (lower bound on plasticity)

``run_ablation`` returns all three plus the learning curves behind plan.md's figure.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..states import NUM_STATES, STATES
from ..window import NUM_FEATURES
from .continual import EWC, ContinualTrainer, ReplayBuffer, evaluate_accuracy
from .dataset import _CLASS_MEANS, FeatureNormalizer
from .model import PersonalFocusNet


@dataclass
class Task:
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor


def _random_rotation(dim: int, rng: np.random.RandomState) -> np.ndarray:
    """A random orthogonal matrix via QR (sign-fixed so it's a proper, reproducible rotation)."""
    q, r = np.linalg.qr(rng.normal(size=(dim, dim)))
    return q * np.sign(np.diag(r))


def _standardized_base() -> np.ndarray:
    """The 4×F class means, z-scored across classes so every channel is on a comparable scale."""
    base = np.array([_CLASS_MEANS[s] for s in STATES], dtype=np.float64)
    mean = base.mean(axis=0, keepdims=True)
    std = base.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (base - mean) / std


def _sample_split(
    means: np.ndarray, count: int, seq_len: int, noise: float, rng: np.random.RandomState
) -> tuple[torch.Tensor, torch.Tensor]:
    """``count`` sequences per class around the (rotated) class means, with Gaussian noise."""
    xs, ys = [], []
    for c in range(NUM_STATES):
        for _ in range(count):
            seq = means[c][None, :] + rng.normal(0, noise, size=(seq_len, NUM_FEATURES))
            xs.append(seq.astype(np.float32))
            ys.append(c)
    order = rng.permutation(len(ys))
    return (
        torch.from_numpy(np.stack(xs)[order]),
        torch.tensor(np.array(ys)[order], dtype=torch.long),
    )


def make_task_sequence(
    n_tasks: int = 5,
    n_per_class: int = 60,
    seq_len: int = 12,
    val_per_class: int = 20,
    noise: float = 0.35,
    seed: int = 0,
) -> list[Task]:
    """A sequence of rotated-feature classification tasks (shared labels, shifting geometry)."""
    rng = np.random.RandomState(seed)
    base = _standardized_base()  # [C, F]
    tasks: list[Task] = []
    for _ in range(n_tasks):
        rot = _random_rotation(NUM_FEATURES, rng)
        means = base @ rot.T  # rotate each class mean into this task's frame
        tx, ty = _sample_split(means, n_per_class, seq_len, noise, rng)
        vx, vy = _sample_split(means, val_per_class, seq_len, noise, rng)
        tasks.append(Task(tx, ty, vx, vy))
    return tasks


@dataclass
class StrategyResult:
    strategy: str
    accuracy_matrix: list[list[float]]  # [after_task_t][eval_task_j]
    backward_transfer: float  # mean drop on earlier tasks (lower = less forgetting)
    final_avg_accuracy: float

    def as_dict(self) -> dict:
        return asdict(self)


def backward_transfer(matrix: list[list[float]]) -> float:
    """Mean over earlier tasks of (acc when just learned) − (acc at the end). Lower is better."""
    n = len(matrix)
    if n < 2:
        return 0.0
    drops = [matrix[j][j] - matrix[n - 1][j] for j in range(n - 1)]
    return float(np.mean(drops))


def run_strategy(
    tasks: list[Task],
    strategy: str,
    *,
    epochs: int = 8,
    ewc_lambda: float = 100.0,
    replay_capacity: int = 500,
    replay_ratio: float = 0.3,
    seed: int = 0,
) -> StrategyResult:
    """Train sequentially under one strategy; return the full accuracy matrix + forgetting."""
    torch.manual_seed(seed)
    model = PersonalFocusNet()
    normalizer = FeatureNormalizer.fit(torch.cat([t.train_x for t in tasks]))

    ewc = EWC(lam=ewc_lambda) if strategy in ("ewc", "ewc_replay") else None
    replay = (
        ReplayBuffer(capacity=replay_capacity, seed=seed)
        if strategy in ("replay", "ewc_replay")
        else None
    )
    trainer = ContinualTrainer(
        model, normalizer=normalizer, ewc=ewc, replay=replay, replay_ratio=replay_ratio
    )

    matrix: list[list[float]] = []
    for t, task in enumerate(tasks):
        if not (strategy == "frozen" and t > 0):  # 'frozen' learns task 0 only
            trainer.fit_task(task.train_x, task.train_y, epochs=epochs, seed=seed + t)
        matrix.append([evaluate_accuracy(model, s.val_x, s.val_y, normalizer) for s in tasks])

    return StrategyResult(
        strategy=strategy,
        accuracy_matrix=matrix,
        backward_transfer=backward_transfer(matrix),
        final_avg_accuracy=float(np.mean([matrix[-1][j] for j in range(len(tasks))])),
    )


@dataclass
class AblationReport:
    n_tasks: int
    results: dict[str, StrategyResult] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"Continual-learning ablation over {self.n_tasks} sessions:"]
        for name, r in self.results.items():
            lines.append(
                f"  {name:11s} | forgetting {r.backward_transfer * 100:5.1f}% | "
                f"final avg acc {r.final_avg_accuracy * 100:5.1f}%"
            )
        return "\n".join(lines)

    def save_learning_curves(self, path: str | Path) -> Path:
        out = Path(path)
        out.write_text(
            json.dumps(
                {
                    "n_tasks": self.n_tasks,
                    "results": {k: v.as_dict() for k, v in self.results.items()},
                },
                indent=2,
            )
        )
        return out


def run_ablation(
    n_tasks: int = 6,
    *,
    strategies: tuple[str, ...] = ("naive", "ewc_replay", "frozen"),
    epochs: int = 8,
    seed: int = 0,
    **task_kwargs: object,
) -> AblationReport:
    """Run the EWC-vs-naive-vs-frozen ablation and return per-strategy forgetting + curves."""
    tasks = make_task_sequence(n_tasks=n_tasks, seed=seed, **task_kwargs)  # type: ignore[arg-type]
    report = AblationReport(n_tasks=n_tasks)
    for strategy in strategies:
        report.results[strategy] = run_strategy(tasks, strategy, epochs=epochs, seed=seed)
    return report


def plot_learning_curves(report: AblationReport, path: str | Path) -> Path | None:
    """Optional matplotlib plot of final-task accuracy per strategy; no-op if matplotlib absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, ax = plt.subplots()
    for name, r in report.results.items():
        ax.plot(range(report.n_tasks), r.accuracy_matrix[-1], marker="o", label=name)
    ax.set_xlabel("task")
    ax.set_ylabel("final accuracy")
    ax.set_title("Backward transfer after all sessions")
    ax.legend()
    out = Path(path)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
