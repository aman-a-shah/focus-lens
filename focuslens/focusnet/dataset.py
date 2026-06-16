"""Window-sequence datasets for PersonalFocusNet (roadmap Phase 7).

``WindowSequenceDataset`` turns a labelled SQLite store (Phase 6 output) into ``[T, 8]`` →
class-index examples: for each labelled window it stacks the trailing ``seq_len`` windows of the
same session (left-padded with zeros at the session start). ``SyntheticFocusDataset`` generates
the same shape from class-conditional feature signatures, so training/eval runs end-to-end
without any logged sessions.

Features are standardized per-channel; the normalizer travels with the checkpoint so the runtime
adapter applies the identical transform.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from ..session import SessionStore
from ..states import NUM_STATES, DistractionState, state_index
from ..window import NUM_FEATURES

# Per-class mean feature vector (canonical order: gaze_x, gaze_y, gaze_vel, gaze_accel,
# blink_rate, blink_dur, head_pose_change_rate, ear) — the signatures the simulator also uses.
_CLASS_MEANS: dict[DistractionState, list[float]] = {
    DistractionState.FOCUSED: [0.0, 0.0, 0.1, 0.0, 12.0, 0.15, 3.0, 0.27],
    DistractionState.DRIFTING: [0.6, 0.0, 1.8, 0.2, 14.0, 0.15, 40.0, 0.26],
    DistractionState.DISTRACTED: [1.1, 0.0, 0.3, 0.0, 12.0, 0.15, 8.0, 0.26],
    DistractionState.FATIGUED: [0.0, 0.0, 0.1, 0.0, 30.0, 0.30, 3.0, 0.16],
}
_CLASS_STD = np.array([0.08, 0.08, 0.3, 0.1, 1.5, 0.02, 6.0, 0.02], dtype=np.float32)


@dataclass
class FeatureNormalizer:
    mean: torch.Tensor  # [F]
    std: torch.Tensor  # [F]

    @classmethod
    def fit(cls, sequences: torch.Tensor) -> FeatureNormalizer:
        flat = sequences.reshape(-1, sequences.shape[-1])
        mean = flat.mean(dim=0)
        std = flat.std(dim=0).clamp_min(1e-6)
        return cls(mean=mean, std=std)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def state_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state_dict(cls, d: dict[str, list[float]]) -> FeatureNormalizer:
        return cls(mean=torch.tensor(d["mean"]), std=torch.tensor(d["std"]))


def _stack_sequence(vectors: list[np.ndarray], end: int, seq_len: int) -> np.ndarray:
    """Trailing ``seq_len`` window-vectors ending at index ``end`` (inclusive), left zero-padded."""
    start = max(0, end - seq_len + 1)
    chunk = vectors[start : end + 1]
    seq = np.zeros((seq_len, NUM_FEATURES), dtype=np.float32)
    seq[seq_len - len(chunk) :] = np.stack(chunk)
    return seq


class WindowSequenceDataset(Dataset):
    def __init__(
        self,
        sequences: torch.Tensor,
        labels: torch.Tensor,
        normalizer: FeatureNormalizer | None = None,
    ) -> None:
        self.raw = sequences.float()
        self.labels = labels.long()
        self.normalizer = normalizer or FeatureNormalizer.fit(self.raw)
        self.sequences = self.normalizer.apply(self.raw)

    @classmethod
    def from_store(
        cls, store: SessionStore, seq_len: int = 30, normalizer: FeatureNormalizer | None = None
    ) -> WindowSequenceDataset:
        seqs: list[np.ndarray] = []
        ys: list[int] = []
        for sid in store.session_ids():
            windows = store.get_windows(sid)
            label_map = store.get_session_labels(sid)
            vectors = [w.features.to_vector() for w in windows]
            for i, w in enumerate(windows):
                label = label_map.get(w.window_id)
                if label is None:
                    continue
                seqs.append(_stack_sequence(vectors, i, seq_len))
                ys.append(state_index(DistractionState(label)))
        if not seqs:
            raise ValueError("no labelled windows found — run `focuslens label` first")
        return cls(torch.from_numpy(np.stack(seqs)), torch.tensor(ys), normalizer)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"seq": self.sequences[idx], "label": self.labels[idx]}


class SyntheticFocusDataset(WindowSequenceDataset):
    """Class-conditional synthetic sequences — a learnable stand-in for logged sessions."""

    def __init__(self, n_per_class: int = 200, seq_len: int = 30, seed: int = 0) -> None:
        rng = np.random.RandomState(seed)
        seqs: list[np.ndarray] = []
        ys: list[int] = []
        for state, mean in _CLASS_MEANS.items():
            mu = np.array(mean, dtype=np.float32)
            for _ in range(n_per_class):
                noise = rng.normal(0, 1, size=(seq_len, NUM_FEATURES)).astype(np.float32)
                seqs.append(mu[None, :] + noise * _CLASS_STD[None, :])
                ys.append(state_index(state))
        order = rng.permutation(len(ys))
        sequences = torch.from_numpy(np.stack(seqs)[order])
        labels = torch.tensor(np.array(ys)[order])
        super().__init__(sequences, labels)
        assert len(_CLASS_MEANS) == NUM_STATES  # signatures cover every class
