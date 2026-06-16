"""Learned-classifier runtime adapter (roadmap Phase 7).

Drop-in for ``RuleClassifier``: exposes ``classify(window) -> DistractionState`` so the
``AttentionPipeline`` runs PersonalFocusNet without any other change. It keeps its own
``SequenceBuffer`` (the model needs the trailing window-sequence, not a single window) and
applies two suppression rules so a fresh model doesn't fire spuriously:

- **Cold start** — "day 1 knows nothing": before ``warmup_windows`` have streamed, never escalate.
- **Uncertainty gate** — when the uncertainty head exceeds the tuned threshold, fall back to
  FOCUSED instead of escalating. High uncertainty ⇒ intervene rarely.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..states import DistractionState, state_from_index
from ..window import NUM_FEATURES, SequenceBuffer, WindowFeatures
from .dataset import FeatureNormalizer
from .model import PersonalFocusNet, predict


class LearnedClassifier:
    def __init__(
        self,
        model: PersonalFocusNet,
        normalizer: FeatureNormalizer | None = None,
        seq_len: int = 30,
        uncertainty_threshold: float = 0.5,
        warmup_windows: int = 10,
    ) -> None:
        self.model = model
        self.model.eval()
        self.normalizer = normalizer
        self.seq_len = seq_len
        self.uncertainty_threshold = uncertainty_threshold
        self.warmup_windows = warmup_windows
        self.buffer = SequenceBuffer(length=seq_len)
        self._seen = 0
        self.last_uncertainty = 1.0

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, **overrides: object) -> LearnedClassifier:
        ckpt = torch.load(checkpoint, map_location="cpu")
        model = PersonalFocusNet()
        model.load_state_dict(ckpt["state_dict"])
        normalizer = (
            FeatureNormalizer.from_state_dict(ckpt["normalizer"]) if "normalizer" in ckpt else None
        )
        params = {
            "normalizer": normalizer,
            "seq_len": int(ckpt.get("seq_len", 30)),
            "uncertainty_threshold": float(ckpt.get("uncertainty_threshold", 0.5)),
        }
        params.update(overrides)
        return cls(model, **params)  # type: ignore[arg-type]

    def _sequence_tensor(self) -> torch.Tensor:
        arr = self.buffer.to_array()  # [<=T, F], oldest first
        seq = np.zeros((self.seq_len, NUM_FEATURES), dtype=np.float32)
        seq[self.seq_len - arr.shape[0] :] = arr
        x = torch.from_numpy(seq).unsqueeze(0)
        return self.normalizer.apply(x) if self.normalizer is not None else x

    def _gate(self, raw: DistractionState, uncertainty: float) -> DistractionState:
        if self._seen < self.warmup_windows:
            return DistractionState.FOCUSED  # cold start — don't intervene yet
        if uncertainty > self.uncertainty_threshold:
            return DistractionState.FOCUSED  # too unsure to escalate
        return raw

    def classify(self, window: WindowFeatures) -> DistractionState:
        self.buffer.append(window)
        self._seen += 1
        idx, _probs, unc = predict(self.model, self._sequence_tensor())
        self.last_uncertainty = float(unc[0])
        raw = state_from_index(int(idx[0]))
        return self._gate(raw, self.last_uncertainty)
