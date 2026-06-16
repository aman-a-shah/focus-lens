"""PersonalFocusNet architecture (roadmap Phase 7).

Consumes a ``[B, T, 8]`` window-sequence (T 200ms windows, 8 features each) and emits both a
state distribution and a learned **uncertainty**:

    Conv1d encoder (local temporal patterns)
      -> temporal self-attention (which windows in the sequence matter)
      -> mean-pool over time
      -> classifier head (4-way logits)  +  uncertainty head (log-precision scalar)

The uncertainty head powers the "day-1 knows nothing" gate (``classifier.py``): on ambiguous or
out-of-distribution sequences the model reports low precision -> high uncertainty -> the runtime
declines to escalate. Training (``train.py``) uses a precision-attenuated cross-entropy so the
head learns when the classifier is likely to be wrong.
"""

from __future__ import annotations

import torch
from torch import nn

from ..states import NUM_STATES
from ..window import NUM_FEATURES


class PersonalFocusNet(nn.Module):
    def __init__(
        self,
        in_features: int = NUM_FEATURES,
        hidden: int = 64,
        num_classes: int = NUM_STATES,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_features, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.cls_head = nn.Linear(hidden, num_classes)
        self.unc_head = nn.Linear(hidden, 1)  # raw log-precision

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [B, T, F] -> (logits [B, C], log_precision [B])."""
        h = self.encoder(x.transpose(1, 2)).transpose(1, 2)  # [B, T, hidden]
        attended, _ = self.attn(h, h, h)
        h = self.norm(h + attended)
        pooled = h.mean(dim=1)  # [B, hidden]
        logits = self.cls_head(pooled)
        log_prec = self.unc_head(pooled).squeeze(-1)  # [B]
        return logits, log_prec


def uncertainty_from(log_prec: torch.Tensor) -> torch.Tensor:
    """Map raw log-precision to an uncertainty score in (0, 1) (high = unsure)."""
    return torch.sigmoid(-log_prec)


@torch.no_grad()
def predict(
    model: PersonalFocusNet, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the model in eval mode -> (class_indices [B], probs [B, C], uncertainty [B])."""
    model.eval()
    logits, log_prec = model(x)
    probs = torch.softmax(logits, dim=-1)
    return probs.argmax(dim=-1), probs, uncertainty_from(log_prec)
