"""Cox proportional-hazards model (roadmap Phase 9).

Models the hazard of distraction as ``h(t | x) = h0(t) · exp(βᵀx)`` — the linear predictor
``βᵀx`` is the per-moment *risk score* the intervention timer thresholds. β is fit by maximizing
the Breslow partial log-likelihood (no baseline hazard needed for ranking), and quality is the
**concordance index** (plan.md §6 headline for this phase): the fraction of comparable
event pairs whose risk ordering matches their time ordering.

Self-contained (NumPy + a torch optimizer); no ``lifelines`` dependency.
"""

from __future__ import annotations

import numpy as np
import torch


def _to_tensor(a: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a, dtype=np.float32))


class CoxPH:
    """Linear Cox proportional-hazards model fit by Breslow partial likelihood."""

    def __init__(self, l2: float = 1e-3) -> None:
        self.l2 = l2
        self.beta: np.ndarray | None = None
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None

    def fit(
        self,
        x: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        epochs: int = 300,
        lr: float = 0.05,
    ) -> CoxPH:
        """Fit β on ``x`` [N, D], ``durations`` [N], ``events`` [N] (1=event, 0=censored)."""
        x = np.asarray(x, dtype=np.float32)
        self.feature_mean = x.mean(axis=0)
        self.feature_std = x.std(axis=0)
        self.feature_std[self.feature_std < 1e-6] = 1.0
        xn = (x - self.feature_mean) / self.feature_std

        # Sort by descending duration so prefix [0:i] is exactly sample i's risk set
        # (all j with t_j >= t_i); cumulative log-sum-exp gives the Breslow denominator.
        order = np.argsort(-durations)
        xt = _to_tensor(xn[order])
        et = _to_tensor((np.asarray(events) != 0).astype(np.float32)[order])

        beta = torch.zeros(xt.shape[1], requires_grad=True)
        optimizer = torch.optim.Adam([beta], lr=lr)
        for _ in range(epochs):
            optimizer.zero_grad()
            risk = xt @ beta  # [N], sorted by descending time
            log_risk_set = torch.logcumsumexp(risk, dim=0)  # denominator over risk set
            partial_ll = (et * (risk - log_risk_set)).sum() / et.sum().clamp_min(1.0)
            loss = -partial_ll + self.l2 * (beta**2).sum()
            loss.backward()
            optimizer.step()
        self.beta = beta.detach().numpy()
        return self

    def predict_risk(self, x: np.ndarray) -> np.ndarray | float:
        """Linear predictor βᵀx (higher = sooner distraction). Accepts [D] or [N, D]."""
        if self.beta is None:
            raise RuntimeError("CoxPH is not fit yet")
        x = np.asarray(x, dtype=np.float32)
        single = x.ndim == 1
        rows = x.reshape(1, -1) if single else x
        xn = (rows - self.feature_mean) / self.feature_std
        risk = xn @ self.beta
        return float(risk[0]) if single else risk

    def state_dict(self) -> dict:
        return {
            "beta": self.beta.tolist() if self.beta is not None else None,
            "feature_mean": self.feature_mean.tolist() if self.feature_mean is not None else None,
            "feature_std": self.feature_std.tolist() if self.feature_std is not None else None,
            "l2": self.l2,
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> CoxPH:
        def _arr(key: str) -> np.ndarray | None:
            return np.array(d[key], dtype=np.float32) if d.get(key) is not None else None

        model = cls(l2=d.get("l2", 1e-3))
        model.beta = _arr("beta")
        model.feature_mean = _arr("feature_mean")
        model.feature_std = _arr("feature_std")
        return model


def concordance_index(risk: np.ndarray, durations: np.ndarray, events: np.ndarray) -> float:
    """Harrell's C-index: over all comparable pairs, fraction with risk ordered like (inverse) time.

    A pair (i, j) is comparable when the earlier of the two had an event. Concordant when the
    earlier-time sample carries the higher risk. Ties in risk count as 0.5.
    """
    risk = np.asarray(risk, dtype=np.float64)
    durations = np.asarray(durations, dtype=np.float64)
    events = np.asarray(events) != 0
    num = den = 0.0
    n = len(risk)
    for i in range(n):
        if not events[i]:
            continue
        for j in range(n):
            if durations[j] > durations[i]:  # i had the (earlier) event vs a longer survivor
                den += 1
                if risk[i] > risk[j]:
                    num += 1
                elif risk[i] == risk[j]:
                    num += 0.5
    return num / den if den else 0.0
