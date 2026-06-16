"""Train + calibrate the intervention timing model (roadmap Phase 9).

Fits the Cox model (synthetic by default), reports the validation **C-index**, and calibrates the
hazard threshold to fire ~``target_lead_s`` before drift. The checkpoint bundles β + the
normalizer + the tuned threshold for the runtime ``InterventionController``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..logging import get_logger
from ..paths import CHECKPOINTS_DIR as _CHECKPOINTS
from .cox import CoxPH, concordance_index
from .synthetic import make_cox_dataset, make_sessions
from .timing import calibrate_threshold, lead_times

log = get_logger(__name__)


@dataclass
class TimingResult:
    c_index: float
    threshold: float
    median_lead_s: float
    checkpoint: str

    def summary(self) -> str:
        return (
            f"Intervention timer: C-index {self.c_index:.3f} | gate risk>{self.threshold:.2f} | "
            f"median lead {self.median_lead_s:.1f}s before drift | checkpoint {self.checkpoint}"
        )


def train_timing(
    *,
    n: int = 800,
    epochs: int = 300,
    target_lead_s: float = 20.0,
    seed: int = 0,
    checkpoint_dir: Path | None = None,
) -> TimingResult:
    # Fit + score the Cox model on an i.i.d. survival split (C-index).
    x, dur, ev, _ = make_cox_dataset(n=n, seed=seed)
    split = int(0.8 * len(x))
    model = CoxPH().fit(x[:split], dur[:split], ev[:split], epochs=epochs)
    risk_val = model.predict_risk(x[split:])
    c_index = concordance_index(risk_val, dur[split:], ev[split:])

    # Calibrate the firing threshold + measure lead time on held-out ramped sessions.
    cal_sessions = make_sessions(n=40, seed=seed + 100)
    eval_sessions = make_sessions(n=40, seed=seed + 200)
    threshold = calibrate_threshold(model, cal_sessions, target_lead_s=target_lead_s)
    leads = lead_times(model, eval_sessions, threshold)
    median_lead = float(np.median(leads)) if leads else 0.0

    ckpt_dir = checkpoint_dir or _CHECKPOINTS
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "intervention_cox.pt"
    ckpt_path.write_text(
        json.dumps(
            {"cox": model.state_dict(), "threshold": threshold, "target_lead_s": target_lead_s},
            indent=2,
        )
    )
    log.info("C-index %.3f | threshold %.3f | median lead %.1fs", c_index, threshold, median_lead)
    return TimingResult(
        c_index=c_index,
        threshold=threshold,
        median_lead_s=median_lead,
        checkpoint=str(ckpt_path),
    )


def load_timing(checkpoint: str | Path) -> tuple[CoxPH, float]:
    """Load (model, threshold) from a saved intervention checkpoint."""
    data = json.loads(Path(checkpoint).read_text())
    return CoxPH.from_state_dict(data["cox"]), float(data["threshold"])
