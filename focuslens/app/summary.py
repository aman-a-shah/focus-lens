"""Post-session distraction summary + heatmap (roadmap Phase 10).

Reduces a logged session to a glanceable picture: what fraction of time was spent in each state,
a distraction **heatmap** over the session timeline (each bucket weighted by how distracting its
state was), and how many interventions fired / were marked helpful. Rendered as an ASCII strip so
it works in the terminal; ``save_png`` adds a matplotlib figure when available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..session import SessionStore
from ..states import STATES, DistractionState

# How "distracting" each state is, for the heatmap intensity (0 = focused, 1 = fully off).
_WEIGHT = {
    DistractionState.FOCUSED: 0.0,
    DistractionState.DRIFTING: 0.5,
    DistractionState.FATIGUED: 0.65,
    DistractionState.DISTRACTED: 1.0,
}
_RAMP = " .:-=+*#%@"  # low → high intensity


@dataclass
class SessionSummary:
    session_id: int
    total_windows: int
    duration_s: float
    state_pct: dict[str, float]
    heatmap: list[float]  # per-bucket distraction intensity in [0, 1]
    interventions: int
    helpful: int

    def ascii_heatmap(self) -> str:
        if not self.heatmap:
            return ""
        return "".join(_RAMP[min(len(_RAMP) - 1, int(v * len(_RAMP)))] for v in self.heatmap)

    def report(self) -> str:
        pct = "  ".join(f"{k} {self.state_pct.get(k, 0.0) * 100:.0f}%" for k in _state_order())
        lines = [
            f"Session {self.session_id}: {self.duration_s:.0f}s, {self.total_windows} windows",
            "  " + pct,
            f"  distraction: [{self.ascii_heatmap()}]",
            f"  interventions: {self.interventions} ({self.helpful} marked helpful)",
        ]
        return "\n".join(lines)

    def save_png(self, path: str | Path) -> Path | None:
        """Optional heatmap figure; no-op (returns None) if matplotlib isn't installed."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None
        fig, ax = plt.subplots(figsize=(8, 1.4))
        ax.imshow([self.heatmap], aspect="auto", cmap="inferno", vmin=0.0, vmax=1.0)
        ax.set_yticks([])
        ax.set_xlabel("session timeline →")
        ax.set_title(f"Session {self.session_id} distraction heatmap")
        out = Path(path)
        fig.savefig(out, bbox_inches="tight", dpi=120)
        plt.close(fig)
        return out


def _state_order() -> list[str]:
    return [str(s) for s in STATES]


def build_summary(store: SessionStore, session_id: int, buckets: int = 40) -> SessionSummary:
    """Aggregate a logged session into state percentages + a distraction heatmap."""
    windows = store.get_windows(session_id)
    interventions = store.get_interventions(session_id)
    n_helpful = sum(1 for *_rest, helpful in interventions if helpful == 1)

    if not windows:
        return SessionSummary(session_id, 0, 0.0, {}, [], len(interventions), n_helpful)

    t0 = windows[0].features.t_start
    t1 = windows[-1].features.t_end
    duration = max(t1 - t0, 1e-6)

    # State percentages.
    counts: dict[str, int] = {}
    for w in windows:
        counts[str(w.state)] = counts.get(str(w.state), 0) + 1
    state_pct = {k: v / len(windows) for k, v in counts.items()}

    # Distraction heatmap: average state weight per time bucket.
    sums = [0.0] * buckets
    hits = [0] * buckets
    for w in windows:
        frac = (w.features.t_start - t0) / duration
        b = min(buckets - 1, int(frac * buckets))
        sums[b] += _WEIGHT.get(w.state, 0.0)
        hits[b] += 1
    heatmap = [sums[i] / hits[i] if hits[i] else 0.0 for i in range(buckets)]

    return SessionSummary(
        session_id=session_id,
        total_windows=len(windows),
        duration_s=duration,
        state_pct=state_pct,
        heatmap=heatmap,
        interventions=len(interventions),
        helpful=n_helpful,
    )
