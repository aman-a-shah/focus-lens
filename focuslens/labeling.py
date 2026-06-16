"""Self-supervised labelling (roadmap Phase 6).

Manufactures per-window training labels from a raw session *without hand-labelling*, by fusing
three weak sources logged during the session:

1. **Retrospective marks** — the user hits a key the moment they *notice* they drifted. But
   distraction starts *before* you notice it, so a mark at time ``T`` propagates a DISTRACTED
   label back over ``[T - onset_lookback, T]`` (plan.md §3.4: infer onset ~10–30s early).
2. **Weak proxy intervals** — idle periods / app switches (``events`` table) directly mark the
   overlapping windows as DISTRACTED.
3. **Heuristic prior** — every other window keeps the rule classifier's logged state as a weak
   label, so the learned model (Phase 7) bootstraps from the skeleton instead of from nothing.

The output is one ``(window_id, label, source)`` row per window, written to ``window_labels``.
Marks/events override the heuristic; the most *specific* source wins (mark onset > weak signal >
heuristic).
"""

from __future__ import annotations

from dataclasses import dataclass

from .context.activity import is_distracting_activity
from .session import LoggedWindow, SessionStore, TimeInterval
from .states import DistractionState


@dataclass(frozen=True)
class LabelConfig:
    # How far back from a "noticed" mark to call the onset DISTRACTED (plan.md: 10–30s).
    onset_lookback_s: float = 20.0
    # Pad weak-signal intervals slightly so a window touching the edge still counts.
    event_pad_s: float = 0.0
    # Which event kinds imply distraction.
    distracting_events: tuple[str, ...] = ("idle", "app_switch")


@dataclass(frozen=True)
class WindowLabel:
    window_id: int
    label: DistractionState
    source: str  # 'mark_onset' | 'weak_<kind>' | 'heuristic'


def _overlaps(w_start: float, w_end: float, a: float, b: float) -> bool:
    """Do [w_start, w_end] and [a, b] overlap?"""
    return w_start <= b and a <= w_end


def propagate_labels(
    windows: list[LoggedWindow],
    marks: list[tuple[float, str]],
    events: list[TimeInterval],
    config: LabelConfig | None = None,
) -> list[WindowLabel]:
    """Fuse marks + weak events + heuristic prior into one label per window.

    Pure function over already-loaded session data, so it unit-tests without a database.
    """
    cfg = config or LabelConfig()
    # Onset intervals derived from "noticed" marks.
    onset_spans = [(t - cfg.onset_lookback_s, t) for t, kind in marks if kind == "noticed_drift"]
    weak_spans = [
        (e.t_start - cfg.event_pad_s, e.t_end + cfg.event_pad_s, e.kind)
        for e in events
        if e.kind in cfg.distracting_events
    ]

    out: list[WindowLabel] = []
    for w in windows:
        ws, we = w.features.t_start, w.features.t_end
        if any(_overlaps(ws, we, a, b) for a, b in onset_spans):
            out.append(WindowLabel(w.window_id, DistractionState.DISTRACTED, "mark_onset"))
            continue
        weak = next((k for a, b, k in weak_spans if _overlaps(ws, we, a, b)), None)
        if weak is not None:
            out.append(WindowLabel(w.window_id, DistractionState.DISTRACTED, f"weak_{weak}"))
            continue
        # The foreground app is a strong weak-signal: a non-work app in front of you means
        # distraction more reliably than the webcam-only heuristic can.
        if is_distracting_activity(w.activity):
            out.append(WindowLabel(w.window_id, DistractionState.DISTRACTED, "weak_activity"))
            continue
        out.append(WindowLabel(w.window_id, w.state, "heuristic"))
    return out


def label_session(store: SessionStore, session_id: int, config: LabelConfig | None = None) -> int:
    """Run propagation for one session and persist the labels. Returns #windows labelled."""
    windows = store.get_windows(session_id)
    if not windows:
        return 0
    labels = propagate_labels(
        windows, store.get_marks(session_id), store.get_events(session_id), config
    )
    store.write_window_labels([(lb.window_id, str(lb.label), lb.source) for lb in labels])
    return len(labels)


def label_all_sessions(store: SessionStore, config: LabelConfig | None = None) -> dict[int, int]:
    """Label every session in the store. Returns {session_id: #windows labelled}."""
    return {sid: label_session(store, sid, config) for sid in store.session_ids()}
