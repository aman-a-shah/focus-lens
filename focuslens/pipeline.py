"""End-to-end attention pipeline (roadmap Phase 3, the walking skeleton).

Wires the per-frame stream into: window aggregation → state classification → debounced state
commit → notification → SQLite logging. It is **camera-agnostic** — it consumes
``FrameFeatures``, so the live runtime and the offline simulator drive the exact same logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .classifier import RuleClassifier
from .context.activity import ActivityCategory
from .features import FrameFeatures
from .notify import Notifier
from .session import SessionStore
from .states import DistractionState
from .window import SequenceBuffer, WindowAggregator, WindowFeatures


class Classifier(Protocol):
    """Anything that maps a window to a state — the rule classifier or PersonalFocusNet.

    ``activity`` is the foreground-app activity for this window; classifiers that don't use it
    (e.g. PersonalFocusNet) simply ignore it. A classifier may expose ``last_reason`` to supply
    a human-readable explanation for the committed state.
    """

    def classify(
        self, window: WindowFeatures, activity: ActivityCategory | None = None
    ) -> DistractionState: ...


@dataclass(frozen=True)
class PipelineOutput:
    """Emitted once per closed 200ms window."""

    window: WindowFeatures
    raw_state: DistractionState  # classifier's instantaneous call
    state: DistractionState  # debounced/committed state
    transitioned: bool  # True if the committed state changed this window
    notified: bool
    activity: ActivityCategory  # foreground-app activity for this window
    reason: str  # human-readable explanation for the state


class _StateDebouncer:
    """Commit a new state only after it persists for ``hold`` consecutive windows."""

    def __init__(self, hold: int = 2, initial: DistractionState = DistractionState.FOCUSED) -> None:
        self.hold = max(1, hold)
        self._committed = initial
        self._candidate = initial
        self._count = 0

    def update(self, raw: DistractionState) -> tuple[DistractionState, bool]:
        if raw == self._committed:
            self._candidate = raw
            self._count = 0
            return self._committed, False
        if raw == self._candidate:
            self._count += 1
        else:
            self._candidate = raw
            self._count = 1
        if self._count >= self.hold:
            self._committed = raw
            self._count = 0
            return self._committed, True
        return self._committed, False


class AttentionPipeline:
    def __init__(
        self,
        store: SessionStore | None = None,
        session_id: int | None = None,
        notifier: Notifier | None = None,
        classifier: Classifier | None = None,
        window_s: float = 0.2,
        sequence_length: int = 30,
        debounce_windows: int = 2,
        intervention: object | None = None,
    ) -> None:
        self.store = store
        self.session_id = session_id
        self.notifier = notifier or Notifier(enabled=False)
        self.classifier = classifier or RuleClassifier()
        self.aggregator = WindowAggregator(window_s=window_s)
        self.sequence = SequenceBuffer(length=sequence_length)
        self.debouncer = _StateDebouncer(hold=debounce_windows)
        # Optional hazard-timer controller (Phase 9); consulted per window for pre-emptive nudges.
        self.intervention = intervention
        # Most recent foreground-app activity; carried onto the window when it closes.
        self._activity = ActivityCategory.UNKNOWN

    def process_frame(
        self, features: FrameFeatures, activity: ActivityCategory | None = None
    ) -> PipelineOutput | None:
        """Feed one frame; returns a result on frames that close a window, else None.

        ``activity`` is the current foreground-app activity (from app context); the latest
        value seen is used when the window closes.
        """
        if activity is not None:
            self._activity = activity
        window = self.aggregator.add(features)
        if window is None:
            return None
        return self._on_window(window)

    def finish(self) -> PipelineOutput | None:
        """Flush any trailing partial window at session end."""
        window = self.aggregator.flush()
        if window is None:
            return None
        return self._on_window(window)

    def _on_window(self, window: WindowFeatures) -> PipelineOutput:
        self.sequence.append(window)
        activity = self._activity
        raw_state = self.classifier.classify(window, activity)
        reason = getattr(self.classifier, "last_reason", "")
        state, transitioned = self.debouncer.update(raw_state)

        notified = False
        if transitioned and self.notifier is not None:
            notified = self.notifier.on_state(state, window.t_end, reason=reason)

        if self.intervention is not None:
            self.intervention.on_window(window, state)

        if self.store is not None and self.session_id is not None:
            self.store.log_window(self.session_id, window, state, activity)

        return PipelineOutput(
            window=window,
            raw_state=raw_state,
            state=state,
            transitioned=transitioned,
            notified=notified,
            activity=activity,
            reason=reason,
        )
