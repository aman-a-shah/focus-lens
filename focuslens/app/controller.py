"""App controller — the runtime behind the GUI (roadmap Phase 10).

Owns the session: builds the ``AttentionPipeline`` from the current settings, feeds it frames
(respecting pause), exposes live sensitivity tuning, and produces the post-session summary. It is
camera-agnostic — it consumes ``FrameFeatures`` exactly like the live runtime and the simulator —
so the whole control surface is testable headlessly; the Tkinter shell is a thin wrapper over it.
"""

from __future__ import annotations

from ..classifier import RuleClassifier
from ..config import Config
from ..features import FrameFeatures
from ..notify import Notifier
from ..pipeline import AttentionPipeline, PipelineOutput
from ..session import SessionStore
from ..states import DistractionState
from .settings import AppSettings
from .summary import SessionSummary, build_summary


class AppController:
    def __init__(
        self,
        config: Config | None = None,
        settings: AppSettings | None = None,
        store: SessionStore | None = None,
        db_path: str = "focuslens.sqlite",
        notify: bool = True,
    ) -> None:
        self.config = config or Config()
        self.settings = settings or AppSettings()
        self._own_store = store is None
        self.store = store or SessionStore(db_path)
        self.notifier = Notifier(enabled=notify and self.settings.notify)
        self.classifier = RuleClassifier(self.settings.thresholds())
        self.session_id: int | None = None
        self.pipeline: AttentionPipeline | None = None
        self.current_state: DistractionState | None = None
        self.frames_processed = 0

    # ---- session lifecycle -------------------------------------------------------------------

    def start_session(self, t0: float = 0.0) -> int:
        self.session_id = self.store.start_session(t0)
        self.pipeline = AttentionPipeline(
            store=self.store,
            session_id=self.session_id,
            notifier=self.notifier,
            classifier=self.classifier,
        )
        self.frames_processed = 0
        return self.session_id

    def process_frame(self, features: FrameFeatures) -> PipelineOutput | None:
        """Drive one frame through the pipeline; a no-op while paused."""
        if self.settings.paused or self.pipeline is None:
            return None
        self.frames_processed += 1
        out = self.pipeline.process_frame(features)
        if out is not None:
            self.current_state = out.state
        return out

    def end_session(self, t: float = 0.0) -> None:
        if self.pipeline is not None:
            self.pipeline.finish()
        if self.session_id is not None:
            self.store.end_session(self.session_id, t)

    # ---- live controls -----------------------------------------------------------------------

    def toggle_pause(self) -> bool:
        self.settings.paused = not self.settings.paused
        return self.settings.paused

    def set_sensitivity(self, value: float) -> float:
        """Update the sensitivity knob and re-tune the live rule classifier in place."""
        s = self.settings.set_sensitivity(value)
        if isinstance(self.classifier, RuleClassifier):
            self.classifier.t = self.settings.thresholds()
        return s

    # ---- summary -----------------------------------------------------------------------------

    def summary(self, buckets: int = 40) -> SessionSummary | None:
        if self.session_id is None:
            return None
        return build_summary(self.store, self.session_id, buckets=buckets)

    def close(self) -> None:
        if self._own_store:
            self.store.close()
