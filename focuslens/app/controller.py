"""App controller — the runtime behind the GUI (roadmap Phase 10).

Owns the session: builds the ``AttentionPipeline`` from the current settings, feeds it frames
(respecting pause), exposes live sensitivity tuning, and produces the post-session summary. It is
camera-agnostic — it consumes ``FrameFeatures`` exactly like the live runtime and the simulator —
so the whole control surface is testable headlessly; the Tkinter shell is a thin wrapper over it.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass

from ..classifier import RuleClassifier
from ..config import Config
from ..context.activity import ActivityCategory
from ..features import FrameFeatures
from ..notify import Notifier
from ..pipeline import AttentionPipeline, PipelineOutput
from ..session import SessionStore
from ..states import STATES, DistractionState
from .settings import AppSettings
from .summary import SessionSummary, build_summary


def _num(x: float) -> float:
    """NaN-safe scalar -> 0.0 (absent signals read as 'nothing happening')."""
    return 0.0 if x is None or math.isnan(x) else float(x)


@dataclass(frozen=True)
class LiveSnapshot:
    """A thread-safe, glanceable picture of the latest frame for the UI to render."""

    state: DistractionState | None = None
    activity: ActivityCategory = ActivityCategory.UNKNOWN
    reason: str = ""
    fps: float = 0.0
    face_present: bool = False
    body_present: bool = False
    # 0..1 signal strengths the dashboard shows as meters.
    gaze_drift: float = 0.0  # how far the eyes are off-centre
    looking_down: float = 0.0  # downward gaze (phone/lap)
    hands_near_face: float = 0.0  # hand-at-face / phone-in-hand
    attention: float = 1.0  # 1 = locked on, 0 = fully distracted


# How "off" each committed state is, for the attention meter (mirrors the summary heatmap weights).
_ATTENTION = {
    DistractionState.FOCUSED: 1.0,
    DistractionState.DRIFTING: 0.5,
    DistractionState.FATIGUED: 0.35,
    DistractionState.DISTRACTED: 0.0,
}


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
        self.current_activity: ActivityCategory = ActivityCategory.UNKNOWN
        self.current_reason: str = ""
        self.frames_processed = 0
        # Live dashboard state (read from the UI thread, written from the capture thread).
        self._lock = threading.Lock()
        self._snapshot = LiveSnapshot()
        self._state_counts: dict[str, int] = {}
        self._recent: deque[DistractionState] = deque(maxlen=160)  # ~30s of committed windows
        self._last_frame_t: float | None = None
        self._fps = 0.0

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

    def process_frame(
        self, features: FrameFeatures, activity: ActivityCategory | None = None
    ) -> PipelineOutput | None:
        """Drive one frame through the pipeline; a no-op while paused."""
        if self.settings.paused or self.pipeline is None:
            return None
        self.frames_processed += 1
        out = self.pipeline.process_frame(features, activity)
        if out is not None:
            self.current_state = out.state
            self.current_activity = out.activity
            self.current_reason = out.reason
        self._update_live(features, out, activity)
        return out

    def _update_live(
        self,
        features: FrameFeatures,
        out: PipelineOutput | None,
        activity: ActivityCategory | None,
    ) -> None:
        """Refresh the per-frame snapshot + rolling stats under the lock."""
        # Smoothed FPS from inter-frame spacing.
        if self._last_frame_t is not None:
            dt = features.timestamp - self._last_frame_t
            if dt > 1e-6:
                inst = 1.0 / dt
                self._fps = inst if self._fps == 0.0 else 0.9 * self._fps + 0.1 * inst
        self._last_frame_t = features.timestamp

        gaze_drift = 0.0
        if features.face_present:
            gaze_drift = min(1.0, max(abs(_num(features.gaze_x)), abs(_num(features.gaze_y))))

        with self._lock:
            if out is not None:  # a window closed: fold into the timeline + histogram
                self._recent.append(out.state)
                self._state_counts[str(out.state)] = self._state_counts.get(str(out.state), 0) + 1
            state = self.current_state
            self._snapshot = LiveSnapshot(
                state=state,
                activity=activity if activity is not None else self.current_activity,
                reason=self.current_reason,
                fps=self._fps,
                face_present=features.face_present,
                body_present=features.body_present,
                gaze_drift=gaze_drift,
                looking_down=min(1.0, _num(features.looking_down)),
                hands_near_face=min(1.0, _num(features.hands_near_face)),
                attention=_ATTENTION.get(state, 1.0) if state is not None else 1.0,
            )

    def snapshot(self) -> LiveSnapshot:
        """Latest per-frame snapshot for the UI (thread-safe)."""
        with self._lock:
            return self._snapshot

    def state_percentages(self) -> dict[str, float]:
        """Share of committed windows spent in each state so far this session."""
        with self._lock:
            total = sum(self._state_counts.values())
            if not total:
                return {str(s): 0.0 for s in STATES}
            return {str(s): self._state_counts.get(str(s), 0) / total for s in STATES}

    def recent_states(self) -> list[DistractionState]:
        """The last ~30s of committed states, oldest first (for the live timeline)."""
        with self._lock:
            return list(self._recent)

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
