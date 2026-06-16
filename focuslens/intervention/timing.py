"""Hazard-threshold firing + the notification controller (roadmap Phase 9).

``InterventionTimer`` thresholds the Cox risk score with a cooldown — the user-tunable knob.
``calibrate_threshold`` picks the threshold that fires a target lead time (default 20s) before
drift, and ``lead_times`` measures how early it actually fires on held-out sessions (the "10–30s
early" exit criterion). ``InterventionController`` is the runtime glue: it assembles features each
window, asks the timer, fires a pre-emptive nudge through the notifier, and logs the intervention
(with a slot for "was this helpful?" feedback).
"""

from __future__ import annotations

import numpy as np

from ..logging import get_logger
from ..notify import Notifier
from ..states import DistractionState
from ..window import WindowFeatures
from .cox import CoxPH
from .features import assemble_hazard_features

log = get_logger(__name__)


class InterventionTimer:
    """Fire when the Cox risk score crosses ``threshold``, respecting a cooldown."""

    def __init__(self, model: CoxPH, threshold: float, cooldown_s: float = 60.0) -> None:
        self.model = model
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._last_fire_t: float | None = None
        self.last_risk = 0.0

    def update(self, features: np.ndarray, t: float) -> bool:
        """Return True if an intervention should fire now."""
        self.last_risk = float(self.model.predict_risk(features))
        if self.last_risk < self.threshold:
            return False
        if self._last_fire_t is not None and t - self._last_fire_t < self.cooldown_s:
            return False
        self._last_fire_t = t
        return True


def calibrate_threshold(model: CoxPH, sessions, target_lead_s: float = 20.0) -> float:
    """Pick the threshold whose risk level is reached ``target_lead_s`` before drift (median)."""
    risks_at_target = []
    for s in sessions:
        if s.drift_at is None:
            continue
        t_target = s.drift_at - target_lead_s
        idx = int(np.argmin(np.abs(s.times - t_target)))
        risks_at_target.append(float(model.predict_risk(s.features[idx])))
    return float(np.median(risks_at_target)) if risks_at_target else 0.0


def lead_times(model: CoxPH, sessions, threshold: float, cooldown_s: float = 1e9) -> list[float]:
    """For each drift session, how many seconds before drift the timer first fires (if it does)."""
    leads = []
    for s in sessions:
        if s.drift_at is None:
            continue
        timer = InterventionTimer(model, threshold, cooldown_s=cooldown_s)
        for i, t in enumerate(s.times):
            if t >= s.drift_at:
                break
            if timer.update(s.features[i], float(t)):
                leads.append(float(s.drift_at - t))
                break
    return leads


class InterventionController:
    """Wires the timer into the live loop: assemble → threshold → notify → log feedback row."""

    def __init__(
        self,
        timer: InterventionTimer,
        notifier: Notifier | None = None,
        store=None,
        session_id: int | None = None,
        trend_windows: int = 300,  # ~60s of 200ms windows
    ) -> None:
        self.timer = timer
        self.notifier = notifier
        self.store = store
        self.session_id = session_id
        self.trend_windows = trend_windows
        self._windows: list[WindowFeatures] = []
        self._states: list[DistractionState] = []
        self._t0: float | None = None

    def on_window(self, window: WindowFeatures, state: DistractionState) -> bool:
        """Feed one classified window; fire + log an intervention if the hazard threshold trips."""
        if self._t0 is None:
            self._t0 = window.t_start
        self._windows.append(window)
        self._states.append(state)
        self._windows = self._windows[-self.trend_windows :]
        self._states = self._states[-self.trend_windows :]

        feats = assemble_hazard_features(
            self._windows, self._states, session_length_s=window.t_end - self._t0
        )
        fired = self.timer.update(feats.to_vector(), window.t_end)
        if fired:
            if self.notifier is not None:
                self.notifier.on_intervention(window.t_end)
            if self.store is not None and self.session_id is not None:
                self.store.log_intervention(
                    self.session_id, window.t_end, self.timer.last_risk, fired=True
                )
            log.info("Intervention fired @ %.1fs (risk %.2f)", window.t_end, self.timer.last_risk)
        return fired
