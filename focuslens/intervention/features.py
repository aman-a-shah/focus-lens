"""Hazard feature assembly for the intervention timer (roadmap Phase 9).

Turns the recent window/state stream into the covariate vector the Cox model scores. The signals
(plan.md §3.5) are deliberately *trend*-oriented — distraction risk is about momentum, not a
single window:

    0 distraction_trend   fraction of the last 60s spent DRIFTING/DISTRACTED
    1 gaze_velocity_60s   mean gaze saccade velocity over the window
    2 blink_rate_60s      mean blink rate (fatigue couples to distraction)
    3 head_pose_60s       mean head-pose change rate
    4 time_of_day         wall-clock hour / 24  (attention dips are diurnal)
    5 session_length_h    elapsed session time in hours (vigilance decays)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..states import DistractionState
from ..window import WindowFeatures

FEATURE_DIM = 6
_DISTRACTED_STATES = {DistractionState.DRIFTING, DistractionState.DISTRACTED}


@dataclass(frozen=True)
class HazardFeatures:
    distraction_trend: float
    gaze_velocity_60s: float
    blink_rate_60s: float
    head_pose_60s: float
    time_of_day: float
    session_length_h: float

    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self.distraction_trend,
                self.gaze_velocity_60s,
                self.blink_rate_60s,
                self.head_pose_60s,
                self.time_of_day,
                self.session_length_h,
            ],
            dtype=np.float32,
        )


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def assemble_hazard_features(
    windows: list[WindowFeatures],
    states: list[DistractionState],
    *,
    time_of_day: float = 0.5,
    session_length_s: float = 0.0,
) -> HazardFeatures:
    """Assemble the covariate vector from the trailing 60s of windows + their classified states."""
    trend = sum(1 for s in states if s in _DISTRACTED_STATES) / len(states) if states else 0.0
    return HazardFeatures(
        distraction_trend=trend,
        gaze_velocity_60s=_mean([w.gaze_velocity for w in windows]),
        blink_rate_60s=_mean([w.blink_rate for w in windows]),
        head_pose_60s=_mean([abs(w.head_pose_change_rate) for w in windows]),
        time_of_day=time_of_day,
        session_length_h=session_length_s / 3600.0,
    )
