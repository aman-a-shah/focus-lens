"""Synthetic survival data for the intervention model (roadmap Phase 9).

Two generators, both camera-free:

- ``make_cox_dataset`` — i.i.d. (covariate, time-to-event, censored?) samples drawn from a true
  proportional-hazards process, for fitting/validating the Cox model and its C-index.
- ``make_sessions`` — time series where a *risk ramp* feature rises in the ~40s before a drift
  event, so the timer's lead time (how early it fires before the drift) can be measured.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import FEATURE_DIM

# A "true" effect: distraction_trend and gaze velocity raise hazard; later features are weak/noise.
_BETA_TRUE = np.array([1.6, 0.7, 0.4, 0.2, 0.0, 0.3], dtype=np.float64)


def make_cox_dataset(
    n: int = 800, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X [n, D], durations [n], events [n], beta_true [D]) from a Cox process."""
    rng = np.random.RandomState(seed)
    x = rng.normal(size=(n, FEATURE_DIM)).astype(np.float32)
    lp = x @ _BETA_TRUE
    # Exponential time-to-event with rate ∝ exp(linear predictor).
    rate = np.exp(lp) / 40.0
    t_event = rng.exponential(1.0 / rate)
    t_censor = rng.uniform(10.0, 120.0, size=n)
    duration = np.minimum(t_event, t_censor)
    event = (t_event <= t_censor).astype(np.int64)
    return x.astype(np.float32), duration, event, _BETA_TRUE


@dataclass
class Session:
    times: np.ndarray  # [T] seconds
    features: np.ndarray  # [T, D]
    drift_at: float | None  # event time, or None if censored (no drift)


def make_session(
    seed: int = 0,
    session_len: float = 120.0,
    dt: float = 1.0,
    ramp_s: float = 40.0,
    has_drift: bool = True,
) -> Session:
    """One session: a risk-ramp feature climbs over the ``ramp_s`` before ``drift_at``."""
    rng = np.random.RandomState(seed)
    times = np.arange(0.0, session_len, dt)
    drift_at = float(rng.uniform(session_len * 0.5, session_len * 0.85)) if has_drift else None

    feats = np.zeros((len(times), FEATURE_DIM), dtype=np.float32)
    for i, t in enumerate(times):
        # Risk ramp climbs 0 -> 1 over the last ``ramp_s`` before drift (flat low if no drift).
        ramp = (
            float(np.clip(1.0 - (drift_at - t) / ramp_s, 0.0, 1.0))
            if drift_at is not None
            else 0.05
        )
        feats[i, 0] = ramp + rng.normal(0, 0.03)  # distraction_trend
        feats[i, 1] = 0.5 + 1.5 * ramp + rng.normal(0, 0.05)  # gaze velocity
        feats[i, 2] = 14.0 + rng.normal(0, 0.5)  # blink rate
        feats[i, 3] = 5.0 + 10.0 * ramp + rng.normal(0, 0.5)  # head pose change
        feats[i, 4] = 0.6  # time of day
        feats[i, 5] = t / 3600.0  # session length (hours)
    return Session(times=times, features=feats, drift_at=drift_at)


def make_sessions(n: int = 40, seed: int = 0, drift_fraction: float = 0.8) -> list[Session]:
    """A mix of drift and no-drift (censored) sessions."""
    rng = np.random.RandomState(seed)
    sessions = []
    for i in range(n):
        has_drift = rng.uniform() < drift_fraction
        sessions.append(make_session(seed=seed + i + 1, has_drift=has_drift))
    return sessions


def sessions_to_survival(
    sessions: list[Session], stride: int = 5, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Landmark-sample sessions into (X, durations, events) for Cox training.

    At every ``stride``-th second we take the feature vector and the time until the session's drift
    (event=1), or time to session end (event=0, censored). A little jitter breaks duration ties.
    """
    rng = np.random.RandomState(seed)
    xs, durs, evs = [], [], []
    for s in sessions:
        end = s.drift_at if s.drift_at is not None else float(s.times[-1])
        for i in range(0, len(s.times), stride):
            t = float(s.times[i])
            if t >= end:
                break
            xs.append(s.features[i])
            durs.append(end - t + rng.uniform(0, 1e-3))
            evs.append(1 if s.drift_at is not None else 0)
    return (
        np.stack(xs).astype(np.float32),
        np.array(durs, dtype=np.float64),
        np.array(evs, dtype=np.int64),
    )
