"""Hazard-threshold firing, calibration + lead time (roadmap Phase 9)."""

import numpy as np

from focuslens.intervention.cox import CoxPH
from focuslens.intervention.synthetic import make_cox_dataset, make_sessions
from focuslens.intervention.timing import (
    InterventionTimer,
    calibrate_threshold,
    lead_times,
)


def _fitted_model(seed: int = 0) -> CoxPH:
    x, dur, ev, _ = make_cox_dataset(n=800, seed=seed)
    return CoxPH().fit(x, dur, ev, epochs=300)


def test_timer_threshold_and_cooldown():
    model = CoxPH()
    model.beta = np.zeros(6, dtype=np.float32)
    model.feature_mean = np.zeros(6, dtype=np.float32)
    model.feature_std = np.ones(6, dtype=np.float32)
    feats = np.ones(6, dtype=np.float32)

    high = InterventionTimer(model, threshold=-1.0, cooldown_s=10.0)
    assert high.update(feats, t=0.0) is True  # risk 0 > -1 -> fire
    assert high.update(feats, t=5.0) is False  # within cooldown
    assert high.update(feats, t=11.0) is True  # cooldown elapsed

    low = InterventionTimer(model, threshold=1.0, cooldown_s=0.0)
    assert low.update(feats, t=0.0) is False  # risk 0 < 1 -> no fire


def test_interventions_fire_10_to_30s_before_drift():
    model = _fitted_model(seed=0)
    cal = make_sessions(n=40, seed=100)
    eval_sessions = make_sessions(n=40, seed=200)

    threshold = calibrate_threshold(model, cal, target_lead_s=20.0)
    leads = np.array(lead_times(model, eval_sessions, threshold))

    assert len(leads) > 0
    assert 10.0 <= float(np.median(leads)) <= 30.0  # roadmap exit criterion
    assert (leads >= 5.0).mean() > 0.9  # essentially all fire pre-emptively, not at the last second
