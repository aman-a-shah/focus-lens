"""Intervention timing (roadmap Phase 9).

Decides *when* to interrupt, not just whether distraction is happening: a Cox proportional-hazards
model over assembled risk features (current state, 60s trend, time-of-day, session length) predicts
time-to-distraction, and a user-tunable hazard threshold fires a pre-emptive nudge 10–30s before
the user would notice they drifted. Helpfulness feedback is logged back for later tuning.
"""

from .cox import CoxPH, concordance_index
from .features import HazardFeatures, assemble_hazard_features
from .timing import InterventionController, InterventionTimer, calibrate_threshold, lead_times

__all__ = [
    "CoxPH",
    "concordance_index",
    "HazardFeatures",
    "assemble_hazard_features",
    "InterventionTimer",
    "InterventionController",
    "calibrate_threshold",
    "lead_times",
]
