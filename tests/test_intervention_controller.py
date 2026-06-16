"""Intervention controller + feedback persistence + pipeline wiring (roadmap Phase 9)."""

import numpy as np

from focuslens.intervention.cox import CoxPH
from focuslens.intervention.timing import InterventionController, InterventionTimer
from focuslens.notify import Notifier
from focuslens.pipeline import AttentionPipeline
from focuslens.session import SessionStore
from focuslens.simulate import generate_frames
from focuslens.states import DistractionState
from focuslens.window import WindowFeatures


def _always_fire_timer() -> InterventionTimer:
    model = CoxPH()
    model.beta = np.zeros(6, dtype=np.float32)
    model.feature_mean = np.zeros(6, dtype=np.float32)
    model.feature_std = np.ones(6, dtype=np.float32)
    return InterventionTimer(model, threshold=-1.0, cooldown_s=0.0)


def _window(t: float) -> WindowFeatures:
    return WindowFeatures(
        t_start=t,
        t_end=t + 0.2,
        face_fraction=1.0,
        gaze_x=0.0,
        gaze_y=0.0,
        gaze_velocity=0.5,
        gaze_accel=0.0,
        blink_rate=12.0,
        blink_duration=0.15,
        head_pose_change_rate=2.0,
        ear=0.27,
    )


def test_controller_fires_and_logs_intervention_with_feedback():
    store = SessionStore(":memory:")
    sid = store.start_session(0.0)
    ctrl = InterventionController(
        _always_fire_timer(), notifier=Notifier(enabled=False), store=store, session_id=sid
    )

    assert ctrl.on_window(_window(0.0), DistractionState.FOCUSED) is True
    rows = store.get_interventions(sid)
    assert len(rows) == 1
    iid, _t, _risk, fired, helpful = rows[0]
    assert fired == 1 and helpful is None  # logged, awaiting feedback

    store.record_feedback(iid, helpful=True)
    assert store.get_interventions(sid)[0][4] == 1  # feedback persisted
    store.close()


def test_controller_wires_into_attention_pipeline():
    store = SessionStore(":memory:")
    sid = store.start_session(0.0)
    ctrl = InterventionController(
        _always_fire_timer(), notifier=Notifier(enabled=False), store=store, session_id=sid
    )
    pipeline = AttentionPipeline(store=store, session_id=sid, intervention=ctrl)

    for f in generate_frames([("focused", 2.0)], fps=30, seed=0):
        pipeline.process_frame(f)
    pipeline.finish()

    # The pipeline consulted the controller every window, so interventions were logged.
    assert len(store.get_interventions(sid)) > 0
    store.close()
