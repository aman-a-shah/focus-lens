"""Hazard feature assembly (roadmap Phase 9)."""

from focuslens.intervention.features import FEATURE_DIM, assemble_hazard_features
from focuslens.states import DistractionState
from focuslens.window import WindowFeatures


def _w(gaze_velocity=0.0, blink_rate=12.0, head_pose=0.0) -> WindowFeatures:
    return WindowFeatures(
        t_start=0.0,
        t_end=0.2,
        face_fraction=1.0,
        gaze_x=0.0,
        gaze_y=0.0,
        gaze_velocity=gaze_velocity,
        gaze_accel=0.0,
        blink_rate=blink_rate,
        blink_duration=0.15,
        head_pose_change_rate=head_pose,
        ear=0.27,
    )


def test_assembles_trend_and_means():
    windows = [_w(gaze_velocity=1.0, head_pose=-10.0), _w(gaze_velocity=3.0, head_pose=10.0)]
    states = [
        DistractionState.FOCUSED,
        DistractionState.DRIFTING,
        DistractionState.DISTRACTED,
        DistractionState.FOCUSED,
    ]
    f = assemble_hazard_features(windows, states, time_of_day=0.6, session_length_s=1800.0)

    assert abs(f.distraction_trend - 0.5) < 1e-6  # 2 of 4 states distracted
    assert abs(f.gaze_velocity_60s - 2.0) < 1e-6
    assert abs(f.head_pose_60s - 10.0) < 1e-6  # mean of |−10|, |10|
    assert abs(f.session_length_h - 0.5) < 1e-6  # 1800s
    assert f.to_vector().shape == (FEATURE_DIM,)


def test_empty_stream_is_zero_risk():
    f = assemble_hazard_features([], [])
    assert f.distraction_trend == 0.0 and f.gaze_velocity_60s == 0.0
