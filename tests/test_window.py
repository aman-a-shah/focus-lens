import numpy as np
import pytest

from focuslens.features import FrameFeatures
from focuslens.window import NUM_FEATURES, SequenceBuffer, WindowAggregator


def _frame(t, gaze_x=0.0, gaze_y=0.0, yaw=0.0, ear=0.27, face=True):
    return FrameFeatures(
        timestamp=t,
        face_present=face,
        ear_right=ear,
        ear_left=ear,
        ear_mean=ear if face else float("nan"),
        eye_closed=False,
        blinks_per_min=12.0,
        last_blink_duration_s=0.15,
        yaw=yaw if face else float("nan"),
        pitch=0.0 if face else float("nan"),
        roll=0.0 if face else float("nan"),
        gaze_x=gaze_x if face else float("nan"),
        gaze_y=gaze_y if face else float("nan"),
    )


def _feed(agg, frames):
    return [w for f in frames if (w := agg.add(f)) is not None]


def test_emits_window_after_window_duration():
    agg = WindowAggregator(window_s=0.2)
    frames = [_frame(t / 10.0, gaze_x=0.1) for t in range(3)]  # 0.0, 0.1, 0.2
    windows = _feed(agg, frames)
    assert len(windows) == 1
    w = windows[0]
    assert w.gaze_x == pytest.approx(0.1)
    assert w.face_fraction == 1.0
    assert w.gaze_velocity == 0.0  # first window has no predecessor


def test_velocity_and_pose_change_are_positive_when_moving():
    agg = WindowAggregator(window_s=0.2)
    # window 1 at origin, window 2 shifted in gaze and yaw
    f1 = [_frame(0.0, gaze_x=0.0, yaw=0.0), _frame(0.1), _frame(0.2)]
    f2 = [
        _frame(0.3, gaze_x=0.3, yaw=10.0),
        _frame(0.4, gaze_x=0.3, yaw=10.0),
        _frame(0.5, gaze_x=0.3, yaw=10.0),
    ]
    windows = _feed(agg, f1 + f2)
    assert len(windows) == 2
    assert windows[1].gaze_velocity > 0
    assert windows[1].head_pose_change_rate > 0


def test_face_fraction_reflects_missing_face():
    agg = WindowAggregator(window_s=0.2)
    frames = [_frame(0.0, face=True), _frame(0.1, face=False), _frame(0.2, face=False)]
    w = _feed(agg, frames)[0]
    assert w.face_fraction == pytest.approx(1 / 3)
    # NaN gaze with no face must not poison the vector
    assert not np.isnan(w.to_vector()).any()


def test_to_vector_has_eight_features():
    agg = WindowAggregator(window_s=0.2)
    w = _feed(agg, [_frame(0.0), _frame(0.1), _frame(0.2)])[0]
    assert w.to_vector().shape == (NUM_FEATURES,)


def test_sequence_buffer_ring_and_array():
    buf = SequenceBuffer(length=3)
    agg = WindowAggregator(window_s=0.2)
    for k in range(5):
        for w in _feed(agg, [_frame(k * 0.3 + i * 0.1) for i in range(3)]):
            buf.append(w)
    assert len(buf) == 3
    assert buf.is_full
    assert buf.to_array().shape == (3, NUM_FEATURES)
