"""App controller: session lifecycle, pause, live tuning (roadmap Phase 10)."""

from focuslens.app.controller import AppController
from focuslens.classifier import RuleClassifier
from focuslens.simulate import DEFAULT_SCENARIO, generate_frames


def _controller(**kw) -> AppController:
    return AppController(db_path=":memory:", notify=False, **kw)


def test_pause_gates_frame_processing():
    ctrl = _controller()
    ctrl.start_session(0.0)
    frames = generate_frames([("focused", 4.0)], fps=30, seed=0)

    half = len(frames) // 2
    for f in frames[:half]:
        ctrl.process_frame(f)
    assert ctrl.frames_processed == half

    assert ctrl.toggle_pause() is True
    for f in frames[half:]:
        assert ctrl.process_frame(f) is None  # paused -> no-op
    assert ctrl.frames_processed == half  # unchanged while paused

    assert ctrl.toggle_pause() is False
    ctrl.process_frame(frames[0])
    assert ctrl.frames_processed == half + 1
    ctrl.close()


def test_set_sensitivity_retunes_live_classifier():
    ctrl = _controller()
    ctrl.start_session(0.0)
    assert isinstance(ctrl.classifier, RuleClassifier)
    before = ctrl.classifier.t.gaze_offscreen
    ctrl.set_sensitivity(0.95)
    assert ctrl.classifier.t.gaze_offscreen < before  # more eager after sliding up
    assert ctrl.settings.sensitivity == 0.95
    ctrl.close()


def test_full_session_produces_summary():
    ctrl = _controller()
    sid = ctrl.start_session(0.0)
    frames = generate_frames(DEFAULT_SCENARIO, fps=30, seed=0)
    for f in frames:
        ctrl.process_frame(f)
    ctrl.end_session(frames[-1].timestamp)

    summary = ctrl.summary()
    assert summary is not None and summary.session_id == sid
    assert summary.total_windows > 0
    assert ctrl.store.window_count(sid) == summary.total_windows
    ctrl.close()


def test_summary_is_none_before_a_session():
    ctrl = _controller()
    assert ctrl.summary() is None
    ctrl.close()
