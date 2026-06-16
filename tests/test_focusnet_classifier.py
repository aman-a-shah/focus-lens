"""Learned-classifier runtime adapter: gating + pipeline swap (roadmap Phase 7)."""

import torch

from focuslens.focusnet.classifier import LearnedClassifier
from focuslens.focusnet.dataset import SyntheticFocusDataset
from focuslens.focusnet.model import PersonalFocusNet
from focuslens.focusnet.train import train_focusnet
from focuslens.pipeline import AttentionPipeline
from focuslens.simulate import DEFAULT_SCENARIO, generate_frames
from focuslens.states import DistractionState
from focuslens.window import WindowFeatures


def _clf(**kw) -> LearnedClassifier:
    return LearnedClassifier(PersonalFocusNet(), **kw)


def _window() -> WindowFeatures:
    return WindowFeatures(
        t_start=0.0,
        t_end=0.2,
        face_fraction=1.0,
        gaze_x=1.1,  # off-screen-ish
        gaze_y=0.0,
        gaze_velocity=0.0,
        gaze_accel=0.0,
        blink_rate=12.0,
        blink_duration=0.15,
        head_pose_change_rate=0.0,
        ear=0.26,
    )


def test_cold_start_suppresses_escalation():
    clf = _clf(warmup_windows=5, uncertainty_threshold=1.0)
    clf._seen = 2  # still warming up
    assert clf._gate(DistractionState.DISTRACTED, uncertainty=0.0) == DistractionState.FOCUSED


def test_high_uncertainty_gates_to_focused():
    clf = _clf(warmup_windows=0, uncertainty_threshold=0.3)
    clf._seen = 50
    assert clf._gate(DistractionState.DISTRACTED, uncertainty=0.9) == DistractionState.FOCUSED
    # Below threshold, the raw call passes through.
    assert clf._gate(DistractionState.DISTRACTED, uncertainty=0.1) == DistractionState.DISTRACTED


def test_classify_matches_classifier_interface_and_warms_up():
    clf = _clf(warmup_windows=4, uncertainty_threshold=1.0, seq_len=8)
    # During warmup every call is suppressed to FOCUSED regardless of the model output.
    for _ in range(3):
        assert clf.classify(_window()) == DistractionState.FOCUSED
    # After warmup it returns a valid state (interface check).
    out = clf.classify(_window())
    assert isinstance(out, DistractionState)
    assert 0.0 <= clf.last_uncertainty <= 1.0


def test_trained_model_swaps_into_pipeline(tmp_path):
    ds = SyntheticFocusDataset(n_per_class=60, seq_len=12, seed=0)
    result = train_focusnet(dataset=ds, epochs=8, seed=0, checkpoint_dir=tmp_path)

    clf = LearnedClassifier.from_checkpoint(result.checkpoint, warmup_windows=3)
    pipeline = AttentionPipeline(classifier=clf)
    frames = generate_frames(DEFAULT_SCENARIO, fps=30, seed=0)
    n_windows = 0
    for f in frames:
        out = pipeline.process_frame(f)
        if out is not None:
            n_windows += 1
            assert isinstance(out.state, DistractionState)
    assert n_windows > 0  # the learned classifier drove the whole session


def test_from_checkpoint_without_normalizer(tmp_path):
    path = tmp_path / "bare.pt"
    torch.save({"state_dict": PersonalFocusNet().state_dict(), "seq_len": 10}, path)
    clf = LearnedClassifier.from_checkpoint(path)
    assert clf.normalizer is None and clf.seq_len == 10
    assert clf.classify(_window()) in set(DistractionState)
