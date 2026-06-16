"""EWC + experience replay + checkpoint rotation (roadmap Phase 8)."""

import torch

from focuslens.focusnet.ablation import make_task_sequence
from focuslens.focusnet.continual import (
    EWC,
    ContinualTrainer,
    ReplayBuffer,
    evaluate_accuracy,
    save_session_checkpoint,
)
from focuslens.focusnet.dataset import FeatureNormalizer
from focuslens.focusnet.model import PersonalFocusNet
from focuslens.window import NUM_FEATURES


def test_ewc_penalty_zero_at_snapshot_positive_after_drift():
    model = PersonalFocusNet()
    x = torch.randn(32, 10, NUM_FEATURES)
    y = torch.randint(0, 4, (32,))
    ewc = EWC(lam=10.0)

    assert float(ewc.penalty(model)) == 0.0  # no consolidated task yet
    ewc.consolidate(model, x, y)
    assert float(ewc.penalty(model).detach()) < 1e-6  # weights still at the snapshot

    with torch.no_grad():
        next(iter(model.parameters())).add_(1.0)  # drift a weight
    assert float(ewc.penalty(model).detach()) > 0.0

    # Fisher entries are non-negative and shaped like the parameters.
    _snap, fisher = ewc.tasks[0]
    for name, p in model.named_parameters():
        assert fisher[name].shape == p.shape
        assert float(fisher[name].min()) >= 0.0


def test_replay_buffer_reservoir_capacity_and_sampling():
    buf = ReplayBuffer(capacity=50, seed=0)
    x = torch.randn(200, 8, NUM_FEATURES)
    y = torch.randint(0, 4, (200,))
    buf.add_many(x, y)
    assert len(buf) == 50  # capacity respected despite 200 seen

    sx, sy = buf.sample(16)
    assert sx.shape == (16, 8, NUM_FEATURES) and sy.shape == (16,)


def test_replay_buffer_state_roundtrip():
    buf = ReplayBuffer(capacity=20, seed=1)
    buf.add_many(torch.randn(30, 6, NUM_FEATURES), torch.randint(0, 4, (30,)))
    restored = ReplayBuffer()
    restored.load_state_dict(buf.state_dict())
    assert len(restored) == len(buf)
    sx, _ = restored.sample(5)
    assert sx.shape == (5, 6, NUM_FEATURES)


def test_continual_trainer_learns_single_task_and_grows_state():
    tasks = make_task_sequence(n_tasks=1, n_per_class=40, seq_len=10, seed=0)
    task = tasks[0]
    model = PersonalFocusNet()
    normalizer = FeatureNormalizer.fit(task.train_x)
    ewc, replay = EWC(lam=50.0), ReplayBuffer(capacity=100, seed=0)
    trainer = ContinualTrainer(model, normalizer, ewc=ewc, replay=replay)

    trainer.fit_task(task.train_x, task.train_y, epochs=10, seed=0)
    acc = evaluate_accuracy(model, task.val_x, task.val_y, normalizer)
    assert acc >= 0.80  # learns a single rotated task
    assert len(ewc.tasks) == 1 and len(replay) > 0  # state grew after the task


def test_checkpoint_rotation_keeps_newest(tmp_path):
    model = PersonalFocusNet()
    for i in range(7):
        save_session_checkpoint(model, i, tmp_path, keep=5)
    remaining = sorted(p.name for p in tmp_path.glob("focusnet_session_*.pt"))
    assert len(remaining) == 5
    assert remaining[0] == "focusnet_session_0002.pt"  # oldest two rotated out
    assert remaining[-1] == "focusnet_session_0006.pt"


def test_session_checkpoint_roundtrips_ewc_and_replay(tmp_path):
    model = PersonalFocusNet()
    ewc = EWC(lam=10.0)
    ewc.consolidate(model, torch.randn(16, 10, NUM_FEATURES), torch.randint(0, 4, (16,)))
    replay = ReplayBuffer(capacity=30, seed=0)
    replay.add_many(torch.randn(40, 10, NUM_FEATURES), torch.randint(0, 4, (40,)))

    path = save_session_checkpoint(model, 0, tmp_path, ewc=ewc, replay=replay)
    ckpt = torch.load(path, map_location="cpu")
    assert "ewc" in ckpt and "replay" in ckpt

    ewc2, replay2 = EWC(), ReplayBuffer()
    ewc2.load_state_dict(ckpt["ewc"])
    replay2.load_state_dict(ckpt["replay"])
    assert len(ewc2.tasks) == 1 and len(replay2) == 30
