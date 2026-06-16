"""Backward-transfer ablation: EWC+replay vs naive vs frozen (roadmap Phase 8).

The headline "Done when": after a sequence of sessions, EWC+replay keeps backward degradation
small (<5% target) while naive fine-tuning forgets badly.
"""

from focuslens.focusnet.ablation import (
    backward_transfer,
    make_task_sequence,
    run_ablation,
)


def test_backward_transfer_metric():
    # Task 0 learned to 1.0, then decayed to 0.4 by the end -> 0.6 forgetting.
    matrix = [[1.0, 0.0], [0.4, 1.0]]
    assert abs(backward_transfer(matrix) - 0.6) < 1e-9
    assert backward_transfer([[1.0]]) == 0.0  # single task -> nothing to forget


def test_task_sequence_shapes():
    tasks = make_task_sequence(n_tasks=3, n_per_class=10, seq_len=8, val_per_class=5, seed=0)
    assert len(tasks) == 3
    assert tasks[0].train_x.shape == (40, 8, 8)  # 4 classes * 10
    assert tasks[0].val_x.shape == (20, 8, 8)


def test_ewc_replay_forgets_far_less_than_naive():
    report = run_ablation(n_tasks=5, epochs=10, seed=0, n_per_class=50, seq_len=10)
    naive = report.results["naive"]
    ewc = report.results["ewc_replay"]

    assert ewc.backward_transfer < 0.05  # roadmap target: <5% backward degradation
    assert naive.backward_transfer > 0.20  # naive catastrophically forgets
    assert ewc.backward_transfer < 0.5 * naive.backward_transfer
    assert ewc.final_avg_accuracy > naive.final_avg_accuracy + 0.2


def test_frozen_baseline_retains_first_task_only():
    report = run_ablation(
        n_tasks=4, epochs=6, seed=0, n_per_class=40, seq_len=10, strategies=("frozen",)
    )
    frozen = report.results["frozen"]
    # Never updates after task 0 -> no forgetting, but poor average (later tasks unlearned).
    assert frozen.backward_transfer <= 0.01
    assert frozen.final_avg_accuracy < 0.6


def test_learning_curves_serialize(tmp_path):
    report = run_ablation(n_tasks=3, epochs=4, seed=0, n_per_class=20, seq_len=8)
    path = report.save_learning_curves(tmp_path / "curves.json")
    assert path.exists()
    import json

    data = json.loads(path.read_text())
    assert data["n_tasks"] == 3 and "ewc_replay" in data["results"]
