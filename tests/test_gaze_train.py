"""Training smoke test — small + fast, but real (no mocks).

Confirms the gaze training loop actually learns the synthetic mapping and that the angular
loss stays finite (regression guard for the arccos-gradient NaN we hit and fixed).
"""

from pathlib import Path

from focuslens.gaze.train import train_gaze


def test_mlp_training_reduces_error(tmp_path):
    result = train_gaze(
        arch="mlp", n_samples=1500, epochs=12, batch_size=128, seed=0, checkpoint_dir=tmp_path
    )
    # Clear learning: best error is a small fraction of the first epoch's, and low in absolute.
    assert result.best_val_mae_deg < 0.5 * result.history[0].val_mae_deg
    assert result.best_val_mae_deg < 6.0  # learnable synthetic mapping
    assert result.latency_ms_p50 > 0.0
    assert Path(result.checkpoint).exists()
    assert (tmp_path / "gaze_mlp_history.json").exists()


def test_losses_stay_finite(tmp_path):
    result = train_gaze(
        arch="mlp", n_samples=400, epochs=5, batch_size=64, seed=1, checkpoint_dir=tmp_path
    )
    assert all(m.train_loss == m.train_loss for m in result.history)  # not NaN
    assert all(m.val_mae_deg == m.val_mae_deg for m in result.history)
