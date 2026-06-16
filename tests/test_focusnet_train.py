"""PersonalFocusNet training + metrics (roadmap Phase 7).

Confirms the headline "Done when": the learned model hits the precision/recall bar on the
DISTRACTED class (>70% P @ >60% R) and bundles a usable checkpoint.
"""

from pathlib import Path

from focuslens.focusnet.dataset import SyntheticFocusDataset
from focuslens.focusnet.train import train_focusnet


def test_training_meets_distracted_precision_recall_target(tmp_path):
    ds = SyntheticFocusDataset(n_per_class=120, seq_len=16, seed=0)
    result = train_focusnet(dataset=ds, epochs=15, seed=0, checkpoint_dir=tmp_path)

    assert result.distracted.precision >= 0.70  # roadmap target
    assert result.distracted.recall >= 0.60
    assert result.best_val_accuracy >= 0.80
    assert Path(result.checkpoint).exists()
    assert (tmp_path / "focusnet_history.json").exists()
    assert 0.0 <= result.uncertainty_threshold <= 1.0


def test_checkpoint_bundles_normalizer_and_threshold(tmp_path):
    import torch

    ds = SyntheticFocusDataset(n_per_class=40, seq_len=12, seed=1)
    result = train_focusnet(dataset=ds, epochs=4, seed=1, checkpoint_dir=tmp_path)
    ckpt = torch.load(result.checkpoint, map_location="cpu")
    assert "state_dict" in ckpt and "normalizer" in ckpt
    assert ckpt["seq_len"] == 12
    assert "uncertainty_threshold" in ckpt
