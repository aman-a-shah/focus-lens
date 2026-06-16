"""PersonalFocusNet architecture (roadmap Phase 7)."""

import torch

from focuslens.focusnet.model import PersonalFocusNet, predict, uncertainty_from
from focuslens.states import NUM_STATES
from focuslens.window import NUM_FEATURES


def test_forward_shapes():
    model = PersonalFocusNet()
    x = torch.randn(4, 30, NUM_FEATURES)
    logits, log_prec = model(x)
    assert logits.shape == (4, NUM_STATES)
    assert log_prec.shape == (4,)


def test_predict_returns_valid_classes_and_bounded_uncertainty():
    model = PersonalFocusNet()
    x = torch.randn(8, 30, NUM_FEATURES)
    idx, probs, unc = predict(model, x)
    assert idx.shape == (8,) and int(idx.min()) >= 0 and int(idx.max()) < NUM_STATES
    assert torch.allclose(probs.sum(dim=-1), torch.ones(8), atol=1e-5)
    assert float(unc.min()) >= 0.0 and float(unc.max()) <= 1.0


def test_uncertainty_monotonic_in_log_precision():
    # Higher precision -> lower uncertainty.
    u = uncertainty_from(torch.tensor([-3.0, 0.0, 3.0]))
    assert u[0] > u[1] > u[2]
