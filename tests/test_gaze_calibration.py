"""Personal gaze calibration (roadmap Phase 5).

Demonstrates the headline "Done when": fine-tuning the population head on a user's labelled
calibration samples cuts angular error by ≥20% vs. the uncalibrated baseline.
"""

import torch

from focuslens.gaze.calibration import (
    CALIB_POINTS_5,
    CALIB_POINTS_9,
    CalibrationData,
    calibrate_user,
    fine_tune,
)
from focuslens.gaze.model import MLPGazeNet
from focuslens.gaze.predictor import CalibratedGazePredictor


def _population_base(seed: int = 0) -> MLPGazeNet:
    """A base head fit to a *population* map y ≈ 0.9·iris — deliberately off for our user."""
    torch.manual_seed(seed)
    model = MLPGazeNet()
    x = torch.empty(2000, 5).uniform_(-1.0, 1.0)
    y = torch.stack([0.9 * x[:, 0], 0.9 * x[:, 1]], dim=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(120):
        opt.zero_grad()
        loss = torch.nn.functional.smooth_l1_loss(model(x), y)
        loss.backward()
        opt.step()
    return model


def _user_calibration_data(n: int = 400, seed: int = 1) -> CalibrationData:
    """Samples from a user whose true mapping (gain 0.55 + offset) differs from the population."""
    g = torch.Generator().manual_seed(seed)
    x = torch.empty(n, 5).uniform_(-1.0, 1.0, generator=g)
    data = CalibrationData()
    for row in x:
        tx = 0.55 * float(row[0]) + 0.30
        ty = 0.55 * float(row[1]) - 0.20
        data.add(tuple(float(v) for v in row), (tx, ty))
    return data


def test_calib_point_sets():
    assert len(CALIB_POINTS_5) == 5
    assert len(CALIB_POINTS_9) == 9
    assert (0.0, 0.0) in CALIB_POINTS_5  # centre target present


def test_calibration_data_to_tensors_shapes():
    data = CalibrationData()
    data.add((0.1, 0.2, 0.0, 0.0, 0.0), (0.5, -0.5))
    data.add((0.3, 0.4, 0.1, 0.1, 0.0), (-0.5, 0.5))
    x, y = data.to_tensors()
    assert x.shape == (2, 5) and y.shape == (2, 2)


def test_fine_tune_cuts_error_by_at_least_20pct():
    base = _population_base()
    data = _user_calibration_data()
    _, before, after = fine_tune(base, data, epochs=80, seed=0)
    assert after < before
    improvement = (before - after) / before
    assert improvement >= 0.20  # roadmap Phase 5 target: 20–40% drop


def test_calibrate_user_saves_loadable_per_user_checkpoint(tmp_path):
    base = _population_base()
    base_path = tmp_path / "gaze_mlp.pt"
    torch.save({"arch": "mlp", "state_dict": base.state_dict()}, base_path)

    data = _user_calibration_data()
    result = calibrate_user(
        data,
        user_id="alice",
        base_checkpoint=base_path,
        checkpoint_dir=tmp_path,
        epochs=80,
    )
    assert result.improvement_pct >= 20.0
    assert (tmp_path / "gaze_user_alice.pt").exists()

    # The saved checkpoint loads into the runtime predictor and produces calibrated gaze.
    predictor = CalibratedGazePredictor.from_checkpoint(result.checkpoint)
    from focuslens.features.gaze import GazeProxy

    out = predictor.predict(GazeProxy(0.5, -0.3), pose=None)
    assert out is not None and abs(out.x) < 2.0 and abs(out.y) < 2.0
