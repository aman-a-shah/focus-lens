import math

import pytest
import torch

from focuslens.gaze.metrics import angular_error_deg, mean_angular_error_deg


def test_zero_error_for_identical_gaze():
    g = torch.tensor([[0.1, -0.2], [0.0, 0.3]])
    assert mean_angular_error_deg(g, g) == pytest.approx(0.0, abs=1e-4)


def test_known_angular_offset():
    pred = torch.tensor([[0.0, 0.0]])
    target = torch.tensor([[0.1, 0.0]])  # 0.1 rad apart in yaw
    err = angular_error_deg(pred, target)
    assert float(err[0]) == pytest.approx(math.degrees(0.1), abs=1e-3)


def test_batch_mean():
    pred = torch.zeros(4, 2)
    target = torch.tensor([[0.1, 0.0], [0.0, 0.1], [-0.1, 0.0], [0.0, -0.1]])
    # every sample is 0.1 rad off -> mean is 0.1 rad in degrees
    assert mean_angular_error_deg(pred, target) == pytest.approx(math.degrees(0.1), abs=1e-3)
