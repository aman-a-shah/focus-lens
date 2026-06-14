import pytest
import torch

from focuslens.gaze.dataset import FEATURE_DIM
from focuslens.gaze.model import build_model, model_input_key


def test_mlp_forward_shape():
    model = build_model("mlp")
    out = model(torch.zeros(8, FEATURE_DIM))
    assert out.shape == (8, 2)


def test_cnn_forward_shape():
    model = build_model("cnn")
    out = model(torch.zeros(8, 1, 36, 60))
    assert out.shape == (8, 2)


def test_cnn_accepts_other_input_sizes():
    # AdaptiveAvgPool makes the head input-size agnostic
    out = build_model("cnn")(torch.zeros(2, 1, 48, 72))
    assert out.shape == (2, 2)


def test_input_key_routing():
    assert model_input_key("mlp") == "features"
    assert model_input_key("cnn") == "image"


def test_unknown_arch_raises():
    with pytest.raises(ValueError):
        build_model("transformer")
