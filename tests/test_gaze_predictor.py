"""Gaze predictor strategy + its integration into the feature extractor (roadmap Phase 5)."""

import numpy as np
import torch

from focuslens.face_mesh import FaceMeshResult
from focuslens.features import FeatureExtractor
from focuslens.features.gaze import GazeProxy
from focuslens.features.head_pose import HeadPose
from focuslens.gaze.model import MLPGazeNet
from focuslens.gaze.predictor import (
    CalibratedGazePredictor,
    NaiveGazePredictor,
    features_from,
)


def test_features_from_layout_and_pose_scaling():
    feats = features_from(GazeProxy(0.2, -0.4), HeadPose(yaw=90.0, pitch=-45.0, roll=0.0))
    assert feats == [0.2, -0.4, 1.0, -0.5, 0.0]  # pose scaled by /90


def test_features_from_handles_missing_pose():
    assert features_from(GazeProxy(0.1, 0.2), None) == [0.1, 0.2, 0.0, 0.0, 0.0]


def test_naive_predictor_is_identity():
    proxy = GazeProxy(0.3, -0.1)
    assert NaiveGazePredictor().predict(proxy, None) is proxy
    assert NaiveGazePredictor().predict(None, None) is None


def test_calibrated_predictor_runs_and_handles_no_iris():
    predictor = CalibratedGazePredictor(MLPGazeNet())
    out = predictor.predict(GazeProxy(0.4, 0.1), HeadPose(0.0, 0.0, 0.0))
    assert out is not None and isinstance(out.x, float) and isinstance(out.y, float)
    assert predictor.predict(None, None) is None  # no iris -> no gaze


def test_calibrated_predictor_checkpoint_roundtrip(tmp_path):
    model = MLPGazeNet()
    path = tmp_path / "gaze_user_test.pt"
    torch.save({"arch": "mlp", "state_dict": model.state_dict()}, path)
    loaded = CalibratedGazePredictor.from_checkpoint(path)
    same = loaded.predict(GazeProxy(0.2, 0.2), None)
    ref = CalibratedGazePredictor(model).predict(GazeProxy(0.2, 0.2), None)
    assert abs(same.x - ref.x) < 1e-6 and abs(same.y - ref.y) < 1e-6


class _StubPredictor:
    """Returns a fixed gaze regardless of input — proves the extractor consults the predictor."""

    def predict(self, offset, pose):
        return GazeProxy(0.123, -0.456)


def _synthetic_face_result(seed: int = 0) -> FaceMeshResult:
    rng = np.random.RandomState(seed)
    landmarks = rng.uniform(0.2, 0.8, size=(478, 3)).astype(np.float32)  # has_iris -> True
    return FaceMeshResult(landmarks=landmarks)


def test_extractor_uses_gaze_predictor_to_override_proxy():
    result = _synthetic_face_result()
    shape = (720, 1280)

    naive = FeatureExtractor().extract(result, shape, timestamp=0.0)
    calibrated = FeatureExtractor(gaze_predictor=_StubPredictor()).extract(
        result, shape, timestamp=0.0
    )

    assert calibrated.gaze_x == 0.123 and calibrated.gaze_y == -0.456
    # The predictor changed the emitted gaze relative to the naive proxy.
    assert (naive.gaze_x, naive.gaze_y) != (calibrated.gaze_x, calibrated.gaze_y)
