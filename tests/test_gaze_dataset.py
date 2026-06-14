import torch

from focuslens.gaze.dataset import FEATURE_DIM, GAZE_RANGE, SyntheticGazeDataset


def test_sample_shapes_and_keys():
    ds = SyntheticGazeDataset(n_samples=50, image_size=(36, 60), seed=1)
    assert len(ds) == 50
    item = ds[0]
    assert item["features"].shape == (FEATURE_DIM,)
    assert item["image"].shape == (1, 36, 60)
    assert item["gaze"].shape == (2,)


def test_gaze_within_range_and_images_normalized():
    ds = SyntheticGazeDataset(n_samples=100, seed=2)
    assert ds.gaze.abs().max() <= GAZE_RANGE + 1e-6
    assert float(ds.images.min()) >= 0.0
    assert float(ds.images.max()) <= 1.0


def test_deterministic_with_seed():
    a = SyntheticGazeDataset(n_samples=20, seed=3)
    b = SyntheticGazeDataset(n_samples=20, seed=3)
    assert torch.equal(a.gaze, b.gaze)
    assert torch.equal(a.images, b.images)
