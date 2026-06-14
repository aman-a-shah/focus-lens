"""Gaze datasets (roadmap Phase 4).

``SyntheticGazeDataset`` generates a fully self-contained dataset so the training pipeline is
runnable and testable without the gated MPIIFaceGaze download. Each sample carries both
representations the two architectures need:

- ``features`` — [iris_dx, iris_dy, head_yaw, head_pitch, head_roll]  (for MLPGazeNet)
- ``image``    — a 1×H×W eye crop with a dark pupil whose position encodes gaze (for the CNN)
- ``gaze``     — the (yaw, pitch) label in radians

Both representations are deterministic functions of the same latent gaze + head pose (plus
noise), so a model that inverts them recovers the gaze — exactly the structure the real
pipeline exploits. ``MPIIFaceGazeDataset`` loads the real normalized data when present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

GAZE_RANGE = 0.6  # radians (~34°), comparable to MPIIFaceGaze
FEATURE_DIM = 5


class SyntheticGazeDataset(Dataset):
    def __init__(
        self,
        n_samples: int = 4000,
        image_size: tuple[int, int] = (36, 60),
        noise: float = 0.02,
        seed: int = 0,
    ) -> None:
        rng = np.random.RandomState(seed)
        h, w = image_size
        self.image_size = image_size

        gaze = rng.uniform(-GAZE_RANGE, GAZE_RANGE, size=(n_samples, 2)).astype(np.float32)
        head = rng.uniform(-0.3, 0.3, size=(n_samples, 3)).astype(np.float32)
        gaze_yaw, gaze_pitch = gaze[:, 0], gaze[:, 1]

        # Geometric features: iris offset depends on gaze, slightly on head pose, plus noise.
        iris_dx = 0.8 * gaze_yaw - 0.2 * head[:, 0] + rng.normal(0, noise, n_samples)
        iris_dy = 0.8 * gaze_pitch - 0.2 * head[:, 1] + rng.normal(0, noise, n_samples)
        features = np.stack([iris_dx, iris_dy, head[:, 0], head[:, 1], head[:, 2]], axis=1)

        # Eye-crop images: a dark pupil blob positioned by gaze (and a little by head yaw).
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        cx = (w / 2) + (gaze_yaw / GAZE_RANGE) * (w * 0.3) + head[:, 0] * 3.0
        cy = (h / 2) - (gaze_pitch / GAZE_RANGE) * (h * 0.3)
        radius = min(h, w) * 0.18
        dist2 = (xs[None] - cx[:, None, None]) ** 2 + (ys[None] - cy[:, None, None]) ** 2
        pupil = np.exp(-dist2 / (2 * radius**2))  # [N, H, W]
        images = 1.0 - 0.9 * pupil  # light sclera, dark pupil
        images += rng.normal(0, 0.01, images.shape)
        images = np.clip(images, 0.0, 1.0).astype(np.float32)

        self.features = torch.from_numpy(features.astype(np.float32))
        self.images = torch.from_numpy(images).unsqueeze(1)  # [N, 1, H, W]
        self.gaze = torch.from_numpy(gaze)

    def __len__(self) -> int:
        return self.gaze.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "features": self.features[idx],
            "image": self.images[idx],
            "gaze": self.gaze[idx],
        }


class MPIIFaceGazeDataset(Dataset):
    """Loader for pre-normalized MPIIFaceGaze (.npz with 'image'/'features'/'gaze' arrays).

    The raw dataset requires manual download + agreement; see
    ``scripts/download_mpiifacegaze.py``. This expects an already-normalized ``.npz`` produced
    by that script's preprocessing step.
    """

    def __init__(self, npz_path: str | Path) -> None:
        path = Path(npz_path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run scripts/download_mpiifacegaze.py first "
                "(MPIIFaceGaze requires manual registration)."
            )
        data = np.load(path)
        self.images = torch.from_numpy(data["image"].astype(np.float32))
        if self.images.ndim == 3:  # [N, H, W] -> [N, 1, H, W]
            self.images = self.images.unsqueeze(1)
        self.features = torch.from_numpy(data["features"].astype(np.float32))
        self.gaze = torch.from_numpy(data["gaze"].astype(np.float32))

    def __len__(self) -> int:
        return self.gaze.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "features": self.features[idx],
            "image": self.images[idx],
            "gaze": self.gaze[idx],
        }
