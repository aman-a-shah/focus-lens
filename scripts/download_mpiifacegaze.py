#!/usr/bin/env python3
"""Fetch + normalize MPIIFaceGaze for the gaze-regression pipeline (roadmap Phase 4).

MPIIFaceGaze requires accepting a usage agreement, so it cannot be auto-downloaded. This
script prints the steps, and — once you've placed the raw archive — normalizes it into the
``checkpoints/mpiifacegaze.npz`` that ``focuslens train-gaze --data`` consumes.

Usage:
    python scripts/download_mpiifacegaze.py            # print instructions
    python scripts/download_mpiifacegaze.py --raw PATH # normalize a downloaded archive/dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DATASET_URL = "https://www.perceptualui.org/research/datasets/MPIIFaceGaze/"
OUT = Path(__file__).resolve().parent.parent / "checkpoints" / "mpiifacegaze.npz"

INSTRUCTIONS = f"""
MPIIFaceGaze setup
==================
1. Open {DATASET_URL} and accept the usage agreement.
2. Download "MPIIFaceGaze_normalized" (the normalized HDF5/mat release is easiest).
3. Re-run this script pointing at the extracted folder:

       python scripts/download_mpiifacegaze.py --raw /path/to/MPIIFaceGaze_normalized

   It will write a normalized bundle to:
       {OUT}

4. Train on it:
       focuslens train-gaze --arch cnn --data {OUT}

Until then, the pipeline runs on the built-in synthetic dataset:
       focuslens train-gaze --arch mlp
"""


def normalize(raw: Path) -> None:
    """Convert the raw normalized release into a single .npz of image/features/gaze.

    Left as an explicit, dataset-version-specific step: the public MPIIFaceGaze release ships
    per-subject .mat/HDF5 files whose exact keys vary by mirror. Implement the read here once
    you have the files in hand (the expected output arrays are documented in
    focuslens/gaze/dataset.py::MPIIFaceGazeDataset).
    """
    raise NotImplementedError(
        f"Point this at your downloaded MPIIFaceGaze and implement the per-subject read for "
        f"your release layout, writing image/features/gaze arrays to {OUT}. "
        f"Raw path given: {raw}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=None, help="Path to the extracted MPIIFaceGaze release")
    args = parser.parse_args()

    if args.raw is None:
        print(INSTRUCTIONS)
        return 0
    normalize(Path(args.raw))
    return 0


if __name__ == "__main__":
    sys.exit(main())
