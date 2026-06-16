"""Reflection-masking + low-light TTA (roadmap Phase 5, PRD §3.1 failure modes)."""

import numpy as np

from focuslens.features.robust import (
    SPECULAR_THRESHOLD,
    low_light_tta,
    mask_specular,
    robust_iris_center,
)


def test_low_light_tta_brightens_dark_crop():
    dark = np.full((10, 10), 20, dtype=np.uint8)
    dark[5, 5] = 40  # a little contrast so min != max
    out = low_light_tta(dark, target_mean=128.0)
    assert out.mean() > dark.mean()
    assert out.max() <= 255.0 + 1e-3 and out.min() >= 0.0


def test_low_light_tta_flat_crop_returns_target():
    flat = np.full((4, 4), 7, dtype=np.uint8)
    out = low_light_tta(flat, target_mean=128.0)
    assert np.allclose(out, 128.0)


def test_mask_specular_flags_bright_pixels():
    inten = np.array([10, 250, 128, 255])
    keep = mask_specular(inten, threshold=SPECULAR_THRESHOLD)
    assert keep.tolist() == [True, False, True, False]


def test_robust_iris_center_drops_glare_point():
    # Four iris points; the last sits on a specular highlight and should be dropped.
    pts = np.array([[10.0, 10.0], [12.0, 10.0], [10.0, 12.0], [40.0, 40.0]])
    gray = np.zeros((64, 64), dtype=np.uint8)
    gray[40, 40] = 255  # blow out the outlier point's pixel
    iris_idx = [0, 1, 2, 3]

    plain = robust_iris_center(pts, iris_idx, gray=None)
    masked = robust_iris_center(pts, iris_idx, gray=gray)
    # Masking pulls the centre back toward the tight cluster (away from the glare outlier).
    assert masked[0] < plain[0] and masked[1] < plain[1]
    assert np.allclose(masked, pts[:3].mean(axis=0))


def test_robust_iris_center_all_specular_falls_back():
    pts = np.array([[1.0, 1.0], [2.0, 2.0]])
    gray = np.full((8, 8), 255, dtype=np.uint8)
    out = robust_iris_center(pts, [0, 1], gray=gray)
    assert np.allclose(out, pts.mean(axis=0))  # fall back, don't return empty
