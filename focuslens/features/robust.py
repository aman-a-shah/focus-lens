"""Perception robustness utilities for the iris/gaze path (roadmap Phase 5, PRD §3.1 failure modes).

Two real-world failure modes degrade the iris signal the gaze head consumes:

- **Reflections** — glasses glare / specular highlights land *on top of* iris landmarks, pulling
  the estimated iris centre toward the bright spot. ``mask_specular`` / ``robust_iris_center``
  drop the blown-out points before averaging.
- **Low light** — under-exposed frames flatten the sclera/iris contrast the threshold above
  relies on. ``low_light_tta`` is a test-time augmentation that stretches contrast back before
  any thresholding or CNN inference, so the same specular threshold stays meaningful.

All three operate on an 8-bit grayscale eye crop (``uint8`` or float in [0, 255]); they are pure
NumPy so they unit-test without a camera.
"""

from __future__ import annotations

import numpy as np

SPECULAR_THRESHOLD = 240  # 8-bit intensity above which a pixel reads as a reflection
_EPS = 1e-6


def low_light_tta(gray: np.ndarray, target_mean: float = 128.0) -> np.ndarray:
    """Gamma-correct a (possibly dark) grayscale crop so its mean lands near ``target_mean``.

    Picks the gamma that maps the crop's current mean intensity to ``target_mean`` — brightening
    under-exposed crops, leaving well-exposed ones roughly unchanged (gamma ≈ 1). A flat crop
    (no contrast) maps to the target directly. Returns float32 in [0, 255].
    """
    g = gray.astype(np.float32)
    if float(g.max()) - float(g.min()) < _EPS:
        return np.full_like(g, target_mean)
    norm = g / 255.0  # -> [0, 1]
    mean = float(norm.mean())
    if mean <= _EPS:
        return np.full_like(g, target_mean)
    # gamma so that mean ** gamma == target_mean/255 (brighten when dark, darken when too bright)
    gamma = float(np.clip(np.log(target_mean / 255.0) / np.log(mean), 0.2, 5.0))
    return (norm**gamma * 255.0).astype(np.float32)


def mask_specular(intensities: np.ndarray, threshold: int = SPECULAR_THRESHOLD) -> np.ndarray:
    """Boolean keep-mask: True for pixels that are *not* specular highlights."""
    return np.asarray(intensities) <= threshold


def robust_iris_center(
    points_px: np.ndarray,
    iris_idx: list[int],
    gray: np.ndarray | None = None,
    threshold: int = SPECULAR_THRESHOLD,
) -> np.ndarray:
    """Iris centre as the mean of its landmark points, dropping points hit by reflections.

    Without ``gray`` this is just the landmark centroid (the Phase-2 behaviour). With a grayscale
    frame it samples intensity at each iris landmark, masks out specular points, and averages the
    survivors — falling back to the plain centroid if a reflection swallows the whole iris.
    """
    pts = points_px[iris_idx]
    if gray is None:
        return pts.mean(axis=0)
    h, w = gray.shape[:2]
    xi = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    keep = mask_specular(gray[yi, xi], threshold)
    if keep.sum() < 1:
        return pts.mean(axis=0)
    return pts[keep].mean(axis=0)
