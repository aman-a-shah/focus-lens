import math
from pathlib import Path

import pytest

from focuslens.features import FeatureExtractor, FrameFeatures

PORTRAIT = Path(__file__).resolve().parent.parent / "checkpoints" / "sample_portrait.jpg"


def test_header_matches_row_width():
    f = FrameFeatures(
        timestamp=0.0,
        face_present=False,
        ear_right=0.0,
        ear_left=0.0,
        ear_mean=0.0,
        eye_closed=False,
        blinks_per_min=0.0,
        last_blink_duration_s=0.0,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        gaze_x=0.0,
        gaze_y=0.0,
    )
    assert len(f.header()) == len(f.to_row())
    assert "ear_mean" in f.header() and "yaw" in f.header()


def test_no_face_yields_nan_features():
    extractor = FeatureExtractor()
    feats = extractor.extract(None, (720, 1280), timestamp=1.0)
    assert feats.face_present is False
    assert math.isnan(feats.ear_mean)
    assert math.isnan(feats.yaw)
    assert "no face" in feats.console_line()


@pytest.mark.skipif(not PORTRAIT.exists(), reason="sample portrait not cached")
def test_extracts_plausible_features_on_real_face():
    import cv2

    from focuslens.config import load_config
    from focuslens.face_mesh import FaceMeshTracker

    image = cv2.imread(str(PORTRAIT))
    cfg = load_config()
    with FaceMeshTracker(cfg.face_mesh) as tracker:
        result = tracker.process(image, timestamp_ms=0)
    feats = FeatureExtractor().extract(result, image.shape, timestamp=0.0)

    assert feats.face_present is True
    assert 0.05 < feats.ear_mean < 0.6  # plausible open-eye EAR
    assert all(math.isfinite(v) for v in (feats.yaw, feats.pitch, feats.roll))
    assert all(math.isfinite(v) for v in (feats.gaze_x, feats.gaze_y))
