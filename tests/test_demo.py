"""Public demo inference (roadmap Phase 11)."""

import math
from pathlib import Path

import numpy as np
import pytest

from focuslens.demo.app import _to_bgr
from focuslens.demo.inference import DemoResult, _frame_to_window
from focuslens.features import FrameFeatures
from focuslens.states import DistractionState

PORTRAIT = Path(__file__).resolve().parent.parent / "checkpoints" / "sample_portrait.jpg"


def _frame(face: bool, **kw) -> FrameFeatures:
    nan = float("nan")
    base = dict(
        timestamp=0.0,
        face_present=face,
        ear_right=nan,
        ear_left=nan,
        ear_mean=nan,
        eye_closed=False,
        blinks_per_min=nan,
        last_blink_duration_s=nan,
        yaw=nan,
        pitch=nan,
        roll=nan,
        gaze_x=nan,
        gaze_y=nan,
    )
    base.update(kw)
    return FrameFeatures(**base)


def test_frame_to_window_replaces_nan_and_sets_face_fraction():
    w = _frame_to_window(_frame(False))
    assert w.face_fraction == 0.0
    assert not math.isnan(w.gaze_x) and w.gaze_x == 0.0  # NaN -> 0
    present = _frame_to_window(_frame(True, ear_mean=0.27, gaze_x=0.3))
    assert present.face_fraction == 1.0 and present.gaze_x == 0.3


def test_to_bgr_flips_channels():
    rgb = np.dstack([np.full((2, 2), 10), np.full((2, 2), 20), np.full((2, 2), 30)]).astype(
        np.uint8
    )
    bgr = _to_bgr(rgb)
    assert bgr[0, 0, 0] == 30 and bgr[0, 0, 2] == 10  # R<->B swapped


def test_caption_handles_no_face():
    res = DemoResult(
        face_present=False,
        state=DistractionState.FOCUSED,
        features=_frame(False),
        annotated=np.zeros((4, 4, 3), np.uint8),
    )
    assert "No face" in res.caption()


@pytest.mark.skipif(not PORTRAIT.exists(), reason="sample portrait not cached")
def test_analyzer_on_real_face():
    import cv2

    from focuslens.demo.inference import Analyzer

    image = cv2.imread(str(PORTRAIT))
    with Analyzer() as analyzer:
        result = analyzer.analyze(image)
    assert result.face_present is True
    assert result.state in set(DistractionState)
    assert result.annotated.shape == image.shape
    assert "State:" in result.caption()
