import pytest

from focuslens.features.blink import BlinkDetector


def test_single_blink_is_detected_with_duration():
    det = BlinkDetector(ear_close=0.21, ear_open=0.24)
    assert det.update(0.30, 0.00).eye_closed is False  # open
    assert det.update(0.10, 0.10).eye_closed is True  # closes at t=0.10
    det.update(0.10, 0.20)  # still closed
    final = det.update(0.30, 0.35)  # reopens at t=0.35
    assert final.blink_completed is True
    assert final.eye_closed is False
    assert det.last_blink_duration_s == pytest.approx(0.25)


def test_hysteresis_band_does_not_toggle():
    det = BlinkDetector(ear_close=0.21, ear_open=0.24)
    # EAR sitting in the dead band between the two thresholds: never blinks
    for t in range(5):
        state = det.update(0.225, float(t))
        assert state.eye_closed is False
        assert state.blink_completed is False


def test_blink_rate_counts_within_window():
    det = BlinkDetector(rate_window_s=60.0)
    t = 0.0
    for _ in range(3):  # three blinks within the window
        det.update(0.10, t + 0.1)
        last = det.update(0.30, t + 0.2)
        t += 1.0
    assert last.blinks_per_min == pytest.approx(3.0)


def test_old_blinks_fall_out_of_window():
    det = BlinkDetector(rate_window_s=10.0)
    det.update(0.10, 0.1)
    det.update(0.30, 0.2)  # one blink at t~0.2
    assert det.blinks_per_min(5.0) > 0
    assert det.blinks_per_min(100.0) == 0.0  # aged out


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError):
        BlinkDetector(ear_close=0.3, ear_open=0.2)
