import pytest

from focuslens.runtime import BenchResult, _FpsMeter


def test_fps_meter_needs_two_samples():
    meter = _FpsMeter()
    assert meter.tick(0.0) == 0.0  # single sample -> unknown


def test_fps_meter_computes_rate_over_window():
    meter = _FpsMeter(window=10)
    # 5 frames evenly spaced 0.05s apart => 20 fps
    fps = 0.0
    for i in range(5):
        fps = meter.tick(i * 0.05)
    assert fps == pytest.approx(20.0, rel=1e-6)


def test_fps_meter_handles_duplicate_timestamp():
    meter = _FpsMeter()
    meter.tick(1.0)
    assert meter.tick(1.0) == 0.0  # zero span -> guarded, no ZeroDivisionError


def test_bench_result_summary_mentions_key_stats():
    r = BenchResult(
        frames=120, faces_found=120, mean_fps=42.0, p50_latency_ms=23.0, p95_latency_ms=30.0
    )
    s = r.summary()
    assert "120 frames" in s
    assert "42.0 FPS" in s
    assert "p50" in s and "p95" in s
