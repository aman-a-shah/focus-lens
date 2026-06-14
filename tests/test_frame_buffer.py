import pytest

from focuslens.capture import Frame, FrameBuffer


def _frame(t: float) -> Frame:
    return Frame(timestamp=t, image=object())


def test_capacity_derived_from_seconds_and_fps():
    buf = FrameBuffer(seconds=5.0, fps=30)
    assert buf.capacity == 150


def test_rounds_and_floors_to_at_least_one():
    assert FrameBuffer(seconds=0.01, fps=1).capacity == 1


def test_evicts_oldest_when_full():
    buf = FrameBuffer(seconds=1.0, fps=3)  # capacity 3
    for t in range(5):
        buf.append(_frame(float(t)))
    assert len(buf) == 3
    assert buf.is_full
    timestamps = [f.timestamp for f in buf]
    assert timestamps == [2.0, 3.0, 4.0]


def test_latest_returns_most_recent_or_none():
    buf = FrameBuffer(seconds=1.0, fps=30)
    assert buf.latest() is None
    buf.append(_frame(1.0))
    buf.append(_frame(2.0))
    assert buf.latest().timestamp == 2.0


@pytest.mark.parametrize("seconds,fps", [(0, 30), (5, 0), (-1, 30)])
def test_rejects_nonpositive_args(seconds, fps):
    with pytest.raises(ValueError):
        FrameBuffer(seconds=seconds, fps=fps)
