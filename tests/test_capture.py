import pytest

from focuslens.capture import WebcamCapture


def test_is_camera_distinguishes_int_from_path():
    assert WebcamCapture(source=0).is_camera is True
    assert WebcamCapture(source="clip.mp4").is_camera is False


def test_missing_video_file_raises_clear_error():
    with (
        pytest.raises(RuntimeError, match="Could not open video source"),
        WebcamCapture(source="/no/such/file.mp4"),
    ):
        pass
