import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np
import cv2


@pytest.fixture
def test_image(tmp_path):
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    path = str(tmp_path / "frame_0001.jpg")
    cv2.imwrite(path, img)
    return path


SAMPLE_DETECTIONS = [
    {"label": "person", "confidence": 0.92, "bbox": [10, 20, 100, 180]},
]


def test_annotate_frame_creates_output_file(test_image, tmp_path):
    out_path = str(tmp_path / "annotated_0001.jpg")
    from visualizer import annotate_frame
    annotate_frame(test_image, SAMPLE_DETECTIONS, out_path)
    assert Path(out_path).exists()


def test_annotate_frame_with_empty_detections(test_image, tmp_path):
    out_path = str(tmp_path / "annotated_0001.jpg")
    from visualizer import annotate_frame
    annotate_frame(test_image, [], out_path)
    assert Path(out_path).exists()


def test_assemble_video_calls_ffmpeg(tmp_path):
    annotated_dir = tmp_path / "annotated"
    annotated_dir.mkdir()
    output_path = str(tmp_path / "output.mp4")

    with patch("visualizer.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        from visualizer import assemble_video
        assemble_video(str(annotated_dir), output_path, fps=1)

    args = mock_run.call_args[0][0]
    assert args[0] == "ffmpeg"
    assert output_path in args


def test_assemble_video_raises_on_ffmpeg_failure(tmp_path):
    annotated_dir = tmp_path / "annotated"
    annotated_dir.mkdir()

    with patch("visualizer.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="error")
        from visualizer import assemble_video
        with pytest.raises(RuntimeError, match="FFmpeg video assembly failed"):
            assemble_video(str(annotated_dir), str(tmp_path / "out.mp4"), fps=1)
