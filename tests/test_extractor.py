import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


def test_extract_frames_raises_if_video_not_found(tmp_path):
    from extractor import extract_frames
    with pytest.raises(FileNotFoundError, match="Video file not found"):
        extract_frames("nonexistent.mp4", str(tmp_path / "frames"), fps=1)


def test_extract_frames_calls_ffmpeg_with_correct_args(tmp_path):
    video = tmp_path / "video.mp4"
    video.touch()
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    # Pre-create fake frame files so the function has something to return
    for i in range(1, 4):
        (frames_dir / f"frame_{i:04d}.jpg").touch()

    with patch("extractor.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        from extractor import extract_frames
        extract_frames(str(video), str(frames_dir), fps=2)

    args = mock_run.call_args[0][0]
    assert args[0] == "ffmpeg"
    assert "-i" in args
    assert str(video) in args
    assert "fps=2" in " ".join(args)


def test_extract_frames_returns_sorted_jpg_paths(tmp_path):
    video = tmp_path / "video.mp4"
    video.touch()
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i in [3, 1, 2]:
        (frames_dir / f"frame_{i:04d}.jpg").touch()

    with patch("extractor.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        from extractor import extract_frames
        result = extract_frames(str(video), str(frames_dir), fps=1)

    assert result == sorted(result)
    assert all(p.endswith(".jpg") for p in result)


def test_extract_frames_raises_on_ffmpeg_failure(tmp_path):
    video = tmp_path / "video.mp4"
    video.touch()
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()

    with patch("extractor.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="error message")
        from extractor import extract_frames
        with pytest.raises(RuntimeError, match="FFmpeg failed"):
            extract_frames(str(video), str(frames_dir), fps=1)
