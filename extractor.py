import subprocess
import shutil
from pathlib import Path


def extract_frames(video_path: str, output_dir: str, fps: int) -> list[str]:
    """Extract frames from video using FFmpeg at the given FPS.

    Returns a sorted list of extracted frame file paths.
    Raises FileNotFoundError if video_path does not exist.
    Raises RuntimeError if FFmpeg returns a non-zero exit code.
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if Path(output_dir).exists():
        shutil.rmtree(output_dir)
    Path(output_dir).mkdir(parents=True)
    output_pattern = str(Path(output_dir) / "frame_%04d.jpg")

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={fps}",
        output_pattern,
        "-y",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{result.stderr}")

    frame_paths = sorted(str(p) for p in Path(output_dir).glob("frame_*.jpg"))
    return frame_paths
