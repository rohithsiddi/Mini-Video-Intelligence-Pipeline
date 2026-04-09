"""
Bonus: Analyze the effect of FPS and bitrate on detection count.
Runs the pipeline across FPS values (1, 2, 5) and bitrates (2000k, 500k, 100k),
then prints comparison tables for each.
"""

import subprocess
import time
import tempfile
import yaml

from extractor import extract_frames
from detector import load_detector, detect_frame


def reencode_video(input_path: str, output_path: str, bitrate: str) -> None:
    """Re-encode video at a given bitrate using FFmpeg."""
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-b:v", bitrate,
        output_path,
        "-y",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg re-encode failed:\n{result.stderr}")


def count_detections(detector, video_path: str, fps: int, confidence_threshold: float) -> tuple[int, int, set, float]:
    """Extract frames and run detection. Returns (frame_count, detection_count, unique_labels, elapsed_sec)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        t0 = time.time()
        frame_paths = extract_frames(video_path, tmp_dir, fps=fps)
        total_detections = 0
        unique_labels = set()
        for frame_path in frame_paths:
            dets = detect_frame(detector, frame_path, confidence_threshold)
            total_detections += len(dets)
            unique_labels.update(d["label"] for d in dets)
        elapsed = round(time.time() - t0, 2)
        return len(frame_paths), total_detections, unique_labels, elapsed


def analyze_fps(detector, video_path: str, confidence_threshold: float) -> None:
    """Print detection results across FPS values (1, 2, 5) at original bitrate."""
    print("\n--- FPS Impact (original bitrate) ---")
    print(f"{'FPS':>5}  {'Frames':>8}  {'Detections':>12}  {'Time (s)':>10}  Labels Detected")
    print("-" * 75)
    for fps in [1, 2, 5]:
        frames, dets, labels, elapsed = count_detections(detector, video_path, fps, confidence_threshold)
        print(f"{fps:>5}  {frames:>8}  {dets:>12}  {elapsed:>10}  {', '.join(sorted(labels))}")
    print()


def analyze_bitrate(detector, video_path: str, confidence_threshold: float) -> None:
    """Re-encode video at different bitrates and compare detection counts at FPS=1."""
    bitrates = ["2000k", "500k", "100k"]
    print("\n--- Bitrate Impact (FPS=1) ---")
    print(f"{'Bitrate':>10}  {'Frames':>8}  {'Detections':>12}  {'Time (s)':>10}  Labels Detected")
    print("-" * 79)
    for bitrate in bitrates:
        with tempfile.TemporaryDirectory() as tmp_dir:
            reencoded = f"{tmp_dir}/reencoded.mp4"
            reencode_video(video_path, reencoded, bitrate)
            frames, dets, labels, elapsed = count_detections(detector, reencoded, fps=1, confidence_threshold=confidence_threshold)
            print(f"{bitrate:>10}  {frames:>8}  {dets:>12}  {elapsed:>10}  {', '.join(sorted(labels))}")
    print()


def analyze(config_path: str = "config.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    video_path = cfg["video_path"]
    model_name = cfg["model"]
    confidence_threshold = cfg["confidence_threshold"]

    print(f"Loading detector: {model_name}")
    detector = load_detector(model_name)

    analyze_fps(detector, video_path, confidence_threshold)
    analyze_bitrate(detector, video_path, confidence_threshold)


if __name__ == "__main__":
    analyze()
