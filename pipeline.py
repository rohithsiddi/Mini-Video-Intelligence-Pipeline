import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from extractor import extract_frames
from detector import load_detector, detect_frame
from visualizer import annotate_frame, assemble_video


def load_config(config_path: str) -> dict:
    """Load YAML config file. Raises FileNotFoundError if missing."""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_pipeline(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)

    video_path = cfg["video_path"]
    fps = cfg["fps"]
    model_name = cfg["model"]
    confidence_threshold = cfg["confidence_threshold"]
    output_dir = Path(cfg["output_dir"])
    viz = cfg.get("visualization", {})

    frames_dir = output_dir / "frames"
    annotated_dir = output_dir / "annotated"
    output_dir.mkdir(parents=True, exist_ok=True)

    for d in (frames_dir, annotated_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir()

    print(f"[1/4] Extracting frames from '{video_path}' at {fps} FPS...")
    frame_paths = extract_frames(video_path, str(frames_dir), fps=fps)
    print(f"      Extracted {len(frame_paths)} frames.")

    print(f"[2/4] Loading detector: {model_name}")
    detector = load_detector(model_name)

    print(f"[3/4] Running detection on {len(frame_paths)} frames...")
    frames_output = []
    annotation_tasks = []
    for idx, frame_path in enumerate(frame_paths, start=1):
        detections = detect_frame(detector, frame_path, confidence_threshold)
        timestamp = round(idx / fps, 2)
        frames_output.append({
            "frame": idx,
            "timestamp_sec": timestamp,
            "detections": detections,
        })
        print(f"      Frame {idx}: {len(detections)} detection(s)")

        if viz.get("save_annotated_frames", True):
            out_name = Path(frame_path).name
            annotation_tasks.append((frame_path, detections, str(annotated_dir / out_name)))

    if annotation_tasks:
        with ThreadPoolExecutor() as executor:
            executor.map(lambda args: annotate_frame(*args), annotation_tasks)

    result = {
        "video": video_path,
        "fps": fps,
        "model": model_name,
        "total_frames": len(frame_paths),
        "frames": frames_output,
    }

    json_path = output_dir / "detections.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[4/4] Detections saved to '{json_path}'")

    if viz.get("generate_annotated_video", True):
        video_out = str(output_dir / "annotated_video.mp4")
        assemble_video(str(annotated_dir), video_out, fps=fps)
        print(f"      Annotated video saved to '{video_out}'")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    run_pipeline(config_path)
