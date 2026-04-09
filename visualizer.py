import subprocess
from pathlib import Path
import cv2


def annotate_frame(frame_path: str, detections: list[dict], output_path: str) -> None:
    """Draw bounding boxes and labels on frame, save to output_path."""
    img = cv2.imread(frame_path)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        confidence = det["confidence"]

        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        text = f"{label} {confidence:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - text_h - 4), (x1 + text_w, y1), (0, 255, 0), -1)
        cv2.putText(img, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, img)


def assemble_video(annotated_dir: str, output_path: str, fps: int) -> None:
    """Assemble annotated frames into a video using FFmpeg.

    Input frames are at the extracted FPS (e.g. 1). Output is upsampled to
    30fps so the video plays smoothly, with each annotated frame held for
    its full duration.
    """
    input_pattern = str(Path(annotated_dir) / "frame_%04d.jpg")

    cmd = [
        "ffmpeg",
        "-framerate", str(fps),
        "-i", input_pattern,
        "-vf", "fps=30",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_path,
        "-y",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg video assembly failed:\n{result.stderr}")
