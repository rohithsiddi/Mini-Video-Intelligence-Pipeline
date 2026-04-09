# Mini Video Intelligence Pipeline

A simple pipeline that extracts frames from a video using FFmpeg, runs object detection using RF-DETR (Roboflow, ICLR 2026), and produces structured JSON output with annotated visualizations.

## Prerequisites

- Python 3.12+
- FFmpeg installed and on PATH (`brew install ffmpeg` on macOS)

## Setup

**Option 1: uv (recommended, faster)**
```bash
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

**Option 2: pip**
```bash
pip install -r requirements.txt
```

## Usage

1. Place your MP4 video in the project root as `sample_video.mp4`
2. (Optional) Edit `config.yaml` to adjust FPS, model, or confidence threshold
3. Run the pipeline:

```bash
python pipeline.py
# or with a custom config:
python pipeline.py config.yaml
```

## Output

After running, the `output/` directory contains:
- `frames/`: raw extracted frames (JPEGs)
- `annotated/`: frames with bounding boxes drawn
- `annotated_video.mp4`: assembled annotated video
- `detections.json`: structured detection results

### Example `detections.json`

```json
{
  "video": "sample_video.mp4",
  "fps": 1,
  "model": "rfdetr-medium",
  "total_frames": 4,
  "frames": [
    {
      "frame": 1,
      "timestamp_sec": 1.0,
      "detections": [
        { "label": "person", "confidence": 0.92, "bbox": [120, 80, 340, 460] }
      ]
    }
  ]
}
```

## Configuration (`config.yaml`)

| Key | Default | Description |
|---|---|---|
| `video_path` | `sample_video.mp4` | Input video file |
| `fps` | `1` | Frames per second to extract |
| `confidence_threshold` | `0.7` | Minimum detection confidence |
| `model` | `rfdetr-medium` | Model identifier (RF-DETR medium) |
| `output_dir` | `output` | Output directory |
| `visualization.save_annotated_frames` | `true` | Save frames with bboxes |
| `visualization.generate_annotated_video` | `true` | Assemble annotated video |

## Bonus: Analysis

```bash
python analyze.py
```

Runs detection across FPS values (1, 2, 5) and bitrates (2000k, 500k, 100k), reporting frame count, detection count, and unique labels detected for each setting.

Sample output on the included video:

```
--- FPS Impact (original bitrate) ---
  FPS    Frames    Detections    Time (s)  Labels Detected
---------------------------------------------------------------------------
    1        11           117        2.59  handbag, person
    2        22           227        2.72  handbag, person
    5        55           555        6.23  handbag, person


--- Bitrate Impact (FPS=1) ---
   Bitrate    Frames    Detections    Time (s)  Labels Detected
-------------------------------------------------------------------------------
     2000k        11           118        1.45  handbag, person
      500k        11            98         1.4  backpack, handbag, person
      100k        11            19        1.56  person
```

A few observations from the results:
- At 100k bitrate, detections drop from ~118 to 19 and only `person` is detected. Handbags and backpacks are missed entirely due to compression artifacts degrading fine detail.
- RF-DETR correctly identifies handbags without confusing them with suitcases, which was a recurring misclassification with the previous model.
- At 500k bitrate, `backpack` appears which is absent at higher quality, likely because compression changes texture in a way that shifts the prediction for some bags.

## Running Tests

```bash
python -m pytest tests/ -v
```

## Approach

FFmpeg was new to me, so frame extraction was the learning curve. I used it via `subprocess` to extract JPEG frames at a configurable FPS, and again to reassemble annotated frames into a video at the end.

For detection I chose RF-DETR Medium from Roboflow, published at ICLR 2026. This was my first time using a Roboflow model. I have used YOLO and RCNN before, so DETR-based architectures were a new area to explore. RF-DETR combines a transformer decoder with a ResNet backbone using a multi-scale feature fusion approach. It sets state-of-the-art accuracy on COCO among real-time detectors while remaining fast enough for practical use.

OpenCV handles drawing bounding boxes on each frame. Annotation is done in parallel using `ThreadPoolExecutor` since it is purely I/O bound, which speeds things up when there are more frames.

All tunables (FPS, model, confidence threshold) are in `config.yaml` so the pipeline can be adjusted without touching code. A separate `analyze.py` script runs the pipeline across different FPS and bitrate settings to show how they affect detection results.

## Challenges

- **FFmpeg CLI**: First time using it directly. Getting the filter syntax right (`-vf fps=N`) and understanding the frame output pattern (`frame_%04d.jpg`) took some reading.
- **Detection count**: RF-DETR produces 8-13 detections per frame at a 0.7 confidence threshold. Some of these are overlapping boxes on the same object. YOLO handles this more aggressively with NMS, which is something to consider for a production use case where clean bounding boxes per object matter.
- **Missed detections**: In some frames handbags are not detected at all when they are partially occluded or at an angle. The model correctly distinguishes handbags from suitcases, but visibility-dependent misses are expected and would need calibration against labelled retail footage in production.
- **Bitrate degradation**: At 100k bitrate, detections dropped from ~118 to 19 and only persons were detected. Smaller objects like handbags and backpacks were missed entirely due to compression artifacts.

## Tradeoffs

- **RF-DETR vs YOLO**: RF-DETR was the more interesting choice to explore since it uses a transformer decoder rather than the anchor-based or anchor-free CNN approach YOLO uses. RF-DETR achieves higher mAP on COCO than comparable YOLO variants while still being real-time capable. For a short clip at 1 FPS this is well within its performance envelope, but for a high-throughput retail system YOLO would still be the practical choice due to lower per-frame latency.
- **FPS=1**: Keeps frame count low for a short video while still capturing enough variation. Higher FPS adds mostly redundant frames for slow-moving scenes.
- **Confidence threshold at 0.7**: A lower threshold catches more objects but increases false positives. 0.7 was a reasonable starting point given the results, but this would need calibration against a labelled dataset in production.
- **Annotated video via FFmpeg**: Chosen over OpenCV VideoWriter which has codec availability issues on macOS. FFmpeg reuses a tool already in the pipeline and gives reliable H.264 output.
