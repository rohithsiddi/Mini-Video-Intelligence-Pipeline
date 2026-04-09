import pytest
import json
from unittest.mock import patch, MagicMock
from pathlib import Path


@pytest.fixture
def config_file(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "video_path: sample.mp4\n"
        "fps: 1\n"
        "confidence_threshold: 0.7\n"
        "model: facebook/detr-resnet-50\n"
        "output_dir: output\n"
        "visualization:\n"
        "  save_annotated_frames: true\n"
        "  generate_annotated_video: true\n"
    )
    return str(config)


def test_load_config_returns_expected_keys(config_file):
    from pipeline import load_config
    cfg = load_config(config_file)
    assert cfg["fps"] == 1
    assert cfg["confidence_threshold"] == 0.7
    assert cfg["model"] == "facebook/detr-resnet-50"
    assert "visualization" in cfg


def test_load_config_raises_on_missing_file():
    from pipeline import load_config
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config("nonexistent.yaml")


def test_run_pipeline_writes_detections_json(tmp_path, config_file):
    import yaml
    cfg = yaml.safe_load(open(config_file))
    cfg["output_dir"] = str(tmp_path / "output")
    cfg["video_path"] = str(tmp_path / "video.mp4")
    (tmp_path / "video.mp4").touch()
    updated_config = tmp_path / "config.yaml"
    updated_config.write_text(yaml.dump(cfg))

    fake_frames = [str(tmp_path / f"frame_{i:04d}.jpg") for i in range(1, 3)]
    fake_detections = [{"label": "person", "confidence": 0.92, "bbox": [10, 20, 50, 80]}]

    with patch("pipeline.extract_frames", return_value=fake_frames), \
         patch("pipeline.load_detector", return_value=MagicMock()), \
         patch("pipeline.detect_frame", return_value=fake_detections), \
         patch("pipeline.annotate_frame"), \
         patch("pipeline.assemble_video"):
        from pipeline import run_pipeline
        run_pipeline(str(updated_config))

    json_path = tmp_path / "output" / "detections.json"
    assert json_path.exists()
    data = json.loads(json_path.read_text())
    assert data["fps"] == 1
    assert data["model"] == "facebook/detr-resnet-50"
    assert "total_frames" in data
    assert len(data["frames"]) == 2
    frame = data["frames"][0]
    assert frame["frame"] == 1
    assert "timestamp_sec" in frame
    assert "detections" in frame


def test_run_pipeline_json_includes_empty_detection_frames(tmp_path, config_file):
    import yaml
    cfg = yaml.safe_load(open(config_file))
    cfg["output_dir"] = str(tmp_path / "output")
    cfg["video_path"] = str(tmp_path / "video.mp4")
    cfg["visualization"]["generate_annotated_video"] = False
    (tmp_path / "video.mp4").touch()
    updated_config = tmp_path / "config.yaml"
    updated_config.write_text(yaml.dump(cfg))

    fake_frames = [str(tmp_path / "frame_0001.jpg")]

    with patch("pipeline.extract_frames", return_value=fake_frames), \
         patch("pipeline.load_detector", return_value=MagicMock()), \
         patch("pipeline.detect_frame", return_value=[]), \
         patch("pipeline.annotate_frame"), \
         patch("pipeline.assemble_video"):
        from pipeline import run_pipeline
        run_pipeline(str(updated_config))

    data = json.loads((tmp_path / "output" / "detections.json").read_text())
    assert data["frames"][0]["detections"] == []
