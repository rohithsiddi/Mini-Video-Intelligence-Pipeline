import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock, patch


@pytest.fixture
def test_image(tmp_path):
    """Create a small blank JPEG for testing."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    path = str(tmp_path / "frame.jpg")
    cv2.imwrite(path, img)
    return path


@pytest.fixture
def mock_detector():
    """RF-DETR detector mock.

    RF-DETR filters by threshold inside .predict(), so the mock simulates
    that: called at threshold=0.7 it returns only the high-confidence detection.
    """
    high_conf = MagicMock()
    high_conf.xyxy = np.array([[10, 20, 50, 80]], dtype=float)
    high_conf.class_id = np.array([1])
    high_conf.confidence = np.array([0.95])

    detector = MagicMock()
    detector.predict.return_value = high_conf
    return detector


def test_detect_frame_filters_below_confidence(test_image, mock_detector):
    from detector import detect_frame
    results = detect_frame(mock_detector, test_image, confidence_threshold=0.7)
    assert len(results) == 1
    assert results[0]["label"] == "person"


def test_detect_frame_output_shape(test_image, mock_detector):
    from detector import detect_frame
    results = detect_frame(mock_detector, test_image, confidence_threshold=0.7)
    item = results[0]
    assert set(item.keys()) == {"label", "confidence", "bbox"}
    assert isinstance(item["confidence"], float)
    assert len(item["bbox"]) == 4
    assert all(isinstance(v, int) for v in item["bbox"])


def test_detect_frame_bbox_order(test_image, mock_detector):
    from detector import detect_frame
    results = detect_frame(mock_detector, test_image, confidence_threshold=0.7)
    x1, y1, x2, y2 = results[0]["bbox"]
    assert [x1, y1, x2, y2] == [10, 20, 50, 80]


def test_detect_frame_returns_empty_list_when_nothing_above_threshold(test_image, mock_detector):
    from detector import detect_frame
    # Override mock to return empty detections
    empty = MagicMock()
    empty.xyxy = np.array([]).reshape(0, 4)
    empty.class_id = np.array([])
    empty.confidence = np.array([])
    mock_detector.predict.return_value = empty

    results = detect_frame(mock_detector, test_image, confidence_threshold=0.99)
    assert results == []


def test_load_detector_returns_model():
    with patch("detector.RFDETRMedium") as mock_cls:
        mock_cls.return_value = MagicMock()
        from detector import load_detector
        det = load_detector("rfdetr-medium")
        mock_cls.assert_called_once()
        assert det is mock_cls.return_value
