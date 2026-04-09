from PIL import Image
from rfdetr import RFDETRMedium
from rfdetr.assets.coco_classes import COCO_CLASSES


def load_detector(model_name: str):
    """Load and return an RF-DETR detector. model_name is accepted for interface compatibility."""
    return RFDETRMedium()


def detect_frame(detector, frame_path: str, confidence_threshold: float) -> list[dict]:
    """Run object detection on a single frame.

    Returns a list of detections above confidence_threshold.
    Each detection: {"label": str, "confidence": float, "bbox": [x1, y1, x2, y2]}
    """
    image = Image.open(frame_path).convert("RGB")
    detections = detector.predict(image, threshold=confidence_threshold)

    results = []
    for i in range(len(detections.xyxy)):
        x1, y1, x2, y2 = detections.xyxy[i]
        class_id = int(detections.class_id[i])
        label = COCO_CLASSES.get(class_id, f"class_{class_id}")
        results.append({
            "label": label,
            "confidence": round(float(detections.confidence[i]), 2),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
        })
    return results
