"""Compare exported ONNX predictions with the PyTorch reference model.

The validation runs both inference backends on the same validation images
and checks:
- top detection class agreement
- bounding-box IoU
- confidence-score differences
- average number of detections

The ONNX model does not need to be bit-identical to PyTorch, but its
predictions should remain logically consistent.

Output:
    outputs/evaluation/final/onnx_consistency.json

Usage:
    python training\\validate_onnx.py [--images 40]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import IMAGES_VAL_DIR, MODELS_DIR, PROJECT_ROOT  # noqa: E402


def calculate_iou(box_a, box_b) -> float:
    """Calculate IoU between two XYXY bounding boxes."""
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    intersection_width = max(0.0, x_right - x_left)
    intersection_height = max(0.0, y_bottom - y_top)
    intersection_area = intersection_width * intersection_height

    if intersection_area <= 0:
        return 0.0

    area_a = (
        (box_a[2] - box_a[0])
        * (box_a[3] - box_a[1])
    )
    area_b = (
        (box_b[2] - box_b[0])
        * (box_b[3] - box_b[1])
    )

    union_area = area_a + area_b - intersection_area

    return intersection_area / (union_area + 1e-9)


def normalize_detections(detections):
    """Convert detector results into comparable tuples."""
    normalized = []

    for detection in detections:
        bbox = detection["bbox"]

        normalized.append(
            (
                detection["class"],
                detection["confidence"],
                [
                    bbox["x1"],
                    bbox["y1"],
                    bbox["x2"],
                    bbox["y2"],
                ],
            )
        )

    return normalized


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="PyTorch vs ONNX prediction consistency check"
    )

    parser.add_argument(
        "--images",
        type=int,
        default=40,
        help="Maximum number of validation images to compare",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Detection confidence threshold",
    )

    return parser.parse_args()


def load_image(path: Path):
    """Read an image safely using OpenCV."""
    image = cv2.imdecode(
        np.fromfile(str(path), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    return image


def select_validation_images(max_images: int) -> list[Path]:
    """Select validation images using deterministic sampling."""
    images = sorted(IMAGES_VAL_DIR.glob("*.jpg"))

    if not max_images:
        return images

    sampling_step = max(1, len(images) // max_images)

    return images[::sampling_step][:max_images]


def load_image_size() -> int | None:
    """Read the trained model image size from metadata."""
    metadata_path = MODELS_DIR / "best_metadata.json"

    if not metadata_path.exists():
        return None

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    return int(metadata.get("image_size", 320))


def build_detectors(confidence: float, image_size: int | None):
    """Initialize and warm up both inference backends."""
    import torch

    from app.inference.detector import Detector

    # Preserve the original CUDA/CPU selection.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    detector_pt = Detector(
        backend="pytorch",
        conf_threshold=confidence,
        imgsz=image_size,
    )
    detector_pt.warmup()

    detector_onnx = Detector(
        backend="onnx",
        conf_threshold=confidence,
        imgsz=image_size,
    )
    detector_onnx.warmup()

    return detector_pt, detector_onnx, device


def compare_predictions(detector_pt, detector_onnx, images):
    """Compare PyTorch and ONNX predictions across validation images."""
    total_images = 0
    class_agreements = 0
    both_empty = 0

    ious: list[float] = []
    confidence_deltas: list[float] = []

    pytorch_counts: list[int] = []
    onnx_counts: list[int] = []

    for image_path in images:
        image = load_image(image_path)

        if image is None:
            continue

        pytorch_detections = normalize_detections(
            detector_pt.detect_raw(image)
        )
        onnx_detections = normalize_detections(
            detector_onnx.detect_raw(image)
        )

        pytorch_detections.sort(
            key=lambda detection: -detection[1]
        )
        onnx_detections.sort(
            key=lambda detection: -detection[1]
        )

        total_images += 1

        pytorch_counts.append(len(pytorch_detections))
        onnx_counts.append(len(onnx_detections))

        if not pytorch_detections and not onnx_detections:
            both_empty += 1
            class_agreements += 1
            continue

        top_pytorch = (
            pytorch_detections[0]
            if pytorch_detections
            else None
        )

        top_onnx = (
            onnx_detections[0]
            if onnx_detections
            else None
        )

        if (
            top_pytorch
            and top_onnx
            and top_pytorch[0] == top_onnx[0]
        ):
            class_agreements += 1

            ious.append(
                calculate_iou(
                    top_pytorch[2],
                    top_onnx[2],
                )
            )

            confidence_deltas.append(
                abs(
                    top_pytorch[1]
                    - top_onnx[1]
                )
            )

    return {
        "images": total_images,
        "class_agreements": class_agreements,
        "both_empty": both_empty,
        "ious": ious,
        "confidence_deltas": confidence_deltas,
        "pytorch_counts": pytorch_counts,
        "onnx_counts": onnx_counts,
    }


def create_result(stats: dict) -> dict:
    """Build the final ONNX consistency report."""
    image_count = stats["images"]

    agreement = (
        stats["class_agreements"] / image_count
        if image_count
        else 0.0
    )

    median_iou = (
        float(np.median(stats["ious"]))
        if stats["ious"]
        else None
    )

    mean_confidence_delta = (
        float(np.mean(stats["confidence_deltas"]))
        if stats["confidence_deltas"]
        else None
    )

    mean_pytorch_detections = (
        sum(stats["pytorch_counts"]) / image_count
        if image_count
        else 0
    )

    mean_onnx_detections = (
        sum(stats["onnx_counts"]) / image_count
        if image_count
        else 0
    )

    is_consistent = (
        agreement >= 0.9
        and (
            median_iou is None
            or median_iou >= 0.8
        )
    )

    return {
        "images_compared": image_count,
        "top_class_agreement": round(agreement, 4),
        "images_both_empty": stats["both_empty"],
        "median_top_box_iou": (
            round(median_iou, 4)
            if median_iou is not None
            else None
        ),
        "mean_top_confidence_delta": (
            round(mean_confidence_delta, 4)
            if mean_confidence_delta is not None
            else None
        ),
        "mean_detections_pt": round(
            mean_pytorch_detections,
            2,
        ),
        "mean_detections_onnx": round(
            mean_onnx_detections,
            2,
        ),
        "verdict": (
            "CONSISTENT"
            if is_consistent
            else "REVIEW NEEDED"
        ),
    }


def save_report(result: dict) -> Path:
    """Save the consistency report to the final evaluation directory."""
    output_path = (
        PROJECT_ROOT
        / "outputs"
        / "evaluation"
        / "final"
        / "onnx_consistency.json"
    )

    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    return output_path


def print_report(result: dict, output_path: Path) -> None:
    """Display the final consistency report."""
    print("=" * 62)
    print("PYTORCH vs ONNX CONSISTENCY")
    print("=" * 62)

    for key, value in result.items():
        print(f"  {key:<28} {value}")

    print(f"\nSaved: {output_path}")


def main() -> int:
    """Run the complete PyTorch vs ONNX consistency validation."""
    args = parse_arguments()

    pytorch_model = MODELS_DIR / "best.pt"
    onnx_model = MODELS_DIR / "best.onnx"

    if not pytorch_model.exists() or not onnx_model.exists():
        print(
            "both models/best.pt and models/best.onnx are required"
        )
        return 1

    image_size = load_image_size()

    detector_pt, detector_onnx, _ = build_detectors(
        args.conf,
        image_size,
    )

    validation_images = select_validation_images(
        args.images
    )

    statistics = compare_predictions(
        detector_pt,
        detector_onnx,
        validation_images,
    )

    result = create_result(statistics)

    output_path = save_report(result)

    print_report(result, output_path)

    return (
        0
        if result["verdict"] == "CONSISTENT"
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
