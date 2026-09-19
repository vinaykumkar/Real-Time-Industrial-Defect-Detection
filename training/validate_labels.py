"""Validate YOLO labels and generate a final class-distribution report.

Validates all converted YOLO label files for:
- class IDs 0..5
- normalized coordinates in [0, 1]
- positive bounding-box dimensions
- missing image/label pairs
- class distribution across train/validation splits
- multi-object image counts

The script exits with a non-zero status when validation errors are found.

Usage:
    python training\\validate_labels.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import DATA_YAML  # noqa: E402


def load_class_names() -> dict[int, str]:
    """Load and validate the configured YOLO class names."""
    config = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    class_names = {int(index): name for index, name in config["names"].items()}

    if len(class_names) != 6:
        raise AssertionError(
            f"expected 6 classes, got {len(class_names)}"
        )

    return class_names


def validate_label_file(
    label_path: Path,
    split_name: str,
    class_names: dict[int, str],
    errors: list[str],
    class_boxes: Counter,
    class_image_counts: Counter,
) -> tuple[int, int]:
    """Validate one YOLO label file and return box/image statistics."""
    box_count = 0
    classes_in_image: set[int] = set()

    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue

        fields = raw_line.split()

        if len(fields) != 5:
            errors.append(
                f"{split_name}/{label_path.name}:{line_number}: "
                "expected 5 fields"
            )
            continue

        try:
            class_id = int(fields[0])
            center_x, center_y, width, height = map(float, fields[1:])
        except ValueError:
            errors.append(
                f"{split_name}/{label_path.name}:{line_number}: "
                "invalid numeric value"
            )
            continue

        if class_id not in class_names:
            errors.append(
                f"{split_name}/{label_path.name}:{line_number}: "
                f"class id {class_id} not in 0-5"
            )
            continue

        if not (0.0 <= center_x <= 1.0 and 0.0 <= center_y <= 1.0):
            errors.append(
                f"{split_name}/{label_path.name}:{line_number}: "
                "center out of range"
            )

        if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            errors.append(
                f"{split_name}/{label_path.name}:{line_number}: "
                "size out of range"
            )

        class_boxes[class_id] += 1
        classes_in_image.add(class_id)
        box_count += 1

    for class_id in classes_in_image:
        class_image_counts[class_id] += 1

    return box_count, int(box_count > 1)


def validate_split(
    config: dict,
    split_name: str,
    class_names: dict[int, str],
    errors: list[str],
) -> dict:
    """Validate all images and labels belonging to one dataset split."""
    image_dir = Path(config["path"]) / f"images/{split_name}"
    label_dir = Path(config["path"]) / f"labels/{split_name}"

    images = sorted(image_dir.glob("*.jpg"))
    labels = sorted(label_dir.glob("*.txt"))

    image_stems = {image.stem for image in images}
    label_stems = {label.stem for label in labels}

    for stem in sorted(image_stems - label_stems):
        errors.append(
            f"{split_name}/{stem}: image without label file"
        )

    for stem in sorted(label_stems - image_stems):
        errors.append(
            f"{split_name}/{stem}: label without image"
        )

    class_boxes: Counter = Counter()
    class_image_counts: Counter = Counter()

    total_boxes = 0
    multi_object_images = 0

    for label_file in labels:
        boxes, is_multi_object = validate_label_file(
            label_file,
            split_name,
            class_names,
            errors,
            class_boxes,
            class_image_counts,
        )

        total_boxes += boxes
        multi_object_images += is_multi_object

    return {
        "images": len(images),
        "labels": len(labels),
        "boxes": total_boxes,
        "multi_object_images": multi_object_images,
        "class_boxes": class_boxes,
        "class_img_counts": class_image_counts,
    }


def print_report(
    class_names: dict[int, str],
    train_report: dict,
    val_report: dict,
) -> None:
    """Print the final validation and class-distribution report."""
    print("=" * 78)
    print("FINAL LABEL VALIDATION")
    print("=" * 78)

    print(
        f"{'Class':<18}"
        f"{'Train objects':>14}"
        f"{'Val objects':>13}"
        f"{'Train imgs':>12}"
        f"{'Val imgs':>10}"
    )

    for class_id, class_name in class_names.items():
        print(
            f"{class_name:<18}"
            f"{train_report['class_boxes'][class_id]:>14}"
            f"{val_report['class_boxes'][class_id]:>13}"
            f"{train_report['class_img_counts'][class_id]:>12}"
            f"{val_report['class_img_counts'][class_id]:>10}"
        )

    print(
        f"\n{'TOTALS':<18}"
        f"{train_report['boxes']:>14}"
        f"{val_report['boxes']:>13}"
        f"{train_report['images']:>12}"
        f"{val_report['images']:>10}"
    )

    print(
        f"{'multi-object imgs':<18}"
        f"{'':>14}"
        f"{'':>13}"
        f"{train_report['multi_object_images']:>12}"
        f"{val_report['multi_object_images']:>10}"
    )

    train_counts = train_report["class_boxes"]

    maximum_class = max(train_counts, key=train_counts.get)
    minimum_class = min(train_counts, key=train_counts.get)

    imbalance_ratio = max(train_counts.values()) / max(
        1,
        min(train_counts.values()),
    )

    print(
        f"\nTrain class imbalance ratio (max/min objects): "
        f"{imbalance_ratio:.2f}x "
        f"(max: {maximum_class}, min: {minimum_class})"
    )


def main() -> int:
    """Run final YOLO label validation."""
    config = yaml.safe_load(
        DATA_YAML.read_text(encoding="utf-8")
    )

    class_names = {
        int(index): name
        for index, name in config["names"].items()
    }

    if len(class_names) != 6:
        raise AssertionError(
            f"expected 6 classes, got {len(class_names)}"
        )

    errors: list[str] = []

    train_report = validate_split(
        config,
        "train",
        class_names,
        errors,
    )

    val_report = validate_split(
        config,
        "val",
        class_names,
        errors,
    )

    print_report(
        class_names,
        train_report,
        val_report,
    )

    if errors:
        print(f"\n{len(errors)} LABEL ERRORS:")

        for error in errors[:30]:
            print(f"  {error}")

        return 2

    print(
        "\nAll labels valid: class ids 0-5, coords in [0,1], "
        "positive sizes, complete image/label pairs."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
