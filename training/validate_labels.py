"""Final pre-training YOLO label validation + class-distribution report.

Checks every converted label file: class ids 0..5, coords in [0,1],
positive box sizes, missing label files, missing images, duplicate files.
Prints a compact class/table report and exits non-zero on any error.

Usage (from project root):
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


def main() -> int:
    cfg = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    names: dict[int, str] = {int(k): v for k, v in cfg["names"].items()}
    assert len(names) == 6, f"expected 6 classes, got {len(names)}"

    errors: list[str] = []
    report: dict[str, dict] = {}
    for split_key, split in (("train", "images/train"), ("val", "images/val")):
        img_dir = Path(cfg["path"]) / split
        lbl_dir = Path(cfg["path"]) / ("labels/train" if split_key == "train" else "labels/val")
        images = sorted(img_dir.glob("*.jpg"))
        labels = sorted(lbl_dir.glob("*.txt"))
        img_stems = {p.stem for p in images}
        lbl_stems = {p.stem for p in labels}

        for stem in sorted(img_stems - lbl_stems):
            errors.append(f"{split_key}/{stem}: image without label file")
        for stem in sorted(lbl_stems - img_stems):
            errors.append(f"{split_key}/{stem}: label without image")

        class_boxes: Counter = Counter()
        class_img_counts: Counter = Counter()
        n_boxes = 0
        multi = 0
        for lbl in labels:
            seen: set[int] = set()
            n_in_file = 0
            for ln, line in enumerate(lbl.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"{split_key}/{lbl.name}:{ln}: expected 5 fields")
                    continue
                cid, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
                if cid not in names:
                    errors.append(f"{split_key}/{lbl.name}:{ln}: class id {cid} not in 0-5")
                    continue
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                    errors.append(f"{split_key}/{lbl.name}:{ln}: center out of range")
                if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    errors.append(f"{split_key}/{lbl.name}:{ln}: size out of range")
                class_boxes[cid] += 1
                n_boxes += 1
                n_in_file += 1
                seen.add(cid)
            if n_in_file > 1:
                multi += 1
            for cid in seen:
                class_img_counts[cid] += 1

        report[split_key] = {
            "images": len(images),
            "labels": len(labels),
            "boxes": n_boxes,
            "multi_object_images": multi,
            "class_boxes": class_boxes,
            "class_img_counts": class_img_counts,
        }

    print("=" * 78)
    print("FINAL LABEL VALIDATION")
    print("=" * 78)
    header = f"{'Class':<18}{'Train objects':>14}{'Val objects':>13}{'Train imgs':>12}{'Val imgs':>10}"
    print(header)
    t, v = report["train"], report["val"]
    for cid, name in names.items():
        print(
            f"{name:<18}{t['class_boxes'][cid]:>14}{v['class_boxes'][cid]:>13}"
            f"{t['class_img_counts'][cid]:>12}{v['class_img_counts'][cid]:>10}"
        )
    print(f"\n{'TOTALS':<18}{t['boxes']:>14}{v['boxes']:>13}{t['images']:>12}{v['images']:>10}")
    print(f"{'multi-object imgs':<18}{'':>14}{'':>13}{t['multi_object_images']:>12}{v['multi_object_images']:>10}")

    imbalance = max(t["class_boxes"].values()) / max(1, min(t["class_boxes"].values()))
    print(f"\nTrain class imbalance ratio (max/min objects): {imbalance:.2f}x "
          f"(max: {max(t['class_boxes'], key=lambda c: t['class_boxes'][c])}, "
          f"min: {min(t['class_boxes'], key=lambda c: t['class_boxes'][c])})")

    if errors:
        print(f"\n{len(errors)} LABEL ERRORS:")
        for e in errors[:30]:
            print(f"  {e}")
        return 2
    print("\nAll labels valid: class ids 0-5, coords in [0,1], positive sizes, complete image/label pairs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
