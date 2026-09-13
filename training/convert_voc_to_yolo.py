"""Convert NEU-DET Pascal VOC XML annotations to YOLO format.

Reads:   data/raw/NEU-DET/{split}/annotations/*.xml + {split}/images/**
Writes:  data/yolo/images/{train,val}/*.jpg
         data/yolo/labels/{train,val}/*.txt
         data/yolo/data.yaml

Every bounding box is validated before and after normalisation.
Original XML files are never modified.

Usage (from project root):
    python training\\convert_voc_to_yolo.py
"""
from __future__ import annotations

import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (  # noqa: E402
    CLASS_NAMES,
    DATA_YAML,
    IMAGES_TRAIN_DIR,
    IMAGES_VAL_DIR,
    LABELS_TRAIN_DIR,
    LABELS_VAL_DIR,
    RAW_DATA_DIR,
    YOLO_DATA_DIR,
)
from training.inspect_dataset import _iter_splits  # noqa: E402

CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

SPLIT_MAP = {
    "train": (IMAGES_TRAIN_DIR, LABELS_TRAIN_DIR),
    "training": (IMAGES_TRAIN_DIR, LABELS_TRAIN_DIR),
    "val": (IMAGES_VAL_DIR, LABELS_VAL_DIR),
    "validation": (IMAGES_VAL_DIR, LABELS_VAL_DIR),
    "valid": (IMAGES_VAL_DIR, LABELS_VAL_DIR),
}


def voc_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h) -> tuple[float, float, float, float] | None:
    """Convert absolute VOC pixel coords to normalised YOLO (cx, cy, w, h).

    Returns None if the box is degenerate or falls outside the image.
    """
    xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
    # Clamp small overflows, reject grossly invalid boxes
    if xmax <= xmin or ymax <= ymin:
        return None
    if xmin < -1 or ymin < -1 or xmax > img_w + 1 or ymax > img_h + 1:
        return None
    xmin = max(0.0, xmin)
    ymin = max(0.0, ymin)
    xmax = min(float(img_w), xmax)
    ymax = min(float(img_h), ymax)
    cx = (xmin + xmax) / 2.0 / img_w
    cy = (ymin + ymax) / 2.0 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
        return None
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
        return None
    return cx, cy, w, h


def parse_voc_xml(xml_path: Path) -> tuple[int, int, list[tuple[int, float, float, float, float]], list[str]]:
    """Parse a VOC file; returns (width, height, yolo_boxes, warnings)."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"{xml_path.name}: missing <size> element")
    img_w = int(float(size.findtext("width", "0")))
    img_h = int(float(size.findtext("height", "0")))
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"{xml_path.name}: invalid image size {img_w}x{img_h}")

    boxes: list[tuple[int, float, float, float, float]] = []
    warnings: list[str] = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        if name not in CLASS_TO_ID:
            warnings.append(f"{xml_path.name}: unknown class '{name}' skipped")
            continue
        box = obj.find("bndbox")
        if box is None:
            warnings.append(f"{xml_path.name}: object '{name}' has no bndbox, skipped")
            continue
        try:
            xmin = box.findtext("xmin")
            ymin = box.findtext("ymin")
            xmax = box.findtext("xmax")
            ymax = box.findtext("ymax")
            yolo = voc_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
        except (TypeError, ValueError) as exc:
            warnings.append(f"{xml_path.name}: bad bbox for '{name}' ({exc}), skipped")
            continue
        if yolo is None:
            warnings.append(f"{xml_path.name}: invalid bbox for '{name}' skipped")
            continue
        boxes.append((CLASS_TO_ID[name], *yolo))
    return img_w, img_h, boxes, warnings


def find_image_for_stem(images_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        for cand in (images_dir / f"{stem}{ext}",):
            if cand.exists():
                return cand
    # images may live in per-class subfolders (NEU-DET layout)
    for cand in sorted(images_dir.rglob(f"{stem}.*")):
        if cand.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            return cand
    return None


def convert(split_name: str, ann_dir: Path, img_dir: Path) -> dict:
    stats = {"images": 0, "labels": 0, "boxes": 0, "skipped_images": 0, "skipped_boxes": 0, "warnings": []}
    out_img_dir, out_lbl_dir = SPLIT_MAP.get(split_name.lower(), (IMAGES_TRAIN_DIR, LABELS_TRAIN_DIR))
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    for xml_path in sorted(ann_dir.glob("*.xml")):
        stem = xml_path.stem
        img_path = find_image_for_stem(img_dir, stem)
        if img_path is None:
            stats["skipped_images"] += 1
            stats["warnings"].append(f"{xml_path.name}: no matching image, skipped")
            continue
        try:
            _, _, boxes, warns = parse_voc_xml(xml_path)
        except (ET.ParseError, ValueError) as exc:
            stats["skipped_images"] += 1
            stats["warnings"].append(str(exc))
            continue
        stats["warnings"].extend(warns)

        # Copy image (flat layout: class prefixes in filename keep them unique)
        shutil.copyfile(img_path, out_img_dir / f"{stem}{img_path.suffix.lower()}")

        label_path = out_lbl_dir / f"{stem}.txt"
        with label_path.open("w", encoding="utf-8") as fh:
            for cls_id, cx, cy, w, h in boxes:
                fh.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        stats["images"] += 1
        stats["labels"] += 1
        stats["boxes"] += len(boxes)
    return stats


def write_data_yaml() -> Path:
    yaml_data = {
        "path": YOLO_DATA_DIR.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
    }
    with DATA_YAML.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(yaml_data, fh, sort_keys=False, default_flow_style=False)
    return DATA_YAML


def validate_converted(out_lbl_dir: Path) -> list[str]:
    """Re-check every converted label file for normalisation validity."""
    problems = []
    for lbl in sorted(out_lbl_dir.glob("*.txt")):
        for ln, line in enumerate(lbl.read_text().splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                problems.append(f"{lbl.name}:{ln}: expected 5 fields, got {len(parts)}")
                continue
            try:
                cid = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
            except ValueError:
                problems.append(f"{lbl.name}:{ln}: non-numeric values")
                continue
            if not 0 <= cid < len(CLASS_NAMES):
                problems.append(f"{lbl.name}:{ln}: class id {cid} out of range")
            if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
                problems.append(f"{lbl.name}:{ln}: coordinates outside [0,1]")
            if w <= 0 or h <= 0:
                problems.append(f"{lbl.name}:{ln}: zero/negative box size")
    return problems


def main() -> int:
    splits = list(_iter_splits(RAW_DATA_DIR))
    if not splits:
        print(f"ERROR: no dataset found under {RAW_DATA_DIR}. Extract the archive first.")
        return 1

    all_warnings: list[str] = []
    for split_name, ann_dir, img_dir in splits:
        print(f"Converting split '{split_name}' ...")
        stats = convert(split_name, ann_dir, img_dir)
        print(f"  images converted : {stats['images']}")
        print(f"  label files      : {stats['labels']}")
        print(f"  boxes written    : {stats['boxes']}")
        print(f"  skipped images   : {stats['skipped_images']}")
        print(f"  skipped boxes    : {stats['skipped_boxes']}")
        for w in stats["warnings"]:
            print(f"  WARN {w}")
        all_warnings.extend(stats["warnings"])

    yaml_path = write_data_yaml()
    print(f"\nWrote {yaml_path}")

    print("\nValidating converted labels...")
    problems = []
    for d in (LABELS_TRAIN_DIR, LABELS_VAL_DIR):
        if d.exists():
            problems.extend(validate_converted(d))
    if problems:
        print(f"  {len(problems)} PROBLEMS FOUND:")
        for p in problems[:20]:
            print(f"  {p}")
        return 2
    print("  All converted labels valid (class ids and coords within range).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
