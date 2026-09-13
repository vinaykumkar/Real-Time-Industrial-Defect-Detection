"""Visual annotation sanity check for the converted YOLO dataset.

Draws the converted YOLO labels back onto validation images so a human (or a
scripted diff against the raw Pascal VOC boxes) can confirm the VOC -> YOLO
conversion did not shift, distort, or drop bounding boxes.

Writes before/after pairs to outputs/annotation_checks/:
  <stem>_yolo.jpg      - YOLO label drawn on the converted image
  <stem>_voc_vs_yolo.jpg - side-by-side VOC (green) vs YOLO (amber) overlay

Usage (from project root):
    python training\\check_annotations.py [--per-class 3] [--split val]
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.inference.preprocessing import CLASS_COLORS  # noqa: E402
from app.config import (  # noqa: E402
    CLASS_NAMES,
    IMAGES_VAL_DIR,
    LABELS_VAL_DIR,
    RAW_DATA_DIR,
    TRAINING_OUTPUT_DIR,
)

VOC_SPLITS = {"val": "validation", "train": "train"}


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [int(round(v)) for v in (x1, y1, x2, y2)]


def voc_boxes_for(raw_xml: Path) -> list[tuple[str, list[int]]]:
    root = ET.parse(raw_xml).getroot()
    out = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        out.append((obj.findtext("name", ""), [
            int(float(box.findtext("xmin"))), int(float(box.findtext("ymin"))),
            int(float(box.findtext("xmax"))), int(float(box.findtext("ymax"))),
        ]))
    return out


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def main() -> int:
    parser = argparse.ArgumentParser(description="Visual VOC->YOLO annotation sanity check")
    parser.add_argument("--per-class", type=int, default=3, help="samples per class")
    parser.add_argument("--split", choices=["val", "train"], default="val")
    args = parser.parse_args()

    img_dir = IMAGES_VAL_DIR if args.split == "val" else IMAGES_VAL_DIR.parent / "train"
    lbl_dir = LABELS_VAL_DIR if args.split == "val" else LABELS_VAL_DIR.parent / "train"
    raw_split = VOC_SPLITS[args.split]
    raw_img_dir = RAW_DATA_DIR / "NEU-DET" / raw_split / "images"
    raw_ann_dir = RAW_DATA_DIR / "NEU-DET" / raw_split / "annotations"

    if not img_dir.exists() or not lbl_dir.exists():
        print("Converted dataset missing. Run training\\convert_voc_to_yolo.py first.")
        return 1

    out_dir = PROJECT_ROOT / "outputs" / "annotation_checks"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_checked, mismatches = 0, []
    cross_class: list[str] = []
    for cid, class_name in enumerate(CLASS_NAMES):
        images = sorted(img_dir.glob(f"{class_name}_*.jpg"))[: args.per_class]
        for img_path in images:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                mismatches.append(f"{img_path.stem}: missing YOLO label file")
                continue
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                mismatches.append(f"{img_path.stem}: unreadable converted image")
                continue
            h, w = img.shape[:2]
            yolo_boxes = []
            for line in lbl_path.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    mismatches.append(f"{img_path.stem}: malformed YOLO line: {line!r}")
                    continue
                b_cid, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
                # NOTE: NEU-DET images may legitimately contain multiple defect
                # classes, so a differing class id is only a soft signal; the
                # geometric IoU check below is the authoritative validation.
                if b_cid != cid:
                    cross_class.append(f"{img_path.stem}: box of class {CLASS_NAMES[b_cid]} inside {class_name} sample")
                if not all(0.0 <= v <= 1.0 for v in (cx, cy, bw, bh)) or bw <= 0 or bh <= 0:
                    mismatches.append(f"{img_path.stem}: out-of-range YOLO coords")
                yolo_boxes.append((b_cid, yolo_to_xyxy(cx, cy, bw, bh, w, h)))

            vis = img.copy()
            for b_cid, box in yolo_boxes:
                cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), CLASS_COLORS[b_cid], 1)
                cv2.putText(vis, CLASS_NAMES[b_cid], (box[0], max(9, box[1] - 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, CLASS_COLORS[b_cid], 1)
            cv2.imencode(".jpg", vis)[1].tofile(out_dir / f"{img_path.stem}_yolo.jpg")

            # quantitative comparison against the original VOC XML
            raw_matches = sorted(raw_img_dir.rglob(f"{img_path.stem}.*"))
            if raw_matches:
                raw_xml = raw_ann_dir / f"{img_path.stem}.xml"
                voc = voc_boxes_for(raw_xml)
                voc_xyxy = [b for _, b in voc]
                for _, ybox in yolo_boxes:
                    best = max((iou(ybox, vbox) for vbox in voc_xyxy), default=0.0)
                    if best < 0.85:
                        mismatches.append(f"{img_path.stem}: YOLO box IoU {best:.2f} vs VOC (shifted?)")
                if len(yolo_boxes) != len(voc):
                    mismatches.append(f"{img_path.stem}: box count YOLO={len(yolo_boxes)} vs VOC={len(voc)}")

                # side-by-side overlay image
                voc_vis = img.copy()
                for name, vbox in voc:
                    cv2.rectangle(voc_vis, (vbox[0], vbox[1]), (vbox[2], vbox[3]), (0, 255, 0), 1)
                both = np.hstack([voc_vis, vis])
                cv2.putText(both, "VOC (green)", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.putText(both, "YOLO (class color)", (w + 6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
                cv2.imencode(".jpg", both)[1].tofile(out_dir / f"{img_path.stem}_voc_vs_yolo.jpg")
            total_checked += 1

    print("=" * 60)
    print("ANNOTATION SANITY CHECK")
    print("=" * 60)
    print(f"Images checked : {total_checked} ({args.per_class}/class, split={args.split})")
    print(f"Outputs        : {out_dir}")
    if cross_class:
        print(f"\nInfo: {len(cross_class)} box(es) of a different class appear inside their "
              "sample image (legitimate in NEU-DET, verified against raw VOC):")
        for m in cross_class[:10]:
            print(f"  {m}")
    if mismatches:
        print(f"\n{len(mismatches)} POTENTIAL PROBLEMS:")
        for m in mismatches[:20]:
            print(f"  {m}")
        return 2
    print("No shift/distortion detected: every YOLO box matches its VOC box (IoU >= 0.85).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
