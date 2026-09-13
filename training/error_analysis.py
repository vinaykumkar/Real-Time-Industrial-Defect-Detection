"""Error analysis for the trained NEU defect model.

Runs validation-set inference, matches predictions against ground truth by
IoU, and saves example images of false positives, false negatives, and
low-confidence predictions, plus a class-confusion summary.

Usage (from project root):
    python training\\error_analysis.py [--model models\\best.pt] [--limit 200]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (  # noqa: E402
    CLASS_NAMES,
    ERROR_ANALYSIS_DIR,
    IMAGES_VAL_DIR,
    LABELS_VAL_DIR,
    PYTORCH_MODEL_PATH,
    PROJECT_ROOT,
)

IOU_MATCH_THRESHOLD = 0.5
LOW_CONFIDENCE = 0.35


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def draw_error(img, boxes, color, label) -> np.ndarray:
    out = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for b in boxes:
        cv2.rectangle(out, (b[0], b[1]), (b[2], b[3]), color, 1)
        cv2.putText(out, label, (b[0], max(10, b[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Error analysis: FP / FN / low-confidence examples")
    parser.add_argument("--model", type=str, default=str(PYTORCH_MODEL_PATH))
    parser.add_argument("--limit", type=int, default=200, help="max validation images to analyse")
    parser.add_argument("--conf", type=float, default=0.25, help="inference confidence for analysis")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    from ultralytics import YOLO

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not model_path.exists():
        print("CUSTOM DEFECT MODEL NOT TRAINED")
        print(f"Expected: {model_path}")
        print("Run: python training\\train.py --epochs 1")
        return 1

    images = sorted(IMAGES_VAL_DIR.glob("*.jpg"))[: args.limit]
    if not images:
        print(f"No validation images at {IMAGES_VAL_DIR}. Run training\\convert_voc_to_yolo.py first.")
        return 1

    for d in ("false_positives", "false_negatives", "low_confidence"):
        (ERROR_ANALYSIS_DIR / d).mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    stats = {
        "images_analysed": 0,
        "gt_boxes": 0,
        "pred_boxes": 0,
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "low_confidence_predictions": 0,
        "fp_by_class": Counter(),
        "fn_by_class": Counter(),
        "confusion_gt_to_pred": Counter(),  # "gt->pred" when matched with wrong class
    }
    examples = {"fp": 0, "fn": 0, "low": 0}
    MAX_EXAMPLES = 15

    for img_path in images:
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        lbl_path = LABELS_VAL_DIR / f"{img_path.stem}.txt"
        gt = []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.split()
                if len(parts) == 5:
                    cid, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
                    x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
                    x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
                    gt.append((cid, [x1, y1, x2, y2]))
        stats["gt_boxes"] += len(gt)
        stats["images_analysed"] += 1

        result = model.predict(img, conf=args.conf, verbose=False)[0]
        preds = []
        if result.boxes is not None:
            for xyxy, conf, cls in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy().astype(int),
            ):
                preds.append((int(cls), [int(v) for v in xyxy], float(conf)))
        stats["pred_boxes"] += len(preds)

        # greedy match preds to gts
        used_gt = set()
        fp_preds = []
        for pc, pbox, pconf in sorted(preds, key=lambda t: -t[2]):
            best_iou, best_j = 0.0, -1
            for j, (gc, gbox) in enumerate(gt):
                if j in used_gt:
                    continue
                i = iou_xyxy(pbox, gbox)
                if i > best_iou:
                    best_iou, best_j = i, j
            if best_iou >= IOU_MATCH_THRESHOLD and best_j >= 0:
                used_gt.add(best_j)
                if pc != gt[best_j][0]:
                    stats["confusion_gt_to_pred"][f"{CLASS_NAMES[gt[best_j][0]]}->{CLASS_NAMES[pc]}"] += 1
                stats["true_positives"] += 1
            else:
                stats["false_positives"] += 1
                stats["fp_by_class"][CLASS_NAMES[pc]] += 1
                fp_preds.append((pc, pbox, pconf))
                if examples["fp"] < MAX_EXAMPLES:
                    vis = draw_error(img, [pbox], (0, 0, 255), f"FP {CLASS_NAMES[pc]} {pconf:.2f}")
                    cv2.imencode(".jpg", vis)[1].tofile(ERROR_ANALYSIS_DIR / "false_positives" / f"{img_path.stem}_fp.jpg")
                    examples["fp"] += 1
            if pconf < LOW_CONFIDENCE:
                stats["low_confidence_predictions"] += 1
                if examples["low"] < MAX_EXAMPLES:
                    vis = draw_error(img, [pbox], (0, 165, 255), f"LOW {CLASS_NAMES[pc]} {pconf:.2f}")
                    cv2.imencode(".jpg", vis)[1].tofile(ERROR_ANALYSIS_DIR / "low_confidence" / f"{img_path.stem}_low.jpg")
                    examples["low"] += 1

        for j, (gc, gbox) in enumerate(gt):
            if j not in used_gt:
                stats["false_negatives"] += 1
                stats["fn_by_class"][CLASS_NAMES[gc]] += 1
                if examples["fn"] < MAX_EXAMPLES:
                    vis = draw_error(img, [gbox], (0, 255, 0), f"FN {CLASS_NAMES[gc]}")
                    cv2.imencode(".jpg", vis)[1].tofile(ERROR_ANALYSIS_DIR / "false_negatives" / f"{img_path.stem}_fn.jpg")
                    examples["fn"] += 1

    # summary
    stats["fp_by_class"] = dict(stats["fp_by_class"])
    stats["fn_by_class"] = dict(stats["fn_by_class"])
    stats["confusion_gt_to_pred"] = dict(stats["confusion_gt_to_pred"])
    out = ERROR_ANALYSIS_DIR / "error_summary.json"
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("=" * 60)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Images analysed   : {stats['images_analysed']}")
    print(f"Ground-truth boxes: {stats['gt_boxes']}")
    print(f"Predictions       : {stats['pred_boxes']} (conf >= {args.conf})")
    print(f"True positives    : {stats['true_positives']}")
    print(f"False positives   : {stats['false_positives']}")
    print(f"False negatives   : {stats['false_negatives']}")
    print(f"Low-conf preds    : {stats['low_confidence_predictions']} (< {LOW_CONFIDENCE})")
    print(f"FP by class       : {stats['fp_by_class']}")
    print(f"FN by class       : {stats['fn_by_class']}")
    if stats["confusion_gt_to_pred"]:
        print(f"GT->Pred confusion: {stats['confusion_gt_to_pred']}")
    print(f"\nExample images saved under {ERROR_ANALYSIS_DIR}\\{{false_positives,false_negatives,low_confidence}}")
    print(f"Summary saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
