"""Confidence-threshold analysis for the defect model (manufacturing tradeoff).

Runs validation-set inference ONCE at a low confidence, then evaluates
TP/FP/FN by IoU-matching at several thresholds. In quality control missing a
real defect (FN) is usually costlier than a false alarm (FP), so the table is
meant to pick a recall-favouring operating point — not just max precision.

Usage (from project root):
    python training\\threshold_analysis.py [--model models\\best.pt] [--imgsz 640]
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

from app.config import (  # noqa: E402
    CLASS_NAMES,
    IMAGES_VAL_DIR,
    LABELS_VAL_DIR,
    PROJECT_ROOT,
    TRAINING_OUTPUT_DIR,
)

THRESHOLDS = [0.10, 0.25, 0.35, 0.50, 0.60]
IOU_MATCH = 0.5


def iou_xyxy(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def main() -> int:
    parser = argparse.ArgumentParser(description="Confidence-threshold operating-point analysis")
    parser.add_argument("--model", type=str, default=str(PROJECT_ROOT / "models" / "best.pt"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--limit", type=int, default=0, help="limit val images (0 = all)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"model not found: {model_path}")
        return 1
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(str(model_path))

    images = sorted(IMAGES_VAL_DIR.glob("*.jpg"))
    if args.limit:
        images = images[: args.limit]
    if not images:
        print("no validation images")
        return 1

    # One pass at low conf; bucket predictions per image
    records = []  # (img_stem, class_id, conf, box_xyxy)
    gts = {}      # stem -> list[(class_id, box_xyxy)]
    for img_path in images:
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        lbl = LABELS_VAL_DIR / f"{img_path.stem}.txt"
        gt = []
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) == 5:
                    cid, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
                    gt.append((cid, [(cx - bw / 2) * w, (cy - bh / 2) * h,
                                     (cx + bw / 2) * w, (cy + bh / 2) * h]))
        gts[img_path.stem] = gt
        res = model.predict(img, conf=min(THRESHOLDS), iou=0.45, imgsz=args.imgsz, device=device, verbose=False)[0]
        if res.boxes is not None and len(res.boxes):
            for xyxy, conf, cls in zip(
                res.boxes.xyxy.cpu().numpy(),
                res.boxes.conf.cpu().numpy(),
                res.boxes.cls.cpu().numpy().astype(int),
            ):
                records.append((img_path.stem, int(cls), float(conf), [float(v) for v in xyxy]))

    def evaluate(threshold: float) -> dict:
        tp = fp = fn = 0
        by_class = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASS_NAMES}
        for stem, gt in gts.items():
            preds = [(conf, c, box) for s, c, conf, box in records if s == stem and conf >= threshold]
            used = set()
            for conf, pc, pbox in sorted(preds, key=lambda t: -t[0]):
                best_iou, best_j = 0.0, -1
                for j, (gc, gbox) in enumerate(gt):
                    if j in used or gc != pc:
                        continue
                    i = iou_xyxy(pbox, gbox)
                    if i > best_iou:
                        best_iou, best_j = i, j
                if best_iou >= IOU_MATCH and best_j >= 0:
                    used.add(best_j)
                    tp += 1
                    by_class[CLASS_NAMES[pc]]["tp"] += 1
                else:
                    fp += 1
                    by_class[CLASS_NAMES[pc]]["fp"] += 1
            for j, (gc, _) in enumerate(gt):
                if j not in used:
                    fn += 1
                    by_class[CLASS_NAMES[gc]]["fn"] += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "threshold": threshold,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "false_negative_rate": round(fn / (tp + fn), 4) if tp + fn else 0.0,
            "by_class": by_class,
        }

    results = [evaluate(t) for t in THRESHOLDS]

    print("=" * 76)
    print("CONFIDENCE THRESHOLD ANALYSIS (measured on validation set)")
    print("=" * 76)
    print(f"{'conf':>6}{'TP':>7}{'FP':>7}{'FN':>7}{'precision':>11}{'recall':>9}{'F1':>8}{'FNR':>8}")
    for r in results:
        print(f"{r['threshold']:>6.2f}{r['tp']:>7}{r['fp']:>7}{r['fn']:>7}"
              f"{r['precision']:>11.4f}{r['recall']:>9.4f}{r['f1']:>8.4f}{r['false_negative_rate']:>8.4f}")

    out = Path(PROJECT_ROOT) / "outputs" / "evaluation" / "final"
    out.mkdir(parents=True, exist_ok=True)
    (out / "threshold_analysis.json").write_text(
        json.dumps({"model": str(model_path), "imgsz": args.imgsz, "iou_match": IOU_MATCH,
                    "images": len(images), "results": results}, indent=2), encoding="utf-8")
    print(f"\nSaved: {out / 'threshold_analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
