"""Validate the exported ONNX model against the PyTorch reference.

Runs both backends on the same validation images and compares detections:
class agreement, box overlap (IoU), and confidence deltas. ONNX need not be
bit-identical to PyTorch, but must be logically consistent (spec §4/§61).

Writes outputs/evaluation/final/onnx_consistency.json.

Usage (from project root):
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


def iou_xyxy(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def to_lists(dets):
    return [
        (d["class"], d["confidence"], [d["bbox"]["x1"], d["bbox"]["y1"], d["bbox"]["x2"], d["bbox"]["y2"]])
        for d in dets
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="PyTorch vs ONNX prediction consistency check")
    parser.add_argument("--images", type=int, default=40)
    parser.add_argument("--conf", type=float, default=0.35)
    args = parser.parse_args()

    import torch

    from app.inference.detector import Detector

    pt_path = MODELS_DIR / "best.pt"
    onnx_path = MODELS_DIR / "best.onnx"
    if not pt_path.exists() or not onnx_path.exists():
        print("both models/best.pt and models/best.onnx are required")
        return 1

    imgsz = None
    meta = MODELS_DIR / "best_metadata.json"
    if meta.exists():
        imgsz = int(json.loads(meta.read_text(encoding="utf-8")).get("image_size", 320))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    det_pt = Detector(backend="pytorch", conf_threshold=args.conf, imgsz=imgsz)
    det_pt.warmup()
    det_onnx = Detector(backend="onnx", conf_threshold=args.conf, imgsz=imgsz)
    det_onnx.warmup()

    images = sorted(IMAGES_VAL_DIR.glob("*.jpg"))
    if args.images:
        step = max(1, len(images) // args.images)
        images = images[::step][: args.images]

    n_images = 0
    class_match = 0          # images where top defect class agrees (or both empty)
    ious, conf_deltas = [], []
    pt_counts, onnx_counts = [], []
    both_empty = 0
    for img_path in images:
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        d_pt = sorted(to_lists(det_pt.detect_raw(img)), key=lambda t: -t[1])
        d_on = sorted(to_lists(det_onnx.detect_raw(img)), key=lambda t: -t[1])
        n_images += 1
        pt_counts.append(len(d_pt))
        onnx_counts.append(len(d_on))
        if not d_pt and not d_on:
            both_empty += 1
            class_match += 1
            continue
        top_pt = d_pt[0] if d_pt else None
        top_on = d_on[0] if d_on else None
        if top_pt and top_on and top_pt[0] == top_on[0]:
            class_match += 1
            ious.append(iou_xyxy(top_pt[2], top_on[2]))
            conf_deltas.append(abs(top_pt[1] - top_on[1]))

    agree = class_match / n_images if n_images else 0.0
    med_iou = float(np.median(ious)) if ious else None
    mean_conf_delta = float(np.mean(conf_deltas)) if conf_deltas else None
    result = {
        "images_compared": n_images,
        "top_class_agreement": round(agree, 4),
        "images_both_empty": both_empty,
        "median_top_box_iou": round(med_iou, 4) if med_iou is not None else None,
        "mean_top_confidence_delta": round(mean_conf_delta, 4) if mean_conf_delta is not None else None,
        "mean_detections_pt": round(sum(pt_counts) / n_images, 2) if n_images else 0,
        "mean_detections_onnx": round(sum(onnx_counts) / n_images, 2) if n_images else 0,
        "verdict": "CONSISTENT" if agree >= 0.9 and (med_iou is None or med_iou >= 0.8) else "REVIEW NEEDED",
    }

    out = PROJECT_ROOT / "outputs" / "evaluation" / "final" / "onnx_consistency.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("=" * 62)
    print("PYTORCH vs ONNX CONSISTENCY")
    print("=" * 62)
    for k, v in result.items():
        print(f"  {k:<28} {v}")
    print(f"\nSaved: {out}")
    return 0 if result["verdict"] == "CONSISTENT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
