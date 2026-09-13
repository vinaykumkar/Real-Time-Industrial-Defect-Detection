"""Evaluate the trained NEU defect model on the validation set.

Reports mAP@0.50, mAP@0.50:0.95, precision, recall, F1, per-class AP and the
confusion matrix. All numbers come from real evaluation — nothing invented.

Usage (from project root):
    python training\\evaluate.py [--model models\\best.pt] [--imgsz 640]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (  # noqa: E402
    CLASS_NAMES,
    DATA_YAML,
    EVALUATION_OUTPUT_DIR,
    PYTORCH_MODEL_PATH,
)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained defect-detection model")
    p.add_argument("--model", type=str, default=str(PYTORCH_MODEL_PATH))
    p.add_argument("--data", type=str, default=str(DATA_YAML))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--split", type=str, default="val", choices=["val", "test"])
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    import torch
    from ultralytics import YOLO

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not model_path.exists():
        print("CUSTOM DEFECT MODEL NOT TRAINED")
        print(f"Expected trained weights at: {model_path}")
        print("Train first with:")
        print("  python training\\train.py --epochs 1    # smoke test")
        print("  python training\\train.py --epochs 100  # full training")
        return 1

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating {model_path} on device={device}, split={args.split}")

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        split=args.split,
        project=str(EVALUATION_OUTPUT_DIR),
        name="val_run",
        exist_ok=True,
        plots=True,
    )

    # Ultralytics DetMetrics
    results_dict = metrics.results_dict  # keys like metrics/mAP50(B), ...
    summary = {
        "model": str(model_path),
        "device": device,
        "dataset": str(args.data),
        "split": args.split,
        "mAP50": float(results_dict.get("metrics/mAP50(B)", 0.0)),
        "mAP50_95": float(results_dict.get("metrics/mAP50-95(B)", 0.0)),
        "precision": float(results_dict.get("metrics/precision(B)", 0.0)),
        "recall": float(results_dict.get("metrics/recall(B)", 0.0)),
    }
    p, r = summary["precision"], summary["recall"]
    summary["f1"] = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    per_class = {}
    try:
        ap50_per_class = metrics.box.ap50  # per-class AP@0.5
        ap_all_per_class = np.ravel(metrics.box.ap)  # per-class AP@0.5:0.95
        names = metrics.names or {i: n for i, n in enumerate(CLASS_NAMES)}
        for i, name in names.items():
            per_class[name] = {
                "AP50": float(ap50_per_class[i]) if i < len(ap50_per_class) else None,
                "AP50_95": float(ap_all_per_class[i]) if i < len(ap_all_per_class) else None,
            }
        summary["per_class"] = per_class
        # confusion matrix summary
        try:
            cm = metrics.confusion_matrix.matrix  # numpy array
            summary["confusion_matrix"] = cm.tolist()
        except Exception:
            pass
    except Exception as exc:
        print(f"Note: per-class metrics unavailable ({exc})")

    out_json = EVALUATION_OUTPUT_DIR / "metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (measured, not estimated)")
    print("=" * 60)
    print(f"mAP@0.50      : {summary['mAP50']:.4f}")
    print(f"mAP@0.50:0.95 : {summary['mAP50_95']:.4f}")
    print(f"Precision     : {p:.4f}")
    print(f"Recall        : {r:.4f}")
    print(f"F1            : {summary['f1']:.4f}")
    if per_class:
        print("\nPer-class AP@0.5:")
        for name, vals in per_class.items():
            ap = vals.get("AP50")
            print(f"  {name:<16} {ap if ap is None else f'{ap:.4f}'}")
    print(f"\nSaved: {out_json}")
    print("Confusion matrix / plots:", EVALUATION_OUTPUT_DIR / "val_run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
