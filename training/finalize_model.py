"""Finalize the selected experiment as the project's best model.

Copies the chosen run's best.pt to models/best.pt (preserving the original
run), writes models/best_metadata.json with the measured metrics and training
config, exports ONNX, and assembles outputs/evaluation/final/:
metrics.json, per_class_metrics.csv, plots, confusion matrix.

Usage (from project root):
    python training\\finalize_model.py --run outputs\\training\\runs\\<run_name> ^
        --experiment baseline_yolov8n_640 [--confidence-threshold 0.35] [--imgsz 640]
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (  # noqa: E402
    CLASS_NAMES,
    DATA_YAML,
    MODELS_DIR,
    PROJECT_ROOT,
    TRAINING_OUTPUT_DIR,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote a run to models/best.pt + final eval bundle")
    parser.add_argument("--run", required=True, help="run directory (contains weights/best.pt, args.yaml)")
    parser.add_argument("--experiment", required=True, help="experiment name for metadata")
    parser.add_argument("--confidence-threshold", type=float, default=0.35,
                        help="recommended operating confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--skip-onnx", action="store_true", help="skip ONNX re-export")
    args = parser.parse_args()

    import torch
    import yaml
    from ultralytics import YOLO

    import train  # noqa: F401  (ensures package context)

    run_dir = Path(args.run)
    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        print(f"ERROR: {weights} not found")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(weights))

    # fresh, honest evaluation of exactly the weights being promoted
    print(f"Evaluating {weights} on device={device} ...")
    metrics = model.val(data=str(DATA_YAML), imgsz=args.imgsz, device=device,
                        project=str(PROJECT_ROOT / "outputs" / "evaluation"),
                        name="final", exist_ok=True, plots=True)
    rd = metrics.results_dict
    pr, rc = float(rd.get("metrics/precision(B)", 0.0)), float(rd.get("metrics/recall(B)", 0.0))
    f1 = 2 * pr * rc / (pr + rc) if pr + rc > 0 else 0.0

    per_class = {}
    try:
        ap50 = metrics.box.ap50
        ap = metrics.box.ap
        for i, name in (metrics.names or {}).items():
            per_class[name] = {
                "AP50": round(float(ap50[i]), 4) if i < len(ap50) else None,
                "AP50_95": round(float(ap[i][0]), 4) if i < len(ap) else None,
            }
    except Exception as exc:
        print(f"note: per-class extraction failed ({exc})")

    cm = None
    try:
        cm = metrics.confusion_matrix.matrix.tolist()
    except Exception:
        pass

    # args.yaml from the run
    try:
        train_args = yaml.safe_load((run_dir / "args.yaml").read_text()) or {}
    except OSError:
        train_args = {}

    # 1) copy weights
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest_pt = MODELS_DIR / "best.pt"
    shutil.copyfile(weights, dest_pt)

    # 2) metadata
    metadata = {
        "architecture": train_args.get("model", "yolov8n"),
        "training_date": datetime.now().isoformat(timespec="seconds"),
        "epochs_trained": len((run_dir / "results.csv").read_text().strip().splitlines()) - 1
        if (run_dir / "results.csv").exists() else train_args.get("epochs"),
        "epochs_requested": train_args.get("epochs"),
        "image_size": train_args.get("imgsz", args.imgsz),
        "batch": train_args.get("batch"),
        "optimizer": train_args.get("optimizer"),
        "lr0": train_args.get("lr0"),
        "augmentation": {
            "mosaic": train_args.get("mosaic"), "mixup": train_args.get("mixup"),
            "degrees": train_args.get("degrees"), "fliplr": train_args.get("fliplr"),
            "flipud": train_args.get("flipud"), "hsv_v": train_args.get("hsv_v"),
        },
        "classes": CLASS_NAMES,
        "mAP50": round(float(rd.get("metrics/mAP50(B)", 0.0)), 4),
        "mAP50_95": round(float(rd.get("metrics/mAP50-95(B)", 0.0)), 4),
        "precision": round(pr, 4),
        "recall": round(rc, 4),
        "f1": round(f1, 4),
        "per_class": per_class,
        "recommended_confidence_threshold": args.confidence_threshold,
        "iou_threshold": 0.45,
        "device_used": device,
        "source_experiment": args.experiment,
        "source_run": str(run_dir),
        "weights": str(dest_pt),
        "note": "Trained and evaluated on the NEU dataset (research demonstration — not production-certified).",
    }
    (MODELS_DIR / "best_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # 3) ONNX export of the final weights
    if not args.skip_onnx:
        print("Exporting ONNX ...")
        try:
            exported = model.export(format="onnx", imgsz=args.imgsz, simplify=True, opset=13)
            src = Path(exported)
            dest_onnx = MODELS_DIR / "best.onnx"
            if src.resolve() != dest_onnx.resolve():
                shutil.copyfile(src, dest_onnx)
            print(f"ONNX written: {dest_onnx}")
        except Exception as exc:
            print(f"WARNING: ONNX export failed: {exc}")

    # 4) final evaluation bundle
    final_dir = PROJECT_ROOT / "outputs" / "evaluation" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    summary = {k: metadata[k] for k in ("mAP50", "mAP50_95", "precision", "recall", "f1")}
    summary.update({"model": str(dest_pt), "device": device, "dataset": str(DATA_YAML),
                    "experiment": args.experiment, "per_class": per_class})
    if cm is not None:
        summary["confusion_matrix"] = cm
    (final_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if per_class:
        with (final_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["class", "AP50", "AP50_95"])
            for name, v in per_class.items():
                w.writerow([name, v.get("AP50"), v.get("AP50_95")])

    # copy plots from the fresh val run + training curves from the source run
    val_dir = PROJECT_ROOT / "outputs" / "evaluation" / "final"  # ultralytics writes into .../final itself? (name=final)
    val_run_dir = PROJECT_ROOT / "outputs" / "evaluation" / "final"
    for fname in ("confusion_matrix.png", "confusion_matrix_normalized.png", "PR_curve.png",
                  "F1_curve.png", "P_curve.png", "R_curve.png"):
        p = val_run_dir / fname
        if p.exists():
            break
    else:
        val_run_dir = PROJECT_ROOT / "outputs" / "evaluation" / "final"
    # Ultralytics saved val plots under outputs/evaluation/final/ already (name=final)
    for src in (run_dir / "results.csv", run_dir / "results.png", run_dir / "confusion_matrix.png",
                run_dir / "train_batch0.jpg", run_dir / "val_batch0_pred.jpg", run_dir / "val_batch0_labels.jpg"):
        if src.exists():
            shutil.copyfile(src, final_dir / src.name)
    # sample predictions (annotated val batches)
    for src in run_dir.glob("val_batch*_pred.jpg"):
        shutil.copyfile(src, final_dir / src.name)

    print("=" * 64)
    print("FINAL MODEL SELECTED:", args.experiment)
    print("=" * 64)
    print(f"  weights        : {dest_pt}")
    print(f"  mAP50          : {metadata['mAP50']}")
    print(f"  mAP50-95       : {metadata['mAP50_95']}")
    print(f"  precision      : {metadata['precision']}")
    print(f"  recall         : {metadata['recall']}")
    print(f"  f1             : {metadata['f1']}")
    for name, v in per_class.items():
        print(f"    {name:<16} AP50={v.get('AP50')}  AP50-95={v.get('AP50_95')}")
    print(f"  metadata       : {MODELS_DIR / 'best_metadata.json'}")
    print(f"  final eval dir : {final_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
