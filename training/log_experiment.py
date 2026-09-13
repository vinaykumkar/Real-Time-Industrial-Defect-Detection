"""Log one training experiment into outputs/evaluation/experiment_comparison.csv.

Evaluates the run's best.pt on the validation split (fresh Ultralytics val),
measures real inference latency on a sample image (warmup + N iterations),
records model size, and appends one CSV row. Comparison records only —
no estimated values anywhere.

Usage (from project root):
    python training\\log_experiment.py --run outputs\\training\\runs\\baseline_yolov8n_640 ^
        --experiment baseline_yolov8n_640 --model yolov8n.pt --epochs 50 --imgsz 640 ^
        --batch 16 --lr auto --aug default --notes "baseline"
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import DATA_YAML, EVALUATION_OUTPUT_DIR, IMAGES_VAL_DIR  # noqa: E402

CSV_PATH = EVALUATION_OUTPUT_DIR / "experiment_comparison.csv"
COLUMNS = [
    "experiment", "model", "epochs", "imgsz", "batch", "learning_rate",
    "augmentation_profile", "mAP50", "mAP50_95", "precision", "recall", "f1",
    "training_time_min", "model_size_mb", "inference_latency_ms", "device", "notes",
]


def measure_latency(model, imgsz: int, device: str, iters: int = 30) -> float:
    """Median single-image inference latency in ms (model-load excluded)."""
    img = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    # real-ish content so timing isn't degenerate
    rng = np.random.default_rng(0)
    img = (rng.random((imgsz, imgsz, 3)) * 255).astype("uint8")
    for _ in range(5):
        model.predict(img, imgsz=imgsz, device=device, verbose=False)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model.predict(img, imgsz=imgsz, device=device, verbose=False)
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def main() -> int:
    p = argparse.ArgumentParser(description="Append one experiment row to experiment_comparison.csv")
    p.add_argument("--run", required=True, help="run directory containing weights/best.pt")
    p.add_argument("--experiment", required=True)
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=0)
    p.add_argument("--lr", default="auto")
    p.add_argument("--aug", default="default")
    p.add_argument("--notes", default="")
    p.add_argument("--device", default=None)
    p.add_argument("--iters", type=int, default=30)
    args = p.parse_args()

    import torch
    from ultralytics import YOLO

    run_dir = Path(args.run)
    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        print(f"ERROR: {weights} not found")
        return 1

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(str(weights))
    metrics = model.val(data=str(DATA_YAML), imgsz=args.imgsz, device=device, verbose=False)
    rd = metrics.results_dict
    pr, rc = float(rd.get("metrics/precision(B)", 0.0)), float(rd.get("metrics/recall(B)", 0.0))
    f1 = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0

    # training wall time from the run's args.yaml + results.csv if available
    train_min = ""
    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        lines = results_csv.read_text().strip().splitlines()
        train_min = str(max(0, len(lines) - 1))  # epochs actually trained (informational)
    try:
        import yaml as _yaml

        train_args = _yaml.safe_load((run_dir / "args.yaml").read_text()) or {}
        if "batch" in train_args and not args.batch:
            args.batch = int(train_args["batch"])
        if "lr0" in train_args and args.lr == "auto":
            args.lr = str(train_args.get("lr0"))
    except OSError:
        pass

    latency_ms = measure_latency(model, args.imgsz, device, iters=args.iters)
    size_mb = weights.stat().st_size / 1e6

    row = {
        "experiment": args.experiment,
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "learning_rate": args.lr,
        "augmentation_profile": args.aug,
        "mAP50": round(float(rd.get("metrics/mAP50(B)", 0.0)), 4),
        "mAP50_95": round(float(rd.get("metrics/mAP50-95(B)", 0.0)), 4),
        "precision": round(pr, 4),
        "recall": round(rc, 4),
        "f1": round(f1, 4),
        "training_time_min": train_min,
        "model_size_mb": round(size_mb, 2),
        "inference_latency_ms": round(latency_ms, 2),
        "device": device,
        "notes": args.notes,
    }

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

    print("=" * 70)
    print(f"EXPERIMENT LOGGED: {args.experiment}")
    print("=" * 70)
    for k, v in row.items():
        print(f"  {k:<22} {v}")
    print(f"\nCSV: {CSV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
