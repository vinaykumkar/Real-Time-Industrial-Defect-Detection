"""One-off: extract per-class AP for the final model and patch
models/best_metadata.json, outputs/evaluation/final/metrics.json and write
per_class_metrics.csv.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from ultralytics import YOLO

from app.config import DATA_YAML, MODELS_DIR, PROJECT_ROOT


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(MODELS_DIR / "best.pt"))
    m = model.val(data=str(DATA_YAML), imgsz=320, device=device, workers=0, verbose=False)
    ap50 = m.box.ap50
    ap = m.box.ap  # per-class AP@0.5:0.95 (1-D array in ultralytics 8.4)
    per_class = {}
    for i, name in m.names.items():
        per_class[name] = {
            "AP50": round(float(ap50[i]), 4) if i < len(ap50) else None,
            "AP50_95": round(float(np.ravel(ap)[i]) if i < len(np.ravel(ap)) else 0.0, 4),
        }
    print("per_class:", per_class)

    meta_path = MODELS_DIR / "best_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["per_class"] = per_class
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    final_dir = PROJECT_ROOT / "outputs" / "evaluation" / "final"
    fm_path = final_dir / "metrics.json"
    fm = json.loads(fm_path.read_text(encoding="utf-8"))
    fm["per_class"] = per_class
    fm_path.write_text(json.dumps(fm, indent=2), encoding="utf-8")

    with (final_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "AP50", "AP50_95"])
        for name, v in per_class.items():
            w.writerow([name, v["AP50"], v["AP50_95"]])
    print("patched metadata + final metrics + per_class_metrics.csv")


if __name__ == "__main__":
    main()
