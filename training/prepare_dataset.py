"""Prepare the YOLO dataset: inspect -> convert -> validate -> data.yaml.

One-stop runner for the dataset pipeline.

Usage (from project root):
    python training\\prepare_dataset.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import DATA_YAML, LABELS_TRAIN_DIR, LABELS_VAL_DIR  # noqa: E402


def run_step(script: Path, description: str) -> None:
    print(f"\n{'=' * 70}\nSTEP: {description}\n{'=' * 70}")
    result = subprocess.run([sys.executable, str(script)], cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"\nFAILED: {description} (exit code {result.returncode})")
        raise SystemExit(result.returncode)


def main() -> int:
    run_step(PROJECT_ROOT / "training" / "inspect_dataset.py", "Inspect raw dataset")
    run_step(PROJECT_ROOT / "training" / "convert_voc_to_yolo.py", "Convert VOC -> YOLO")

    print(f"\n{'=' * 70}\nFINAL CHECKS\n{'=' * 70}")
    ok = True
    if DATA_YAML.exists():
        print(f"[OK] {DATA_YAML}")
    else:
        print(f"[MISSING] {DATA_YAML}")
        ok = False

    for d in (LABELS_TRAIN_DIR, LABELS_VAL_DIR):
        n = len(list(d.glob("*.txt"))) if d.exists() else 0
        n_img = len(list((d.parent.parent / "images" / d.name).glob("*.jpg"))) if d.exists() else 0
        print(f"[{'OK' if n else 'MISSING'}] {d}: {n} label files")
        if n == 0:
            ok = False

    if not ok:
        return 1
    print("\nDataset preparation complete.")
    print("Next: python training\\train.py --epochs 1   (smoke test)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
