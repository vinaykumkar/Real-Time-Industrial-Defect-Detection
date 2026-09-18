```python
"""YOLOv8 training pipeline for the NEU defect detection dataset.

Smoke test:
    python training\\train.py --epochs 1

Full training:
    python training\\train.py --epochs 100 --batch 16 --imgsz 640

CUDA is used automatically when available; otherwise training falls back to CPU.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (  # noqa: E402
    DATA_YAML,
    MODELS_DIR,
    TRAINING_OUTPUT_DIR,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLOv8 model on the NEU defect dataset."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Training batch size.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Training device. Automatically selects CUDA or CPU when omitted.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of dataloader workers. Use 0 on some Windows systems.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Base YOLO model weights or architecture file.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name of the training run.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of the dataset used for training.",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Keep generated weights separate and do not overwrite models/best.pt.",
    )
    parser.add_argument(
        "--mosaic",
        type=float,
        default=None,
        help="Mosaic augmentation probability.",
    )
    parser.add_argument(
        "--mixup",
        type=float,
        default=None,
        help="MixUp augmentation probability.",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=None,
        help="Initial learning rate override.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def select_device(requested_device: str | None, torch) -> str:
    """Return the requested device or automatically select CUDA/CPU."""
    if requested_device:
        return requested_device

    return "cuda" if torch.cuda.is_available() else "cpu"


def build_optional_training_args(args: argparse.Namespace) -> dict:
    """Build optional Ultralytics training parameters."""
    optional = {}

    if args.mosaic is not None:
        optional["mosaic"] = args.mosaic

    if args.mixup is not None:
        optional["mixup"] = args.mixup

    if args.lr0 is not None:
        optional["lr0"] = args.lr0

    return optional


def copy_training_outputs(best_weights: Path) -> None:
    """Copy the best model and selected training metrics to project outputs."""
    destination = MODELS_DIR / "best.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(best_weights, destination)
    print(f"\nBest weights copied to {destination}")

    run_directory = best_weights.parent

    for filename in (
        "results.csv",
        "confusion_matrix.png",
        "results.png",
    ):
        source_file = run_directory / filename

        if source_file.exists():
            shutil.copyfile(
                source_file,
                TRAINING_OUTPUT_DIR / filename,
            )

    print(f"Training curves/metrics copied to {TRAINING_OUTPUT_DIR}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    # Heavy dependencies are imported only after argument parsing.
    import torch
    from ultralytics import YOLO, settings

    print("=" * 70)
    print("NEU DEFECT DETECTION — TRAINING")
    print("=" * 70)

    # Verify that the prepared dataset exists.
    if not DATA_YAML.exists():
        print(f"ERROR: {DATA_YAML} not found.")
        print("Run first: python training\\prepare_dataset.py")
        return 1

    device = select_device(args.device, torch)

    print(f"Device        : {device}")

    if device.startswith("cuda"):
        print(f"GPU           : {torch.cuda.get_device_name(0)}")

    print(f"Base model    : {args.model}")
    print(f"Epochs        : {args.epochs}")
    print(f"Batch / imgsz : {args.batch} / {args.imgsz}")
    print(f"Dataset       : {DATA_YAML}")

    # Store all Ultralytics training runs inside the project.
    runs_directory = TRAINING_OUTPUT_DIR / "runs"
    runs_directory.mkdir(parents=True, exist_ok=True)

    settings.update({
        "runs_dir": str(runs_directory),
    })

    model = YOLO(args.model)

    run_name = (
        args.name
        if args.name
        else f"defect_{Path(args.model).stem}_e{args.epochs}"
    )

    training_options = {
        "data": str(DATA_YAML),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": device,
        "workers": args.workers,
        "name": run_name,
        "patience": args.patience,
        "project": str(runs_directory),
        "exist_ok": True,
        "seed": 42,
        "fraction": args.fraction,

        # Moderate augmentation for 200x200 grayscale defect images.
        "degrees": 5.0,
        "fliplr": 0.5,
        "flipud": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.2,
    }

    training_options.update(
        build_optional_training_args(args)
    )

    results = model.train(**training_options)

    # -----------------------------------------------------------------------
    # Handle generated model weights
    # -----------------------------------------------------------------------

    best_weights = None

    if hasattr(model, "trainer") and model.trainer.best:
        best_weights = Path(model.trainer.best)

    if args.no_copy:
        if best_weights and best_weights.exists():
            print(
                f"\nSmoke-test weights kept at {best_weights} "
                "(models/best.pt NOT overwritten)"
            )

    elif best_weights and best_weights.exists():
        copy_training_outputs(best_weights)

    else:
        print(
            "\nWARNING: training finished but no best.pt was produced."
        )

    print(f"\nResults object: {results}")

    print("Next steps:")
    print(
        "  python training\\evaluate.py "
        "         # evaluate on the validation set"
    )
    print(
        "  python training\\export_model.py "
        "      # export to ONNX for edge inference"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```
