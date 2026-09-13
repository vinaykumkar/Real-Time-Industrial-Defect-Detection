"""YOLOv8 training pipeline for the NEU defect dataset.

Smoke test (fast sanity check):
    python training\\train.py --epochs 1

Full training:
    python training\\train.py --epochs 100 --batch 16 --imgsz 640

Automatically selects CUDA if available, otherwise CPU.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import DATA_YAML, MODELS_DIR, PROJECT_ROOT, TRAINING_OUTPUT_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8 on the NEU defect dataset")
    p.add_argument("--epochs", type=int, default=1, help="number of training epochs (1 = smoke test)")
    p.add_argument("--batch", type=int, default=16, help="batch size")
    p.add_argument("--imgsz", type=int, default=640, help="training image size")
    p.add_argument("--device", type=str, default=None, help="device: auto-detect when omitted (cuda or cpu)")
    p.add_argument("--workers", type=int, default=4, help="dataloader workers (use 0 on some Windows setups)")
    p.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="base model weights (yolov8n.pt recommended; a .yaml architecture also works)",
    )
    p.add_argument("--name", type=str, default=None, help="run name (default: defect_yolov8n)")
    p.add_argument("--patience", type=int, default=20, help="early stopping patience")
    p.add_argument(
        "--fraction",
        type=float, default=1.0,
        help="fraction of the dataset to train on (e.g. 0.2 = fast pipeline smoke test)",
    )
    p.add_argument(
        "--no-copy",
        action="store_true",
        help="do NOT overwrite models/best.pt with this run's weights "
        "(use for pipeline smoke tests so real trained weights are preserved)",
    )
    p.add_argument("--mosaic", type=float, default=None, help="mosaic augmentation probability (Ultralytics default 1.0)")
    p.add_argument("--mixup", type=float, default=None, help="mixup augmentation probability (Ultralytics default 0.0)")
    p.add_argument("--lr0", type=float, default=None, help="override initial learning rate")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    import torch  # heavy import, deferred until after arg parsing
    from ultralytics import YOLO, settings

    print("=" * 70)
    print("NEU DEFECT DETECTION — TRAINING")
    print("=" * 70)

    if not DATA_YAML.exists():
        print(f"ERROR: {DATA_YAML} not found.")
        print("Run first:  python training\\prepare_dataset.py")
        return 1

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device        : {device}")
    if device.startswith("cuda"):
        print(f"GPU           : {torch.cuda.get_device_name(0)}")
    print(f"Base model    : {args.model}")
    print(f"Epochs        : {args.epochs}")
    print(f"Batch / imgsz : {args.batch} / {args.imgsz}")
    print(f"Dataset       : {DATA_YAML}")

    # Keep all training artefacts inside the project (outputs/training)
    runs_dir = TRAINING_OUTPUT_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    settings.update({"runs_dir": str(runs_dir)})

    model = YOLO(args.model)
    run_name = args.name or f"defect_{Path(args.model).stem}_e{args.epochs}"

    results = model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        workers=args.workers,
        name=run_name,
        patience=args.patience,
        project=str(runs_dir),
        exist_ok=True,
        seed=42,
        fraction=args.fraction,
        # defect images are 200x200 grayscale; modest augmentation is enough
        degrees=5.0,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.2,
        **(
            {}
            if all(v is None for v in (args.mosaic, args.mixup, args.lr0))
            else {
                **({"mosaic": args.mosaic} if args.mosaic is not None else {}),
                **({"mixup": args.mixup} if args.mixup is not None else {}),
                **({"lr0": args.lr0} if args.lr0 is not None else {}),
            }
        ),
    )

    # Locate the best weights produced by this run
    best_pt = Path(model.trainer.best) if hasattr(model, "trainer") and model.trainer.best else None
    if args.no_copy:
        if best_pt and best_pt.exists():
            print(f"\nSmoke-test weights kept at {best_pt} (models/best.pt NOT overwritten)")
    elif best_pt and best_pt.exists():
        dest = MODELS_DIR / "best.pt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(best_pt, dest)
        print(f"\nBest weights copied to {dest}")
        metrics_dir = best_pt.parent
        for fname in ("results.csv", "confusion_matrix.png", "results.png"):
            src = metrics_dir / fname
            if src.exists():
                shutil.copyfile(src, TRAINING_OUTPUT_DIR / fname)
        print(f"Training curves/metrics copied to {TRAINING_OUTPUT_DIR}")
    else:
        print("\nWARNING: training finished but no best.pt was produced.")

    print(f"\nResults object: {results}")
    print("Next steps:")
    print("  python training\\evaluate.py          # evaluate on the validation set")
    print("  python training\\export_model.py      # export to ONNX for edge inference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
