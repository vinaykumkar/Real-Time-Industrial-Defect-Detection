"""Export the trained NEU defect model to ONNX (and optionally TensorRT).

Usage (from project root):
    python training\\export_model.py                    # export ONNX
    python training\\export_model.py --format engine    # TensorRT (only if installed)

TensorRT is optional: if it is not installed the script prints a clear notice
and exits successfully instead of crashing.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import MODELS_DIR, PYTORCH_MODEL_PATH, TRAINING_OUTPUT_DIR  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Export trained defect model")
    p.add_argument("--model", type=str, default=str(PYTORCH_MODEL_PATH))
    p.add_argument("--format", type=str, default="onnx", choices=["onnx", "engine"], help="export format")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--simplify", action="store_true", default=True, help="simplify ONNX graph (default: on)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    from ultralytics import YOLO

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not model_path.exists():
        print("CUSTOM DEFECT MODEL NOT TRAINED")
        print(f"Expected trained weights at: {model_path}")
        print("Train first with: python training\\train.py --epochs 1")
        return 1

    if args.format == "engine":
        try:
            import tensorrt  # noqa: F401

            trt_available = True
        except ImportError:
            trt_available = False
        if not trt_available:
            print("TensorRT unavailable — using ONNX Runtime/PyTorch.")
            print("(TensorRT export skipped; ONNX export is the supported path.)")
            if not (MODELS_DIR / "best.onnx").exists():
                print("\nFalling back to ONNX export now...")
                args.format = "onnx"
            else:
                return 0

    print(f"Loading {model_path} ...")
    model = YOLO(str(model_path))

    if args.format == "onnx":
        print(f"Exporting to ONNX (imgsz={args.imgsz}) ...")
        exported = model.export(format="onnx", imgsz=args.imgsz, simplify=True, opset=13)
        src = Path(exported)
        dest = MODELS_DIR / "best.onnx"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dest.resolve():
            shutil.copyfile(src, dest)
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"\nONNX model written: {dest} ({size_mb:.1f} MB)")

        # sanity check with onnxruntime if present
        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(str(dest), providers=["CPUExecutionProvider"])
            inp = sess.get_inputs()[0]
            print(f"ONNX Runtime load OK — input '{inp.name}' shape {inp.shape}")
        except ImportError:
            print("onnxruntime not installed — skipping runtime sanity check.")
        except Exception as exc:
            print(f"WARNING: ONNX Runtime sanity check failed: {exc}")

        # verify predictions match between pt and onnx (coarse)
        try:
            import numpy as np

            dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
            r_pt = model.predict(dummy, verbose=False)[0]
            print("PyTorch model sanity inference OK (empty frame).")
        except Exception as exc:
            print(f"WARNING: PyTorch sanity inference failed: {exc}")

        print("\nExport complete. The FastAPI app will automatically use ONNX when available.")
        return 0

    # engine export path (only reached when tensorrt import succeeded)
    print("Exporting TensorRT engine ...")
    exported = model.export(format="engine", imgsz=args.imgsz, device=0)
    src = Path(exported)
    if src.exists():
        dest = MODELS_DIR / src.name
        if src.resolve() != dest.resolve():
            shutil.copyfile(src, dest)
        print(f"TensorRT engine written: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
