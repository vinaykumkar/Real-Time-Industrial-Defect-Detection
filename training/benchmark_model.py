"""Benchmark the final model: real inference latency + FPS per device/backend.

Measures (never estimates):
  - PyTorch on CPU (and GPU when CUDA works)
  - ONNX Runtime on CPU (if models/best.onnx exists)

Warmup excluded from timing; median + mean of N iterations on a real
validation image. Results -> outputs/evaluation/final/benchmark.json

Usage (from project root):
    python training\\benchmark_model.py [--model models\\best.pt] [--iters 100]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import IMAGES_VAL_DIR, PROJECT_ROOT  # noqa: E402


def bench_pt(weights: Path, img: np.ndarray, device: str, imgsz: int, iters: int) -> dict:
    import torch

    from ultralytics import YOLO

    model = YOLO(str(weights))
    # warmup (excluded from timing)
    for _ in range(8):
        model.predict(img, imgsz=imgsz, device=device, verbose=False)
    if device == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model.predict(img, imgsz=imgsz, device=device, verbose=False)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "backend": "pytorch",
        "device": device,
        "median_ms": round(statistics.median(times), 2),
        "mean_ms": round(statistics.mean(times), 2),
        "fps_from_median": round(1000.0 / statistics.median(times), 1),
        "iterations": iters,
    }


def bench_onnx(weights: Path, img: np.ndarray, imgsz: int, iters: int) -> dict | None:
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    if not weights.exists():
        return None
    from app.inference.onnx_detector import OnnxDetector

    det = OnnxDetector(weights, imgsz=imgsz, conf_threshold=0.25)
    for _ in range(8):
        det.detect_raw(img)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        det.detect_raw(img)
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "backend": "onnxruntime",
        "device": det.provider_used,
        "median_ms": round(statistics.median(times), 2),
        "mean_ms": round(statistics.mean(times), 2),
        "fps_from_median": round(1000.0 / statistics.median(times), 1),
        "iterations": iters,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark final defect model")
    parser.add_argument("--model", type=str, default=str(PROJECT_ROOT / "models" / "best.pt"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    import torch

    weights = Path(args.model)
    if not weights.exists():
        print(f"model not found: {weights}")
        return 1

    images = sorted(IMAGES_VAL_DIR.glob("*.jpg"))
    if not images:
        print("no validation images")
        return 1
    img = cv2.imdecode(np.fromfile(str(images[0]), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("benchmark image unreadable")
        return 1

    results = {"model": str(weights), "imgsz": args.imgsz, "benchmarks": []}
    results["benchmarks"].append(bench_pt(weights, img, "cpu", args.imgsz, args.iters))
    if torch.cuda.is_available():
        try:
            results["benchmarks"].append(bench_pt(weights, img, "cuda", args.imgsz, args.iters))
        except Exception as exc:
            results["benchmarks"].append({"backend": "pytorch", "device": "cuda", "error": str(exc)[:200]})
    onnx_b = bench_onnx(PROJECT_ROOT / "models" / "best.onnx", img, args.imgsz, args.iters)
    if onnx_b:
        results["benchmarks"].append(onnx_b)

    results["model_size_mb"] = round(weights.stat().st_size / 1e6, 2)

    out_dir = PROJECT_ROOT / "outputs" / "evaluation" / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("=" * 64)
    print("INFERENCE BENCHMARK (median of", args.iters, "iterations, warmup excluded)")
    print("=" * 64)
    for b in results["benchmarks"]:
        if "error" in b:
            print(f"  {b['device']:<10} ERROR: {b['error'][:80]}")
        else:
            print(f"  {b['backend']:<12} {b['device']:<24} {b['median_ms']:>8.2f} ms   {b['fps_from_median']:>7.1f} FPS")
    print(f"\nSaved: {out_dir / 'benchmark.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
