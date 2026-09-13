"""Edge inference benchmark: every available backend on identical inputs.

Measures (never estimates) for each backend/device/precision combination:
  avg / median / p95 latency and FPS over N steady-state iterations
  (warmup excluded) on representative validation images.

Writes:
  outputs/evaluation/edge_benchmark.csv  (comparison table)
  outputs/evaluation/final/benchmark.json (full details)

Usage (from project root):
    python training\\benchmark_edge.py [--iters 100]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import IMAGES_VAL_DIR, MODELS_DIR, PROJECT_ROOT  # noqa: E402

CSV_PATH = PROJECT_ROOT / "outputs" / "evaluation" / "edge_benchmark.csv"
COLUMNS = ["backend", "device", "provider", "precision", "imgsz", "avg_inference_ms",
           "median_inference_ms", "p95_inference_ms", "estimated_fps", "model_size_mb", "notes"]


def bench_predict(model, img: np.ndarray, imgsz: int, iters: int, half: bool, device: str) -> list[float]:
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model.predict(img, imgsz=imgsz, device=device, half=half, verbose=False)
        if device == "cuda":
            import torch

            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def bench_onnx(img: np.ndarray, imgsz: int, iters: int) -> list[float]:
    from app.inference.onnx_detector import OnnxDetector

    det = OnnxDetector(MODELS_DIR / "best.onnx", imgsz=imgsz, conf_threshold=0.35)
    for _ in range(8):
        det.detect_raw(img)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        det.detect_raw(img)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def summarize(name: str, device: str, provider: str, precision: str, imgsz: int,
              times: list[float], model_path: Path, notes: str = "") -> dict:
    times_sorted = sorted(times)
    p95 = times_sorted[int(0.95 * (len(times_sorted) - 1))]
    return {
        "backend": name,
        "device": device,
        "provider": provider,
        "precision": precision,
        "imgsz": imgsz,
        "avg_inference_ms": round(statistics.mean(times), 2),
        "median_inference_ms": round(statistics.median(times), 2),
        "p95_inference_ms": round(p95, 2),
        "estimated_fps": round(1000.0 / statistics.median(times), 1),
        "model_size_mb": round(model_path.stat().st_size / 1e6, 2),
        "notes": notes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Edge backend benchmark suite")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=None, help="defaults to metadata image size")
    parser.add_argument("--bench-images", type=int, default=20, help="rotation of val images used as input")
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    imgsz = args.imgsz
    if imgsz is None:
        meta = MODELS_DIR / "best_metadata.json"
        imgsz = int(json.loads(meta.read_text(encoding="utf-8")).get("image_size", 640)) if meta.exists() else 640

    images = sorted(IMAGES_VAL_DIR.glob("*.jpg"))
    if not images:
        print("no validation images")
        return 1
    # rotate a handful of real frames so input isn't degenerate
    step = max(1, len(images) // args.bench_images)
    bench_images = [cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
                    for p in images[::step][: args.bench_images]]
    bench_images = [b for b in bench_images if b is not None]

    pt_path = MODELS_DIR / "best.pt"
    onnx_path = MODELS_DIR / "best.onnx"
    rows: list[dict] = []

    # ---- PyTorch CPU (FP32)
    model = YOLO(str(pt_path))
    model.model.eval()
    times = []
    for i in range(args.iters):
        img = bench_images[i % len(bench_images)]
        t0 = time.perf_counter()
        model.predict(img, imgsz=imgsz, device="cpu", half=False, verbose=False)
        times.append((time.perf_counter() - t0) * 1000.0)
    rows.append(summarize("pytorch", "cpu", "CPU", "FP32", imgsz, times, pt_path))

    # ---- PyTorch CUDA (FP32 + FP16) when available
    if torch.cuda.is_available():
        times = []
        for i in range(args.iters):
            img = bench_images[i % len(bench_images)]
            t0 = time.perf_counter()
            model.predict(img, imgsz=imgsz, device="cuda", half=False, verbose=False)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
        rows.append(summarize("pytorch", "cuda", "CUDA", "FP32", imgsz, times, pt_path))

        times = []
        ok_fp16 = True
        try:
            for i in range(8):
                model.predict(bench_images[i % len(bench_images)], imgsz=imgsz, device="cuda", half=True, verbose=False)
            torch.cuda.synchronize()
            for i in range(args.iters):
                img = bench_images[i % len(bench_images)]
                t0 = time.perf_counter()
                model.predict(img, imgsz=imgsz, device="cuda", half=True, verbose=False)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000.0)
        except Exception as exc:
            ok_fp16 = False
            rows.append({"backend": "pytorch", "device": "cuda", "provider": "CUDA", "precision": "FP16",
                         "imgsz": imgsz, "avg_inference_ms": "", "median_inference_ms": "", "p95_inference_ms": "",
                         "estimated_fps": "", "model_size_mb": round(pt_path.stat().st_size / 1e6, 2),
                         "notes": f"FP16 unstable: {str(exc)[:80]}"})
        if ok_fp16 and times:
            rows.append(summarize("pytorch", "cuda", "CUDA", "FP16", imgsz, times, pt_path))

    # ---- ONNX Runtime CPU
    if onnx_path.exists():
        times = bench_onnx(bench_images[0], imgsz, args.iters)
        rows.append(summarize("onnxruntime", "cpu", "CPUExecutionProvider", "FP32", imgsz, times, onnx_path))
        providers = __import__("onnxruntime").get_available_providers()
        if "CUDAExecutionProvider" not in providers:
            rows.append({"backend": "onnxruntime", "device": "cuda", "provider": "CUDAExecutionProvider",
                         "precision": "FP32", "imgsz": imgsz, "avg_inference_ms": "", "median_inference_ms": "",
                         "p95_inference_ms": "", "estimated_fps": "",
                         "model_size_mb": round(onnx_path.stat().st_size / 1e6, 2),
                         "notes": "unavailable: onnxruntime built without CUDA EP"})
        # TensorRT
        try:
            import tensorrt  # noqa: F401

            trt_ok = True
        except ImportError:
            trt_ok = False
        if not trt_ok:
            rows.append({"backend": "tensorrt", "device": "cuda", "provider": "TensorRT",
                         "precision": "FP16", "imgsz": imgsz, "avg_inference_ms": "", "median_inference_ms": "",
                         "p95_inference_ms": "", "estimated_fps": "",
                         "model_size_mb": "", "notes": "TensorRT unavailable — continuing with ONNX Runtime/PyTorch"})

    # ---- write outputs
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    out_dir = PROJECT_ROOT / "outputs" / "evaluation" / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark.json").write_text(
        json.dumps({"imgsz": imgsz, "iters": args.iters, "benchmarks": rows}, indent=2), encoding="utf-8")

    print("=" * 80)
    print(f"EDGE BENCHMARK (median of {args.iters} steady-state iterations, warmup excluded, imgsz {imgsz})")
    print("=" * 80)
    for r in rows:
        if r.get("avg_inference_ms") == "":
            print(f"  {r['backend']:<12} {r['device']:<5} {r['precision']:<5} — {r['notes']}")
        else:
            print(f"  {r['backend']:<12} {r['device']:<5} {r['precision']:<5} "
                  f"avg {r['avg_inference_ms']:>7.2f}  med {r['median_inference_ms']:>7.2f}  "
                  f"p95 {r['p95_inference_ms']:>7.2f}  {r['estimated_fps']:>6.1f} FPS")
    print(f"\nCSV: {CSV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
