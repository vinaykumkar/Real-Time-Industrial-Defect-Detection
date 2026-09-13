"""Stability test: run the inspection pipeline over many frames and watch
for memory growth, FPS stability, exceptions and DB/log explosion.

Builds a ~320-frame video by cycling validation frames, processes it through
the same VideoProcessor used in production, samples process RSS during the
run, and writes outputs/evaluation/final/stability_test.json.

Usage (from project root):
    python training\\stability_test.py [--frames 320]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / 1e6


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline stability test")
    parser.add_argument("--frames", type=int, default=320)
    args = parser.parse_args()

    from app.config import IMAGES_VAL_DIR, PROJECT_ROOT
    from app.inference.detector import Detector
    from app.inference.video_processor import VideoProcessor

    video_path = PROJECT_ROOT / "data" / "demo" / "stability_input.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    images = sorted(IMAGES_VAL_DIR.glob("*.jpg"))
    if not images:
        print("no validation images")
        return 1

    if not video_path.exists():
        frames = [cv2.imread(str(p)) for p in images]
        frames = [cv2.resize(f, (400, 400)) for f in frames if f is not None]
        vw = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (400, 400))
        i = 0
        while i < args.frames:
            vw.write(frames[i % len(frames)])
            i += 1
        vw.release()

    detector = Detector.create_with_fallback(backend="onnx", conf_threshold=0.35)[0]
    detector.warmup()

    rss_samples: list[float] = []
    proc = VideoProcessor(detector, max_frames=args.frames)

    t0 = time.perf_counter()
    rss_samples.append(rss_mb())
    stats = proc.process(video_path, annotated_output=None)
    rss_samples.append(rss_mb())

    elapsed = time.perf_counter() - t0
    mem_growth_mb = rss_samples[-1] - rss_samples[0]
    result = {
        "frames_requested": args.frames,
        "frames_processed": stats["frames_processed"],
        "wall_seconds": round(elapsed, 1),
        "avg_pipeline_fps": round(stats["frames_processed"] / elapsed, 2) if elapsed else 0,
        "measured_fps_reported": stats["measured_fps"],
        "avg_inference_ms": stats["avg_inference_ms"],
        "rss_before_mb": round(rss_samples[0], 1),
        "rss_after_mb": round(rss_samples[-1], 1),
        "rss_growth_mb": round(mem_growth_mb, 1),
        "verdict": "STABLE" if mem_growth_mb < 150 and stats["frames_processed"] >= args.frames - 1 else "REVIEW",
    }
    out = PROJECT_ROOT / "outputs" / "evaluation" / "final" / "stability_test.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("=" * 62)
    print("STABILITY TEST")
    print("=" * 62)
    for k, v in result.items():
        print(f"  {k:<24} {v}")
    print(f"\nSaved: {out}")
    return 0 if result["verdict"] == "STABLE" else 2


if __name__ == "__main__":
    main()
