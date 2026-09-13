"""Inspect a video file frame-by-frame and write an annotated output.

Usage (from project root):
    python scripts\\run_video.py path\\to\\video.mp4 [--max-frames 300]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Video defect inspection")
    parser.add_argument("video", help="path to the video file")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    args = parser.parse_args()

    from app.inference.detector import ModelNotTrainedError
    from app.inference.video_processor import VideoProcessor
    from app.services.detection_service import detection_service

    try:
        detector = detection_service.ensure_loaded()
    except ModelNotTrainedError as exc:
        print("CUSTOM DEFECT MODEL NOT TRAINED")
        print(str(exc))
        return 1

    out = PROJECT_ROOT / "outputs" / "detections" / "video_processed" / f"{Path(args.video).stem}_annotated.mp4"
    processor = VideoProcessor(detector, max_frames=args.max_frames, frame_stride=args.stride)
    stats = processor.process(args.video, annotated_output=out)

    print("=" * 60)
    print("VIDEO INSPECTION COMPLETE")
    print("=" * 60)
    for k in ("video", "frames_processed", "frames_with_defects", "total_detections",
              "class_counts", "reject_frames", "measured_fps", "avg_inference_ms"):
        print(f"{k:>22}: {stats[k]}")
    print(f"{'annotated_video':>22}: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
