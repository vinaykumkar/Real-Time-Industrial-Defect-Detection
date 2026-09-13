"""Video file inspection: frame-by-frame defect detection with measured FPS.

Used by the dashboard's Video Inspection page and the CLI:
    python scripts\\run_video.py path\\to\\video.mp4
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from app.inference.detector import Detector
from app.services.qc import InspectionResult, inspect_frame

class VideoProcessor:
    """Processes a video file frame by frame through the defect detector."""

    def __init__(self, detector: Detector, max_frames: int | None = None, frame_stride: int = 1):
        self.detector = detector
        self.max_frames = max_frames
        self.frame_stride = max(1, frame_stride)

    def process(self, video_path: str | Path, annotated_output: str | Path | None = None) -> dict:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"video not found: {video_path}")
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise IOError(f"cannot open video: {video_path}")

        src_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if annotated_output is not None:
            out_path = Path(annotated_output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, src_fps, (width, height))

        stats = {
            "video": video_path.name,
            "source_frames": total_frames,
            "frames_processed": 0,
            "frames_with_defects": 0,
            "reject_frames": 0,
            "total_detections": 0,
            "class_counts": Counter(),
            "reject_events": [],
            "measured_fps": 0.0,
            "avg_inference_ms": 0.0,
        }
        inference_ms_samples: list[float] = []
        frame_idx = 0
        processed = 0
        import time

        t_start = time.perf_counter()
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_idx % self.frame_stride == 0:
                    result: InspectionResult = inspect_frame(self.detector, frame, source=video_path.name)
                    stats["frames_processed"] += 1
                    processed += 1
                    stats["total_detections"] += len(result.detections)
                    inference_ms_samples.append(result.timings.inference_ms)
                    if result.detections:
                        stats["frames_with_defects"] += 1
                        for d in result.detections:
                            stats["class_counts"][d["class"]] += 1
                    if result.inspection_status == "REJECT":
                        stats["reject_frames"] += 1
                        stats["reject_events"].append(
                            {
                                "frame": frame_idx,
                                "dominant_defect": result.dominant_defect,
                                "confidence": result.highest_confidence,
                            }
                        )
                    if writer is not None:
                        annotated = frame.copy()
                        for d in result.detections:
                            from app.inference.preprocessing import draw_detection

                            b = d["bbox"]
                            draw_detection(
                                annotated,
                                int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"]),
                                d["class"], d["confidence"],
                            )
                        status_color = (0, 200, 0) if result.inspection_status == "PASS" else (0, 0, 255)
                        cv2.putText(annotated, result.inspection_status, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, status_color, 3)
                        writer.write(annotated)
                    if self.max_frames and processed >= self.max_frames:
                        break
                frame_idx += 1
        finally:
            capture.release()
            if writer is not None:
                writer.release()

        elapsed = time.perf_counter() - t_start
        if processed > 0:
            stats["measured_fps"] = round(processed / elapsed, 2) if elapsed > 0 else 0.0
            stats["avg_inference_ms"] = round(sum(inference_ms_samples) / len(inference_ms_samples), 2)
        stats["class_counts"] = dict(stats["class_counts"])
        stats["reject_events"] = stats["reject_events"][:50]
        if annotated_output is not None:
            stats["annotated_video"] = str(Path(annotated_output))
        return stats
