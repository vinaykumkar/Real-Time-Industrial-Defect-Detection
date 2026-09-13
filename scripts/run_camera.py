"""Real-time edge inspection runtime: webcam / video file / RTSP.

Single reusable pipeline (also used by the API's live camera service):
    capture -> preprocess -> backend inference -> QC -> temporal confirmation
    -> PASS/REJECT -> PLC dispatch (debounced) -> overlay -> display/write

Usage (from project root, venv activated):
    python scripts\\run_camera.py                       # webcam 0, auto backend
    python scripts\\run_camera.py --source 1
    python scripts\\run_camera.py --source rtsp://user:pass@ip/stream
    python scripts\\run_camera.py --source path\\to\\video.mp4 --output out.mp4
    python scripts\\run_camera.py --no-display          # headless edge mode
    python scripts\\run_camera.py --backend onnx --conf 0.35 --process-every 2

Press q or ESC to quit in display mode.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="Real-time edge defect inspection")
    p.add_argument("--source", default="0", help="camera index, rtsp:// URL, or video file path")
    p.add_argument("--backend", default="auto", choices=["auto", "onnx", "pytorch"],
                   help="inference backend (TensorRT engine is used automatically when present)")
    p.add_argument("--conf", type=float, default=None, help="confidence threshold (default from config/env)")
    p.add_argument("--iou", type=float, default=None, help="NMS IoU threshold (default from config/env)")
    p.add_argument("--process-every", type=int, default=None,
                   help="process every Nth frame, display keeps last result (default: config)")
    p.add_argument("--temporal-window", type=int, default=None, help="temporal confirmation window (default: config)")
    p.add_argument("--min-defect-frames", type=int, default=None, help="defect frames required to confirm REJECT")
    p.add_argument("--plc-cooldown-ms", type=float, default=None, help="min ms between PLC REJECT commands")
    p.add_argument("--no-display", action="store_true", help="headless edge mode: no GUI window")
    p.add_argument("--save", action="store_true", help="save annotated frames of REJECT events")
    p.add_argument("--output", type=str, default=None, help="write annotated video to this mp4 path")
    p.add_argument("--max-frames", type=int, default=None, help="stop after N processed frames")
    p.add_argument("--fps-window", type=int, default=30, help="rolling FPS window (frames)")
    return p.parse_args()


def main() -> int:
    from app.config import (
        CONFIDENCE_THRESHOLD,
        IOU_THRESHOLD,
        MIN_DEFECT_FRAMES,
        PLC_REJECT_COOLDOWN_MS,
        PROCESS_EVERY_N_FRAMES,
        TEMPORAL_WINDOW,
        PROJECT_ROOT,
    )
    from app.inference.detector import ModelNotTrainedError
    from app.inference.preprocessing import draw_detection
    from app.services.logging_service import setup_logging, get_logger
    from app.api.plc import plc as plc_line
    from app.services.metrics_service import FRAMES_PROCESSED, PLC_EVENTS
    from app.services.qc import PLCDebouncer, TemporalValidator, inspect_frame

    logger = get_logger("defect.edge")
    setup_logging()
    args = parse_args()

    conf = args.conf if args.conf is not None else CONFIDENCE_THRESHOLD
    iou = args.iou if args.iou is not None else IOU_THRESHOLD
    process_every = args.process_every if args.process_every is not None else PROCESS_EVERY_N_FRAMES
    temporal = TemporalValidator(
        args.temporal_window if args.temporal_window is not None else TEMPORAL_WINDOW,
        args.min_defect_frames if args.min_defect_frames is not None else MIN_DEFECT_FRAMES,
    )
    debouncer = PLCDebouncer(args.plc_cooldown_ms if args.plc_cooldown_ms is not None else PLC_REJECT_COOLDOWN_MS)

    # ------------------------------------------------------------- model
    from app.inference.detector import Detector

    try:
        detector, fallback_log = Detector.create_with_fallback(
            backend=args.backend, conf_threshold=conf, iou_threshold=iou
        )
    except (ModelNotTrainedError, RuntimeError) as exc:
        print("=" * 60)
        print("CUSTOM DEFECT MODEL NOT AVAILABLE")
        print("=" * 60)
        print(str(exc))
        print("Train it with:  python training\\train.py --epochs 1")
        return 1
    perf = detector.get_performance_info()
    print(f"Backend: {perf['backend']}  device: {perf['device']}  model: {Path(perf['model_path']).name}  imgsz: {perf['imgsz']}")
    for entry in fallback_log:
        print(f"  fallback: {entry}")
    print(f"Temporal confirmation: {temporal.min_defect_frames}/{temporal.window} frames | "
          f"PLC cooldown: {debouncer.cooldown_ms:.0f} ms | process-every: {process_every}")

    # ------------------------------------------------------------- source
    source: int | str
    if str(args.source).isdigit():
        source = int(args.source)
    else:
        source = args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"ERROR: camera source unavailable: {args.source!r}")
        return 1
    cam_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = capture.get(cv2.CAP_PROP_FPS)
    print(f"Source opened: {args.source} ({cam_w}x{cam_h}" + (f" @ {cam_fps:.1f} fps" if cam_fps else "") + ")")

    writer = None
    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 cam_fps or 25.0, (cam_w, cam_h))
        print(f"Annotated output: {out_path}")
    save_dir = PROJECT_ROOT / "outputs" / "detections"
    save_dir.mkdir(parents=True, exist_ok=True)

    # steady-state timing accumulators (warmup excluded)
    timings = {"capture_ms": deque(maxlen=120), "preprocess_ms": deque(maxlen=120),
               "inference_ms": deque(maxlen=120), "postprocess_ms": deque(maxlen=120),
               "render_ms": deque(maxlen=120)}
    fps_hist: deque[float] = deque(maxlen=max(5, args.fps_window))
    fps_value = 0.0
    fps_t0, fps_frames = time.perf_counter(), 0
    frame_idx = processed = plc_rejects = 0
    last_confirmed_state = "PASS"

    print("Running live inspection — press q or ESC to quit." + ("" if not args.no_display else " (headless: Ctrl+C to stop)"))
    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                print("stream ended / camera disconnected")
                break
            timings["capture_ms"].append((time.perf_counter() - t0) * 1000.0)
            frame_idx += 1
            if frame_idx % max(1, process_every) != 0:
                continue

            result = inspect_frame(detector, frame, source=f"camera:{args.source}")
            timings["inference_ms"].append(result.timings.inference_ms)
            timings["preprocess_ms"].append(result.timings.preprocess_ms)
            timings["postprocess_ms"].append(result.timings.postprocess_ms)

            confirmed_reject = temporal.update(result.defect_detected)

            # PLC dispatch on CONFIRMED state change, debounced
            decision_latency_ms = None
            plc_event = None
            confirmed_state = "REJECT" if confirmed_reject else "PASS"
            if confirmed_state != last_confirmed_state:
                t_decision = time.perf_counter()
                if confirmed_state == "REJECT":
                    if debouncer.allow("REJECT"):
                        plc_event = plc_line.send_sort_command(
                            "REJECT", defect=result.dominant_defect, confidence=result.highest_confidence)
                        decision_latency_ms = (time.perf_counter() - t_decision) * 1000.0
                        PLC_EVENTS.labels(action="REJECT").inc()
                        plc_rejects += 1
                        logger.info("PLC REJECT dispatched (%.1f ms decision)", decision_latency_ms)
                else:
                    plc_event = plc_line.send_sort_command("PASS")
                    PLC_EVENTS.labels(action="PASS").inc()
                last_confirmed_state = confirmed_state

            # ------------------------------------------------------ render
            t_r = time.perf_counter()
            annotated = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            for d in result.detections:
                b = d["bbox"]
                draw_detection(annotated, int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"]),
                               d["class"], d["confidence"])
                cv2.putText(annotated, d["severity"], (int(b["x2"]) + 2, int(b["y1"]) + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 215, 255), 1, cv2.LINE_AA)
            # rolling FPS
            fps_frames += 1
            now = time.perf_counter()
            if now - fps_t0 >= 0.5:
                fps_value = fps_frames / (now - fps_t0)
                fps_hist.append(fps_value)
                fps_t0, fps_frames = now, 0

            status_color = (0, 170, 0) if confirmed_state == "PASS" else (0, 0, 220)
            cv2.rectangle(annotated, (0, 0), (330, 44), status_color, -1)
            cv2.putText(annotated, confirmed_state, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            hud = [
                f"Defects: {result.defect_count}  Backend: {detector.get_backend_name()}  Camera: ONLINE",
                f"FPS: {fps_value:.1f}  Inference: {result.timings.inference_ms:.1f} ms  "
                f"Pipeline: {timings['capture_ms'][-1] + result.timings.total_ms + (time.perf_counter() - t_r) * 1000:.1f} ms",
                f"Temporal: {sum(temporal.recent)}/{len(temporal.recent)} defect frames  PLC rejects: {plc_rejects}",
            ]
            y = 66
            for ln in hud:
                cv2.putText(annotated, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 255, 255), 1, cv2.LINE_AA)
                y += 20
            render_ms = (time.perf_counter() - t_r) * 1000.0
            timings["render_ms"].append(render_ms)

            if writer is not None:
                writer.write(annotated)
            if args.save and confirmed_state == "REJECT":
                cv2.imwrite(str(save_dir / f"reject_{time.strftime('%Y%m%d_%H%M%S')}.jpg"), annotated)
            FRAMES_PROCESSED.inc()
            processed += 1

            if not args.no_display:
                cv2.imshow("Industrial Defect Detection  (q=quit)", annotated)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
            if args.max_frames and processed >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("interrupted — shutting down cleanly")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    # ------------------------------------------------------------- summary
    def med(key):
        vals = sorted(timings[key])
        return sum(vals) / len(vals) if vals else 0.0, (vals[len(vals) // 2] if vals else 0.0)

    print("=" * 62)
    print("SESSION SUMMARY (steady-state, measured)")
    print("=" * 62)
    print(f"frames captured/processed : {frame_idx} / {processed}")
    if processed:
        parts = []
        for key in ("capture_ms", "preprocess_ms", "inference_ms", "postprocess_ms", "render_ms"):
            avg, med_v = med(key)
            parts.append(f"{key.replace('_ms','')} avg {avg:.1f} / med {med_v:.1f} ms")
        for line in parts:
            print("  " + line)
        fps_vals = sorted(fps_hist)
        print(f"  rolling FPS: last {fps_value:.1f}, median {fps_vals[len(fps_vals)//2] if fps_vals else 0:.1f}")
        print(f"  PLC REJECT commands (debounced): {plc_rejects}")
    if writer is not None and args.output:
        print(f"annotated video saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
