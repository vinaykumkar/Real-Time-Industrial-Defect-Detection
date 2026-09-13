"""Background camera worker for live inspection.

Runs an OpenCV capture loop in a daemon thread, inspects every frame through
the DetectionService, publishes the latest annotated JPEG for the MJPEG
stream endpoint, and updates camera Prometheus metrics. FPS and latency are
always measured, never fabricated.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.config import (
    CAMERA_SOURCE,
    MIN_DEFECT_FRAMES,
    PLC_REJECT_COOLDOWN_MS,
    PROCESS_EVERY_N_FRAMES,
    TEMPORAL_WINDOW,
)
from app.services.logging_service import get_logger
from app.services.metrics_service import set_camera_state
from app.services.qc import PLCDebouncer, TemporalValidator

logger = get_logger("defect.camera")


def _parse_source(raw: str | int):
    if raw is None:
        raw = CAMERA_SOURCE
    if isinstance(raw, int):
        return raw
    s = str(raw).strip()
    if s.lower() in ("", "none", "null"):
        s = str(CAMERA_SOURCE)
    if s.isdigit():
        return int(s)
    return s  # rtsp://... or file path


@dataclass
class CameraSnapshot:
    online: bool = False
    source: str = ""
    fps: float = 0.0
    frames_processed: int = 0
    frames_captured: int = 0
    last_result: dict | None = None
    last_timings: dict = field(default_factory=dict)
    confirmed_status: str = "PASS"
    last_plc_event: dict | None = None
    last_decision_latency_ms: float | None = None
    started_at: float = 0.0
    error: str | None = None


class CameraWorker(threading.Thread):
    """Continuous capture + inspect loop."""

    def __init__(self, detection_service, source=CAMERA_SOURCE):
        super().__init__(daemon=True, name="camera-worker")
        self.detection_service = detection_service
        self.source = _parse_source(source)
        self._stop_event = threading.Event()
        self._jpeg_lock = threading.Lock()
        self._jpeg: bytes | None = None
        self.snapshot = CameraSnapshot(source=str(self.source))
        self._proc_times: list[float] = []

    # ------------------------------------------------------------- control
    def stop(self) -> None:
        self._stop_event.set()

    # -------------------------------------------------------------- stream
    def latest_jpeg(self) -> bytes | None:
        with self._jpeg_lock:
            return self._jpeg

    # ---------------------------------------------------------------- main
    def run(self) -> None:
        logger.info("camera connecting to source=%s", self.source)
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            msg = f"camera unavailable: source={self.source}"
            logger.error(msg)
            self.snapshot.error = msg
            self.snapshot.online = False
            set_camera_state(False)
            return

        self.snapshot.online = True
        self.snapshot.started_at = time.time()
        self.snapshot.error = None
        set_camera_state(True)
        logger.info("camera online: %s", self.source)

        detector = self.detection_service.ensure_loaded()
        temporal = TemporalValidator(TEMPORAL_WINDOW, MIN_DEFECT_FRAMES)
        debouncer = PLCDebouncer(PLC_REJECT_COOLDOWN_MS)
        from app.api.plc import plc as plc_line

        fps_t0 = time.perf_counter()
        fps_frames = 0
        last_status: str = "PASS"
        last_recorded_status: str = "PASS"
        last_history_event_ms = -float("inf")
        HISTORY_MIN_INTERVAL_MS = 1500.0  # event throttling for live inspection
        frame_idx = 0

        try:
            while not self._stop_event.is_set():
                t_capture = time.perf_counter()
                ok, frame = capture.read()
                if not ok:
                    logger.warning("camera frame grab failed; stopping loop")
                    self.snapshot.error = "frame grab failed (stream ended?)"
                    break
                capture_ms = (time.perf_counter() - t_capture) * 1000.0
                self.snapshot.frames_captured += 1
                frame_idx += 1

                if frame_idx % max(1, PROCESS_EVERY_N_FRAMES) != 0:
                    continue  # frame-skipping mode: display stays on last result

                if not self.detection_service.model_loaded:
                    self._publish_placeholder(frame, "CUSTOM DEFECT MODEL NOT TRAINED")
                    continue

                try:
                    result = self.detection_service.inspect_image(
                        frame, source=f"camera:{self.source}", save_annotated=False
                    )
                    inference_ms = result.timings.inference_ms

                    # temporal confirmation (anti-flicker) for PLC decisions
                    confirmed = temporal.update(result.defect_detected)
                    confirmed_status = "REJECT" if confirmed else "PASS"

                    decision_latency_ms = None
                    plc_event = self.snapshot.last_plc_event
                    if confirmed != (last_status == "REJECT"):
                        t_decision = time.perf_counter()
                        action = "REJECT" if confirmed else "PASS"
                        if debouncer.allow(action):
                            dominant = result.dominant_defect
                            plc_event = plc_line.send_sort_command(
                                action,
                                defect=dominant if action == "REJECT" else None,
                                confidence=result.highest_confidence if action == "REJECT" else None,
                            )
                            from app.services.metrics_service import PLC_DECISION_LATENCY, PLC_EVENTS

                            decision_latency_ms = (time.perf_counter() - t_decision) * 1000.0
                            PLC_EVENTS.labels(action=action).inc()
                            if decision_latency_ms is not None:
                                PLC_DECISION_LATENCY.set(decision_latency_ms)
                            logger.info(
                                "PLC dispatched %s (decision_latency_ms=%.2f)", action, decision_latency_ms
                            )
                        last_status = confirmed_status

                    # event-throttled history writes: record on any status
                    # change or when defects are present, at most every 1.5 s
                    # in live mode (one persistent defect ≠ 100s of events)
                    now_ms = time.perf_counter() * 1000.0
                    status_changed = result.inspection_status != last_recorded_status
                    if (status_changed or result.detections) and \
                            (now_ms - last_history_event_ms) >= HISTORY_MIN_INTERVAL_MS:
                        self.detection_service.history.record_inspection(result.as_dict())
                        last_recorded_status = result.inspection_status
                        last_history_event_ms = now_ms

                    t_render = time.perf_counter()
                    annotated = self._overlay(frame, result, confirmed_status)
                    self._publish_jpeg(annotated)
                    render_ms = (time.perf_counter() - t_render) * 1000.0

                    snap = result.as_dict()
                    snap.pop("detections", None)
                    self.snapshot.last_result = snap
                    self.snapshot.last_timings = {
                        **result.timings.as_dict(),
                        "capture_ms": round(capture_ms, 2),
                        "render_ms": round(render_ms, 2),
                        "total_pipeline_ms": round(capture_ms + result.timings.total_ms + render_ms, 2),
                    }
                    self.snapshot.confirmed_status = confirmed_status
                    if decision_latency_ms is not None:
                        self.snapshot.last_decision_latency_ms = round(decision_latency_ms, 2)
                    if plc_event is not None:
                        self.snapshot.last_plc_event = plc_event
                    self.snapshot.frames_processed += 1

                    from app.services.metrics_service import FRAMES_PROCESSED

                    FRAMES_PROCESSED.inc()

                    fps_frames += 1
                    now = time.perf_counter()
                    if now - fps_t0 >= 1.0:
                        self.snapshot.fps = fps_frames / (now - fps_t0)
                        set_camera_state(True, fps=self.snapshot.fps, uptime_seconds=time.time() - self.snapshot.started_at)
                        fps_t0, fps_frames = now, 0
                except Exception as exc:
                    logger.exception("frame inspection failed: %s", exc)
                    self.snapshot.error = str(exc)
                    self._publish_placeholder(frame, f"INSPECTION ERROR: {exc}")
        finally:
            capture.release()
            self.snapshot.online = False
            set_camera_state(False)
            logger.info("camera offline")

    # ------------------------------------------------------------- helpers
    def _overlay(self, frame: np.ndarray, result, confirmed_status: str) -> np.ndarray:
        from app.inference.preprocessing import draw_detection

        annotated = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        for d in result.detections:
            b = d["bbox"]
            draw_detection(
                annotated, int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"]),
                d["class"], d["confidence"],
            )
            # severity tag next to the box
            cv2.putText(annotated, d["severity"], (int(b["x2"]) + 2, int(b["y1"]) + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 215, 255), 1, cv2.LINE_AA)
        timings = self.snapshot.last_timings or {}
        status = confirmed_status
        color = (0, 170, 0) if status == "PASS" else (0, 0, 220)
        cv2.rectangle(annotated, (0, 0), (300, 44), color, -1)
        cv2.putText(annotated, status, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        backend = self.detection_service._detector.get_backend_name() if self.detection_service.model_loaded else "none"
        lines = [
            f"Defects: {result.defect_count}   Backend: {backend}",
            f"FPS: {self.snapshot.fps:.1f}   Latency: {result.timings.total_ms:.1f} ms",
            f"Inference: {result.timings.inference_ms:.1f} ms   Pipeline: {timings.get('total_pipeline_ms', 0):.1f} ms",
            f"Camera: ONLINE   Confirmed: {confirmed_status}",
        ]
        y = 66
        for ln in lines:
            cv2.putText(annotated, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 255, 255), 1, cv2.LINE_AA)
            y += 20
        return annotated

    def _publish_jpeg(self, frame: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._jpeg_lock:
                self._jpeg = buf.tobytes()

    def _publish_placeholder(self, frame: np.ndarray, message: str) -> None:
        annotated = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (annotated.shape[1], annotated.shape[0]), (0, 0, 120), -1)
        annotated = cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0)
        cv2.putText(annotated, message, (12, annotated.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated, "python training\\train.py --epochs 1", (12, annotated.shape[0] // 2 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        self._publish_jpeg(annotated)


class CameraManager:
    """Starts/stops the single camera worker for the app."""

    def __init__(self, detection_service):
        self.detection_service = detection_service
        self._worker: CameraWorker | None = None
        self._lock = threading.Lock()

    def start(self, source=CAMERA_SOURCE) -> dict:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return {"online": True, "already_running": True, "source": self._worker.snapshot.source}
            worker = CameraWorker(self.detection_service, source=source)
            worker.start()
            self._worker = worker
            return {"online": True, "already_running": False, "source": str(worker.source)}

    def stop(self) -> dict:
        with self._lock:
            if self._worker is None:
                return {"online": False, "was_running": False}
            self._worker.stop()
            self._worker.join(timeout=5)
            self._worker = None
            return {"online": False, "was_running": True}

    def snapshot(self) -> dict:
        if self._worker is None or not self._worker.is_alive():
            return {
                "online": False, "fps": 0.0, "frames_processed": 0, "frames_captured": 0,
                "last_result": None, "last_timings": {}, "confirmed_status": "PASS",
                "last_plc_event": None, "last_decision_latency_ms": None,
            }
        s = self._worker.snapshot
        return {
            "online": s.online,
            "source": s.source,
            "fps": round(s.fps, 2),
            "frames_processed": s.frames_processed,
            "frames_captured": s.frames_captured,
            "last_result": s.last_result,
            "last_timings": s.last_timings,
            "confirmed_status": s.confirmed_status,
            "last_plc_event": s.last_plc_event,
            "last_decision_latency_ms": s.last_decision_latency_ms,
            "error": s.error,
            "uptime_seconds": round(time.time() - s.started_at, 1) if s.started_at else 0.0,
        }

    def latest_jpeg(self) -> bytes | None:
        return self._worker.latest_jpeg() if self._worker else None


camera_manager = None  # initialised in app.main with the detection service singleton
