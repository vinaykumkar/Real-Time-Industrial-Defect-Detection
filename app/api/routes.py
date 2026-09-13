"""FastAPI routes for the defect-detection system."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from app.api import schemas
from app.api.plc import plc
from app.config import (
    CLASS_DISPLAY_NAMES,
    CLASS_NAMES,
    DETECTIONS_OUTPUT_DIR,
    IMAGES_VAL_DIR,
    LABELS_VAL_DIR,
    PROJECT_ROOT,
)
from app.services.detection_service import detection_service
from app.services.logging_service import get_logger
from app.services.metrics_service import record_api_error

logger = get_logger("defect.api")
router = APIRouter()

ALLOWED_UPLOAD_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def _read_image_upload(upload: UploadFile) -> np.ndarray:
    """Decode an uploaded image; raise 400 with a clear message on failure."""
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix and suffix not in ALLOWED_UPLOAD_TYPES:
        raise HTTPException(status_code=400, detail=f"unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_UPLOAD_TYPES)}")
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file too large (max 25 MB)")
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="corrupted or unreadable image")
    return img


# ------------------------------------------------------------------ health
@router.get("/health", response_model=schemas.HealthResponse, tags=["system"])
def health():
    db_ok = detection_service.history.db_path.exists()
    if detection_service.model_loaded:
        backend = detection_service._detector.backend_name
        device = detection_service._detector.get_device()
        status = "ok"
    else:
        backend, device, status = "none", "cpu", "degraded"
    from app.main import camera_manager

    camera_state = camera_manager.snapshot()
    return {
        "status": status,
        "model_loaded": detection_service.model_loaded,
        "inference_backend": backend,
        "device": device,
        "camera_online": bool(camera_state.get("online")),
        "database_ok": db_ok,
        "version": "1.0.0",
    }


@router.get("/api/status", response_model=schemas.StatusResponse, tags=["system"])
def status():
    payload = detection_service.status_payload()
    if not payload["model_loaded"]:
        payload["training_hint"] = (
            "CUSTOM DEFECT MODEL NOT TRAINED — run: python training\\train.py --epochs 1"
        )
    return payload


@router.get("/api/classes", response_model=schemas.ClassesResponse, tags=["system"])
def classes():
    return {
        "classes": CLASS_NAMES,
        "display_names": CLASS_DISPLAY_NAMES,
        "num_classes": len(CLASS_NAMES),
    }


@router.get("/api/metrics-summary", response_model=schemas.MetricsSummaryResponse, tags=["system"])
def metrics_summary():
    return detection_service.history_summary()


# -------------------------------------------------------------- detection
@router.post("/api/detect/image", response_model=schemas.ImageInspectionResponse, tags=["inspection"])
async def detect_image(file: UploadFile = File(..., description="metal surface image to inspect")):
    try:
        img = _read_image_upload(file)
    except HTTPException as exc:
        record_api_error("/api/detect/image")
        raise
    try:
        result = detection_service.inspect_image(img, source=file.filename or "upload")
    except Exception as exc:
        record_api_error("/api/detect/image")
        logger.exception("detection failed")
        raise HTTPException(status_code=500, detail=f"detection failed: {exc}") from exc

    payload = result.as_dict()
    # find the annotated image we just saved so the UI can show it
    safe_source = Path(payload["source"] or "upload").stem.replace(" ", "_") or "upload"
    stamp = payload["timestamp"].replace(":", "").replace(" ", "_").replace("-", "")
    payload["annotated_image_url"] = f"/outputs/detections/{safe_source}_{stamp}_{payload['inspection_status'].lower()}.jpg"
    return payload


@router.post("/api/detect/video", tags=["inspection"])
async def detect_video(
    file: UploadFile = File(..., description="video file to inspect"),
    max_frames: int = Query(300, ge=1, le=3000, description="cap on frames processed"),
):
    if not detection_service.model_loaded:
        raise HTTPException(status_code=503, detail="CUSTOM DEFECT MODEL NOT TRAINED — run python training\\train.py --epochs 1")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".wmv"}:
        raise HTTPException(status_code=400, detail=f"unsupported video type '{suffix}'")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(data) > 200 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="video too large (max 200 MB)")

    tmp_in = DETECTIONS_OUTPUT_DIR / "video_tmp" / f"upload_{int(time.time())}{suffix}"
    tmp_out_dir = DETECTIONS_OUTPUT_DIR / "video_processed"
    tmp_out_dir.mkdir(parents=True, exist_ok=True)
    tmp_in.parent.mkdir(parents=True, exist_ok=True)
    tmp_in.write_bytes(data)

    try:
        from app.inference.detector import Detector  # noqa: F401
        from app.inference.video_processor import VideoProcessor

        out_video = tmp_out_dir / f"{Path(file.filename).stem}_annotated_{int(time.time())}.mp4"
        processor = VideoProcessor(detection_service.ensure_loaded(), max_frames=max_frames)
        stats = processor.process(tmp_in, annotated_output=out_video)
    except Exception as exc:
        record_api_error("/api/detect/video")
        logger.exception("video processing failed")
        raise HTTPException(status_code=500, detail=f"video processing failed: {exc}") from exc
    finally:
        try:
            tmp_in.unlink(missing_ok=True)
        except OSError:
            pass

    stats["annotated_video_url"] = f"/outputs/detections/video_processed/{Path(stats['annotated_video']).name}"
    return JSONResponse(stats)


# -------------------------------------------------------------------- plc
@router.post("/api/plc/test", response_model=schemas.PLCCommandResponse, tags=["plc"])
def plc_test(command: schemas.PLCCommandRequest):
    try:
        event = plc.send_sort_command(command.action, defect=command.defect, confidence=command.confidence)
        return event
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/api/plc/stats", tags=["plc"])
def plc_stats():
    stats = plc.stats()
    stats["last_event"] = plc.last_event()
    return stats


@router.post("/api/plc/dispatch-inspection", tags=["plc"])
def plc_dispatch_latest():
    """Send the most recent inspection result to the simulated PLC line."""
    latest = detection_service.history.query(limit=1)
    if not latest:
        raise HTTPException(status_code=404, detail="no inspection history yet — run an inspection first")
    ev = latest[0]
    action = ev["inspection_status"]
    event = plc.send_sort_command(action, defect=ev["defect_class"], confidence=ev["confidence"])
    return event


# ---------------------------------------------------------------- history
@router.get("/api/history", response_model=schemas.HistoryResponse, tags=["history"])
def history(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    status_filter: str | None = Query(None, alias="status", pattern="^(PASS|REJECT)$"),
    defect_class: str | None = None,
    severity: str | None = Query(None, pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$"),
):
    events = detection_service.history.query(
        limit=limit, offset=offset, status=status_filter, defect_class=defect_class, severity=severity
    )
    total = detection_service.history.count(
        status=status_filter, defect_class=defect_class, severity=severity
    )
    return {"events": events, "count": len(events), "total": total}


@router.delete("/api/history", tags=["history"])
def clear_history():
    n = detection_service.history.clear()
    logger.info("history cleared (%d events)", n)
    return {"cleared": n}


# ------------------------------------------------------- model performance
@router.get("/api/model/performance", tags=["model"])
def model_performance():
    metrics_path = Path(__file__).resolve().parent.parent.parent / "outputs" / "evaluation" / "metrics.json"
    if not metrics_path.exists():
        return {
            "available": False,
            "message": "No evaluation metrics yet. Train the model and run: python training\\evaluate.py",
        }
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"metrics file unreadable: {exc}") from exc
    return {"available": True, "metrics": data}


# ------------------------------------------------------- analytics summary
@router.get("/api/analytics/summary", tags=["analytics"])
def analytics_summary():
    """Aggregated real analytics for the dashboard (SQL-computed, no fake data)."""
    import time as _time

    hist = detection_service.history
    with hist._lock, __import__("contextlib").closing(hist._connect()) as conn:
        severity_counts = dict(conn.execute(
            "SELECT severity, COUNT(*) FROM detection_events WHERE severity IS NOT NULL GROUP BY severity"
        ).fetchall())
        latency_trend = dict(conn.execute(
            "SELECT substr(timestamp, 1, 16), ROUND(AVG(inference_latency_ms), 1) FROM detection_events "
            "WHERE inference_latency_ms > 0 GROUP BY substr(timestamp, 1, 16) ORDER BY substr(timestamp, 1, 16) DESC LIMIT 40"
        ).fetchall())
        pr_trend = {
            ts: {"REJECT": rej, "PASS": pas}
            for ts, rej, pas in conn.execute(
                "SELECT substr(timestamp, 1, 16), SUM(inspection_status='REJECT'), SUM(inspection_status='PASS') "
                "FROM detection_events GROUP BY substr(timestamp, 1, 16) ORDER BY substr(timestamp, 1, 16) DESC LIMIT 40"
            ).fetchall()
        }
        hourly = dict(conn.execute(
            "SELECT substr(timestamp, 12, 2) || 'h', COUNT(*) FROM detection_events "
            "WHERE defect_class IS NOT NULL GROUP BY substr(timestamp, 12, 2) ORDER BY substr(timestamp, 12, 2)"
        ).fetchall())
    return {
        "severity_counts": severity_counts,
        "latency_trend": latency_trend,
        "pass_reject_trend": pr_trend,
        "detections_by_hour": hourly,
        "generated_at": _time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ------------------------------------------------------------ system info
@router.get("/api/system/info", tags=["system"])
def system_info():
    """Lightweight operational info for the System Health page (cached 5 s)."""
    import time as _time

    global _system_info_cache
    now = _time.time()
    if _system_info_cache and now - _system_info_cache[0] < 5.0:
        return _system_info_cache[1]

    import platform

    import torch

    from app.main import camera_manager

    info: dict = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_loaded": detection_service.model_loaded,
        "backend": detection_service._detector.get_backend_name() if detection_service.model_loaded else None,
        "device": detection_service._detector.get_device() if detection_service.model_loaded else None,
        "camera": camera_manager.snapshot(),
        "database_ok": detection_service.history.db_path.exists(),
        "app_uptime_seconds": round(now - _APP_START_TIME, 1),
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        info["ram_total_gb"] = round(vm.total / 1e9, 1)
        info["ram_used_pct"] = vm.percent
    except ImportError:
        pass
    _system_info_cache = (now, info)
    return info


# ------------------------------------------------------------- model info
@router.get("/api/model/info", tags=["model"])
def model_info():
    """Model metadata + edge benchmark for the Model Performance page."""
    import csv as _csv

    from app.config import MODELS_DIR, PROJECT_ROOT

    meta_path = MODELS_DIR / "best_metadata.json"
    if not meta_path.exists():
        return {"available": False, "message": "Model metadata not found — train a model first."}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"metadata unreadable: {exc}") from exc

    # recommended backend from the measured edge benchmark (never hard-coded):
    # fastest MEDIAN among backends whose avg and p95 stay within 3x of the
    # median (filters out stall-spike backends like shared-GPU CUDA here)
    recommended = None
    bench_rows = []
    bench_csv = PROJECT_ROOT / "outputs" / "evaluation" / "edge_benchmark.csv"
    if bench_csv.exists():
        stable: list[tuple[float, str]] = []
        with bench_csv.open(newline="", encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                bench_rows.append(row)
                try:
                    med = float(row.get("median_inference_ms") or 0)
                    avg = float(row.get("avg_inference_ms") or 0)
                    p95 = float(row.get("p95_inference_ms") or 0)
                except ValueError:
                    continue
                if med > 0 and avg <= med * 3 and p95 <= med * 3:
                    stable.append((med, f"{row['backend']} / {row['device']} / {row['precision']}"))
        if stable:
            stable.sort()
            recommended = stable[0][1]

    running = detection_service._detector.get_performance_info() if detection_service.model_loaded else None
    sizes = {}
    for key, p in (("pt", MODELS_DIR / "best.pt"), ("onnx", MODELS_DIR / "best.onnx")):
        if p.exists():
            sizes[key] = round(p.stat().st_size / 1e6, 2)
    return {
        "available": True,
        "metadata": meta,
        "model_sizes_mb": sizes,
        "edge_benchmark": bench_rows,
        "recommended_backend": recommended or "ONNX Runtime (default fallback)",
        "running_backend": running,
    }


# ------------------------------------------------- dataset sample image
@router.get("/api/dataset/sample-image", tags=["dataset"])
def dataset_sample_image(defect_class: str = Query(...), random_sel: str | None = Query(None, alias="random")):
    """Serve one validation image of a class (for the image-inspection Load Sample button)."""
    if defect_class not in CLASS_NAMES:
        raise HTTPException(status_code=400, detail=f"unknown class '{defect_class}'")
    if not _ensure_dataset_mount():
        raise HTTPException(status_code=404, detail="converted dataset not found — run python training\\convert_voc_to_yolo.py")
    images = sorted(IMAGES_VAL_DIR.glob(f"{defect_class}_*.jpg"))
    if not images:
        raise HTTPException(status_code=404, detail="no sample images for this class")
    pick = images[hash(random_sel or "") % len(images)] if random_sel else images[0]
    return {"image_url": f"/dataset/val/{pick.name}", "filename": pick.name, "class": defect_class}


_system_info_cache: tuple[float, dict] | None = None
_APP_START_TIME = time.time()
def _ensure_dataset_mount() -> bool:
    """Mount /dataset lazily so the explorer works even if the server was
    started before the dataset was converted."""
    from app.main import app as fastapi_app

    if not IMAGES_VAL_DIR.exists():
        return False
    if not any(getattr(r, "path", "").startswith("/dataset") for r in fastapi_app.routes):
        from fastapi.staticfiles import StaticFiles

        fastapi_app.mount(
            "/dataset",
            StaticFiles(directory=str(PROJECT_ROOT / "data" / "yolo" / "images")),
            name="dataset",
        )
    return True


@router.get("/api/dataset/sample", tags=["dataset"])
def dataset_sample(defect_class: str = Query(..., description=f"one of {CLASS_NAMES}"), count: int = Query(4, ge=1, le=12)):
    if defect_class not in CLASS_NAMES:
        raise HTTPException(status_code=400, detail=f"unknown class '{defect_class}'")
    if not _ensure_dataset_mount():
        raise HTTPException(status_code=404, detail="converted dataset not found — run python training\\convert_voc_to_yolo.py")
    images = sorted(IMAGES_VAL_DIR.glob(f"{defect_class}_*.jpg"))[:count]
    if not images:
        images = sorted(IMAGES_VAL_DIR.glob("*.jpg"))[:0]
    samples = []
    for img_path in images:
        lbl = LABELS_VAL_DIR / f"{img_path.stem}.txt"
        boxes = []
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) == 5:
                    boxes.append({"class_id": int(parts[0]), "class": CLASS_NAMES[int(parts[0])], "xywh": [float(x) for x in parts[1:]]})
        samples.append({"image_url": f"/dataset/val/{img_path.name}", "boxes": boxes})
    return {"class": defect_class, "samples": samples}


# -------------------------------------------------------------- MJPEG live
@router.get("/api/video_feed", tags=["live"])
def video_feed():
    """MJPEG stream of the live camera inspection."""
    from app.main import camera_manager

    def _placeholder_jpeg() -> bytes:
        import numpy as np

        img = np.zeros((480, 720, 3), dtype=np.uint8)
        cv2.putText(img, "CAMERA OFFLINE", (200, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2, cv2.LINE_AA)
        cv2.putText(img, "Press START CAMERA on the Live Inspection page", (150, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img)
        return buf.tobytes() if ok else b""

    boundary = "frame"

    def generate():
        offline_sent_at = 0.0
        while True:
            jpeg = camera_manager.latest_jpeg()
            if jpeg is not None:
                yield (
                    b"--" + boundary.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
                time.sleep(0.04)
            else:
                now = time.time()
                if now - offline_sent_at >= 1.0:
                    offline_sent_at = now
                    ph = _placeholder_jpeg()
                    if ph:
                        yield (
                            b"--" + boundary.encode() + b"\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + ph + b"\r\n"
                        )
                time.sleep(0.2)

    return StreamingResponse(generate(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")


@router.post("/api/camera/start", tags=["live"])
def camera_start(source: str | None = None):
    from app.main import camera_manager

    src = source if source is not None else None
    return camera_manager.start(source=src)


@router.post("/api/camera/stop", tags=["live"])
def camera_stop():
    from app.main import camera_manager

    return camera_manager.stop()


@router.get("/api/live/status", tags=["live"])
def live_status():
    from app.main import camera_manager

    return camera_manager.snapshot()
