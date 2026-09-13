"""FastAPI application entry point.

Run from the project root:
    uvicorn app.main:app --reload

Dashboard:      http://127.0.0.1:8000/
API docs:       http://127.0.0.1:8000/docs
Prometheus:     http://127.0.0.1:8000/metrics
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api.routes import router
from app.config import (
    CLASS_NAMES,
    IMAGES_VAL_DIR,
    LOG_LEVEL,
    PROJECT_ROOT,
    ensure_dirs,
)
from app.services.camera_service import CameraManager
from app.services.detection_service import detection_service
from app.services.logging_service import get_logger, setup_logging
from app.services.metrics_service import record_api_error

logger = get_logger("defect.main")

templates = Jinja2Templates(directory=str(PROJECT_ROOT / "app" / "templates"))
camera_manager = CameraManager(detection_service)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(LOG_LEVEL)
    ensure_dirs()
    logger.info("application starting (project root: %s)", PROJECT_ROOT)
    # Load the custom model only if trained weights actually exist.
    # Never substitute a generic COCO model for defect detection.
    try:
        info = detection_service.load_model()
        logger.info("defect model ready: %s", info)
    except Exception as exc:
        logger.warning("%s", exc)
        logger.warning("dashboard will show CUSTOM DEFECT MODEL NOT TRAINED until you run: python training\\train.py --epochs 1")
    yield
    logger.info("application shutting down")
    camera_manager.stop()


app = FastAPI(
    title="Real-Time Industrial Defect Detection System",
    description=(
        "YOLOv8-based surface-defect detection for manufacturing quality "
        "control: six NEU defect classes, PASS/REJECT inspection, PLC "
        "sorting simulation, and Prometheus monitoring."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Static mounts: dashboard assets, annotated outputs, dataset images
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "app" / "static")), name="static")
app.mount("/outputs", StaticFiles(directory=str(PROJECT_ROOT / "outputs")), name="outputs")
if IMAGES_VAL_DIR.exists():
    app.mount("/dataset", StaticFiles(directory=str(PROJECT_ROOT / "data" / "yolo" / "images")), name="dataset")
else:
    # mount will be added lazily once the dataset is converted; explorer returns 404 until then
    pass

app.include_router(router)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"classes": CLASS_NAMES},
    )


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics():
    from fastapi.responses import Response

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    record_api_error(request.url.path)
    logger.exception("unhandled API error on %s", request.url.path)
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=500, content={"detail": f"internal error: {exc}"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
