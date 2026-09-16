"""FastAPI application entry point for the Industrial Defect Detection System.

Start from the project root:
    python -m uvicorn app.main:app --reload

Web Dashboard:
    http://127.0.0.1:8000/

API Documentation:
    http://127.0.0.1:8000/docs

Prometheus Metrics:
    http://127.0.0.1:8000/metrics
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
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


# ------------------------------------------------------------------
# Application configuration
# ------------------------------------------------------------------

logger = get_logger("defect.main")

TEMPLATES_DIR = PROJECT_ROOT / "app" / "templates"
STATIC_DIR = PROJECT_ROOT / "app" / "static"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATASET_DIR = PROJECT_ROOT / "data" / "yolo" / "images"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
camera_manager = CameraManager(detection_service)


# ------------------------------------------------------------------
# Application lifecycle
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Initialize and clean up application resources."""

    setup_logging(LOG_LEVEL)
    ensure_dirs()

    logger.info(
        "Starting Industrial Defect Detection System | root=%s",
        PROJECT_ROOT,
    )

    try:
        model_info = detection_service.load_model()
        logger.info("Custom defect detection model loaded: %s", model_info)

    except Exception as exc:
        logger.warning("Unable to load custom defect model: %s", exc)
        logger.warning(
            "Custom defect model is not trained. "
            "Run: python training\\train.py --epochs 1"
        )

    yield

    logger.info("Stopping Industrial Defect Detection System")

    try:
        camera_manager.stop()
    except Exception:
        logger.exception("Error while stopping camera manager")


# ------------------------------------------------------------------
# FastAPI application
# ------------------------------------------------------------------

app = FastAPI(
    title="Real-Time Industrial Defect Detection System",
    description=(
        "YOLOv8-powered industrial surface-defect inspection system "
        "for manufacturing quality control. Supports six NEU defect "
        "classes, PASS/REJECT inspection, PLC sorting simulation, "
        "and Prometheus monitoring."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ------------------------------------------------------------------
# Static resources
# ------------------------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)

app.mount(
    "/outputs",
    StaticFiles(directory=str(OUTPUTS_DIR)),
    name="outputs",
)

# Dataset images become available after dataset conversion.
if IMAGES_VAL_DIR.exists():
    app.mount(
        "/dataset",
        StaticFiles(directory=str(DATASET_DIR)),
        name="dataset",
    )
    logger.info("Dataset image directory mounted: %s", DATASET_DIR)


# ------------------------------------------------------------------
# API routes
# ------------------------------------------------------------------

app.include_router(router)


# ------------------------------------------------------------------
# Dashboard
# ------------------------------------------------------------------

@app.get(
    "/",
    response_class=HTMLResponse,
    include_in_schema=False,
)
async def dashboard(request: Request) -> HTMLResponse:
    """Render the defect detection dashboard."""

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "classes": CLASS_NAMES,
        },
    )


# ------------------------------------------------------------------
# Monitoring
# ------------------------------------------------------------------

@app.get(
    "/metrics",
    include_in_schema=False,
)
async def metrics() -> Response:
    """Expose Prometheus application metrics."""

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ------------------------------------------------------------------
# Global exception handling
# ------------------------------------------------------------------

@app.exception_handler(Exception)
async def handle_unexpected_error(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle unexpected application errors."""

    record_api_error(request.url.path)

    logger.exception(
        "Unhandled exception while processing %s",
        request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
        },
    )


# ------------------------------------------------------------------
# Local development entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
