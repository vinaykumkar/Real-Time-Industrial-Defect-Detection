"""Detection service: ties the detector engine, QC logic, history and metrics together.

Used by the FastAPI routes, the camera script and the video processor.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.config import (
    CONFIDENCE_THRESHOLD,
    DATABASE,
    IOU_THRESHOLD,
    LOG_LEVEL,
    SAVE_DETECTIONS,
    resolve_path,
)
from app.inference.detector import Detector, ModelNotTrainedError
from app.services.detection_history import DetectionHistory
from app.services.logging_service import get_logger, setup_logging
from app.services.metrics_service import record_inspection
from app.services.qc import InspectionResult, inspect_frame

logger = get_logger("defect.detection_service")


class DetectionService:
    """Single owning object for the model + database + metrics."""

    def __init__(self):
        setup_logging(LOG_LEVEL)
        self._detector: Detector | None = None
        self._history = DetectionHistory(DATABASE)
        self.save_detections = SAVE_DETECTIONS

    # ------------------------------------------------------------- model
    @property
    def model_loaded(self) -> bool:
        return self._detector is not None

    def load_model(self, backend: str = "auto", force: bool = False) -> dict:
        if self._detector is not None and not force:
            return self._detector.info
        try:
            detector, fallback_log = Detector.create_with_fallback(backend=backend, conf_threshold=CONFIDENCE_THRESHOLD, iou_threshold=IOU_THRESHOLD)
            self._detector = detector
            logger.info("Model loaded: %s", detector.info)
            for entry in fallback_log:
                logger.warning("backend fallback entry: %s", entry)
            return detector.info
        except ModelNotTrainedError as exc:
            logger.warning("Model not available: %s", exc)
            self._detector = None
            raise
        except RuntimeError as exc:
            logger.error("All backends failed: %s", exc)
            self._detector = None
            raise

    def ensure_loaded(self) -> Detector:
        if self._detector is None:
            self.load_model()
        return self._detector

    # --------------------------------------------------------- inference
    def inspect_image(self, image_bgr: np.ndarray, source: str = "image", save_annotated: bool | None = None) -> InspectionResult:
        detector = self.ensure_loaded()
        result = inspect_frame(detector, image_bgr, source=source)
        record_inspection(result.as_dict())
        logger.info(
            "inspection source=%s status=%s defects=%d inference_ms=%.1f",
            source, result.inspection_status, result.defect_count, result.timings.inference_ms,
        )
        save = self.save_detections if save_annotated is None else save_annotated
        if save:
            self._save_annotated(image_bgr, result, source)
        self._history.record_inspection(result.as_dict())
        return result

    def _save_annotated(self, image_bgr: np.ndarray, result: InspectionResult, source: str) -> None:
        from app.inference.preprocessing import draw_detection
        from app.services.qc import annotate as _overlay  # noqa: F401  (status overlay used in video paths)

        out_dir = resolve_path("outputs/detections")
        out_dir.mkdir(parents=True, exist_ok=True)
        annotated = image_bgr.copy() if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        for d in result.detections:
            b = d["bbox"]
            draw_detection(
                annotated, int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"]),
                d["class"], d["confidence"],
            )
        status_color = (0, 200, 0) if result.inspection_status == "PASS" else (0, 0, 255)
        cv2.putText(annotated, result.inspection_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2, cv2.LINE_AA)
        safe_source = Path(source).stem.replace(" ", "_") or "frame"
        stamp = result.timestamp.replace(":", "").replace(" ", "_").replace("-", "")
        out_path = out_dir / f"{safe_source}_{stamp}_{result.inspection_status.lower()}.jpg"
        try:
            cv2.imencode(".jpg", annotated)[1].tofile(out_path)
        except OSError as exc:
            logger.warning("could not save annotated image: %s", exc)

    # ----------------------------------------------------------- history
    @property
    def history(self) -> DetectionHistory:
        return self._history

    def history_summary(self) -> dict:
        return self._history.summary()

    def status_payload(self) -> dict:
        """GET /api/status — honest status even when no model is trained."""
        return {
            "model_loaded": self._detector is not None,
            "model": self._detector.get_performance_info() if self._detector else None,
            "history": self.history_summary(),
        }


# Shared singleton for the FastAPI app
detection_service = DetectionService()
