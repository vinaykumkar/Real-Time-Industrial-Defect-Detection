"""Reusable defect-detection engine with automatic backend selection.

Backend priority (only when the corresponding model file actually exists):

    TensorRT engine (models/*.engine, requires TensorRT + CUDA)
      -> ONNX Runtime (models/best.onnx)
      -> PyTorch / Ultralytics (models/best.pt)

If no custom NEU-trained model exists, constructing a Detector raises
ModelNotTrainedError — the system NEVER silently falls back to a generic
COCO model, because COCO weights cannot detect the six NEU defect classes.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.config import (
    CLASS_NAMES,
    MODELS_DIR,
    MODEL_PATH,
    ONNX_MODEL,
    TENSORRT_MODEL_PATH,
    model_file_ok,
    resolve_path,
)


class ModelNotTrainedError(RuntimeError):
    """Raised when no custom NEU-trained model weights exist."""

    def __init__(self):
        super().__init__(
            "CUSTOM DEFECT MODEL NOT TRAINED — expected models/best.pt. "
            "Train it with:  python training\\train.py --epochs 1   (smoke test) "
            "or  python training\\train.py --epochs 100   (full training)"
        )


def _find_engine_model() -> Path | None:
    if MODELS_DIR.exists():
        for p in sorted(MODELS_DIR.glob("*.engine")):
            return p
    return None


# Priority order for INFERENCE_BACKEND=auto (spec §8)
BACKEND_PRIORITY = ("tensorrt", "onnx", "pytorch")


def resolve_backend(backend_pref: str = "auto") -> tuple[str, Path]:
    """Return (backend_name, model_path) based on what actually exists on disk.

    Honours MODEL_PATH / ONNX_MODEL_PATH / TENSORRT_MODEL_PATH configuration
    (env-overridable). Only file presence is checked here; actual loadability
    is verified by the fallback loop in Detector.create_with_fallback.
    """
    pref = (backend_pref or "auto").lower()
    engine_cfg = resolve_path(TENSORRT_MODEL_PATH)
    engine = engine_cfg if model_file_ok(engine_cfg) else _find_engine_model()
    onnx = resolve_path(ONNX_MODEL)
    pt = resolve_path(MODEL_PATH)

    candidates: dict[str, Path | None] = {
        "tensorrt": engine,
        "onnx": onnx if model_file_ok(onnx) else None,
        "pytorch": pt if model_file_ok(pt) else None,
    }
    if pref == "auto":
        order = BACKEND_PRIORITY
    elif pref in candidates:
        order = (pref, *BACKEND_PRIORITY)
    else:
        order = BACKEND_PRIORITY
    for name in order:
        p = candidates.get(name)
        if p is not None:
            return name, p
    raise ModelNotTrainedError()


def _default_imgsz() -> int:
    """Prefer the trained image size recorded in models/best_metadata.json."""
    meta = MODELS_DIR / "best_metadata.json"
    try:
        import json

        return int(json.loads(meta.read_text(encoding="utf-8")).get("image_size", 640))
    except (OSError, ValueError, TypeError):
        return 640


class Detector:
    """High-level defect detector used by the API, CLI and camera scripts."""

    def __init__(self, backend: str = "auto", conf_threshold: float = 0.5, iou_threshold: float = 0.45, imgsz: int | None = None):
        self.backend_name, self.model_path = resolve_backend(backend)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz if imgsz is not None else _default_imgsz()
        self._impl = None
        self._load()

    def _load(self) -> None:
        if self.backend_name == "onnx":
            from app.inference.onnx_detector import OnnxDetector

            self._impl = OnnxDetector(
                self.model_path,
                imgsz=self.imgsz,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
            )
        else:
            # pytorch and tensorrt both run through Ultralytics YOLO
            from ultralytics import YOLO

            model = YOLO(str(self.model_path))
            if self.backend_name == "tensorrt":
                if not model.device or "cuda" not in str(model.device).lower():
                    # TensorRT engines can only execute on NVIDIA GPUs
                    self.backend_name = "pytorch"
                    pt = resolve_path(MODEL_PATH)
                    if pt.exists():
                        model = YOLO(str(pt))
                        self.model_path = pt
                    else:
                        self.backend_name = "onnx"
                        onnx = resolve_path(ONNX_MODEL)
                        if not onnx.exists():
                            raise ModelNotTrainedError()
                        from app.inference.onnx_detector import OnnxDetector

                        self._impl = OnnxDetector(
                            onnx,
                            imgsz=self.imgsz,
                            conf_threshold=self.conf_threshold,
                            iou_threshold=self.iou_threshold,
                        )
                        return
            self._impl = model

    # ------------------------------------------------------------------ api
    @property
    def info(self) -> dict:
        return {
            "backend": self.backend_name,
            "model_path": str(self.model_path),
            "model_format": self.model_path.suffix.lstrip("."),
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "imgsz": self.imgsz,
        }

    def get_backend_name(self) -> str:
        return self.backend_name

    def get_device(self) -> str:
        """Human-readable device actually in use (never a guess)."""
        if self.backend_name == "onnx":
            return str(getattr(self._impl, "provider_used", "CPUExecutionProvider"))
        if self.backend_name == "pytorch":
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        return "tensorrt-gpu"

    def get_performance_info(self) -> dict:
        """Backend/device facts for /api/status — no estimates."""
        info = self.info
        info["device"] = self.get_device()
        if self.backend_name == "onnx":
            info["provider"] = str(getattr(self._impl, "provider_used", "CPUExecutionProvider"))
        return info

    def warmup(self, iters: int = 3) -> None:
        """Run dummy inferences so the first real frame is not the slow one."""
        import numpy as np

        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for _ in range(max(1, iters)):
            try:
                self.detect_raw(dummy)
            except Exception:
                break

    @classmethod
    def create_with_fallback(cls, backend: str = "auto", **kwargs) -> tuple["Detector", list[str]]:
        """Build a Detector trying each candidate backend in priority order.

        A backend is selected only if its model actually loads (spec §8/§48):
        load failures are logged and the next safe fallback is tried.
        Returns (detector, fallback_log).
        """
        from app.services.logging_service import get_logger

        logger = get_logger("defect.detector")
        log: list[str] = []
        pref = (backend or "auto").lower()

        # candidate order identical to resolve_backend, but load-validated
        engine_path = resolve_path(TENSORRT_MODEL_PATH)
        candidates: dict[str, Path | None] = {
            "tensorrt": engine_path if model_file_ok(engine_path) else (_find_engine_model() if MODELS_DIR.exists() else None),
            "onnx": resolve_path(ONNX_MODEL) if model_file_ok(ONNX_MODEL) else None,
            "pytorch": resolve_path(MODEL_PATH) if model_file_ok(MODEL_PATH) else None,
        }
        order = BACKEND_PRIORITY if pref == "auto" else (pref, *BACKEND_PRIORITY)

        last_error: Exception | None = None
        for name in order:
            if candidates.get(name) is None:
                continue
            try:
                det = cls(backend=name, **kwargs)
                det.warmup()
                if log:
                    logger.info("backend fallback: %s", " -> ".join(log))
                return det, log
            except Exception as exc:  # corrupt/unloadable backend -> next
                log.append(f"{name}: {type(exc).__name__}: {str(exc)[:120]}")
                logger.warning("backend '%s' failed to load, trying fallback: %s", name, exc)
                last_error = exc
        raise ModelNotTrainedError() if last_error is None else RuntimeError(
            f"all inference backends failed to load: {log}"
        )

    def detect_raw(self, image_bgr: np.ndarray, conf_threshold: float | None = None) -> list[dict]:
        """Run inference; returns raw detection dicts."""
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        if image_bgr is None:
            raise ValueError("detect_raw called with None image")
        if self.backend_name == "onnx":
            old = self._impl.conf_threshold
            self._impl.conf_threshold = conf
            try:
                return self._impl.detect_raw(image_bgr)
            finally:
                self._impl.conf_threshold = old
        results = self._impl.predict(image_bgr, conf=conf, iou=self.iou_threshold, imgsz=self.imgsz, verbose=False)[0]
        detections = []
        if results.boxes is not None and len(results.boxes) > 0:
            for xyxy, confidence, cls_id in zip(
                results.boxes.xyxy.cpu().numpy(),
                results.boxes.conf.cpu().numpy(),
                results.boxes.cls.cpu().numpy().astype(int),
            ):
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                h, w = image_bgr.shape[:2]
                detections.append(
                    {
                        "class": CLASS_NAMES[int(cls_id)] if 0 <= int(cls_id) < len(CLASS_NAMES) else str(int(cls_id)),
                        "confidence": float(confidence),
                        "bbox": {
                            "x1": max(0.0, x1),
                            "y1": max(0.0, y1),
                            "x2": min(float(w), x2),
                            "y2": min(float(h), y2),
                        },
                    }
                )
        return detections

    def detect_annotated(self, image_bgr: np.ndarray, conf_threshold: float | None = None) -> tuple[list[dict], np.ndarray]:
        """Detect and return (detections, annotated_bgr_image)."""
        detections = self.detect_raw(image_bgr, conf_threshold)
        annotated = image_bgr.copy() if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        from app.inference.preprocessing import draw_detection

        for det in detections:
            b = det["bbox"]
            draw_detection(annotated, int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"]), det["class"], det["confidence"])
        return detections, annotated

    def predict(self, image_bgr: np.ndarray, **kwargs):
        """Ultralytics-style passthrough (used by evaluate / error analysis)."""
        if self.backend_name == "onnx":
            raise NotImplementedError("use detect_raw for the ONNX backend")
        return self._impl.predict(image_bgr, **kwargs)
