"""ONNX Runtime detection backend for the NEU defect model.

Runs a YOLOv8-style ONNX detection graph (Ultralytics export) with pure
NumPy + OpenCV postprocessing: decode, confidence filter, class-aware NMS.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.config import CLASS_NAMES


class OnnxDetector:
    """YOLOv8 ONNX inference with the same interface as the PyTorch backend."""

    def __init__(self, onnx_path: str | Path, imgsz: int = 640, conf_threshold: float = 0.5, iou_threshold: float = 0.45):
        import onnxruntime as ort

        self.onnx_path = Path(onnx_path)
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        providers = ort.get_available_providers()
        # Prefer CUDA EP when available, fall back to CPU
        chosen = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in providers]
        self.session = ort.InferenceSession(str(self.onnx_path), providers=chosen)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # e.g. [1,3,640,640]
        self.provider_used = self.session.get_providers()[0]
        # output layout (1, 4+nc, N) from Ultralytics YOLOv8 export
        self.num_classes = len(CLASS_NAMES)

    def _preprocess(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        from app.inference.preprocessing import letterbox

        img = image_bgr
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        letterboxed, scale, pad_x, pad_y = letterbox(img, (self.imgsz, self.imgsz))
        blob = letterboxed[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0  # BGR->RGB, HWC->CHW
        blob = np.ascontiguousarray(blob[None])  # NCHW
        return blob, scale, pad_x, pad_y

    def _postprocess(self, output: np.ndarray, scale: float, pad_x: float, pad_y: float, orig_shape) -> list[dict]:
        # output: (1, 4+nc, N) -> (4+nc, N)
        pred = output[0]
        boxes_cxcywh = pred[:4, :]  # xywh in letterboxed pixel coords
        class_scores = pred[4 : 4 + self.num_classes, :]
        if class_scores.shape[0] == 0:
            return []
        class_ids = np.argmax(class_scores, axis=0)
        confidences = class_scores[class_ids, np.arange(class_scores.shape[1])]

        keep = confidences >= self.conf_threshold
        if not np.any(keep):
            return []
        boxes_cxcywh = boxes_cxcywh[:, keep]
        class_ids = class_ids[keep]
        confidences = confidences[keep]

        # cx,cy,w,h -> xyxy (letterbox space), then map back to original image
        cx, cy, w, h = boxes_cxcywh
        x1 = (cx - w / 2 - pad_x) / scale
        y1 = (cy - h / 2 - pad_y) / scale
        x2 = (cx + w / 2 - pad_x) / scale
        y2 = (cy + h / 2 - pad_y) / scale

        orig_h, orig_w = orig_shape[:2]
        boxes_px = np.stack([x1, y1, x2, y2], axis=1)
        boxes_px[:, [0, 2]] = boxes_px[:, [0, 2]].clip(0, orig_w)
        boxes_px[:, [1, 3]] = boxes_px[:, [1, 3]].clip(0, orig_h)

        # class-aware NMS via OpenCV
        indices = cv2.dnn.NMSBoxes(
            boxes_px.tolist(), confidences.tolist(), self.conf_threshold, self.iou_threshold
        )
        detections = []
        if indices is not None and len(indices) > 0:
            for i in np.array(indices).flatten():
                bx = boxes_px[i]
                detections.append(
                    {
                        "class": CLASS_NAMES[int(class_ids[i])],
                        "confidence": float(confidences[i]),
                        "bbox": {
                            "x1": float(bx[0]),
                            "y1": float(bx[1]),
                            "x2": float(bx[2]),
                            "y2": float(bx[3]),
                        },
                    }
                )
        return detections

    def detect_raw(self, image_bgr: np.ndarray) -> list[dict]:
        """Run inference; returns raw detection dicts (no severity/status logic)."""
        blob, scale, pad_x, pad_y = self._preprocess(image_bgr)
        outputs = self.session.run(None, {self.input_name: blob})
        return self._postprocess(outputs[0], scale, pad_x, pad_y, image_bgr.shape)
