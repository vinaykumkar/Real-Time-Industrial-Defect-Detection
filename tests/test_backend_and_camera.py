"""Integration tests: ONNX backend, camera source parsing, upload edge cases."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestOnnxBackend:
    """Loads the real exported ONNX model if present (skipped otherwise)."""

    ONNX = PROJECT_ROOT / "models" / "best.onnx"

    @staticmethod
    def _imgsz() -> int:
        from app.inference.detector import _default_imgsz

        return _default_imgsz()

    @pytest.mark.skipif(not ONNX.exists(), reason="models/best.onnx not exported yet")
    def test_onnx_loads_and_detects_valid_output(self):
        from app.inference.onnx_detector import OnnxDetector

        det = OnnxDetector(self.ONNX, imgsz=self._imgsz(), conf_threshold=0.25)
        assert det.session is not None
        rng = np.random.default_rng(1)
        img = (rng.random((200, 200, 3)) * 255).astype("uint8")
        dets = det.detect_raw(img)
        # detections list may be empty on a blank image, but the call must work
        for d in dets:
            assert d["class"] in {
                "crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches",
            }
            assert 0.0 <= d["confidence"] <= 1.0
            for v in d["bbox"].values():
                assert v >= 0.0

    @pytest.mark.skipif(not ONNX.exists(), reason="models/best.onnx not exported yet")
    def test_real_validation_image_detection(self):
        """End-to-end: a real NEU image through the ONNX backend must either
        detect something or return an empty list — never crash."""
        import cv2

        from app.inference.onnx_detector import OnnxDetector

        images = sorted((PROJECT_ROOT / "data" / "yolo" / "images" / "val").glob("*.jpg"))
        if not images:
            pytest.skip("converted dataset missing")
        img = cv2.imdecode(np.fromfile(str(images[0]), dtype=np.uint8), cv2.IMREAD_COLOR)
        det = OnnxDetector(self.ONNX, imgsz=self._imgsz(), conf_threshold=0.25)
        assert det.detect_raw(img) is not None


class TestCameraSourceParsing:
    def test_numeric_string_becomes_int(self):
        from app.services.camera_service import _parse_source

        assert _parse_source("0") == 0
        assert _parse_source("1") == 1

    def test_rtsp_and_file_pass_through(self):
        from app.services.camera_service import _parse_source

        assert _parse_source("rtsp://cam/stream") == "rtsp://cam/stream"
        assert _parse_source("video.mp4") == "video.mp4"

    def test_none_uses_configured_default(self):
        from app.config import CAMERA_SOURCE
        from app.services.camera_service import _parse_source

        assert _parse_source(None) == _parse_source(str(CAMERA_SOURCE))


class TestDetectionHistoryPersistence:
    def test_insert_and_read_roundtrip(self, tmp_path):
        from app.services.detection_history import DetectionHistory

        db = DetectionHistory(tmp_path / "sub" / "auto_created.db")
        db.record_inspection({
            "timestamp": "2026-01-01T10:00:00",
            "source": "unit-test",
            "inspection_status": "REJECT",
            "performance": {"inference_ms": 12.5},
            "detections": [{
                "class": "scratches", "confidence": 0.9, "severity": "HIGH",
                "bbox": {"x1": 1, "y1": 2, "x2": 3, "y2": 4},
            }],
        })
        rows = db.query(limit=10)
        assert len(rows) == 1
        assert rows[0]["defect_class"] == "scratches"
        assert rows[0]["inspection_status"] == "REJECT"
        s = db.summary()
        assert s["total_events"] == 1 and s["rejected"] == 1
        assert db.clear() == 1
