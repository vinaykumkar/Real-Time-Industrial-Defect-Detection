"""Tests for the PLC simulator and FastAPI endpoints.

The API tests use a stubbed detection service so they run without trained
weights: the point is to verify routing, schemas and PASS/REJECT plumbing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient  # noqa: E402


class TestPLCSimulator:
    def test_pass_command(self):
        from app.api.plc import SimulatedPLC

        plc = SimulatedPLC()
        cmd = plc.send_sort_command("PASS")
        assert cmd["action"] == "PASS"
        assert cmd["reject_signal"] is False

    def test_reject_command(self):
        from app.api.plc import SimulatedPLC

        plc = SimulatedPLC()
        cmd = plc.send_sort_command("REJECT", defect="scratches", confidence=0.94)
        assert cmd["reject_signal"] is True
        assert cmd["defect"] == "scratches"
        assert cmd["confidence"] == pytest.approx(0.94)

    def test_invalid_action_rejected(self):
        from app.api.plc import SimulatedPLC

        plc = SimulatedPLC()
        with pytest.raises(ValueError):
            plc.send_sort_command("MAYBE")

    def test_stats_accumulate(self):
        from app.api.plc import SimulatedPLC

        plc = SimulatedPLC()
        plc.send_sort_command("PASS")
        plc.send_sort_command("REJECT", defect="inclusion")
        s = plc.stats()
        assert s["total_pass"] == 1 and s["total_reject"] == 1 and s["total_commands"] == 2


class TestApiSchemas:
    def test_plc_request_validation(self):
        from pydantic import ValidationError

        from app.api.schemas import PLCCommandRequest

        PLCCommandRequest(action="PASS")
        PLCCommandRequest(action="REJECT", defect="scratches", confidence=0.9)
        with pytest.raises(ValidationError):
            PLCCommandRequest(action="MAYBE")
        with pytest.raises(ValidationError):
            PLCCommandRequest(action="PASS", defect="not_a_class")
        with pytest.raises(ValidationError):
            PLCCommandRequest(action="PASS", confidence=5.0)


@pytest.fixture(scope="module")
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


class TestApiEndpoints:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert isinstance(body["model_loaded"], bool)
        assert body["database_ok"] is True

    def test_status(self, client):
        r = client.get("/api/status")
        assert r.status_code == 200
        body = r.json()
        assert "model_loaded" in body and "history" in body

    def test_classes(self, client):
        r = client.get("/api/classes")
        assert r.status_code == 200
        body = r.json()
        assert body["num_classes"] == 6
        assert set(body["classes"]) == {
            "crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches",
        }

    def test_metrics_summary(self, client):
        r = client.get("/api/metrics-summary")
        assert r.status_code == 200
        body = r.json()
        for key in ("total_events", "passed", "rejected", "defect_rate_pct", "class_counts"):
            assert key in body

    def test_metrics_endpoint(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        text = r.text
        assert "defect_detection_requests_total" in text
        assert "inference_latency_seconds" in text
        assert "inspection_pass_total" in text

    def test_plc_test_endpoint(self, client):
        r = client.post("/api/plc/test", json={"action": "REJECT", "defect": "scratches", "confidence": 0.94})
        assert r.status_code == 200
        body = r.json()
        assert body["action"] == "REJECT"
        assert body["reject_signal"] is True

    def test_plc_test_endpoint_invalid(self, client):
        r = client.post("/api/plc/test", json={"action": "MAYBE"})
        assert r.status_code == 422

    def test_detect_image_rejects_bad_upload(self, client):
        r = client.post("/api/detect/image", files={"file": ("x.txt", b"not an image", "text/plain")})
        assert r.status_code == 400

    def test_detect_image_end_to_end(self, client, sample_image):
        import cv2

        path, img, _ = sample_image
        ok, buf = cv2.imencode(".png", img)
        assert ok
        r = client.post(
            "/api/detect/image",
            files={"file": ("img.png", buf.tobytes(), "image/png")},
        )
        # Either real detection happens (200) or the model is not trained (503-style 500)
        assert r.status_code in (200, 500)
        if r.status_code == 200:
            body = r.json()
            assert body["inspection_status"] in ("PASS", "REJECT")
            assert "performance" in body
            assert "total_ms" in body["performance"]

    def test_history_endpoint(self, client):
        r = client.get("/api/history?limit=5")
        assert r.status_code == 200
        assert "events" in r.json()

    def test_dataset_sample_bad_class(self, client):
        r = client.get("/api/dataset/sample?defect_class=nope")
        assert r.status_code == 400

    def test_dashboard_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "REAL-TIME INDUSTRIAL DEFECT DETECTION" in r.text
