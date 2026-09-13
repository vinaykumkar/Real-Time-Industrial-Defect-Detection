"""Edge-pipeline tests: backend fallback, temporal confirmation, PLC debounce,
timing schema, camera source parsing, config flags."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestTemporalValidator:
    def test_confirms_after_min_frames(self):
        from app.services.qc import TemporalValidator

        tv = TemporalValidator(window=3, min_defect_frames=2)
        assert tv.update(False) is False      # 0/3
        assert tv.update(True) is False       # 1/3
        assert tv.update(True) is True        # 2/3 confirmed

    def test_single_frame_noise_does_not_confirm(self):
        from app.services.qc import TemporalValidator

        tv = TemporalValidator(window=3, min_defect_frames=2)
        tv.update(False)
        assert tv.update(True) is False
        assert tv.update(False) is False

    def test_window_slides(self):
        from app.services.qc import TemporalValidator

        tv = TemporalValidator(window=2, min_defect_frames=2)
        assert tv.update(True) is False
        assert tv.update(True) is True
        assert tv.update(False) is False      # oldest True slid out

    def test_invalid_config_rejected(self):
        from app.services.qc import TemporalValidator

        with pytest.raises(ValueError):
            TemporalValidator(window=3, min_defect_frames=4)
        with pytest.raises(ValueError):
            TemporalValidator(window=0, min_defect_frames=1)

    def test_disable_by_window_one(self):
        from app.services.qc import TemporalValidator

        tv = TemporalValidator(window=1, min_defect_frames=1)
        assert tv.update(True) is True        # disabled = instant confirmation


class TestPLCDebouncer:
    def test_first_reject_allowed_then_suppressed(self):
        from app.services.qc import PLCDebouncer

        d = PLCDebouncer(cooldown_ms=1000)
        assert d.allow("REJECT", now_ms=0) is True
        assert d.allow("REJECT", now_ms=500) is False
        assert d.allow("REJECT", now_ms=999) is False
        assert d.allow("REJECT", now_ms=1000) is True

    def test_pass_always_allowed(self):
        from app.services.qc import PLCDebouncer

        d = PLCDebouncer(cooldown_ms=10_000)
        assert d.allow("PASS", now_ms=0) is True
        assert d.allow("PASS", now_ms=1) is True


class TestBackendFallback:
    def test_auto_selects_loadable_backend(self):
        """auto must pick a backend that actually loads (ONNX or PT on this box)."""
        from app.inference.detector import Detector

        det, log = Detector.create_with_fallback(backend="auto")
        assert det.get_backend_name() in ("onnx", "pytorch", "tensorrt")
        assert det.get_device()  # non-empty real device string
        info = det.get_performance_info()
        assert info["backend"] == det.get_backend_name()
        assert "imgsz" in info and info["imgsz"] > 0

    def test_corrupt_onnx_falls_back(self, tmp_path, monkeypatch):
        """A corrupt ONNX file must be logged as failed and fall back to PT."""
        bad = tmp_path / "bad.onnx"
        bad.write_bytes(b"not an onnx model")
        from app.config import PROJECT_ROOT
        from app.inference import detector as det_mod

        real_resolve = det_mod.resolve_path
        target = real_resolve("models/best.onnx")

        def fake_resolve(p):
            if Path(p) in (target, Path("models/best.onnx")):
                return bad
            return real_resolve(p)

        monkeypatch.setattr(det_mod, "resolve_path", fake_resolve)
        det, log = det_mod.Detector.create_with_fallback(backend="onnx")
        assert det.get_backend_name() == "pytorch"
        assert any("onnx" in entry for entry in log)

    def test_missing_model_raises_clear_error(self, monkeypatch):
        from app.inference import detector as det_mod

        monkeypatch.setattr(det_mod, "model_file_ok", lambda p: False)
        monkeypatch.setattr(det_mod, "_find_engine_model", lambda: None)
        with pytest.raises(det_mod.ModelNotTrainedError):
            det_mod.resolve_backend("auto")


class TestTimingSchema:
    def test_inspection_result_has_full_timing_breakdown(self, sample_image):
        import cv2

        from app.services.qc import inspect_frame

        _, img, _ = sample_image

        from app.inference.detector import Detector

        det = Detector.create_with_fallback(backend="onnx")[0]
        res = inspect_frame(det, img, source="test")
        perf = res.timings.as_dict()
        for key in ("preprocess_ms", "inference_ms", "postprocess_ms", "total_ms"):
            assert key in perf
            assert isinstance(perf[key], (int, float)) and perf[key] >= 0
        d = res.as_dict()
        assert d["inspection_status"] in ("PASS", "REJECT")
        assert d["defect_count"] == len(d["detections"])
        # measured, not constant: two runs should not both be exactly 0.0
        res2 = inspect_frame(det, img, source="test")
        assert res2.timings.inference_ms > 0


class TestEdgeConfig:
    def test_new_flags_exist_with_sane_defaults(self):
        from app import config

        assert config.PROCESS_EVERY_N_FRAMES >= 1
        assert 1 <= config.MIN_DEFECT_FRAMES <= config.TEMPORAL_WINDOW
        assert config.PLC_REJECT_COOLDOWN_MS >= 0
        assert config.TEMPORAL_WINDOW >= 1

    def test_model_file_ok(self, tmp_path):
        from app.config import model_file_ok

        f = tmp_path / "m.pt"
        assert model_file_ok(f) is False          # missing
        f.write_bytes(b"x")
        assert model_file_ok(f) is True           # non-empty
        (tmp_path / "empty.pt").write_bytes(b"")
        assert model_file_ok(tmp_path / "empty.pt") is False  # zero-size
