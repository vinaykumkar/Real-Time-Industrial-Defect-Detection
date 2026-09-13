"""Tests for configuration loading and PASS/REJECT + severity logic."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (  # noqa: E402
    CLASS_NAMES,
    CLASS_SEVERITY_WEIGHT,
    PROJECT_ROOT,
    resolve_path,
)
from app.services.qc import SEVERITY_LEVELS, compute_severity  # noqa: E402


class TestConfig:
    def test_project_root_resolves(self):
        assert PROJECT_ROOT.name.lower() == "neu" or PROJECT_ROOT.exists()

    def test_resolve_relative(self):
        p = resolve_path("models/best.pt")
        assert p.is_absolute()
        assert "models" in str(p)

    def test_resolve_absolute_passthrough(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            # resolve() both sides: Windows may return 8.3 short paths for temp dirs
            assert resolve_path(d).resolve() == Path(d).resolve()


class TestPassRejectAndSeverity:
    def test_severity_levels_ordered(self):
        assert SEVERITY_LEVELS == ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def test_scratches_high_confidence_is_high_or_critical(self):
        sev = compute_severity("scratches", 0.95, 0.05, 1)
        assert sev in ("HIGH", "CRITICAL")

    def test_crazing_low_confidence_is_low(self):
        sev = compute_severity("crazing", 0.05, 0.001, 1)
        assert sev == "LOW"

    def test_more_defects_increase_severity(self):
        low = compute_severity("crazing", 0.2, 0.01, 1)
        high = compute_severity("crazing", 0.2, 0.01, 10)
        order = {s: i for i, s in enumerate(SEVERITY_LEVELS)}
        assert order[high] >= order[low]

    def test_unknown_class_uses_default_weight(self):
        sev = compute_severity("totally_unknown_class", 0.5, 0.01, 1)
        assert sev in SEVERITY_LEVELS

    def test_every_configured_class_has_weight(self):
        for c in CLASS_NAMES:
            assert c in CLASS_SEVERITY_WEIGHT
