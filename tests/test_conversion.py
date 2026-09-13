"""Tests for VOC -> YOLO conversion and bounding-box normalisation."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.convert_voc_to_yolo import parse_voc_xml, voc_box_to_yolo, CLASS_TO_ID  # noqa: E402


class TestVocBoxToYolo:
    def test_basic_conversion(self):
        # box (10,20)-(110,70) in 200x200 -> center (0.3,0.225) size (0.5,0.25)
        out = voc_box_to_yolo(10, 20, 110, 70, 200, 200)
        cx, cy, w, h = out
        assert cx == pytest.approx(0.30)
        assert cy == pytest.approx(0.225)
        assert w == pytest.approx(0.50)
        assert h == pytest.approx(0.25)

    def test_full_image_box(self):
        out = voc_box_to_yolo(0, 0, 200, 200, 200, 200)
        assert out == pytest.approx((0.5, 0.5, 1.0, 1.0))

    def test_normalisation_bounds(self):
        out = voc_box_to_yolo(0, 0, 200, 200, 200, 200)
        assert all(0.0 <= v <= 1.0 for v in out)

    def test_degenerate_box_rejected(self):
        assert voc_box_to_yolo(10, 10, 10, 50, 200, 200) is None  # zero width
        assert voc_box_to_yolo(10, 10, 50, 10, 200, 200) is None  # zero height

    def test_out_of_bounds_box_rejected(self):
        assert voc_box_to_yolo(-50, 10, 50, 60, 200, 200) is None
        assert voc_box_to_yolo(10, 10, 500, 60, 200, 200) is None


class TestParseVocXml:
    def test_parses_boxes_and_classes(self, sample_voc_xml):
        img_w, img_h, boxes, warnings = parse_voc_xml(sample_voc_xml)
        assert (img_w, img_h) == (200, 200)
        assert len(boxes) == 2
        assert not warnings
        class_ids = {b[0] for b in boxes}
        assert CLASS_TO_ID["scratches"] in class_ids
        assert CLASS_TO_ID["crazing"] in class_ids
        # all boxes normalised
        for b in boxes:
            assert all(0.0 <= v <= 1.0 for v in b[1:])

    def test_malformed_xml_raises(self, tmp_path):
        bad = tmp_path / "bad.xml"
        bad.write_text("<annotation><unclosed>", encoding="utf-8")
        import xml.etree.ElementTree as ET

        with pytest.raises(ET.ParseError):
            parse_voc_xml(bad)


class TestClassNames:
    def test_six_classes(self):
        from app.config import CLASS_NAMES, NUM_CLASSES

        assert NUM_CLASSES == 6
        assert set(CLASS_NAMES) == {
            "crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches",
        }
