"""Shared pytest fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture()
def sample_voc_xml(tmp_path: Path) -> Path:
    """A minimal valid Pascal VOC XML file."""
    xml = """<annotation>
  <filename>scratches_1.jpg</filename>
  <size><width>200</width><height>200</height><depth>1</depth></size>
  <object>
    <name>scratches</name>
    <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>110</xmax><ymax>70</ymax></bndbox>
  </object>
  <object>
    <name>crazing</name>
    <bndbox><xmin>50</xmin><ymin>100</ymin><xmax>150</xmax><ymax>180</ymax></bndbox>
  </object>
</annotation>"""
    p = tmp_path / "scratches_1.xml"
    p.write_text(xml, encoding="utf-8")
    return p


@pytest.fixture()
def sample_image(tmp_path: Path):
    import numpy as np

    from app.config import CLASS_NAMES

    rng = np.random.default_rng(7)
    img = (rng.random((200, 200, 3)) * 255).astype("uint8")
    p = tmp_path / "img.png"
    import cv2

    cv2.imwrite(str(p), img)
    return str(p), img, CLASS_NAMES
