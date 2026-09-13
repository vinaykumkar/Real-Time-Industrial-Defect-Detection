"""Image preprocessing and annotation drawing utilities.

All latency numbers reported by this module are measured with
time.perf_counter() — never estimated.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.config import CLASS_DISPLAY_NAMES

# BGR colours per class id (industrial dashboard palette)
CLASS_COLORS = [
    (0, 200, 255),   # crazing      - amber
    (255, 120, 0),   # inclusion    - blue (BGR)
    (0, 80, 255),    # patches      - red-orange
    (200, 0, 200),   # pitted_surface - magenta
    (0, 255, 200),   # rolled-in_scale - yellow-green
    (0, 0, 255),     # scratches    - red
]


def letterbox(img: np.ndarray, new_shape: tuple[int, int] = (640, 640), color: int = 114):
    """Resize with aspect-ratio-preserving padding.

    Returns (padded_image, scale, pad_x, pad_y) so raw pixel coords can be
    mapped back: x_raw = (x_model - pad_x) / scale.
    """
    if img is None:
        raise ValueError("letterbox called with None image")
    h, w = img.shape[:2]
    target_w, target_h = new_shape
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (target_w - new_w) / 2
    pad_y = (target_h - new_h) / 2
    top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
    left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))
    if img.ndim == 2:
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    else:
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(color,) * 3)
    return padded, scale, left, top


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_detection(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, class_name: str, confidence: float, thickness: int = 2):
    """Draw one bounding box + label onto a BGR image (in place)."""
    h, w = img.shape[:2]
    # thickness scaled by image size, min 1
    t = max(1, int(round(max(h, w) / 400)) * thickness if max(h, w) > 400 else thickness)
    color = CLASS_COLORS[class_display_index(class_name)]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, t)
    label = f"{CLASS_DISPLAY_NAMES.get(class_name, class_name)} {confidence * 100:.1f}%"
    fs = max(0.35, min(h, w) / 500)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    y_top = max(0, y1 - th - baseline - 4)
    cv2.rectangle(img, (x1, y_top), (min(w, x1 + tw + 4), y1), color, -1)
    txt_color = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
    cv2.putText(img, label, (x1 + 2, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, txt_color, 1)


def class_display_index(class_name: str) -> int:
    from app.config import CLASS_NAMES

    try:
        return CLASS_NAMES.index(class_name)
    except ValueError:
        return 0


@dataclass
class Timings:
    """Measured processing timings (milliseconds)."""

    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms

    def as_dict(self) -> dict:
        return {
            "preprocess_ms": round(self.preprocess_ms, 2),
            "inference_ms": round(self.inference_ms, 2),
            "postprocess_ms": round(self.postprocess_ms, 2),
            "total_ms": round(self.total_ms, 2),
        }


class Stopwatch:
    """Context manager measuring elapsed ms."""

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self._t0) * 1000.0
        return False
