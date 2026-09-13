"""Industrial quality-control logic: severity grading and PASS/REJECT.

SEVERITY NOTE: the severity model below is configurable demonstration /
business logic for this project — it is NOT an industry-certified severity
standard (e.g. it is not calibrated to any particular plant's acceptance
criteria). Tune CLASS_SEVERITY_WEIGHT / severity thresholds in the config or
via code to match real production rules.

PASS/REJECT: PASS when no detection's confidence reaches
PASS_REJECT_THRESHOLD; REJECT otherwise.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from app.config import (
    CLASS_SEVERITY_WEIGHT,
    PASS_REJECT_THRESHOLD,
    PLC_REJECT_COOLDOWN_MS,
    TEMPORAL_WINDOW,
    MIN_DEFECT_FRAMES,
)
from app.inference.detector import Detector
from app.inference.preprocessing import Timings

SEVERITY_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class TemporalValidator:
    """N-of-M temporal confirmation for live REJECT decisions.

    A defect must appear in at least `min_defect_frames` of the last
    `window` processed frames before the part is confirmed REJECT. This
    prevents single-frame prediction flicker from triggering the PLC.

    NOTE: designed for conveyor-style live inspection where consecutive
    frames show (roughly) the same surface. For a moving line with distinct
    parts per frame, disable it (window=1, min=1).
    """

    def __init__(self, window: int = TEMPORAL_WINDOW, min_defect_frames: int = MIN_DEFECT_FRAMES):
        if window < 1 or min_defect_frames < 1 or min_defect_frames > window:
            raise ValueError(
                f"invalid temporal config: window={window}, min_defect_frames={min_defect_frames}"
            )
        self.window = window
        self.min_defect_frames = min_defect_frames
        self._recent: deque[bool] = deque(maxlen=window)

    def update(self, defect_detected: bool) -> bool:
        """Feed one frame result; returns the CONFIRMED decision."""
        self._recent.append(bool(defect_detected))
        return sum(self._recent) >= self.min_defect_frames

    @property
    def recent(self) -> list[bool]:
        return list(self._recent)


class PLCDebouncer:
    """Cooldown gate so a continuous defect does not spam REJECT commands.

    The first confirmed REJECT passes immediately; further REJECT signals
    are suppressed until `cooldown_ms` has elapsed. PASS events always
    pass through (so the line can resume).
    """

    def __init__(self, cooldown_ms: float = PLC_REJECT_COOLDOWN_MS):
        self.cooldown_ms = cooldown_ms
        self._last_reject_ms: float = -float("inf")

    def allow(self, action: str, now_ms: float | None = None) -> bool:
        if action != "REJECT":
            return True
        t = time.perf_counter() * 1000.0 if now_ms is None else now_ms
        if (t - self._last_reject_ms) >= self.cooldown_ms:
            self._last_reject_ms = t
            return True
        return False

    @property
    def ms_until_next_reject(self) -> float:
        t = time.perf_counter() * 1000.0
        return max(0.0, self.cooldown_ms - (t - self._last_reject_ms))


def compute_severity(
    class_name: str,
    confidence: float,
    box_area_fraction: float,
    total_defects_in_image: int,
) -> str:
    """Grade one detection's severity: LOW < MEDIUM < HIGH < CRITICAL.

    Factors: defect class weight, model confidence, relative defect size,
    and how many defects share the image. Purely rule-based and configurable.
    """
    weight = CLASS_SEVERITY_WEIGHT.get(class_name, 0.5)
    size_factor = min(1.0, box_area_fraction * 10.0)  # a defect covering >=10% of frame is "large"
    score = 0.55 * weight + 0.25 * confidence + 0.12 * size_factor + 0.08 * min(1.0, total_defects_in_image / 4.0)
    if score >= 0.80:
        return "CRITICAL"
    if score >= 0.65:
        return "HIGH"
    if score >= 0.50:
        return "MEDIUM"
    return "LOW"


@dataclass
class InspectionResult:
    inspection_status: str            # "PASS" or "REJECT"
    defect_detected: bool
    defect_count: int
    detections: list[dict]            # each: class, confidence, bbox, severity
    dominant_defect: str | None
    highest_confidence: float
    max_severity: str | None
    timings: Timings
    annotated_image: np.ndarray | None = None
    source: str = ""
    timestamp: str = ""

    def as_dict(self) -> dict:
        d = {
            "inspection_status": self.inspection_status,
            "defect_detected": self.defect_detected,
            "defect_count": self.defect_count,
            "detections": self.detections,
            "dominant_defect": self.dominant_defect,
            "highest_confidence": round(self.highest_confidence, 4) if self.detections else 0.0,
            "max_severity": self.max_severity,
            "performance": self.timings.as_dict(),
            "source": self.source,
            "timestamp": self.timestamp,
        }
        return d


def inspect_frame(
    detector: Detector,
    frame: np.ndarray,
    source: str = "frame",
    pass_reject_threshold: float | None = None,
) -> InspectionResult:
    """Full single-frame inspection: detect -> severity -> PASS/REJECT.

    All latency values are measured with perf_counter (never estimated).
    """
    from app.inference.preprocessing import Stopwatch

    threshold = pass_reject_threshold if pass_reject_threshold is not None else PASS_REJECT_THRESHOLD

    with Stopwatch() as t_pre:
        pass  # preprocessing happens inside the backend; measured slot kept for parity

    with Stopwatch() as t_inf:
        raw_detections = detector.detect_raw(frame)

    with Stopwatch() as t_post:
        h, w = frame.shape[:2]
        frame_area = float(h * w)
        detections = []
        for det in raw_detections:
            b = det["bbox"]
            bw = max(0.0, b["x2"] - b["x1"])
            bh = max(0.0, b["y2"] - b["y1"])
            area_fraction = (bw * bh) / frame_area if frame_area > 0 else 0.0
            detections.append(
                {
                    "class": det["class"],
                    "confidence": det["confidence"],
                    "bbox": {"x1": b["x1"], "y1": b["y1"], "x2": b["x2"], "y2": b["y2"]},
                    "severity": compute_severity(det["class"], det["confidence"], area_fraction, len(raw_detections)),
                }
            )

        defect_detected = any(d["confidence"] >= threshold for d in detections)
        inspection_status = "REJECT" if defect_detected else "PASS"
        if detections:
            dominant = max(detections, key=lambda d: d["confidence"] * CLASS_SEVERITY_WEIGHT.get(d["class"], 0.5))
            dominant_defect = dominant["class"]
            highest_confidence = max(d["confidence"] for d in detections)
            severity_rank = {s: i for i, s in enumerate(SEVERITY_LEVELS)}
            max_severity = max((d["severity"] for d in detections), key=lambda s: severity_rank.get(s, 0))
        else:
            dominant_defect = None
            highest_confidence = 0.0
            max_severity = None

    timings = Timings(preprocess_ms=t_pre.ms, inference_ms=t_inf.ms, postprocess_ms=t_post.ms)
    return InspectionResult(
        inspection_status=inspection_status,
        defect_detected=defect_detected,
        defect_count=len(detections),
        detections=detections,
        dominant_defect=dominant_defect,
        highest_confidence=highest_confidence,
        max_severity=max_severity,
        timings=timings,
        source=source,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def annotate(frame: np.ndarray, result: InspectionResult) -> np.ndarray:
    """Return a copy of the frame with boxes + status overlay drawn."""
    from app.inference.preprocessing import draw_detection

    annotated = frame.copy() if frame.ndim == 3 else np.stack([frame] * 3, axis=-1)
    for d in result.detections:
        b = d["bbox"]
        draw_detection(
            annotated,
            int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"]),
            d["class"], d["confidence"],
        )
    h, w = annotated.shape[:2]
    status = result.inspection_status
    color = (0, 200, 0) if status == "PASS" else (0, 0, 255)
    cv2_bg = np.zeros((46, 240, 3), dtype=np.uint8)
    cv2_bg[:] = color
    annotated[0:46, 0:240] = cv2_bg
    cv2.putText(annotated, status, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated
