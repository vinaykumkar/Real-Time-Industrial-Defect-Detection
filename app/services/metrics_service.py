"""Prometheus metrics for the defect-detection system.

Exposed on GET /metrics. Metric types chosen per semantics:
Counters for totals, Gauges for instantaneous state, Histogram for latency.

All collectors are created through _get_or_create so importing this module
twice in one process (uvicorn --reload, pytest + live server) never raises
"Duplicate timeseries in CollectorRegistry".
"""
from __future__ import annotations

from prometheus_client import REGISTRY, Counter, Gauge, Histogram


def _get_or_create(cls, name: str, *args, **kwargs):
    existing = REGISTRY._names_to_collectors.get(name)
    if existing is not None:
        return existing
    return cls(name, *args, **kwargs)

DEFECT_DETECTION_REQUESTS = _get_or_create(Counter, 
    "defect_detection_requests_total",
    "Total defect-detection requests processed",
)
DEFECTS_DETECTED = _get_or_create(Counter, 
    "defects_detected_total",
    "Total individual defects detected",
    ["defect_class"],
)
INSPECTION_PASS = _get_or_create(Counter, 
    "inspection_pass_total",
    "Total PASS inspection results",
)
INSPECTION_REJECT = _get_or_create(Counter, 
    "inspection_reject_total",
    "Total REJECT inspection results",
)
INFERENCE_LATENCY = _get_or_create(Histogram, 
    "inference_latency_seconds",
    "End-to-end inspection latency in seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
CAMERA_STATUS = _get_or_create(Gauge, 
    "camera_status",
    "Camera connection status (1 = online, 0 = offline)",
)
CAMERA_UPTIME = _get_or_create(Gauge, 
    "camera_uptime_seconds",
    "Seconds since the camera/stream came online",
)
CURRENT_CAMERA_FPS = _get_or_create(Gauge, 
    "current_camera_fps",
    "Most recent measured camera FPS",
)
API_ERRORS = _get_or_create(Counter, 
    "api_errors_total",
    "Total API errors",
    ["endpoint"],
)
DETECTIONS_IN_LAST_FRAME = _get_or_create(Gauge,
    "detections_last_frame",
    "Number of defects detected in the most recent processed frame",
)
FRAMES_PROCESSED = _get_or_create(Counter,
    "camera_frames_processed_total",
    "Total camera/video frames processed by inspection",
)
PLC_EVENTS = _get_or_create(Counter,
    "plc_events_total",
    "Simulated PLC sorting commands issued",
    ["action"],
)
PLC_DECISION_LATENCY = _get_or_create(Gauge,
    "plc_decision_latency_ms",
    "Milliseconds from confirmed defect decision to PLC command dispatch",
)


def record_inspection(result_dict: dict) -> None:
    """Update counters/gauges/histogram from an InspectionResult.as_dict()."""
    DEFECT_DETECTION_REQUESTS.inc()
    DETECTIONS_IN_LAST_FRAME.set(result_dict.get("defect_count", 0))
    perf = result_dict.get("performance", {})
    total_ms = perf.get("total_ms", 0)
    if total_ms > 0:
        INFERENCE_LATENCY.observe(total_ms / 1000.0)
    if result_dict.get("inspection_status") == "REJECT":
        INSPECTION_REJECT.inc()
    else:
        INSPECTION_PASS.inc()
    for det in result_dict.get("detections", []):
        DEFECTS_DETECTED.labels(defect_class=str(det.get("class", "unknown"))).inc()


def record_api_error(endpoint: str) -> None:
    API_ERRORS.labels(endpoint=endpoint).inc()


def set_camera_state(online: bool, fps: float | None = None, uptime_seconds: float | None = None) -> None:
    CAMERA_STATUS.set(1 if online else 0)
    if fps is not None:
        CURRENT_CAMERA_FPS.set(fps)
    if uptime_seconds is not None:
        CAMERA_UPTIME.set(uptime_seconds)
