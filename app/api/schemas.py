"""Pydantic schemas for the defect-detection API."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class BoundingBox(BaseModel):
    x1: float = Field(..., description="top-left x in pixels")
    y1: float = Field(..., description="top-left y in pixels")
    x2: float = Field(..., description="bottom-right x in pixels")
    y2: float = Field(..., description="bottom-right y in pixels")


class Detection(BaseModel):
    class_name: str = Field(..., alias="class", description="one of the six NEU defect classes")
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox
    severity: str = Field(..., pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$")

    model_config = {"populate_by_name": True}


class Performance(BaseModel):
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_ms: float


class ImageInspectionResponse(BaseModel):
    inspection_status: str = Field(..., pattern="^(PASS|REJECT)$")
    defect_detected: bool
    defect_count: int = Field(..., ge=0)
    detections: list[Detection]
    dominant_defect: str | None = None
    highest_confidence: float = 0.0
    max_severity: str | None = None
    performance: Performance
    source: str = ""
    timestamp: str = ""
    annotated_image_url: str | None = None


class ModelStatus(BaseModel):
    backend: str
    model_path: str
    model_format: str
    conf_threshold: float
    iou_threshold: float
    imgsz: int | None = None
    device: str | None = None
    provider: str | None = None


class StatusResponse(BaseModel):
    model_loaded: bool
    model: ModelStatus | None
    training_hint: str | None = None
    history: dict


class PLCCommandRequest(BaseModel):
    action: str = Field(..., pattern="^(PASS|REJECT)$")
    defect: str | None = None
    confidence: float | None = Field(None, ge=0.0, le=1.0)

    @field_validator("defect")
    @classmethod
    def defect_must_be_known(cls, v):
        if v is not None:
            from app.config import CLASS_NAMES

            if v not in CLASS_NAMES:
                raise ValueError(f"unknown defect class '{v}'; expected one of {CLASS_NAMES}")
        return v


class PLCCommandResponse(BaseModel):
    action: str
    reject_signal: bool
    defect: str | None = None
    confidence: float | None = None
    line_id: str
    timestamp: str


class ClassesResponse(BaseModel):
    classes: list[str]
    display_names: dict[str, str]
    num_classes: int


class MetricsSummaryResponse(BaseModel):
    total_events: int
    passed: int
    rejected: int
    defect_rate_pct: float
    avg_inference_ms: float
    avg_confidence: float
    class_counts: dict[str, int]
    trend: dict[str, int]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    inference_backend: str = "none"
    device: str = "unknown"
    camera_online: bool = False
    database_ok: bool
    version: str


class HistoryEvent(BaseModel):
    event_id: int
    timestamp: str
    source: str
    defect_class: str | None
    confidence: float | None
    severity: str | None
    bbox: str | None
    inference_latency_ms: float | None
    inspection_status: str


class HistoryResponse(BaseModel):
    events: list[HistoryEvent]
    count: int
    total: int = 0
