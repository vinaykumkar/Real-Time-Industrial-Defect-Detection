"""Central configuration for the Real-Time Industrial Defect Detection System.

All paths resolve from the project root (D:\\NEU or wherever this repo is
cloned) via pathlib. Environment variables (see .env.example) override
defaults without touching code.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
YOLO_DATA_DIR = DATA_DIR / "yolo"
IMAGES_TRAIN_DIR = YOLO_DATA_DIR / "images" / "train"
IMAGES_VAL_DIR = YOLO_DATA_DIR / "images" / "val"
LABELS_TRAIN_DIR = YOLO_DATA_DIR / "labels" / "train"
LABELS_VAL_DIR = YOLO_DATA_DIR / "labels" / "val"
DATA_YAML = YOLO_DATA_DIR / "data.yaml"

MODELS_DIR = PROJECT_ROOT / "models"
PYTORCH_MODEL_PATH = MODELS_DIR / "best.pt"
ONNX_MODEL_PATH = MODELS_DIR / "best.onnx"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DETECTIONS_OUTPUT_DIR = OUTPUTS_DIR / "detections"
EVALUATION_OUTPUT_DIR = OUTPUTS_DIR / "evaluation"
ERROR_ANALYSIS_DIR = EVALUATION_OUTPUT_DIR / "errors"
TRAINING_OUTPUT_DIR = OUTPUTS_DIR / "training"

LOGS_DIR = PROJECT_ROOT / "logs"
DATABASE_PATH = DATA_DIR / "detections.db"

# ------------------------------------------------- defect classes (NEU) ----
# Confirmed from the dataset annotation XMLs (see training/inspect_dataset.py).
CLASS_NAMES: list[str] = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_ID_TO_NAME = dict(enumerate(CLASS_NAMES))

# Pretty labels for the UI / reports.
CLASS_DISPLAY_NAMES: dict[str, str] = {
    "crazing": "Crazing",
    "inclusion": "Inclusion",
    "patches": "Patches",
    "pitted_surface": "Pitted Surface",
    "rolled-in_scale": "Rolled-in Scale",
    "scratches": "Scratches",
}

# -------------------------------------------------------- inference env ----
MODEL_PATH = os.getenv("MODEL_PATH", str(PYTORCH_MODEL_PATH.relative_to(PROJECT_ROOT)))
ONNX_MODEL = os.getenv("ONNX_MODEL_PATH", str(ONNX_MODEL_PATH.relative_to(PROJECT_ROOT)))
TENSORRT_MODEL_PATH = os.getenv("TENSORRT_MODEL_PATH", "models/best.engine")
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "auto").lower()

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")
DATABASE = os.getenv("DATABASE_PATH", str(DATABASE_PATH.relative_to(PROJECT_ROOT)))
SAVE_DETECTIONS = os.getenv("SAVE_DETECTIONS", "true").lower() in ("1", "true", "yes")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# --------------------------------------------------- real-time pipeline ----
# Process every Nth captured frame (1 = every frame). >1 trades detection
# latency for throughput on weak hardware; boxes/camera state keep last result.
PROCESS_EVERY_N_FRAMES = int(os.getenv("PROCESS_EVERY_N_FRAMES", "1"))

# Temporal confirmation: REJECT only when a defect is present in
# MIN_DEFECT_FRAMES of the last TEMPORAL_WINDOW processed frames.
# Guards PLC/inspection decisions against single-frame noise.
TEMPORAL_WINDOW = int(os.getenv("TEMPORAL_WINDOW", "3"))
MIN_DEFECT_FRAMES = int(os.getenv("MIN_DEFECT_FRAMES", "2"))

# Minimum milliseconds between two simulated PLC REJECT commands for the same
# continuous line event (debounce/cooldown).
PLC_REJECT_COOLDOWN_MS = float(os.getenv("PLC_REJECT_COOLDOWN_MS", "2000"))

# --------------------------------------------------- PASS/REJECT logic ----
# If any detection's confidence >= PASS_REJECT_THRESHOLD the part is REJECTED.
# Default from measured threshold analysis (see outputs/evaluation/final/).
PASS_REJECT_THRESHOLD = float(os.getenv("PASS_REJECT_THRESHOLD", "0.35"))

# Severity weights per class (configurable demonstration/business logic —
# NOT an industry-certified severity standard; see PROJECT_DOCUMENTATION.md).
CLASS_SEVERITY_WEIGHT: dict[str, float] = {
    "scratches": 0.95,
    "patches": 0.85,
    "inclusion": 0.75,
    "rolled-in_scale": 0.65,
    "pitted_surface": 0.55,
    "crazing": 0.50,
}


def resolve_path(value: str | Path) -> Path:
    """Resolve a possibly-relative configured path against the project root."""
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def model_file_ok(path: str | Path) -> bool:
    """A model file is usable only if it exists and is non-empty."""
    try:
        p = resolve_path(path)
        return p.is_file() and p.stat().st_size > 0
    except OSError:
        return False


def ensure_dirs() -> None:
    """Create all runtime directories the application writes into."""
    for d in (
        MODELS_DIR,
        DETECTIONS_OUTPUT_DIR,
        EVALUATION_OUTPUT_DIR,
        ERROR_ANALYSIS_DIR,
        TRAINING_OUTPUT_DIR,
        LOGS_DIR,
        DATA_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
