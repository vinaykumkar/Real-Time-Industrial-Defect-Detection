# Real-Time Industrial Defect Detection System

YOLOv8-powered surface-defect detection for manufacturing quality control.
The system inspects metal/steel surface images, video files and live camera
streams, localises the **six NEU defect classes** with bounding boxes, grades
severity, issues **PASS / REJECT** decisions, drives a simulated **PLC
sorting line**, and exposes a professional **Industrial Quality Control
dashboard** with Prometheus/Grafana monitoring.

> **Goal:** *Minimises manufacturing waste and prevents defective products
> from reaching end consumers. The system is designed to enable
> high-throughput automated inspection while ensuring consistent, unbiased
> quality control at the edge.*

---

## Problem Statement

Manual visual inspection of rolled steel is slow, inconsistent and biased.
Defective products that escape inspection become waste or reach customers.
This project demonstrates an automated, edge-deployable quality-control
station that inspects every surface consistently and signals a sorting
mechanism in real time.

## Expected Industrial Impact

- **Less waste** — defects are caught at the line, not downstream.
- **Higher throughput** — inference per frame is measured in milliseconds.
- **Consistent, unbiased QC** — the same objective standard for every part.
- **Edge AI** — ONNX Runtime (and optionally TensorRT) run without a server.

## Features

| Area | What you get |
|---|---|
| Detection | YOLOv8n fine-tuned on NEU-DET (6 defect classes) |
| Backends | PyTorch → ONNX Runtime → TensorRT (auto-selected, only what actually exists) |
| Inspection | Image, video file, live webcam/RTSP (OpenCV) |
| Quality gate | PASS/REJECT + configurable severity (LOW/MEDIUM/HIGH/CRITICAL) |
| PLC | Simulated sorting line via a PLC adapter interface (Modbus/OPC UA-ready) |
| Dashboard | Overview, Live, Image, Video, History, Analytics, Model, Dataset Explorer, Settings |
| History | SQLite detection-event database with filtering |
| Monitoring | Prometheus `/metrics` (13 metric families) + Grafana — deployed and verified via Docker Compose |
| Tests | 53 pytest cases (conversion, config, QC logic, temporal confirmation, PLC debounce, backend fallback, API, edge) |

## Technology Stack

Python 3.10+ · PyTorch · Ultralytics YOLOv8 · OpenCV · Albumentations ·
FastAPI · Uvicorn · SQLite · ONNX / ONNX Runtime · prometheus-client ·
Chart.js dashboard · Docker (deployed & verified) · Prometheus + Grafana (deployed & verified)

## Architecture

```
Industrial Camera ──▶ OpenCV Capture ──▶ Preprocessing ──▶ YOLO Detector
      │                                                        │
      │                                          Defect Detection + Localization
      │                                                        │
      └──▶ FastAPI ◀──── PASS / REJECT Logic ◀─────────────────┘
                │                    │
                │              PLC Simulator ──▶ Sorting Mechanism
                │
        Dashboard / Prometheus / Grafana
```

See [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) for full Mermaid
diagrams, concept explanations (IoU, mAP, PLC, edge computing…) and viva
prep material.

---

## The NEU Dataset

[NEU-DET](http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/) — 1,800
grayscale 200×200 JPEGs of hot-rolled steel strip surfaces:

| Class | Defect | Train imgs | Val imgs |
|---|---|---|---|
| crazing | Fine surface cracks | 240 | 60 |
| inclusion | Foreign material embedded | 240 | 60 |
| patches | Patch-like surface damage | 240 | 60 |
| pitted_surface | Pits on the surface | 240 | 60 |
| rolled-in_scale | Scale rolled into surface | 240 | 60 |
| scratches | Linear scratches | 240 | 60 |

Annotations are **Pascal VOC XML**; images ship in per-class subfolders.
Verified facts on this copy: 1,440 train images (1,439 XML — `crazing_240`
has no XML), 360 val images (361 XML — `crazing_240.xml` has no image),
3,332 train + 854 val boxes, all 200×200. Several images contain multiple
defects (1,033/1,439 train images have >1 box).

## Dataset Structure (after conversion)

```
data/
├── raw/                      # extracted original archive (NEU-DET/...)
└── yolo/
    ├── images/train, images/val
    ├── labels/train, labels/val   # class cx cy w h  (normalised 0-1)
    └── data.yaml
```

---

## Installation (Windows PowerShell)

```powershell
cd D:\NEU
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The default `requirements.txt` installs CPU-compatible PyTorch. For NVIDIA
GPU training, install the CUDA build instead:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

### Dataset pipeline

The archive (`archive.zip`) is already extracted to `data\raw`. Re-extract or
re-run any step with:

```powershell
# 1. inspect the raw dataset (statistics + integrity report)
python training\inspect_dataset.py

# 2. convert Pascal VOC XML -> YOLO format + write data.yaml (validates every box)
python training\convert_voc_to_yolo.py

# or run both in one go:
python training\prepare_dataset.py

# 3. visual sanity check: draws YOLO labels over samples (VOC vs YOLO comparison)
python training\check_annotations.py     # -> outputs\annotation_checks

# 4. optional: offline augmentation demo (before/after samples)
python training\augment.py --count 12
```

## Training

**Pipeline smoke test (does not touch existing weights):**

```powershell
python training\train.py --epochs 1 --fraction 0.15 --no-copy
```

**Reproducing the final selected model:**

```powershell
python training\train.py --model yolov8n.pt --epochs 60 --batch 16 --imgsz 320 --workers 0 --patience 12 --name exp_imgsz320
```

The device (CUDA/CPU) is auto-detected and printed at startup. On Windows
with CUDA, use `--workers 0` (dataloader spawn + CUDA is unstable with
workers > 0 on this machine).

**Other training options:**

```powershell
python training\train.py --epochs 100 --batch 16 --imgsz 640
# optional: --device cuda  --model yolov8s.pt  --mosaic 0.0  --lr0 0.01
```

## Evaluation

```powershell
python training\evaluate.py          # mAP50, mAP50-95, precision, recall, F1, per-class AP, confusion matrix
python training\error_analysis.py    # FP / FN / low-confidence examples -> outputs\evaluation\errors
python training\threshold_analysis.py  # confidence operating-point table -> outputs\evaluation\final
```

Results: `outputs\evaluation\final\` (metrics.json, per_class_metrics.csv,
confusion matrix, PR/F1 curves, threshold + consistency + stability reports).
All metrics are **measured** by Ultralytics validation — nothing is invented.

### Measured results (final selected model)

Trained and evaluated on the NEU dataset (research demonstration — not production-certified).

| Metric | Result |
|---|---|
| mAP@0.50 | **0.725** |
| mAP@0.50:0.95 | **0.394** |
| Precision | 0.686 |
| Recall | 0.686 |
| F1 | 0.686 |
| Model | YOLOv8n @ imgsz 320 (60 epochs, batch 16, CUDA) |
| Model size | 6.2 MB (pt) / 11.6 MB (onnx) |
| Inference latency (median) | 16.2 ms ONNX Runtime CPU (61.6 FPS) · 12.8 ms CUDA FP32 · 28.0 ms PyTorch CPU |
| Selected confidence threshold | 0.35 (best F1; 0.25 for recall-priority QC) |

Experiment comparison: `outputs\evaluation\experiment_comparison.csv` —
native-resolution 320 px beat 640 px (the 200×200 source images lose detail
when upscaled 3×) and a yolov8s run trailed nano on this dataset size.

> **Dataset limitation:** NEU-DET contains only defect images — every image
> has at least one annotated defect and there are no normal/no-defect
> samples. PASS decisions ("no defect above threshold") should be validated
> further using normal production-surface samples before real industrial
> deployment.

## Inference

```powershell
# single image (JSON result + annotated output in outputs\detections)
python scripts\run_image.py data\yolo\images\val\scratches_1.jpg

# video file -> annotated mp4 + stats
python scripts\run_video.py path\to\video.mp4

# live webcam (or --source rtsp://... / file)
python scripts\run_camera.py
```

## Edge Deployment

The inference engine auto-selects the fastest *loadable* backend:
**TensorRT engine → ONNX Runtime → PyTorch** (validated by actually loading;
a corrupt model is logged and the next backend is used). Env override:
`INFERENCE_BACKEND=pytorch|onnx`.

### Edge benchmark (measured on this machine, imgsz 320, 100 iters)

| Backend | Device | Precision | Median | p95 | FPS |
|---|---|---|---|---|---|
| pytorch | cuda | FP32 | 12.8 ms | 27.9 ms | 78.3 |
| pytorch | cuda | FP16 | 13.5 ms | 19.1 ms | 74.1 |
| **onnxruntime** | **cpu** | **FP32** | **16.2 ms** | **17.5 ms** | **61.6** |
| pytorch | cpu | FP32 | 28.0 ms | 31.5 ms | 35.7 |

Full table: `outputs\evaluation\edge_benchmark.csv`. ONNX Runtime CPU is the
selected default on shared GPU machines: its p95 ≈ median (no stall spikes),
while CUDA here is a desktop-shared 4 GB GPU subject to WDDM paging stalls.
FP16 gave no meaningful speedup for the nano model.

### Headless / edge runtime

```powershell
# full edge mode: no GUI, PASS/REJECT + PLC simulation + metrics
python scripts\run_camera.py --source 0 --no-display

# every option
python scripts\run_camera.py --source 0 --backend auto --conf 0.35 --iou 0.45 `
    --process-every 2 --temporal-window 3 --min-defect-frames 2 `
    --plc-cooldown-ms 2000 --save --output outputs\detections\annotated.mp4 --max-frames 500
```

Real-time options (all env-configurable, see `.env.example`):
`PROCESS_EVERY_N_FRAMES` (skip frames on weak hardware),
`TEMPORAL_WINDOW`/`MIN_DEFECT_FRAMES` (N-of-M confirmation so single-frame
flicker cannot REJECT a part), `PLC_REJECT_COOLDOWN_MS` (debounce so one
persistent defect triggers one PLC event, with measured
`plc_decision_latency_ms`).

### ONNX validation & consistency

```powershell
python training\validate_onnx.py     # PT vs ONNX agreement (100% top-class match, IoU 1.0)
python training\benchmark_edge.py    # edge_benchmark.csv
python training\stability_test.py    # 320-frame leak/FPS stability run
```

TensorRT: optional. If `models\best.engine` exists and the TensorRT runtime
is installed it is used automatically; otherwise the system reports
"TensorRT unavailable — continuing with ONNX Runtime/PyTorch" (verified on
this machine — TensorRT is not installed).

## ONNX Export / Edge Optimisation

```powershell
python training\export_model.py                    # models\best.pt -> models\best.onnx
python training\export_model.py --format engine    # TensorRT (optional; prints a clear notice if unavailable)
```

The inference engine picks its backend automatically:
**TensorRT engine → ONNX Runtime → PyTorch**, based on what actually exists.
If no custom model is trained, the system refuses to run with generic COCO
weights and prints `CUSTOM DEFECT MODEL NOT TRAINED` plus the training
command.

## FastAPI + Dashboard

```powershell
uvicorn app.main:app --reload
```

### Dashboard (Industrial Quality Control Center)

Dark control-room UI with a sidebar app shell and live status header
(MODEL / CAMERA / BACKEND / DEVICE / FPS):

| Page | What it shows |
|---|---|
| **Overview** | Executive KPIs (total inspected, passed, rejected, defect rate, avg latency, FPS), defects-by-type, pass-vs-reject, defect + latency trends, recent events |
| **Live Inspection** | MJPEG camera feed with boxes/severity, PASS/REJECT status, full latency breakdown, camera controls, PLC simulator panel (clearly labelled SIMULATED) |
| **Image Inspection** | Drag & drop / browse / **Load Sample** (real NEU validation image per class), original vs detected panels, per-defect cards, download link |
| **Video Inspection** | Upload → process → summary metrics + annotated video |
| **Inspection History** | Filter by status / defect type / severity, paginated |
| **Analytics** | Six charts computed from real SQLite history only (empty state until you inspect) |
| **Model Performance** | Real mAP/precision/recall KPIs, model info, **measured edge benchmark** with recommended backend, per-class AP, confusion matrix, training curves |
| **Dataset Explorer** | NEU-DET stats + browse real samples per class |
| **System Health** | Services, camera state, CPU/RAM/GPU, runtime, uptime, /metrics + /docs links |
| **Settings** | Live server configuration, PLC stats, project info |

Every number on the dashboard comes from the API, SQLite history, evaluation
files, or live system state — no fake metrics; empty states are explicit.

| URL | Description |
|---|---|
| http://127.0.0.1:8000/ | Industrial dashboard (all sections) |
| http://127.0.0.1:8000/docs | Interactive OpenAPI docs |
| http://127.0.0.1:8000/health | Health check |
| http://127.0.0.1:8000/metrics | Prometheus metrics |
| http://127.0.0.1:8000/api/status, /api/classes, /api/metrics-summary | Status JSON |
| POST http://127.0.0.1:8000/api/detect/image | Image inspection (multipart upload) |
| POST http://127.0.0.1:8000/api/detect/video | Video inspection |
| POST http://127.0.0.1:8000/api/plc/test | PLC simulator command |
| GET http://127.0.0.1:8000/api/history | Detection history (filterable) |

## PLC Simulation

`SimulatedPLC` implements the `PLCAdapter` interface — a real Modbus TCP
skeleton (`ModbusTCPPLC`) is included in `app\api\plc.py`. No physical
hardware is required. Live REJECT decisions are temporally confirmed
(2-of-3 frames) and debounced (`PLC_REJECT_COOLDOWN_MS`).

```powershell
# manual command
curl -X POST http://127.0.0.1:8000/api/plc/test -H "Content-Type: application/json" `
     -d "{\"action\": \"REJECT\", \"defect\": \"scratches\", \"confidence\": 0.94}"
# -> {"action": "REJECT", "reject_signal": true, "defect": "scratches", ...}
```

## Prometheus / Grafana / Docker (deployed & verified)

```powershell
docker compose up -d
# app: 8000, prometheus: 9090, grafana: 3000 (admin/admin)
```

Verified end-to-end (2026-09-13): the containerised app served a real REJECT
inspection, Prometheus scraped it (target `defect-detection-app` UP, counter
observed via query), and Grafana's Prometheus datasource auto-provisioned.
The app's own dashboard also links `/metrics` and `/docs` from the System
Health page. Core functionality still never requires Docker — `uvicorn
app.main:app` alone runs everything except the monitoring containers.

## Project Documents

| File | Content |
|---|---|
| `PROJECT_TECHNICAL_REPORT.md` | Full technical report (dataset, training, experiments, results, edge, QC, testing, limitations) |
| `PRESENTATION.pptx` | 19-slide presentation deck (regenerate: `node scripts/generate_presentation.js`) |
| `FINAL_TEST_REPORT.md` | Final verification evidence, all checks executed |
| `PROJECT_STATUS.md` | Component-level COMPLETE / PARTIAL / NOT AVAILABLE summary |
| `PROJECT_DOCUMENTATION.md` | Architecture diagrams + concept explanations (viva prep) |

## Tests

```powershell
pytest
```

## Configuration

Copy `.env.example` to `.env` and adjust thresholds, model paths, camera
source, database path and log level. All paths resolve from the project root
via `pathlib` (see `app\config.py`).

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUSTOM DEFECT MODEL NOT TRAINED` | Run `python training\train.py --epochs 1` |
| `data\yolo\data.yaml not found` | Run `python training\prepare_dataset.py` |
| Camera fails to open | Try `--source 1`, close other apps using the webcam, or use an RTSP URL / video file |
| Slow training | It's CPU-bound PyTorch; reduce `--imgsz 320 --batch 8`, or install CUDA torch |
| Dashboard shows no analytics | Analytics only show **real** recorded inspections — run some inspections first |
| `Activate.ps1` blocked | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Port 8000 already in use | `uvicorn app.main:app --port 8001` (or change the mapping in `docker-compose.yml`) |
| Docker Desktop won't start after a crash | Restart Docker Desktop; worst case Troubleshoot → Reset to factory defaults, then `docker compose build` again |
| Temporal confirmation rejects too slowly | Lower `TEMPORAL_WINDOW` / `MIN_DEFECT_FRAMES` in `.env` (window=1, min=1 disables it) |

## Severity Logic Notice

The LOW/MEDIUM/HIGH/CRITICAL severity grading is **configurable
demonstration/business logic** (class weight + confidence + defect size +
defect count) — **not** an industry-certified severity standard.

## Future Improvements

- Per-class threshold tuning from the error-analysis results
- TensorRT INT8 calibration with a representative dataset
- MQTT adapter for the PLC interface
- Model retraining trigger from dashboard drift metrics
- Multi-camera line support with frame-queue load balancing
