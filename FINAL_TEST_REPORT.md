# FINAL TEST REPORT — Real-Time Industrial Defect Detection System

**Date:** 2026-09-12
**Environment:** Windows 11 (10.0.26200) · Python 3.12.6 · PyTorch 2.5.1+cu121 · Ultralytics 8.4.142 · ONNX Runtime 1.29 (CPU EP) · CUDA 12.1 runtime on NVIDIA GTX 1650 (4 GB, driver 528.49)
**Test location:** all commands executed from `D:\NEU`

---

## 1. Dataset Checks — PASS

| Check | Result |
|---|---|
| Original archive preserved | ✅ `archive.zip` (27.7 MB, unmodified since 2026-09-05) |
| Raw extraction intact | ✅ `data/raw/NEU-DET` — 1,440 train + 360 val images, 1,439 + 361 XMLs |
| `training/inspect_dataset.py` | ✅ 1,800 images · 4,189 boxes · 6 classes · known dataset quirks reported (crazing_240 XML/image gap — inherent to the published dataset) |
| `training/validate_labels.py` | ✅ all class ids 0–5, coords in [0,1], positive sizes, complete pairs; imbalance 2.47× (moderate) |
| `training/check_annotations.py` | ✅ every YOLO box matches its VOC source at IoU ≥ 0.85 (no shift/distortion); visual comparisons in `outputs/annotation_checks/` |
| `data/yolo/data.yaml` | ✅ correct paths (POSIX), 6 classes in fixed order |

## 2. Model Checks — PASS

| Check | Result |
|---|---|
| `models/best.pt` loads | ✅ 6 custom NEU classes, no COCO labels |
| Fresh evaluation vs metadata | ✅ exact match — mAP50 **0.7247**, mAP50-95 **0.3939**, P **0.6856**, R **0.6861** (re-measured 2026-09-12) |
| Per-class AP | ✅ patches 0.923 / scratches 0.837 / inclusion 0.825 / pitted_surface 0.792 / crazing 0.487 / rolled-in_scale 0.486 (AP50) |
| `models/best_metadata.json` | ✅ architecture, imgsz 320, epochs 60/60, thresholds 0.35/0.45, device, source experiment |
| `models/best.onnx` | ✅ loads in ONNX Runtime; PT-vs-ONNX consistency 100% top-class agreement, median box IoU 1.0, Δconf 0.0 (`onnx_consistency.json`) |
| TensorRT | ❌ NOT AVAILABLE (not installed) — no `.engine` artifact present, optional path reports cleanly |

## 3. Inference Tests — PASS

| Test | Result |
|---|---|
| PyTorch image inference (multiple classes) | ✅ detections, boxes, confidences, PASS/REJECT, saved outputs |
| ONNX image inference | ✅ consistent with PyTorch (§ onnx_consistency) |
| Video pipeline (`scripts/run_video.py`) | ✅ 12 frames @ 27.9 FPS measured, annotated MP4 written, resources released |
| Webcam (`scripts/run_camera.py --source 0`) | ✅ verified via API live stream + headless run: 15 frames, 13.7 ms avg inference |
| Headless mode (`--no-display`) | ✅ no GUI dependency, metrics/history/logging active, clean exit |
| Bad camera source | ✅ clear error, no hang, system remains usable |
| RTSP | ⚠ accepted as `--source rtsp://…` by the same code path; not tested against a live RTSP source (none available) |

## 4. Edge Benchmarks — PASS (real measurements, `outputs/evaluation/edge_benchmark.csv`)

| Backend | Device | Precision | Median | p95 | FPS |
|---|---|---|---|---|---|
| pytorch | cuda | FP32 | 12.8 ms | 27.9 ms | 78.3 |
| pytorch | cuda | FP16 | 13.5 ms | 19.1 ms | 74.1 |
| **onnxruntime** | **cpu** | **FP32** | **16.2 ms** | **17.5 ms** | **61.6** |
| pytorch | cpu | FP32 | 28.0 ms | 31.5 ms | 35.7 |
| onnxruntime | cuda | FP32 | — unavailable (CPU-only build) | | |
| tensorrt | cuda | FP16 | — unavailable (not installed) | | |

Stability: 320-frame pipeline test → **STABLE** (52 FPS, +2.6 MB RSS).

## 5. Backend Fallback — PASS

- `INFERENCE_BACKEND=auto` → ONNX Runtime selected (load-validated, warmed up)
- Corrupt ONNX → falls back to PyTorch with logged warning (unit-tested)
- All model files missing → clean `CUSTOM DEFECT MODEL NOT TRAINED`; API starts in `degraded` status with training hint; **no COCO fallback ever**

## 6. API Tests — PASS (cold start from clean terminal, port 8000/8001)

| Endpoint | Status |
|---|---|
| GET `/`, `/health`, `/docs`, `/metrics` | 200 |
| GET `/api/status`, `/api/classes`, `/api/metrics-summary`, `/api/history`, `/api/model/info`, `/api/system/info`, `/api/analytics/summary` | 200 |
| POST `/api/detect/image` | 200 (real REJECT with boxes) |
| POST `/api/detect/video` | 200 (verified in Phase 4) |
| POST `/api/plc/test` | 200 (REJECT → `reject_signal: true`) |
| Invalid inputs | missing upload 422 · text file 400 · corrupt image 400 · bad video type 400 · unknown PLC class 422 · unknown path 404 — no stack traces exposed |
| Prometheus `/metrics` | ✅ valid text; 13 metric families present incl. `plc_events_total`, `plc_decision_latency_ms`, `camera_frames_processed_total` |

## 7. UI Tests — PASS (real browser walkthrough, 1600×900)

All 10 pages verified: Overview (real KPIs/charts/tables) · Live Inspection (camera started from UI, live boxes, REJECT, PLC panel with 0.6 ms decision latency) · Image Inspection (Load Sample → REJECT with boxes + severity + download link; error state verified) · Video · History (severity filter + pagination) · Analytics (6 real charts) · Model Performance (real metrics + benchmark) · Dataset Explorer (real NEU images) · System Health (services/resources/runtime) · Settings. Empty/error states verified. Bug found & fixed during testing: dataset image URL prefix (`/dataset/images/val/…` → `/dataset/val/…`).

## 8. Quality Control — PASS

- PASS logic (no detection ≥ 0.35) ✅ · REJECT (1 or N defects) ✅ unit + live tested
- Temporal confirmation (2-of-3) ✅ unit-tested (noise rejection + persistence + slide + disable)
- PLC debounce ✅ unit-tested (first allowed, suppressed within cooldown) · live-verified (12 defect frames → 1 command, 0.6 ms decision latency)
- SQLite ✅ init with/without DB file, inserts, reads, filters, clear (unit-tested on temp DB; real history preserved)

## 9. Static Checks — PASS

`python -m compileall app training scripts tests` ✅ · core imports ✅ · no TODO/FIXME/placeholder implementations · no fake metrics in code · no console.log spam · rotating logs under `logs/` with correct paths · pytest **53/53 passed** (0 failed, 0 skipped)

## 10. Docker / Grafana — PASS (deployed and verified 2026-09-13)

Docker Desktop installed (v29.7.2, WSL2 backend). Executed:

- `docker compose build` → `neu-app` image (CPU-only torch inside container)
- `docker compose up -d` → all three services running:
  - **app**: `defect-app` on :8000 — `/health` ok, ONNX model loaded, real REJECT detection served (4 defects on patches_242)
  - **prometheus**: `defect-prometheus` on :9090 — target `defect-detection-app` **UP**; query verified `defect_detection_requests_total = 1` after a live detection
  - **grafana**: `defect-grafana` on :3000 (v13.2.1, db ok) — provisioned **Prometheus datasource → http://prometheus:9090** loaded as default; login admin/admin

Operational notes: `.dockerignore` added (small build context); Dockerfile installs CPU-only torch; if host port 8000 is occupied, change the mapping in `docker-compose.yml`. Original caveat resolved — Docker deployment is now tested, not just written.

## 11. Known Limitations (honest)

1. NEU-DET contains **no normal/no-defect images** — PASS decisions (absence of detection) are unvalidated against pristine surfaces.
2. Model trained/evaluated only on NEU; performance on unseen factories/lighting is not guaranteed. Research demonstration, not production-certified.
3. TensorRT and ONNX-CUDA-EP not installed in this environment (optional by design, clean fallback verified).
4. PLC is a simulator; Modbus adapter is a skeleton.
5. RTSP path implemented but not exercised against real hardware.
6. Docker/Grafana stack not executed (Docker unavailable).

**FINAL READINESS: READY FOR DEMONSTRATION AND SUBMISSION**
