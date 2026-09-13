/* Presentation generator — Real-Time Industrial Defect Detection System
 * Run: NODE_PATH="$(npm root -g)" node scripts/generate_presentation.js
 * Output: D:\NEU\PRESENTATION.pptx
 */
const path = require("path");
const pptxgen = require(path.join(process.env.NPM_GLOBAL_ROOT || "C:/Users/Vinay Kumkar/AppData/Roaming/npm/node_modules", "pptxgenjs"));

const p = new pptxgen();
p.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
p.author = "Real-Time Industrial Defect Detection System";
p.title = "Real-Time Industrial Defect Detection System";

// palette: steel + safety orange (industrial)
const BG_DARK = "101820", BG = "FFFFFF", CARD = "EAF1F6", CARD2 = "F4F8FB";
const PRIMARY = "1D5F8A", PRIMARY_D = "14486B", ACCENT = "E8590C";
const TEXT = "1E293B", MUTED = "5B6B7C", LIGHT = "C9D8E4";
const F = "Segoe UI";
const W = 13.33, H = 7.5, M = 0.55;

const bu = () => ({ code: "25B8", indent: 12, color: ACCENT });
const shadow = () => ({ type: "outer", color: "1E293B", blur: 7, offset: 2, angle: 45, opacity: 0.18 });

function titleBar(s, kicker, title, dark = false) {
  s.addText(kicker.toUpperCase(), { x: M, y: 0.32, w: W - 2 * M, h: 0.3, fontSize: 11, fontFace: F,
    color: ACCENT, bold: true, charSpacing: 3, margin: 0 });
  s.addText(title, { x: M, y: 0.58, w: W - 2 * M, h: 0.75, fontSize: 30, fontFace: F,
    color: dark ? "FFFFFF" : TEXT, bold: true, margin: 0 });
}
function pageNum(s, n) {
  s.addText(String(n).padStart(2, "0"), { x: W - 0.85, y: H - 0.5, w: 0.5, h: 0.3, fontSize: 10,
    color: MUTED, fontFace: F, align: "right", margin: 0 });
}
function card(s, x, y, w, h, fill = CARD) {
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, rectRadius: 0.07, shadow: shadow() });
}
function kpi(s, x, y, w, big, label, sub, color = PRIMARY) {
  card(s, x, y, w, 1.62, CARD2);
  s.addText(big, { x: x + 0.18, y: y + 0.14, w: w - 0.36, h: 0.85, fontSize: 40, bold: true, color, fontFace: F, margin: 0 });
  s.addText(label.toUpperCase(), { x: x + 0.18, y: y + 0.98, w: w - 0.36, h: 0.28, fontSize: 10.5, bold: true, color: TEXT, charSpacing: 2, fontFace: F, margin: 0 });
  s.addText(sub, { x: x + 0.18, y: y + 1.26, w: w - 0.36, h: 0.26, fontSize: 9.5, color: MUTED, fontFace: F, margin: 0 });
}
function srcNote(s, txt) {
  s.addText(txt, { x: M, y: H - 0.48, w: W - 1.4, h: 0.3, fontSize: 10.5, color: MUTED, fontFace: F, margin: 0 });
}
function bullets(s, items, x, y, w, h, size = 15) {
  s.addText(items.map((t, i) => ({ text: t, options: { bullet: bu(), breakLine: i < items.length - 1 } })),
    { x, y, w, h, fontSize: size, color: TEXT, fontFace: F, paraSpaceAfter: 10, margin: 0, valign: "top" });
}

/* ============ 1. TITLE (dark) ============ */
let s = p.addSlide();
s.background = { color: BG_DARK };
s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: M, y: 1.05, w: 0.9, h: 0.9, fill: { color: ACCENT }, rectRadius: 0.09 });
s.addText("QC", { x: M, y: 1.05, w: 0.9, h: 0.9, fontSize: 26, bold: true, color: "FFFFFF", align: "center", valign: "middle", fontFace: F });
s.addText("REAL-TIME INDUSTRIAL DEFECT\nDETECTION SYSTEM", { x: M, y: 2.25, w: W - 2 * M, h: 1.7, fontSize: 44, bold: true, color: "FFFFFF", fontFace: F, margin: 0 });
s.addText("AI-Powered Manufacturing Quality Control · NEU Surface Defects · YOLOv8", { x: M, y: 4.0, w: W - 2 * M, h: 0.4, fontSize: 16, color: LIGHT, fontFace: F, margin: 0 });
[["72.5%", "mAP@0.50"], ["0.394", "mAP@0.50:0.95"], ["61.6 FPS", "ONNX CPU edge"], ["6", "defect classes"]].forEach((d, i) => {
  const x = M + i * 3.1;
  s.addText(d[0], { x, y: 5.1, w: 2.9, h: 0.7, fontSize: 34, bold: true, color: ACCENT, fontFace: F, margin: 0 });
  s.addText(d[1].toUpperCase(), { x, y: 5.8, w: 2.9, h: 0.3, fontSize: 11, color: LIGHT, charSpacing: 2, fontFace: F, margin: 0 });
});
s.addText("All figures measured on the delivered system — NEU-DET validation set, GTX 1650 / CPU", { x: M, y: 6.6, w: W - 2 * M, h: 0.3, fontSize: 11, italic: true, color: MUTED, fontFace: F, margin: 0 });

/* ============ 2. AGENDA ============ */
s = p.addSlide(); s.background = { color: BG };
titleBar(s, "Agenda", "What this presentation covers");
const agenda = [
  ["Problem & impact", "why automated surface inspection"],
  ["Dataset", "NEU-DET: 1,800 images, 6 defect classes"],
  ["Data pipeline", "VOC→YOLO conversion, validation, augmentation"],
  ["Model & experiments", "YOLOv8n training, 4 measured experiments"],
  ["Results", "mAP 72.5 / 39.4, per-class AP, error analysis"],
  ["Edge optimisation", "ONNX Runtime, 62 FPS on CPU, backend fallback"],
  ["Quality control", "PASS/REJECT, temporal confirmation, PLC simulation"],
  ["System & monitoring", "FastAPI, dashboard, SQLite, Prometheus, Grafana, Docker"],
  ["Testing, limitations, demo", "53 tests, honest constraints, live demo flow"],
];
agenda.forEach((a, i) => {
  const col = i % 3, row = Math.floor(i / 3);
  const x = M + col * 4.18, y = 1.75 + row * 1.72;
  card(s, x, y, 3.95, 1.5, CARD2);
  s.addText(String(i + 1).padStart(2, "0"), { x: x + 0.15, y: y + 0.15, w: 0.8, h: 0.6, fontSize: 26, bold: true, color: ACCENT, fontFace: F, margin: 0 });
  s.addText(a[0], { x: x + 0.15, y: y + 0.7, w: 3.6, h: 0.35, fontSize: 15, bold: true, color: PRIMARY_D, fontFace: F, margin: 0 });
  s.addText(a[1], { x: x + 0.15, y: y + 1.05, w: 3.6, h: 0.35, fontSize: 10.5, color: MUTED, fontFace: F, margin: 0 });
});
pageNum(s, 2);

/* ============ 3. PROBLEM ============ */
s = p.addSlide(); s.background = { color: BG };
titleBar(s, "Problem statement", "Manual steel inspection is slow, inconsistent and costly");
bullets(s, [
  "Human visual inspection of rolled steel is fatiguing and subjective — quality varies across shifts and inspectors",
  "Escaped defects become manufacturing waste or reach end customers: rework, returns, reputation damage",
  "Rule-based machine vision (fixed thresholds) cannot handle the visual variety of the six NEU defect classes",
  "Deep learning object detection learns defect appearance directly from labelled images and runs in milliseconds",
], M, 1.7, 7.0, 4.2, 16);
card(s, 8.0, 1.7, 4.8, 4.6, CARD);
s.addText("PROJECT GOAL", { x: 8.3, y: 2.0, w: 4.2, h: 0.3, fontSize: 11, bold: true, color: ACCENT, charSpacing: 2, fontFace: F, margin: 0 });
s.addText("Minimise manufacturing waste and prevent defective products from reaching end consumers — with consistent, unbiased, high-throughput quality control at the edge.", { x: 8.3, y: 2.4, w: 4.2, h: 2.2, fontSize: 16, color: TEXT, fontFace: F, margin: 0 });
s.addText("No cloud. No paid services. Runs on a laptop-class machine next to the production line.", { x: 8.3, y: 4.8, w: 4.2, h: 1.0, fontSize: 12.5, italic: true, color: MUTED, fontFace: F, margin: 0 });
pageNum(s, 3);

/* ============ 4. SYSTEM PIPELINE ============ */
s = p.addSlide(); s.background = { color: BG };
titleBar(s, "System overview", "From camera frame to sorting decision — in milliseconds");
const steps = [
  ["1", "Capture", "webcam / video / RTSP\n(OpenCV)"],
  ["2", "Preprocess", "letterbox 320×320\nRGB, /255"],
  ["3", "Infer", "TensorRT ▸ ONNX ▸\nPyTorch (auto, load-validated)"],
  ["4", "Detect", "6 classes · boxes ·\nconfidence"],
  ["5", "Decide", "temporal 2-of-3 →\nPASS / REJECT"],
  ["6", "Act & record", "PLC signal · SQLite ·\nmetrics · dashboard"],
];
steps.forEach((st, i) => {
  const x = M + i * 2.12;
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x, y: 2.0, w: 1.92, h: 1.85, fill: { color: i === 4 ? ACCENT : PRIMARY }, rectRadius: 0.08 });
  s.addText(st[0], { x: x + 0.12, y: 2.1, w: 1.0, h: 0.5, fontSize: 22, bold: true, color: i === 4 ? "FFFFFF" : LIGHT, fontFace: F, margin: 0 });
  s.addText(st[1], { x: x + 0.12, y: 2.62, w: 1.7, h: 0.35, fontSize: 15, bold: true, color: "FFFFFF", fontFace: F, margin: 0 });
  s.addText(st[2], { x: x + 0.12, y: 2.98, w: 1.72, h: 0.75, fontSize: 9.5, color: "E4EDF4", fontFace: F, margin: 0 });
  if (i < 5) s.addText("▸", { x: x + 1.9, y: 2.7, w: 0.25, h: 0.4, fontSize: 18, color: MUTED, fontFace: F, margin: 0 });
});
card(s, M, 4.35, 12.23, 2.0, CARD2);
s.addText("PARALLEL MONITORING CHAIN", { x: M + 0.25, y: 4.55, w: 6, h: 0.3, fontSize: 11, bold: true, color: ACCENT, charSpacing: 2, fontFace: F, margin: 0 });
[["FastAPI", "model loaded once,\nshared by all requests"], ["Dashboard", "10-page control-room UI\n(live MJPEG feed)"], ["SQLite", "every inspection recorded,\nfilterable history"], ["Prometheus", "13 metric families\nscraped every 5 s"], ["Grafana", "auto-provisioned\ndatasource + dashboard"]].forEach((b, i) => {
  const x = M + 0.25 + i * 2.42;
  s.addText(b[0], { x, y: 4.95, w: 2.2, h: 0.32, fontSize: 14, bold: true, color: PRIMARY_D, fontFace: F, margin: 0 });
  s.addText(b[1], { x, y: 5.3, w: 2.25, h: 0.8, fontSize: 10, color: MUTED, fontFace: F, margin: 0 });
  if (i < 4) s.addText("▸", { x: x + 2.14, y: 4.95, w: 0.25, h: 0.35, fontSize: 14, color: MUTED, fontFace: F, margin: 0 });
});
srcNote(s, "Measured live snapshot: 25.9 ms total pipeline latency (13.7 ms inference) on ONNX Runtime CPU");
pageNum(s, 4);

/* ============ 5. DATASET ============ */
s = p.addSlide(); s.background = { color: BG };
titleBar(s, "Dataset", "NEU Surface Defect Database (NEU-DET)");
kpi(s, M, 1.7, 2.9, "1,800", "images", "grayscale 200×200 px");
kpi(s, M + 3.05, 1.7, 2.9, "4,189", "bounding boxes", "Pascal VOC XML");
kpi(s, M + 6.1, 1.7, 2.9, "1,440 / 360", "train / val", "built-in split");
kpi(s, M + 9.15, 1.7, 2.93, "6", "defect classes", "balanced per image count");
const rows = [
  [{ text: "Class", options: { bold: true, color: "FFFFFF", fill: { color: PRIMARY } } },
   { text: "Train objects", options: { bold: true, color: "FFFFFF", fill: { color: PRIMARY } } },
   { text: "Val objects", options: { bold: true, color: "FFFFFF", fill: { color: PRIMARY } } },
   { text: "Appearance", options: { bold: true, color: "FFFFFF", fill: { color: PRIMARY } } }],
  ["crazing", "524", "162", "networks of fine cracks"],
  ["inclusion", "852", "159", "embedded foreign material"],
  ["patches", "688", "193", "patch-like surface damage"],
  ["pitted_surface", "345", "87", "small pits on the surface"],
  ["rolled-in_scale", "496", "132", "oxide scale rolled into surface"],
  ["scratches", "427", "121", "linear scratches"],
];
s.addTable(rows, { x: M, y: 3.65, w: 8.1, fontSize: 12, fontFace: F, color: TEXT,
  border: { pt: 0.5, color: "C9D8E4" }, fill: { color: "FFFFFF" }, rowH: 0.38, valign: "middle",
  colW: [2.1, 1.7, 1.5, 2.8] });
card(s, 9.0, 3.65, 3.83, 2.9, CARD);
s.addText("KEY FACTS", { x: 9.25, y: 3.85, w: 3.3, h: 0.3, fontSize: 11, bold: true, color: ACCENT, charSpacing: 2, fontFace: F, margin: 0 });
bullets(s, [
  "1,033 / 1,439 train images have multiple defects",
  "2.47× class imbalance (inclusion vs pitted_surface)",
  "No normal (defect-free) images exist in NEU-DET",
  "Archive preserved untouched; XMLs never modified",
], 9.25, 4.25, 3.4, 2.2, 11.5);
srcNote(s, "Source: NEU Surface Defect Database (NEU-DET), verified by our inspection scripts");
pageNum(s, 5);

/* ============ 6. DATA PIPELINE ============ */
s = p.addSlide(); s.background = { color: BG };
titleBar(s, "Data pipeline", "Verified conversion — no silent data loss");
const pipe = [
  ["Extract", "archive preserved untouched;\nraw data read-only", "1,800 images"],
  ["Convert VOC→YOLO", "normalised class cx cy w h;\ndegenerate boxes rejected", "4,186 boxes kept"],
  ["Validate", "ids 0–5, coords in [0,1],\ncomplete image-label pairs", "100% valid"],
  ["Visual check", "YOLO boxes re-drawn and\ncompared to source VOC", "IoU ≥ 0.85 every box"],
  ["Augment", "online: mosaic, flips, ≤5° rotate;\noffline: CLAHE/gamma/noise", "bbox-safe only"],
];
pipe.forEach((st, i) => {
  const x = M + i * 2.48;
  card(s, x, 1.85, 2.28, 2.5, CARD2);
  s.addText(String(i + 1), { x: x + 0.15, y: 2.0, w: 0.7, h: 0.5, fontSize: 22, bold: true, color: ACCENT, fontFace: F, margin: 0 });
  s.addText(st[0], { x: x + 0.15, y: 2.52, w: 2.0, h: 0.55, fontSize: 14.5, bold: true, color: PRIMARY_D, fontFace: F, margin: 0 });
  s.addText(st[1], { x: x + 0.15, y: 3.08, w: 1.98, h: 0.85, fontSize: 9.5, color: MUTED, fontFace: F, margin: 0 });
  s.addText(st[2], { x: x + 0.15, y: 3.92, w: 1.98, h: 0.3, fontSize: 10.5, bold: true, color: TEXT, fontFace: F, margin: 0 });
});
card(s, M, 4.75, 12.23, 1.75, CARD);
s.addText("WHY THIS MATTERS", { x: M + 0.25, y: 4.95, w: 5, h: 0.3, fontSize: 11, bold: true, color: ACCENT, charSpacing: 2, fontFace: F, margin: 0 });
s.addText("A single mis-converted bounding box silently teaches the model wrong locations. Every one of the 4,186 labels was machine-validated and 120 samples were visually compared against the original Pascal-VOC annotations — zero shift or distortion detected.", { x: M + 0.25, y: 5.3, w: 11.7, h: 1.0, fontSize: 13.5, color: TEXT, fontFace: F, margin: 0 });
pageNum(s, 6);

/* ============ 7. TRAINING SETUP ============ */
s = p.addSlide(); s.background = { color: BG };
titleBar(s, "Model & training", "YOLOv8n — chosen for the edge, tuned for NEU");
s.addTable([
  [{ text: "Setting", options: { bold: true, color: "FFFFFF", fill: { color: PRIMARY } } },
   { text: "Value", options: { bold: true, color: "FFFFFF", fill: { color: PRIMARY } } }],
  ["Architecture", "YOLOv8n (3.0 M params, 8.1 GFLOPs) — COCO-pretrained transfer"],
  ["Image size", "320 px  (native source is 200×200 — 640 blurrs fine texture)"],
  ["Epochs / batch", "60 / 16, early-stopping patience 12"],
  ["Augmentation", "mosaic, h-flip 0.5, ≤5° rotation, HSV-V 0.2"],
  ["Optimizer", "AdamW, lr0 = 0.01 (auto strategy)"],
  ["Hardware", "GTX 1650 4 GB (CUDA 12.1), Windows-safe workers = 0"],
], { x: M, y: 1.75, w: 7.3, fontSize: 12.5, fontFace: F, color: TEXT, border: { pt: 0.5, color: "C9D8E4" },
  fill: { color: "FFFFFF" }, rowH: 0.52, valign: "middle", colW: [2.2, 5.1] });
card(s, 8.2, 1.75, 4.6, 3.4, CARD);
s.addText("WHY 320 px? (measured)", { x: 8.45, y: 2.0, w: 4.1, h: 0.3, fontSize: 11, bold: true, color: ACCENT, charSpacing: 2, fontFace: F, margin: 0 });
s.addText("640 px", { x: 8.45, y: 2.4, w: 1.9, h: 0.75, fontSize: 30, bold: true, color: MUTED, fontFace: F, margin: 0 });
s.addText("mAP50 0.671", { x: 8.45, y: 3.12, w: 2.2, h: 0.3, fontSize: 12, color: MUTED, fontFace: F, margin: 0 });
s.addText("320 px", { x: 10.6, y: 2.4, w: 1.9, h: 0.75, fontSize: 30, bold: true, color: ACCENT, fontFace: F, margin: 0 });
s.addText("mAP50 0.725", { x: 10.6, y: 3.12, w: 2.2, h: 0.3, fontSize: 12, bold: true, color: TEXT, fontFace: F, margin: 0 });
s.addText("Upscaling 200 px source to 640 blurs defect texture and costs 4× compute — the smaller native-scale input trains faster AND scores higher.", { x: 8.45, y: 3.55, w: 4.1, h: 1.4, fontSize: 12, color: TEXT, fontFace: F, margin: 0 });
card(s, 8.2, 5.35, 4.6, 1.15, CARD2);
s.addText("Windows/CUDA pitfalls solved: torch cu126 vs driver 528.49 → cu121; dataloader spawn + CUDA crash → workers = 0; VRAM thrash at batch 16 → batch 8 (640) / 16 (320).", { x: 8.45, y: 5.5, w: 4.1, h: 0.9, fontSize: 10, color: MUTED, fontFace: F, margin: 0 });
pageNum(s, 7);

/* ============ 8. EXPERIMENTS ============ */
s = p.addSlide(); s.background = { color: BG };
titleBar(s, "Experiments", "Four measured runs — the decision was evidence-driven");
s.addChart(p.charts.BAR, [{
  name: "mAP@0.50:0.95",
  labels: ["baseline 640px", "320px (final)", "no-mosaic 320px", "yolov8s 320px"],
  values: [0.329, 0.394, 0.360, 0.316],
}], {
  x: M, y: 1.8, w: 7.2, h: 4.6, barDir: "col",
  chartColors: ["9FB3C8", ACCENT, "9FB3C8", "9FB3C8"],
  chartArea: { fill: { color: "FFFFFF" } },
  catAxisLabelColor: MUTED, valAxisLabelColor: MUTED, catAxisLabelFontSize: 10, valAxisLabelFontSize: 10,
  valGridLine: { color: "E2E8F0", size: 0.5 }, catGridLine: { style: "none" },
  showValue: true, dataLabelPosition: "outEnd", dataLabelColor: TEXT, dataLabelFontSize: 11,
  showLegend: false, valAxisMaxVal: 0.45, valAxisMinVal: 0,
});
const exp = [
  ["baseline 640 px", "mAP50 0.671 · P 0.600 · R 0.661 — converged, VRAM thrash"],
  ["320 px (final)", "mAP50 0.725 · P 0.686 · R 0.686 — best on every key metric"],
  ["mosaic off", "mAP50 0.705 — augmentation tuning rejected mosaic removal"],
  ["yolov8s", "mAP50 0.674, 3.4× larger — rejected (small dataset favours nano)"],
];
card(s, 8.05, 1.8, 4.78, 4.6, CARD2);
s.addText("DECISION LOG", { x: 8.3, y: 2.0, w: 4.2, h: 0.3, fontSize: 11, bold: true, color: ACCENT, charSpacing: 2, fontFace: F, margin: 0 });
exp.forEach((e, i) => {
  s.addText(e[0], { x: 8.3, y: 2.42 + i * 1.0, w: 4.3, h: 0.3, fontSize: 13.5, bold: true, color: i === 1 ? ACCENT : PRIMARY_D, fontFace: F, margin: 0 });
  s.addText(e[1], { x: 8.3, y: 2.72 + i * 1.0, w: 4.3, h: 0.6, fontSize: 10.5, color: MUTED, fontFace: F, margin: 0 });
});
srcNote(s, "Source: outputs/evaluation/experiment_comparison.csv — every run logged with config + metrics");
pageNum(s, 8);

/* ============ 9. FINAL RESULTS ============ */
s = p.addSlide(); s.background = { color: BG };
titleBar(s, "Final results", "YOLOv8n @ 320 px — 60 epochs — evaluated fresh on the validation set");
kpi(s, M, 1.7, 2.95, "72.5%", "mAP@0.50", "val 360 images / 854 boxes");
kpi(s, M + 3.1, 1.7, 2.95, "39.4%", "mAP@0.50:0.95", "strict IoU average");
kpi(s, M + 6.2, 1.7, 2.95, "68.6%", "precision", "equal to recall — balanced");
kpi(s, M + 9.3, 1.7, 2.9, "68.6%", "recall", "F1 = 0.686");
s.addChart(p.charts.BAR, [{
  name: "AP@0.50",
  labels: ["patches", "scratches", "inclusion", "pitted surf.", "crazing", "rolled-in scale"],
  values: [0.923, 0.837, 0.825, 0.792, 0.487, 0.486],
}], {
  x: M, y: 3.62, w: 7.4, h: 3.3, barDir: "bar",
  chartColors: [PRIMARY, PRIMARY, PRIMARY, PRIMARY, ACCENT, ACCENT],
  chartArea: { fill: { color: "FFFFFF" } },
  catAxisLabelColor: MUTED, valAxisLabelColor: MUTED, catAxisLabelFontSize: 10, valAxisLabelFontSize: 10,
  valGridLine: { color: "E2E8F0", size: 0.5 }, catGridLine: { style: "none" },
  showValue: true, dataLabelPosition: "outEnd", dataLabelColor: TEXT, dataLabelFontSize: 10.5,
  showLegend: false, valAxisMaxVal: 1.0, valAxisMinVal: 0,
});
card(s, 8.25, 3.62, 4.58, 3.3, CARD);
s.addText("READING THE RESULTS", { x: 8.5, y: 3.82, w: 4.1, h: 0.3, fontSize: 11, bold: true, color: ACCENT, charSpacing: 2, fontFace: F, margin: 0 });
bullets(s, [
  "Patches, scratches, inclusion all exceed 0.82 AP50 — production-ready on those classes",
  "Crazing & rolled-in scale are the weak pair (fine, low-contrast textures) — 0.49 AP50",
  "Confusion concentrates inside the crazing / pitted / scale texture family",
], 8.5, 4.2, 4.15, 2.6, 11.5);
srcNote(s, "Source: outputs/evaluation/final/metrics.json + per_class_metrics.csv — measured, not estimated");
pageNum(s, 9);

/* ============ 10. ERROR ANALYSIS & THRESHOLD ============ */
s = p.addSlide(); s.background = { color: BG };
titleBar(s, "Error analysis & threshold tuning", "Choosing the operating point for quality control");
card(s, M, 1.75, 5.9, 2.5, CARD2);
s.addText("ERROR PROFILE (360 val images, conf 0.35)", { x: M + 0.25, y: 1.95, w: 5.4, h: 0.3, fontSize: 11, bold: true, color: ACCENT, charSpacing: 1.5, fontFace: F, margin: 0 });
[["561", "true positives", PRIMARY], ["258", "false positives", MUTED], ["293", "false negatives", ACCENT]].forEach((d, i) => {
  const x = M + 0.25 + i * 1.9;
  s.addText(d[0], { x, y: 2.35, w: 1.7, h: 0.7, fontSize: 34, bold: true, color: d[2], fontFace: F, margin: 0 });
  s.addText(d[1].toUpperCase(), { x, y: 3.05, w: 1.8, h: 0.55, fontSize: 9.5, color: MUTED, charSpacing: 1, fontFace: F, margin: 0 });
});
s.addText("Most missed: crazing (95) · rolled-in scale (65).  Most false alarms: scratches (68), crazing (57).", { x: M + 0.25, y: 3.62, w: 5.4, h: 0.55, fontSize: 11, color: MUTED, fontFace: F, margin: 0 });
s.addTable([
  [{ text: "Confidence", options: { bold: true, color: "FFFFFF", fill: { color: PRIMARY } } },
   { text: "Precision", options: { bold: true, color: "FFFFFF", fill: { color: PRIMARY } } },
   { text: "Recall", options: { bold: true, color: "FFFFFF", fill: { color: PRIMARY } } },
   { text: "F1", options: { bold: true, color: "FFFFFF", fill: { color: PRIMARY } } },
   { text: "Use case", options: { bold: true, color: "FFFFFF", fill: { color: PRIMARY } } }],
  ["0.25", "0.683", "0.685", "0.684", "recall-priority QC"],
  [{ text: "0.35  ◄ selected", options: { bold: true, color: ACCENT, fill: { color: CARD } } },
   { text: "0.754", options: { bold: true, fill: { color: CARD } } },
   { text: "0.639", options: { fill: { color: CARD } } },
   { text: "0.692", options: { bold: true, fill: { color: CARD } } },
   { text: "best F1, balanced", options: { fill: { color: CARD } } }],
  ["0.50", "0.843", "0.539", "0.657", "precision-priority"],
], { x: 6.85, y: 1.75, w: 5.95, fontSize: 12, fontFace: F, color: TEXT, border: { pt: 0.5, color: "C9D8E4" },
  fill: { color: "FFFFFF" }, rowH: 0.45, valign: "middle", colW: [1.35, 1.05, 0.95, 0.8, 1.8] });
card(s, M, 4.65, 12.23, 1.85, CARD);
s.addText("QUALITY-CONTROL LOGIC", { x: M + 0.25, y: 4.85, w: 5, h: 0.3, fontSize: 11, bold: true, color: ACCENT, charSpacing: 2, fontFace: F, margin: 0 });
s.addText([
  { text: "Missing a real defect costs more than a false alarm — so the default (0.35) does not maximise precision. ", options: {} },
  { text: "Temporal confirmation (2 of last 3 frames) ", options: { bold: true } },
  { text: "prevents one noisy frame from rejecting a part, and a ", options: {} },
  { text: "2 s PLC debounce", options: { bold: true } },
  { text: " turns 12 consecutive defect frames into exactly 1 reject command (measured decision latency 0.3–0.6 ms).", options: {} },
], { x: M + 0.25, y: 5.2, w: 11.7, h: 1.1, fontSize: 13.5, color: TEXT, fontFace: F, margin: 0 });
srcNote(s, "Source: outputs/evaluation/final/threshold_analysis.json — single-pass multi-threshold evaluation");
pageNum(s, 10);

/* ============ 11. EDGE OPTIMISATION ============ */
s = p.addSlide(); s.background = { color: BG };
titleBar(s, "Edge optimisation", "Same model, same inputs — measured on this machine");
s.addChart(p.charts.BAR, [{
  name: "median latency (ms)",
  labels: ["PyTorch CUDA FP32", "PyTorch CUDA FP16", "ONNX Runtime CPU", "PyTorch CPU FP32"],
  values: [12.8, 13.5, 16.2, 28.0],
}], {
  x: M, y: 1.8, w: 7.1, h: 4.3, barDir: "bar",
  chartColors: ["9FB3C8", "9FB3C8", ACCENT, "9FB3C8"],
  chartArea: { fill: { color: "FFFFFF" } },
  catAxisLabelColor: MUTED, valAxisLabelColor: MUTED, catAxisLabelFontSize: 10, valAxisLabelFontSize: 10,
  valGridLine: { color: "E2E8F0", size: 0.5 }, catGridLine: { style: "none" },
  showValue: true, dataLabelPosition: "outEnd", dataLabelColor: TEXT, dataLabelFontSize: 11,
  showLegend: false,
});
card(s, 8.0, 1.8, 4.83, 4.3, CARD2);
s.addText("WHY ONNX RUNTIME CPU IS THE DEFAULT", { x: 8.25, y: 2.0, w: 4.4, h: 0.5, fontSize: 11, bold: true, color: ACCENT, charSpacing: 1.5, fontFace: F, margin: 0 });
bullets(s, [
  "p95 ≈ median (17.5 vs 16.2 ms) — CUDA here shows rare VRAM-stall spikes (WDDM shared GPU)",
  "identical detections to PyTorch: 100% class agreement, box IoU 1.0, Δconf 0.0",
  "no GPU dependency — runs on any factory PC",
  "61.6 FPS sustained, stable over 320-frame run (+2.6 MB RSS)",
  "TensorRT: optional — auto-detected if an engine exists; reported unavailable here",
], 8.25, 2.55, 4.4, 3.4, 11.5);
srcNote(s, "Source: outputs/evaluation/edge_benchmark.csv + onnx_consistency.json — 100 iterations per backend, warm-up excluded");
pageNum(s, 11);

/* ============ 12. QUALITY CONTROL PIPELINE ============ */
s = p.addSlide(); s.background = { color: BG };
titleBar(s, "Quality control & PLC", "Industrial sorting logic, simulated safely");
const qc = [
  ["PASS / REJECT", "REJECT iff any defect ≥ 0.35 confidence.\nPer-defect severity: LOW→CRITICAL\n(class weight + confidence + area + count)"],
  ["Temporal confirmation", "REJECT needs defects in ≥2 of the last\n3 frames — a single noisy frame can\nnever reject a part (unit-tested)."],
  ["PLC debounce", "2 s cooldown: one continuous defect =\none reject command (12 defect frames\n→ exactly 1 command, measured live)."],
  ["Decision latency", "confirmed decision → PLC dispatch\nmeasured and exposed as a metric:\n0.3–0.6 ms in live verification."],
];
qc.forEach((q, i) => {
  const x = M + (i % 2) * 6.25, y = 1.85 + Math.floor(i / 2) * 2.45;
  card(s, x, y, 5.95, 2.2, CARD2);
  s.addText(q[0], { x: x + 0.25, y: y + 0.2, w: 5.4, h: 0.35, fontSize: 16, bold: true, color: PRIMARY_D, fontFace: F, margin: 0 });
  s.addText(q[1], { x: x + 0.25, y: y + 0.62, w: 5.45, h: 1.4, fontSize: 12.5, color: TEXT, fontFace: F, margin: 0 });
});
s.addText("PLC is clearly labelled SIMULATED in the UI — a PLCAdapter interface with a Modbus TCP skeleton makes a real controller a drop-in replacement.", { x: M, y: 6.75, w: 12.2, h: 0.35, fontSize: 12, italic: true, color: MUTED, fontFace: F, margin: 0 });
pageNum(s, 12);

/* ============ 13. SOFTWARE SYSTEM ============ */
s = p.addSlide(); s.background = { color: BG };
titleBar(s, "Software system", "FastAPI backend + industrial control-room dashboard");
card(s, M, 1.75, 5.95, 4.7, CARD2);
s.addText("BACKEND (FASTAPI)", { x: M + 0.25, y: 1.95, w: 5.4, h: 0.3, fontSize: 11, bold: true, color: ACCENT, charSpacing: 2, fontFace: F, margin: 0 });
bullets(s, [
  "Model loaded once at startup — shared by every request and camera frame",
  "POST /api/detect/image · /api/detect/video (multipart, validated)",
  "MJPEG live stream (/api/video_feed) + camera start/stop",
  "SQLite history: filters (status/class/severity), pagination, aggregates",
  "Clean 400/413/422 errors — no stack traces reach clients",
  "OpenAPI docs auto-generated at /docs",
], M + 0.25, 2.35, 5.5, 4.0, 12.5);
card(s, 6.85, 1.75, 5.95, 4.7, CARD2);
s.addText("DASHBOARD — 10 PAGES, ZERO FAKE DATA", { x: 7.1, y: 1.95, w: 5.5, h: 0.3, fontSize: 11, bold: true, color: ACCENT, charSpacing: 2, fontFace: F, margin: 0 });
bullets(s, [
  "Overview: real KPIs, 6 charts, recent events",
  "Live Inspection: MJPEG feed, boxes + severity, PLC panel",
  "Image Inspection: drag&drop + one-click NEU sample, download link",
  "History: filters + pagination · Analytics: 6 real charts",
  "Model Performance: measured metrics + edge benchmark table",
  "Dataset Explorer · System Health · Settings",
], 7.1, 2.35, 5.5, 4.0, 12.5);
s.addText("Every displayed figure comes from the API, SQLite, evaluation artifacts, benchmark files or live state — empty states are explicit.", { x: M, y: 6.7, w: 12.2, h: 0.35, fontSize: 12, italic: true, color: MUTED, fontFace: F, margin: 0 });
pageNum(s, 13);

/* ============ 14. MONITORING & DEPLOYMENT ============ */
s = p.addSlide(); s.background = { color: BG };
titleBar(s, "Monitoring & deployment", "Production habits applied to a college project");
const mon = [
  ["Prometheus", "13 metric families on /metrics: requests, detections by class,\nPASS/REJECT counters, latency histogram, camera status/uptime/FPS,\nframes processed, PLC events + decision latency, API errors"],
  ["Grafana", "auto-provisioned datasource + dashboard JSON; live verification:\ndetection counter observed by Prometheus scrape (target UP)"],
  ["Docker", "docker compose up -d → app + prometheus + grafana verified\nend-to-end (CPU torch inside the image; models mounted as volumes)"],
  ["Native Windows", "one-command local run (uvicorn) — cold-start tested from a\nclean terminal; identical behaviour to the container"],
];
mon.forEach((m2, i) => {
  const x = M + (i % 2) * 6.25, y = 1.8 + Math.floor(i / 2) * 2.35;
  card(s, x, y, 5.95, 2.1, CARD2);
  s.addText(m2[0], { x: x + 0.25, y: y + 0.18, w: 5.4, h: 0.35, fontSize: 16, bold: true, color: PRIMARY_D, fontFace: F, margin: 0 });
  s.addText(m2[1], { x: x + 0.25, y: y + 0.6, w: 5.45, h: 1.35, fontSize: 11.5, color: TEXT, fontFace: F, margin: 0 });
});
card(s, M, 6.35, 12.23, 0.72, CARD);
s.addText("Verified live: a real detection through the containerised app incremented defect_detection_requests_total observed by the Prometheus scrape.", { x: M + 0.25, y: 6.5, w: 11.7, h: 0.42, fontSize: 12.5, color: TEXT, fontFace: F, margin: 0 });
pageNum(s, 14);

/* ============ 15. TESTING ============ */
s = p.addSlide(); s.background = { color: BG };
titleBar(s, "Testing & quality", "53 automated tests + full manual verification");
kpi(s, M, 1.75, 3.9, "53 / 53", "pytest passed", "0 failed · 0 skipped");
kpi(s, M + 4.05, 1.75, 3.9, "20 / 20", "API checks", "endpoints + invalid inputs");
kpi(s, M + 8.1, 1.75, 4.15, "10 / 10", "dashboard pages", "real browser walkthrough");
card(s, M, 3.75, 12.23, 2.7, CARD2);
s.addText("WHAT THE TESTS COVER", { x: M + 0.25, y: 3.95, w: 6, h: 0.3, fontSize: 11, bold: true, color: ACCENT, charSpacing: 2, fontFace: F, margin: 0 });
s.addText([
  { text: "Data:", options: { bold: true } }, { text: "  VOC→YOLO maths, bbox normalisation bounds, degenerate-box rejection, class mapping", options: { breakLine: true } },
  { text: "Decisions:", options: { bold: true } }, { text: "  PASS/REJECT rules, severity grading, temporal validator (noise/persistence/slide/disable), PLC debounce & cooldown", options: { breakLine: true } },
  { text: "Robustness:", options: { bold: true } }, { text: "  corrupt-ONNX → PyTorch fallback, all-models-missing error, model-file validation, DB round-trip", options: { breakLine: true } },
  { text: "API:", options: { bold: true } }, { text: "  health/status/classes/metrics schemas, invalid uploads (text/corrupt/missing), malformed requests", options: {} },
], { x: M + 0.25, y: 4.35, w: 11.7, h: 1.9, fontSize: 13, color: TEXT, fontFace: F, paraSpaceAfter: 8, margin: 0 });
pageNum(s, 15);

/* ============ 16. LIMITATIONS ============ */
s = p.addSlide(); s.background = { color: BG };
titleBar(s, "Limitations", "Stated honestly — because credibility matters");
const lim = [
  ["No defect-free samples", "NEU-DET contains only defective images — PASS decisions (absence of detection) are unvalidated against pristine surfaces"],
  ["Single-dataset model", "trained & evaluated only on NEU; real-plant lighting/materials require fine-tuning — not production-certified"],
  ["Simulated PLC", "sorting is simulated through an adapter; Modbus TCP skeleton included but untested on hardware"],
  ["Environment bounds", "TensorRT not installed (optional, fallback verified) · RTSP accepted but untested against a live camera"],
  ["Weak classes", "crazing and rolled-in scale sit at ≈0.49 AP50 — fine low-contrast textures remain hard"],
];
lim.forEach((l, i) => {
  const y = 1.8 + i * 1.02;
  s.addText(String(i + 1), { x: M, y: y + 0.02, w: 0.55, h: 0.6, fontSize: 24, bold: true, color: ACCENT, fontFace: F, margin: 0 });
  s.addText(l[0], { x: M + 0.65, y, w: 3.3, h: 0.9, fontSize: 15.5, bold: true, color: PRIMARY_D, fontFace: F, margin: 0, valign: "top" });
  s.addText(l[1], { x: M + 4.1, y, w: 8.6, h: 0.9, fontSize: 12.5, color: TEXT, fontFace: F, margin: 0, valign: "top" });
  if (i < lim.length - 1) s.addShape(p.shapes.LINE, { x: M, y: y + 0.92, w: 12.2, h: 0, line: { color: "E2E8F0", width: 0.75 } });
});
pageNum(s, 16);

/* ============ 17. CONCLUSION & FUTURE ============ */
s = p.addSlide(); s.background = { color: BG };
titleBar(s, "Conclusion", "A complete, measured, demonstrable system");
card(s, M, 1.75, 5.95, 4.85, CARD2);
s.addText("DELIVERED", { x: M + 0.25, y: 1.95, w: 5.4, h: 0.3, fontSize: 11, bold: true, color: ACCENT, charSpacing: 2, fontFace: F, margin: 0 });
bullets(s, [
  "Custom 6-class defect detector: mAP50 0.725 @ 320 px, 6.2 MB",
  "62 FPS CPU inference via ONNX — edge-ready without a GPU",
  "Image / video / live inspection with PASS/REJECT QC",
  "Temporal confirmation + debounced PLC simulation",
  "Dashboard, SQLite history, Prometheus + Grafana monitoring",
  "Docker deployment verified end-to-end; 53 tests green",
], M + 0.25, 2.35, 5.5, 4.1, 13);
card(s, 6.85, 1.75, 5.95, 4.85, CARD);
s.addText("NEXT STEPS", { x: 7.1, y: 1.95, w: 5.4, h: 0.3, fontSize: 11, bold: true, color: ACCENT, charSpacing: 2, fontFace: F, margin: 0 });
bullets(s, [
  "Collect defect-free surfaces; validate PASS decisions on pristine steel",
  "Targeted augmentation + thresholds for crazing / rolled-in scale",
  "TensorRT INT8 with a proper calibration set",
  "Live Modbus TCP PLC over the existing adapter interface",
  "Multi-camera line support with frame-queue load balancing",
], 7.1, 2.35, 5.5, 4.1, 13);
pageNum(s, 17);

/* ============ 18. DEMO FLOW ============ */
s = p.addSlide(); s.background = { color: BG };
titleBar(s, "Live demo", "Seven minutes, in this order");
const demo = [
  ["Open dashboard", "Overview loads with real KPIs — no camera needed"],
  ["Dataset Explorer", "click Scratches: real NEU images + ground-truth boxes"],
  ["Image Inspection", "Load Sample → REJECT banner, confidence + severity cards"],
  ["Live Inspection", "start camera: boxes, FPS, latency breakdown, PASS/REJECT"],
  ["PLC panel", "SIMULATED badge, reject signal, decision latency 0.3–0.6 ms"],
  ["Model Performance", "measured mAP / per-class AP / edge benchmark table"],
  ["Health + /metrics", "services, CPU/RAM/GPU, then Prometheus + Grafana URLs"],
];
demo.forEach((d, i) => {
  const col = i % 2, row = Math.floor(i / 2);
  const x = M + col * 6.25, y = 1.8 + row * 1.25;
  s.addShape(p.shapes.OVAL, { x, y: y + 0.06, w: 0.52, h: 0.52, fill: { color: i === 3 ? ACCENT : PRIMARY } });
  s.addText(String(i + 1), { x, y: y + 0.06, w: 0.52, h: 0.52, fontSize: 16, bold: true, color: "FFFFFF", align: "center", valign: "middle", fontFace: F, margin: 0 });
  s.addText(d[0], { x: x + 0.68, y, w: 5.4, h: 0.32, fontSize: 15, bold: true, color: TEXT, fontFace: F, margin: 0 });
  s.addText(d[1], { x: x + 0.68, y: y + 0.33, w: 5.5, h: 0.32, fontSize: 11, color: MUTED, fontFace: F, margin: 0 });
});
s.addText("Demo is webcam-independent — steps 2, 3, 6, 7 run entirely from real recorded data.", { x: M, y: 6.9, w: 11.5, h: 0.3, fontSize: 12, italic: true, color: MUTED, fontFace: F, margin: 0 });
pageNum(s, 18);

/* ============ 19. CLOSING (dark) ============ */
s = p.addSlide(); s.background = { color: BG_DARK };
s.addText("Minimise waste.\nStop defects at the line.\nInspect at the edge.", { x: M, y: 1.6, w: W - 2 * M, h: 2.6, fontSize: 40, bold: true, color: "FFFFFF", fontFace: F, margin: 0 });
s.addText("Real-Time Industrial Defect Detection System — trained on NEU-DET, evaluated at 72.5% mAP@0.50,\ndeployed locally and in Docker, verified by 53 automated tests.", { x: M, y: 4.5, w: W - 2 * M, h: 0.8, fontSize: 15, color: LIGHT, fontFace: F, margin: 0 });
s.addText("Thank you — questions welcome", { x: M, y: 5.9, w: W - 2 * M, h: 0.5, fontSize: 20, bold: true, color: ACCENT, fontFace: F, margin: 0 });
s.addText("Project: D:\\NEU  ·  Documentation: README.md / PROJECT_DOCUMENTATION.md / FINAL_TEST_REPORT.md", { x: M, y: 6.85, w: W - 2 * M, h: 0.3, fontSize: 11, color: MUTED, fontFace: F, margin: 0 });

p.writeFile({ fileName: path.join(__dirname, "..", "PRESENTATION.pptx") }).then(() => console.log("PRESENTATION.pptx written"));
