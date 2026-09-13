/* Real-Time Industrial Defect Detection System — dashboard application */
"use strict";

const $ = (id) => document.getElementById(id);
const CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"];
const CLASS_PRETTY = {
  crazing: "Crazing", inclusion: "Inclusion", patches: "Patches",
  pitted_surface: "Pitted Surface", "rolled-in_scale": "Rolled-in Scale", scratches: "Scratches",
};
const PALETTE = ["#3fc1ff", "#ffb02e", "#33d17a", "#e05fd0", "#a8e04a", "#ff5252"];
const SEV_COLORS = { LOW: "#9fb3c8", MEDIUM: "#ffb02e", HIGH: "#ff8c42", CRITICAL: "#ff5252" };

const fmt = {
  pct: (v, d = 1) => (v == null ? "—" : (v * 100).toFixed(d) + "%"),
  ms: (v, d = 1) => (v == null ? "—" : Number(v).toFixed(d) + " ms"),
  fps: (v) => (v == null ? "—" : Number(v).toFixed(1)),
  int: (v) => (v == null ? "—" : Number(v).toLocaleString()),
};
function prettyClass(c) { return CLASS_PRETTY[c] || c || "—"; }
function shortTime(ts) { return ts ? ts.replace("T", " ") : "—"; }

// ------------------------------------------------------------------ toasts
function toast(msg, kind = "") {
  const el = document.createElement("div");
  el.className = "toast " + kind;
  el.textContent = msg;
  $("toast-box").appendChild(el);
  setTimeout(() => el.remove(), 3500);
}

// ------------------------------------------------------------------ api
async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : `${r.status} ${r.statusText}`);
  }
  return r.json();
}

// ------------------------------------------------------------------ charts
const charts = {};
const CHART_OPTS = {
  responsive: true, maintainAspectRatio: false, animation: { duration: 250 },
  plugins: { legend: { display: false } },
  scales: {
    x: { ticks: { color: "#7f95ac", font: { size: 10 } }, grid: { color: "#141d27" } },
    y: { ticks: { color: "#7f95ac", font: { size: 10 } }, grid: { color: "#141d27" }, beginAtZero: true },
  },
};
function makeChart(id, type, labels, data, colors, extra = {}) {
  const el = $(id);
  if (!el) return;
  if (charts[id]) { charts[id].destroy(); delete charts[id]; }
  charts[id] = new Chart(el, {
    type,
    data: { labels, datasets: [{ data, backgroundColor: colors, borderColor: colors, borderWidth: type === "line" ? 2 : 0, tension: .3, pointRadius: type === "line" ? 2 : 0, fill: false }] },
    options: deepMerge({}, CHART_OPTS, extra),
  });
}
function deepMerge(target, ...srcs) {
  for (const s of srcs) for (const k of Object.keys(s)) {
    if (s[k] && typeof s[k] === "object" && !Array.isArray(s[k])) target[k] = deepMerge({}, s[k]);
    else target[k] = s[k];
  }
  return target;
}

// ------------------------------------------------------------------ navigation
document.querySelectorAll(".nav-item").forEach((btn) => {
  btn.addEventListener("click", () => activateTab(btn.dataset.tab));
});
function activateTab(name) {
  document.querySelectorAll(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${name}`));
  if (name === "analytics") loadAnalytics();
  if (name === "history") loadHistory();
  if (name === "health") loadHealth();
  if (name === "model") loadModel();
  if (name === "settings") loadSettings();
}

// ------------------------------------------------------------------ header
async function refreshChips() {
  try {
    const [status, live] = await Promise.all([api("/api/status"), api("/api/live/status")]);
    const setChip = (id, label, cls) => { const c = $(id); c.innerHTML = `${label} <b>${label === "" ? "" : ""}</b>`; };
    const modelChip = $("chip-model");
    modelChip.innerHTML = `MODEL <b>${status.model_loaded ? "READY" : "NOT TRAINED"}</b>`;
    modelChip.className = "chip " + (status.model_loaded ? "ok" : "err");

    const camChip = $("chip-camera");
    camChip.innerHTML = `CAMERA <b>${live.online ? "ONLINE" : "OFFLINE"}</b>`;
    camChip.className = "chip " + (live.online ? "ok" : "");
    $("chip-fps").innerHTML = `FPS <b>${fmt.fps(live.fps || 0)}</b>`;

    const m = status.model;
    $("chip-backend").innerHTML = `BACKEND <b>${m ? m.backend.toUpperCase() : "—"}</b>`;
    const dev = m ? (String(m.device).includes("cuda") || String(m.device).includes("CUDA") ? "GPU" : "CPU") : "—";
    $("chip-device").innerHTML = `DEVICE <b>${m ? dev : "—"}</b>`;
    $("chip-device").className = "chip " + (m && dev === "GPU" ? "ok" : "");

    $("sf-model").textContent = m ? m.model_path.split(/[\\/]/).pop() : "not trained";
    $("sf-backend").textContent = m ? m.backend.toUpperCase() : "—";
    return status;
  } catch (e) {
    $("chip-model").innerHTML = 'API <b style="color:var(--red)">OFFLINE</b>';
  }
}

// ------------------------------------------------------------------ overview
async function loadOverview() {
  try {
    const [s, live, recent] = await Promise.all([
      api("/api/metrics-summary"), api("/api/live/status"), api("/api/history?limit=8"),
    ]);
    $("kpi-total").textContent = fmt.int(s.total_events);
    $("kpi-total-sub").textContent = s.total_events === 1 ? "inspection event" : "inspection events";
    $("kpi-pass").textContent = fmt.int(s.passed);
    $("kpi-reject").textContent = fmt.int(s.rejected);
    $("kpi-rate").textContent = s.total_events ? s.defect_rate_pct.toFixed(1) + "%" : "N/A";
    $("kpi-latency").textContent = s.total_events ? fmt.ms(s.avg_inference_ms) : "N/A";
    $("kpi-fps").textContent = fmt.fps(live.fps || 0);
    $("kpi-fps-sub").textContent = live.online ? "camera stream" : "camera offline";

    const has = s.total_events > 0;
    const labels = Object.keys(s.class_counts).map(prettyClass);
    const data = Object.values(s.class_counts);
    toggleEmpty("empty-by-type", has);
    toggleEmpty("empty-pr", has);
    toggleEmpty("empty-trend", has);
    toggleEmpty("empty-lat", has);
    if (has) {
      makeChart("chart-by-type", "bar", labels, data, labels.map((_, i) => PALETTE[i % PALETTE.length]));
      makeChart("chart-pass-reject", "doughnut", ["PASS", "REJECT"], [s.passed, s.rejected], ["#33d17a", "#ff5252"],
        { cutout: "62%", plugins: { legend: { display: true, position: "bottom", labels: { color: "#7f95ac", font: { size: 11 } } } } });
      const tl = Object.keys(s.trend).slice(0, 20).reverse();
      makeChart("chart-trend", "line", tl, tl.map((t) => s.trend[t]), "#3fc1ff");
      const lat = await api("/api/analytics/summary");
      const lt = Object.keys(lat.latency_trend).slice(0, 20).reverse();
      makeChart("chart-lat", "line", lt, lt.map((t) => lat.latency_trend[t]), "#ffb02e");
    }
    renderRecentTable(recent.events);
  } catch (e) { console.error("overview:", e.message); }
}
function toggleEmpty(id, hasData) {
  const el = $(id);
  if (el) el.classList.toggle("hidden", hasData);
}
function renderRecentTable(events) {
  const tb = $("overview-recent-table").querySelector("tbody");
  tb.innerHTML = "";
  toggleEmpty("empty-recent", events.length > 0);
  $("overview-recent-table").classList.toggle("hidden", events.length === 0);
  for (const ev of events) tb.innerHTML += historyRowHTML(ev, true);
}
function historyRowHTML(ev, compact = false) {
  const cls = ev.defect_class ? prettyClass(ev.defect_class) : "No defect";
  const conf = ev.confidence != null ? fmt.pct(ev.confidence) : "—";
  const lat = ev.inference_latency_ms != null ? fmt.ms(ev.inference_latency_ms) : "—";
  return `<tr class="${ev.inspection_status === "REJECT" ? "reject-row" : ""}">
    <td>${shortTime(ev.timestamp)}</td>
    <td>${escapeHtml(String(ev.source || "—"))}</td>
    <td><span class="tag ${ev.inspection_status}">${ev.inspection_status}</span></td>
    <td>${cls}</td>
    <td>${conf}</td>
    ${compact ? "" : `<td>${ev.severity ? `<span class="tag sev-${ev.severity}">${ev.severity}</span>` : "—"}</td>`}
    <td>${lat}</td>
    ${compact ? '<td>—</td>' : ""}
  </tr>`;
}
function escapeHtml(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }

// -------------------------------------------------------------------- live
let liveTimer = null;
async function startCamera() {
  let src = $("cam-source").value;
  if (src === "custom") src = $("cam-source-custom").value.trim();
  if (!src) { toast("Enter a video file path or RTSP URL", "err"); return; }
  try {
    const q = src !== "0" ? `?source=${encodeURIComponent(src)}` : "";
    await api("/api/camera/start" + q, { method: "POST" });
    $("live-stream").src = "/api/video_feed?" + Date.now();
    $("live-placeholder").classList.add("hidden");
    $("live-stream-state").textContent = "LIVE";
    $("live-stream-state").className = "badge ok";
    toast("Camera started", "ok");
    if (!liveTimer) liveTimer = setInterval(pollLive, 1000);
  } catch (e) { toast("Camera start failed: " + e.message, "err"); }
}
async function stopCamera() {
  try {
    await api("/api/camera/stop", { method: "POST" });
    $("live-stream").src = "";
    $("live-placeholder").classList.remove("hidden");
    $("live-stream-state").textContent = "OFFLINE";
    $("live-stream-state").className = "badge";
    $("live-status").textContent = "STANDBY";
    $("live-status").className = "big-status unknown";
    toast("Camera stopped");
  } catch (e) { toast("Camera stop failed: " + e.message, "err"); }
  clearInterval(liveTimer); liveTimer = null;
}
async function pollLive() {
  try {
    const s = await api("/api/live/status");
    $("live-fps").textContent = fmt.fps(s.fps || 0);
    const r = s.last_result;
    if (r) {
      const st = s.confirmed_status || r.inspection_status;
      const el = $("live-status");
      el.textContent = st;
      el.className = "big-status " + st.toLowerCase();
      $("live-badge").textContent = st;
      $("live-badge").className = "badge " + (st === "REJECT" ? "err" : "ok");
      $("live-status-sub").textContent = st === "REJECT"
        ? `${r.defect_count} defect(s) detected — reject signal active`
        : "no defect above threshold";
      const dets = r.detections || [];
      $("live-count").textContent = String(r.defect_count);
      $("live-conf").textContent = dets.length ? fmt.pct(r.highest_confidence) : "—";
      $("live-severity").textContent = r.max_severity || "—";
      $("live-lat").textContent = fmt.ms(r.performance && r.performance.inference_ms);
      $("live-pipe").textContent = s.last_timings && s.last_timings.total_pipeline_ms != null ? fmt.ms(s.last_timings.total_pipeline_ms) : "—";
      const st2 = await api("/api/status");
      $("live-backend").textContent = st2.model ? `${st2.model.backend} (${st2.model.device || ""})` : "—";
      $("live-src").textContent = s.source || "—";
      $("live-ts").textContent = shortTime(r.timestamp);
    }
    // PLC panel
    const plc = await api("/api/plc/stats");
    const last = plc.last_event;
    if (last) {
      $("plc-action").innerHTML = last.reject_signal
        ? '<span style="color:var(--red);font-weight:700">REJECT SIGNAL</span>'
        : '<span style="color:var(--green);font-weight:700">PASS</span>';
      $("plc-last").textContent = shortTime(last.timestamp);
      $("plc-defect").textContent = last.defect ? prettyClass(last.defect) : "—";
    }
    const live = await api("/api/live/status");
    $("plc-lat").textContent = live.last_decision_latency_ms != null ? fmt.ms(live.last_decision_latency_ms) : "—";
  } catch (e) { /* transient poll errors are ignored */ }
}

// ------------------------------------------------------------------- image
function setupImageInspection() {
  const dz = $("dropzone"), input = $("image-input");
  async function inspect(file, sampleInfo) {
    $("img-original").src = URL.createObjectURL(file);
    $("img-detected").src = "";
    $("img-name").textContent = sampleInfo ? sampleInfo.filename : file.name;
    $("sample-note").hidden = !sampleInfo;
    $("image-status").textContent = "ANALYSING…";
    $("image-status").className = "big-status unknown";
    $("image-status-sub").textContent = "running quality-control inference";
    $("image-spinner").classList.remove("hidden");
    const fd = new FormData();
    fd.append("file", file, sampleInfo ? sampleInfo.filename : file.name);
    try {
      const t0 = performance.now();
      const res = await api("/api/detect/image", { method: "POST", body: fd });
      const roundTrip = performance.now() - t0;
      $("img-detected").src = res.annotated_image_url + "?" + Date.now();
      $("img-detected-msg").classList.add("hidden");
      $("img-download").href = res.annotated_image_url;
      $("img-download").classList.remove("hidden");
      const el = $("image-status");
      el.textContent = res.inspection_status;
      el.className = "big-status " + res.inspection_status.toLowerCase();
      $("image-status-sub").textContent = res.inspection_status === "REJECT"
        ? `${res.defect_count} defect(s) detected`
        : "no defect above threshold";
      const st = await api("/api/status");
      $("img-backend-tag").textContent = st.model ? `${st.model.backend} · ${st.model.device || ""}` : "";
      $("image-meta").innerHTML = `
        <span>Inference <b>${fmt.ms(res.performance.inference_ms)}</b></span>
        <span>Total processing <b>${fmt.ms(res.performance.total_ms)}</b></span>
        <span>HTTP round trip <b>${fmt.ms(roundTrip)}</b></span>`;
      const list = $("image-detections");
      list.innerHTML = res.detections.length ? "" : '<p class="empty">No defects detected — part PASSED.</p>';
      for (const d of res.detections) {
        list.innerHTML += `<div class="det-item">
          <span><b>${prettyClass(d.class)}</b></span>
          <span>Confidence <b>${fmt.pct(d.confidence)}</b></span>
          <span>Severity <span class="tag sev-${d.severity}">${d.severity}</span></span>
          <span>BBox (${Math.round(d.bbox.x1)}, ${Math.round(d.bbox.y1)})–(${Math.round(d.bbox.x2)}, ${Math.round(d.bbox.y2)})</span></div>`;
      }
      toast(`Image analyzed — ${res.inspection_status}`, res.inspection_status === "REJECT" ? "err" : "ok");
      loadOverview();
    } catch (e) {
      $("image-status").textContent = "ERROR";
      $("image-status").className = "big-status unknown";
      $("image-status-sub").textContent = e.message;
      $("img-detected-msg").textContent = "Inference failed: " + e.message;
      $("img-detected-msg").classList.remove("hidden");
      toast("Inspection failed: " + e.message, "err");
    } finally {
      $("image-spinner").classList.add("hidden");
    }
  }
  input.addEventListener("change", () => input.files[0] && inspect(input.files[0]));
  ["dragover", "dragenter"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => e.dataTransfer.files[0] && inspect(e.dataTransfer.files[0]));

  $("btn-load-sample").addEventListener("click", async () => {
    const cls = $("sample-class").value;
    try {
      const s = await api(`/api/dataset/sample-image?defect_class=${cls}&random=${Date.now()}`);
      const blob = await fetch(s.image_url).then((r) => r.blob());
      inspect(new File([blob], s.filename, { type: "image/jpeg" }), s);
    } catch (e) { toast("Sample load failed: " + e.message, "err"); }
  });
}

// ------------------------------------------------------------------- video
function setupVideoInspection() {
  const input = $("video-input");
  input.addEventListener("change", async () => {
    const file = input.files[0];
    if (!file) return;
    $("video-spinner").classList.remove("hidden");
    $("video-progress").hidden = false;
    $("video-progress").textContent = "Processing video… frame-by-frame inspection in progress";
    const fd = new FormData();
    fd.append("file", file);
    try {
      const res = await api("/api/detect/video?max_frames=600", { method: "POST", body: fd });
      $("v-frames").textContent = fmt.int(res.frames_processed);
      $("v-fps").textContent = fmt.fps(res.measured_fps);
      $("v-dets").textContent = fmt.int(res.total_detections);
      $("v-rejects").textContent = fmt.int(res.reject_events.length);
      $("v-lat").textContent = fmt.ms(res.avg_inference_ms);
      $("video-output").src = res.annotated_video_url + "?" + Date.now();
      const labels = Object.keys(res.class_counts).map(prettyClass);
      const data = Object.values(res.class_counts);
      makeChart("chart-video-classes", "bar", labels, data, labels.map((_, i) => PALETTE[i % PALETTE.length]));
      const most = labels[data.indexOf(Math.max(...data, 0))];
      $("v-most").textContent = data.length ? `Most common defect: ${most}` : "No defects detected in this video.";
      toast("Video processing complete", "ok");
    } catch (e) {
      $("video-progress").textContent = "Video processing failed: " + e.message;
      toast("Video failed: " + e.message, "err");
      $("video-spinner").classList.add("hidden");
      return;
    }
    $("video-spinner").classList.add("hidden");
    $("video-progress").hidden = true;
  });
}

// ----------------------------------------------------------------- history
let historyPage = 0;
const HISTORY_PAGE_SIZE = 50;
async function loadHistory() {
  const q = new URLSearchParams({
    limit: String(HISTORY_PAGE_SIZE), offset: String(historyPage * HISTORY_PAGE_SIZE),
  });
  if ($("history-status").value) q.set("status", $("history-status").value);
  if ($("history-class").value) q.set("defect_class", $("history-class").value);
  if ($("history-severity").value) q.set("severity", $("history-severity").value);
  try {
    const res = await api("/api/history?" + q.toString());
    const tbody = $("history-table").querySelector("tbody");
    tbody.innerHTML = "";
    toggleEmpty("history-empty", res.events.length > 0);
    $("history-table").classList.toggle("hidden", res.events.length === 0);
    for (const ev of res.events) {
      const cls = ev.defect_class ? prettyClass(ev.defect_class) : "No defect";
      tbody.innerHTML += `<tr class="${ev.inspection_status === "REJECT" ? "reject-row" : ""}">
        <td>${ev.event_id}</td><td>${shortTime(ev.timestamp)}</td>
        <td>${escapeHtml(String(ev.source || "—"))}</td>
        <td><span class="tag ${ev.inspection_status}">${ev.inspection_status}</span></td>
        <td>${cls}</td>
        <td>${ev.confidence != null ? fmt.pct(ev.confidence) : "—"}</td>
        <td>${ev.severity ? `<span class="tag sev-${ev.severity}">${ev.severity}</span>` : "—"}</td>
        <td style="font-family:monospace;font-size:10.5px">${ev.bbox || "—"}</td>
        <td>${ev.inference_latency_ms != null ? fmt.ms(ev.inference_latency_ms) : "—"}</td></tr>`;
    }
    $("history-total").textContent = `${res.total} total events`;
    const maxPage = Math.max(0, Math.ceil(res.total / HISTORY_PAGE_SIZE) - 1);
    $("btn-history-prev").disabled = historyPage === 0;
    $("btn-history-next").disabled = historyPage >= maxPage;
    $("history-page").textContent = `page ${historyPage + 1} / ${maxPage + 1}`;
  } catch (e) { toast("History load failed: " + e.message, "err"); }
}

// --------------------------------------------------------------- analytics
async function loadAnalytics() {
  try {
    const [s, lat] = await Promise.all([api("/api/metrics-summary"), api("/api/analytics/summary")]);
    const has = s.total_events > 0;
    $("analytics-empty").classList.toggle("hidden", has);
    $("analytics-body").classList.toggle("hidden", !has);
    if (!has) return;
    const labels = Object.keys(s.class_counts).map(prettyClass);
    makeChart("chart-an-class", "bar", labels, Object.values(s.class_counts), labels.map((_, i) => PALETTE[i % PALETTE.length]));
    const sev = lat.severity_counts;
    const sevKeys = ["LOW", "MEDIUM", "HIGH", "CRITICAL"].filter((k) => sev[k]);
    makeChart("chart-an-sev", "doughnut", sevKeys, sevKeys.map((k) => sev[k]), sevKeys.map((k) => SEV_COLORS[k]),
      { cutout: "58%", plugins: { legend: { display: true, position: "bottom", labels: { color: "#7f95ac", font: { size: 10 } } } } });
    const pr = lat.pass_reject_trend;
    const prL = Object.keys(pr).slice(0, 25).reverse();
    makeChart("chart-an-pr", "bar", prL, prL.map((t) => pr[t][1]), "#33d17a");
    if (charts["chart-an-pr"]) charts["chart-an-pr"].data.datasets.push({
      label: "REJECT", data: prL.map((t) => pr[t][0]), backgroundColor: "#ff5252", borderWidth: 0,
    }) && charts["chart-an-pr"].update();
    const lt = Object.keys(lat.latency_trend).slice(0, 25).reverse();
    makeChart("chart-an-lat", "line", lt, lt.map((t) => lat.latency_trend[t]), "#ffb02e");
    const events = await api("/api/history?limit=1000");
    const byClass = {};
    for (const ev of events.events) {
      if (!ev.defect_class) continue;
      (byClass[ev.defect_class] = byClass[ev.defect_class] || []).push(ev.confidence);
    }
    const cl = Object.keys(byClass).map(prettyClass);
    const cd = Object.values(byClass).map((a) => a.reduce((x, y) => x + y, 0) / a.length);
    makeChart("chart-an-conf", "bar", cl, cd, cl.map((_, i) => PALETTE[i % PALETTE.length]),
      { scales: { y: { beginAtZero: true, max: 1.0, ticks: { color: "#7f95ac" }, grid: { color: "#141d27" } } } });
    const hr = lat.detections_by_hour;
    const hrL = Object.keys(hr).sort();
    makeChart("chart-an-hour", "bar", hrL, hrL.map((h) => hr[h]), "#3fc1ff");
    let most = "—", mostN = 0;
    for (const [c, n] of Object.entries(s.class_counts)) if (n > mostN) { most = c; mostN = n; }
    $("an-most-common").textContent = most === "—" ? "—" : `${prettyClass(most)} (${fmt.int(mostN)} detections)`;
  } catch (e) { toast("Analytics load failed: " + e.message, "err"); }
}

// ------------------------------------------------------------------- model
async function loadModel() {
  try {
    const info = await api("/api/model/info");
    $("model-missing").classList.toggle("hidden", info.available);
    $("model-body").classList.toggle("hidden", !info.available);
    if (!info.available) return;
    const m = info.metadata;
    $("m-map50").textContent = fmt.pct(m.mAP50);
    $("m-map").textContent = fmt.pct(m.mAP50_95);
    $("m-prec").textContent = fmt.pct(m.precision);
    $("m-rec").textContent = fmt.pct(m.recall);
    $("m-size").textContent = info.model_sizes_mb && info.model_sizes_mb.pt ? info.model_sizes_mb.pt + " MB" : "—";
    $("m-thr").textContent = m.recommended_confidence_threshold != null ? Number(m.recommended_confidence_threshold).toFixed(2) : "—";
    $("m-info-table").innerHTML = [
      ["Architecture", m.architecture],
      ["Weight file", (m.weights || "").split(/[\\/]/).pop()],
      ["Classes", (m.classes || []).length],
      ["Training image size", m.image_size],
      ["Epochs trained", `${m.epochs_trained} / ${m.epochs_requested}`],
      ["Optimizer / lr", `${m.optimizer} / ${m.lr0}`],
      ["Confidence / IoU", `${m.recommended_confidence_threshold} / ${m.iou_threshold}`],
      ["Training device", String(m.device_used).toUpperCase()],
      ["Inference backend (running)", info.running_backend ? `${info.running_backend.backend} (${info.running_backend.device})` : "not loaded"],
      ["TensorRT", info.running_backend && info.running_backend.backend === "tensorrt" ? "ACTIVE" : "Not Available"],
      ["Source experiment", m.source_experiment],
    ].map(([k, v]) => `<tr><td>${k}</td><td>${escapeHtml(String(v ?? "—"))}</td></tr>`).join("");
    $("about-model").textContent = `${(m.architecture || "yolov8n").split(".")[0]} @ ${m.image_size}px`;

    const tb = $("m-bench-table").querySelector("tbody");
    tb.innerHTML = "";
    for (const b of info.edge_benchmark || []) {
      const empty = !b.avg_inference_ms;
      tb.innerHTML += `<tr><td>${b.backend}</td><td>${b.device}</td><td>${b.precision}</td>
        <td>${empty ? "—" : fmt.ms(b.median_inference_ms)}</td><td>${empty ? "—" : fmt.ms(b.p95_inference_ms)}</td>
        <td>${empty ? "—" : b.estimated_fps}</td><td>${b.model_size_mb ? b.model_size_mb + " MB" : "—"}</td></tr>`;
    }
    $("m-recommended").innerHTML = info.recommended_backend
      ? `Recommended edge backend: <b style="color:var(--green)">${info.recommended_backend}</b> — lowest stable latency measured on this device`
      : "Run python training\\benchmark_edge.py to measure backends.";

    const pc = $("m-perclass-table").querySelector("tbody");
    pc.innerHTML = "";
    for (const [cls, v] of Object.entries(m.per_class || {})) {
      pc.innerHTML += `<tr><td>${prettyClass(cls)}</td><td>${fmt.pct(v.AP50)}</td><td>${fmt.pct(v.AP50_95)}</td></tr>`;
    }
    // plots (real artifacts from evaluation)
    const confImg = $("m-conf-img");
    confImg.src = "/outputs/evaluation/final/confusion_matrix.png?" + Date.now();
    confImg.onerror = () => { confImg.classList.add("hidden"); $("m-conf-msg").classList.remove("hidden"); };
    confImg.onload = () => { confImg.classList.remove("hidden"); $("m-conf-msg").classList.add("hidden"); };
    const cur = $("m-curves-img");
    cur.src = "/outputs/evaluation/final/results.png?" + Date.now();
    cur.onerror = () => { cur.classList.add("hidden"); $("m-curves-msg").classList.remove("hidden"); };
    cur.onload = () => { cur.classList.remove("hidden"); $("m-curves-msg").classList.add("hidden"); };
  } catch (e) { toast("Model info failed: " + e.message, "err"); }
}

// ----------------------------------------------------------------- dataset
async function setupDataset() {
  try {
    const s = await api("/api/metrics-summary");
    $("ds-train").textContent = "1,439";
    $("ds-val").textContent = "360";
    $("ds-total").textContent = "1,799";
    $("ds-ann").textContent = "4,186";
    const btnRow = $("dataset-class-buttons");
    for (const c of CLASS_NAMES) {
      const b = document.createElement("button");
      b.className = "btn";
      b.textContent = CLASS_PRETTY[c];
      b.onclick = () => loadDatasetClass(c, b);
      btnRow.appendChild(b);
    }
  } catch (e) { /* dataset counts are static facts; buttons still work */ }
}
async function loadDatasetClass(cls, btn) {
  document.querySelectorAll("#dataset-class-buttons .btn").forEach((b) => (b.style.borderColor = ""));
  if (btn) btn.style.borderColor = "var(--accent)";
  try {
    const res = await api(`/api/dataset/sample?defect_class=${cls}&count=6`);
    const grid = $("dataset-grid");
    grid.innerHTML = res.samples.length ? "" : '<p class="empty">No samples found.</p>';
    for (const s of res.samples) {
      grid.innerHTML += `<div class="dataset-card">
        <img src="${s.image_url}" loading="lazy" alt="${cls}">
        <div class="ds-name">${s.image_url.split("/").pop()} · ${s.boxes.length} annotation(s)</div></div>`;
    }
  } catch (e) { toast("Dataset load failed: " + e.message, "err"); }
}

// ----------------------------------------------------------------- health
async function loadHealth() {
  try {
    const [h, sys, cam] = await Promise.all([api("/health"), api("/api/system/info"), api("/api/live/status")]);
    const row = (name, ok, val) =>
      `<div class="health-row"><span><span class="dot ${ok === true ? "g" : ok === false ? "r" : "a"}"></span>${name}</span><span>${val}</span></div>`;
    $("health-services").innerHTML =
      row("API Server", true, "ONLINE") +
      row("Model", h.model_loaded, h.model_loaded ? "LOADED" : "NOT TRAINED") +
      row("Database", h.database_ok, h.database_ok ? "CONNECTED" : "UNAVAILABLE") +
      row("Camera", cam.online, cam.online ? "ONLINE" : "OFFLINE") +
      row("Prometheus", true, "metrics exposed on /metrics") +
      row("PLC Simulator", true, "SIMULATED (no hardware)");
    $("health-camera").innerHTML = [
      ["State", cam.online ? "ONLINE" : "OFFLINE"],
      ["Source", cam.source || "—"],
      ["Current FPS", fmt.fps(cam.fps || 0)],
      ["Uptime", cam.uptime_seconds ? cam.uptime_seconds + " s" : "—"],
      ["Frames processed", fmt.int(cam.frames_processed || 0)],
      ["Last error", cam.error || "none"],
    ].map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
    $("health-resources").innerHTML = [
      ["CPU utilization", sys.cpu_percent != null ? sys.cpu_percent.toFixed(0) + "%" : "N/A"],
      ["RAM", sys.ram_total_gb ? `${sys.ram_used_pct}% of ${sys.ram_total_gb} GB` : "N/A"],
      ["GPU", sys.gpu_name || "none (CPU-only CUDA stack)"],
      ["CUDA available", sys.cuda_available ? "YES" : "NO"],
      ["Platform", sys.platform],
    ].map(([k, v]) => `<tr><td>${k}</td><td>${escapeHtml(String(v))}</td></tr>`).join("");
    $("health-runtime").innerHTML = [
      ["App uptime", sys.app_uptime_seconds ? Math.round(sys.app_uptime_seconds) + " s" : "—"],
      ["Python", sys.python],
      ["Inference backend", sys.backend ? `${sys.backend} (${sys.device})` : "not loaded"],
      ["TensorRT", sys.backend === "tensorrt" ? "ACTIVE" : "Not Available"],
      ["Model loaded", h.model_loaded ? "YES" : "NO"],
    ].map(([k, v]) => `<tr><td>${k}</td><td>${escapeHtml(String(v))}</td></tr>`).join("");
  } catch (e) {
    $("health-services").innerHTML = '<div class="health-row"><span>Backend unavailable</span></div>';
  }
}

// ---------------------------------------------------------------- settings
async function loadSettings() {
  try {
    const [status, cls, plc, health] = await Promise.all([
      api("/api/status"), api("/api/classes"), api("/api/plc/stats"), api("/health"),
    ]);
    const m = status.model;
    $("settings-table").innerHTML = [
      ["App version", health.version],
      ["Confidence threshold", m ? m.conf_threshold : "—"],
      ["IoU threshold", m ? m.iou_threshold : "—"],
      ["Inference backend", m ? `${m.backend} (${m.device || ""})` : "not loaded"],
      ["Inference image size", m ? m.imgsz : "—"],
      ["Camera source", "0 (or set in .env)"],
      ["Defect classes", cls.classes.map(prettyClass).join(", ")],
      ["Save detection images", "true (outputs\\detections)"],
    ].map(([k, v]) => `<tr><td>${k}</td><td>${escapeHtml(String(v))}</td></tr>`).join("");
    $("plc-table").innerHTML = [
      ["Line", plc.line_id],
      ["PASS commands", plc.total_pass],
      ["REJECT commands", plc.total_reject],
      ["Mode", "SIMULATED — no physical hardware connected"],
    ].map(([k, v]) => `<tr><td>${k}</td><td>${escapeHtml(String(v))}</td></tr>`).join("");
    $("about-backend").textContent = m ? m.backend : "auto-selectable";
  } catch (e) { toast("Settings load failed: " + e.message, "err"); }
}

// -------------------------------------------------------------------- PLC
async function dispatchPLC() {
  try {
    const res = await api("/api/plc/dispatch-inspection", { method: "POST" });
    const out = $("plc-dispatch-out");
    out.textContent = JSON.stringify(res, null, 2);
    out.classList.remove("hidden");
    toast(`PLC ${res.action} dispatched`);
  } catch (e) { toast("PLC dispatch failed: " + e.message, "err"); }
}

// ------------------------------------------------------------------ boot
window.addEventListener("DOMContentLoaded", () => {
  $("btn-cam-start").onclick = startCamera;
  $("btn-cam-stop").onclick = stopCamera;
  $("btn-plc-dispatch").onclick = dispatchPLC;
  $("btn-history-refresh").onclick = () => { historyPage = 0; loadHistory(); };
  $("btn-history-prev").onclick = () => { historyPage = Math.max(0, historyPage - 1); loadHistory(); };
  $("btn-history-next").onclick = () => { historyPage += 1; loadHistory(); };
  $("btn-history-clear").onclick = async () => {
    if (!confirm("Delete all inspection history?")) return;
    try { const r = await api("/api/history", { method: "DELETE" }); toast(`Cleared ${r.cleared} events`); loadHistory(); loadOverview(); }
    catch (e) { toast("Clear failed: " + e.message, "err"); }
  };
  $("cam-source").addEventListener("change", () => {
    $("cam-source-custom").classList.toggle("hidden", $("cam-source").value !== "custom");
  });
  const classSel = $("history-class");
  for (const c of CLASS_NAMES) classSel.innerHTML += `<option value="${c}">${CLASS_PRETTY[c]}</option>`;

  setupImageInspection();
  setupVideoInspection();
  setupDataset();

  api("/health").then((h) => $("sf-version").textContent = "v" + h.version).catch(() => {});
  refreshChips();
  loadOverview();
  setInterval(refreshChips, 3000);
  setInterval(() => { if ($("tab-overview").classList.contains("active")) loadOverview(); }, 5000);
});
