import { $, api, download, fmt, setMsg, uploadWithProgress } from "/static/js/api.js";

const root = $("#m4");
const mode = root.dataset.mode;
const KEY = "m4.prompts";

const stage = $("#stage");
const ctx = stage.getContext("2d");
const placeholder = $("#placeholder");
let photo = null;
let box = null;
let drag = null;
let imageId = null;
let sampleName = null;
let fileName = null;
let backgroundId = null;
let maskId = null;
let samples = [];

function storedPrompts() {
  try {
    return JSON.parse(localStorage.getItem(KEY) || "[]");
  } catch {
    return [];
  }
}

function rememberPrompt(entry) {
  const all = storedPrompts().filter((item) => item.image !== entry.image);
  all.push(entry);
  localStorage.setItem(KEY, JSON.stringify(all));
  return all;
}

function currentBox() {
  if (!box) return null;
  const x = Math.round(Math.min(box.x, box.x + box.w));
  const y = Math.round(Math.min(box.y, box.y + box.h));
  const w = Math.round(Math.abs(box.w));
  const h = Math.round(Math.abs(box.h));
  if (w < 2 || h < 2) return null;
  return [x, y, w, h];
}

function draw() {
  if (!photo) return;
  ctx.clearRect(0, 0, stage.width, stage.height);
  ctx.drawImage(photo, 0, 0);
  const live = currentBox();
  if (live) {
    ctx.strokeStyle = "#3dffa0";
    ctx.lineWidth = Math.max(2, stage.width / 280);
    ctx.strokeRect(live[0], live[1], live[2], live[3]);
  }
  const label = $("#box-label");
  label.textContent = live ? `box ${live.join(", ")}` : "no box yet";
}

function show(url) {
  photo = new Image();
  photo.onload = () => {
    stage.width = photo.naturalWidth;
    stage.height = photo.naturalHeight;
    placeholder.hidden = true;
    draw();
  };
  photo.src = url;
}

function eventPos(event) {
  const rect = stage.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * (stage.width / rect.width),
    y: (event.clientY - rect.top) * (stage.height / rect.height),
  };
}

stage.addEventListener("pointerdown", (event) => {
  if (!photo) return;
  stage.setPointerCapture(event.pointerId);
  const p = eventPos(event);
  drag = p;
  box = { x: p.x, y: p.y, w: 0, h: 0 };
  draw();
});
stage.addEventListener("pointermove", (event) => {
  if (!drag) return;
  const p = eventPos(event);
  box = { x: drag.x, y: drag.y, w: p.x - drag.x, h: p.y - drag.y };
  draw();
});
stage.addEventListener("pointerup", () => {
  drag = null;
});

function fillSelect(select, items, placeholderText) {
  select.innerHTML = "";
  const blank = document.createElement("option");
  blank.value = "";
  blank.textContent = placeholderText;
  select.appendChild(blank);
  for (const item of items) {
    const option = document.createElement("option");
    option.value = item.name;
    option.textContent = item.label || item.name;
    select.appendChild(option);
  }
}

function selectedSample() {
  return samples.find((item) => item.name === $("#sample").value) || null;
}

async function loadSamples() {
  const data = await api("/api/m4/samples");
  samples = data.samples.filter((item) => item.kind === mode);
  fillSelect($("#sample"), samples, "Upload instead…");
  const sam = $("#sam2");
  fillSelect(sam, samples.filter((item) => item.sam2), "None");
  if (mode === "rgb") {
    const backgrounds = samples.filter((item) => item.background).map((item) => ({
      name: item.background,
      label: item.background,
    }));
    fillSelect($("#bg-sample"), backgrounds, "None");
  }
  if (samples.length) {
    $("#sample").value = samples[0].name;
    applySample(samples[0]);
    $("#btn-run").click();
  }
}

function applySample(item) {
  imageId = null;
  sampleName = item.name;
  fileName = item.name;
  maskId = null;
  $("#photo-name").textContent = item.label || item.name;
  $("#mask-name").textContent = "";
  if (item.box) {
    box = { x: item.box[0], y: item.box[1], w: item.box[2], h: item.box[3] };
  }
  if (item.sam2) $("#sam2").value = item.sam2;
  if (mode === "rgb" && item.background) $("#bg-sample").value = item.background;
  show(item.url);
}

$("#sample").addEventListener("change", () => {
  const item = selectedSample();
  if (item) applySample(item);
});

async function uploadPhoto(file, which) {
  const form = new FormData();
  form.append("file", file);
  setMsg($("#msg"), "Uploading…");
  const meta = await uploadWithProgress("/api/m4/images", form, () => {});
  setMsg($("#msg"), "");
  if (which === "background") {
    backgroundId = meta.id;
    $("#bg-name").textContent = meta.name;
    $("#bg-sample").value = "";
    return;
  }
  imageId = meta.id;
  sampleName = null;
  fileName = meta.name;
  $("#sample").value = "";
  $("#photo-name").textContent = `${meta.name} · ${meta.width}×${meta.height}`;
  box = null;
  show(meta.url);
}

$("#photo").addEventListener("change", () => {
  const file = $("#photo").files[0];
  if (file) uploadPhoto(file, "image").catch((err) => setMsg($("#msg"), err.message, "error"));
});
const bgPhoto = $("#bg-photo");
if (bgPhoto) {
  bgPhoto.addEventListener("change", () => {
    const file = bgPhoto.files[0];
    if (file) uploadPhoto(file, "background").catch((err) => setMsg($("#msg"), err.message, "error"));
  });
}

$("#mask-file").addEventListener("change", async () => {
  const file = $("#mask-file").files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  try {
    const meta = await api("/api/m4/masks", { form });
    maskId = meta.id;
    $("#sam2").value = "";
    $("#mask-name").textContent = `${file.name} · ${meta.width}×${meta.height}`;
  } catch (err) {
    setMsg($("#msg"), err.message, "error");
  }
});

const method = $("#method");
if (method && method.tagName === "SELECT") {
  const sync = () => {
    $("#bg-fields").hidden = method.value !== "background";
    $("#field-iterations").hidden = method.value !== "grabcut";
  };
  method.addEventListener("change", sync);
  sync();
}

function payload() {
  const body = {
    method: $("#method").value,
    box: currentBox(),
    image_id: imageId,
    sample: sampleName,
    open_k: Number($("#open-k").value),
    close_k: Number($("#close-k").value),
  };
  if (mode === "rgb") {
    body.iterations = Number($("#iterations").value);
    body.threshold = Number($("#threshold").value);
    if (method.value === "background") {
      body.background_id = backgroundId;
      body.background_sample = $("#bg-sample").value || null;
    }
  } else {
    body.invert = $("#invert").checked;
    body.split = $("#split").checked;
    body.max_aspect = Number($("#max-aspect").value);
    body.clahe_clip = Number($("#clahe").value);
    body.blur = Number($("#blur").value);
    const area = $("#min-area").value;
    if (area) body.min_area = Number(area);
  }
  if (maskId) body.mask_id = maskId;
  else if ($("#sam2").value) body.sam2_sample = $("#sam2").value;
  return body;
}

function showMetrics(metrics) {
  const host = $("#metrics");
  host.innerHTML = "";
  if (!metrics) return;
  const rows = [
    ["IoU", metrics.iou],
    ["Dice", metrics.dice],
    ["Precision", metrics.precision],
    ["Recall", metrics.recall],
    ["Boundary F", metrics.boundary_f],
    ["Hausdorff px", metrics.hausdorff_px],
    ["Mean boundary px", metrics.mean_boundary_px],
  ];
  for (const [label, value] of rows) {
    const cell = document.createElement("div");
    cell.className = "stat";
    cell.innerHTML = `<div class="v">${fmt(value, 3)}</div><div class="l">${label}</div>`;
    host.appendChild(cell);
  }
}

$("#btn-run").addEventListener("click", async () => {
  const body = payload();
  if (!body.sample && !body.image_id) {
    setMsg($("#msg"), "Choose a sample or upload a photo first.", "warn");
    return;
  }
  if (body.method !== "thermal" && body.method !== "background" && !body.box) {
    setMsg($("#msg"), "Drag a box around the person first.", "warn");
    return;
  }
  if (body.method === "background" && !body.background_id && !body.background_sample) {
    setMsg($("#msg"), "Background differencing needs the empty frame.", "warn");
    return;
  }
  $("#btn-run").disabled = true;
  setMsg($("#msg"), "Segmenting…");
  try {
    const result = await api("/api/m4/segment", { json: body });
    $("#img-mask").src = result.mask_url;
    $("#img-overlay").src = result.overlay_url;
    $("#img-compare").src = result.comparison_url || "";
    $("#img-compare").style.visibility = result.comparison_url ? "visible" : "hidden";
    showMetrics(result.metrics);
    const note = result.metrics
      ? "Outline ready. Scores use the mask selected below. The built-in sample masks are a synthetic stand-in, not SAM2, until you replace them."
      : "Outline ready. Add a SAM2 mask to see the scores.";
    setMsg($("#msg"), note, "ok");
    if (body.box) {
      rememberPrompt({
        image: sampleName ? `HW3/samples/${sampleName}` : fileName,
        kind: mode,
        method: body.method,
        box: body.box,
        background: body.background_sample ? `HW3/samples/${body.background_sample}` : null,
      });
    }
  } catch (err) {
    setMsg($("#msg"), err.message, "error");
  } finally {
    $("#btn-run").disabled = false;
  }
});

$("#btn-prompts").addEventListener("click", () => {
  const prompts = storedPrompts();
  if (!prompts.length && currentBox()) {
    rememberPrompt({
      image: sampleName ? `HW3/samples/${sampleName}` : (fileName || "photo.jpg"),
      kind: mode,
      method: $("#method").value,
      box: currentBox(),
    });
  }
  const saved = storedPrompts();
  if (!saved.length) {
    setMsg($("#msg"), "Draw a box first so the file has something to save.", "warn");
    return;
  }
  download("prompts.json", JSON.stringify({ version: 1, prompts: saved }, null, 2), "application/json");
});

loadSamples().catch((err) => setMsg($("#msg"), err.message, "error"));
