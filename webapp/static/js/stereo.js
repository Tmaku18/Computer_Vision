// Click an object in the left view. The server returns depth and size in the same response.

const $ = (id) => document.getElementById(id);

let points = [];
let guides = [];
let usingSample = true;
let demoMeta = null;
let leftBlob = null;
let rightBlob = null;

function msg(text, kind = "") {
  const el = $("msg");
  el.className = "msg" + (kind ? " " + kind : "");
  el.textContent = text;
}

async function shrink(file, maxSide = 1600) {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  return await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.85));
}

function paint() {
  const img = $("img-left");
  const canvas = $("marks");
  if (!img.naturalWidth) return;
  canvas.width = img.clientWidth;
  canvas.height = img.clientHeight;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const sx = img.clientWidth / img.naturalWidth;
  const sy = img.clientHeight / img.naturalHeight;
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = "rgba(255,255,255,0.7)";
  ctx.setLineDash([4, 3]);
  if (guides.length >= 2) {
    ctx.beginPath();
    guides.forEach(([u, v], i) => {
      const x = u * sx;
      const y = v * sy;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#2952cc";
  ctx.fillStyle = "#2952cc";
  ctx.font = "12px sans-serif";
  if (points.length >= 2) {
    ctx.beginPath();
    points.forEach(([u, v], i) => {
      const x = u * sx;
      const y = v * sy;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.stroke();
  }
  points.forEach(([u, v], i) => {
    ctx.beginPath();
    ctx.arc(u * sx, v * sy, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillText(String(i + 1), u * sx + 6, v * sy - 6);
  });
  const need = { rectangle: "4 corners", circle: "at least 5 rim points", polygon: "at least 3 corners" }[$("shape").value];
  $("click-count").textContent = `${points.length} point${points.length === 1 ? "" : "s"}. A ${$("shape").value} needs ${need}.`;
}

function clickPoint(event) {
  const img = event.currentTarget;
  const rect = img.getBoundingClientRect();
  const u = ((event.clientX - rect.left) * img.naturalWidth) / rect.width;
  const v = ((event.clientY - rect.top) * img.naturalHeight) / rect.height;
  points.push([u, v]);
  paint();
}

function showGuides() {
  guides = usingSample && demoMeta ? demoMeta.shapes[$("shape").value] : [];
  paint();
}

async function useSample() {
  usingSample = true;
  leftBlob = rightBlob = null;
  points = [];
  demoMeta = await (await fetch("/api/m7/demo")).json();
  $("img-left").src = demoMeta.left_url;
  $("img-right").src = demoMeta.right_url;
  $("baseline").value = (demoMeta.baseline_m * 100).toFixed(1);
  $("unit").value = "cm";
  $("source-note").textContent = "Sample pair: a flat plane, baseline 20 cm. Dashed guides show the corners. Click them, or use the button.";
  $("img-left").onload = showGuides;
  showGuides();
}

function showTapeFields() {
  const circle = $("shape").value === "circle";
  $("tape-rect").hidden = circle;
  $("tape-circle").hidden = !circle;
}

async function measure() {
  const shape = $("shape").value;
  const need = shape === "rectangle" ? 4 : shape === "circle" ? 5 : 3;
  if (points.length < need || (shape === "rectangle" && points.length !== 4)) {
    msg(`Need ${need} points${shape === "rectangle" ? " exactly" : ""}.`, "error");
    return;
  }
  const form = new FormData();
  form.set("shape", shape);
  form.set("points", JSON.stringify(points));
  form.set("baseline", $("baseline").value);
  form.set("unit", $("unit").value);
  form.set("tape_width", $("tape-width").value);
  form.set("tape_height", $("tape-height").value);
  form.set("tape_diameter", $("tape-diameter").value);
  if (usingSample) form.set("sample", "1");
  else {
    form.set("left", leftBlob, "left.jpg");
    form.set("right", rightBlob, "right.jpg");
  }
  msg("Measuring…");
  const res = await fetch("/api/m7/measure", { method: "POST", body: form });
  const body = await res.json();
  if (!res.ok) {
    msg(body.detail || "Measurement failed.", "error");
    return;
  }
  $("img-left").src = body.left_url;
  $("img-right").src = body.right_url;
  points = [];
  const stats = [];
  const meters = (value) => (value == null ? "—" : `${(value * 100).toFixed(2)} cm`);
  if (body.width_m != null) stats.push(["Width", meters(body.width_m)]);
  if (body.height_m != null) stats.push(["Height", meters(body.height_m)]);
  if (body.diameter_m != null) stats.push(["Diameter", meters(body.diameter_m)]);
  stats.push(["Mean depth Z", meters(body.mean_z_m)]);
  const dz = body.points[0] && body.points[0].z_uncertainty_m;
  if (dz != null) stats.push(["Depth uncertainty (0.5 px)", meters(dz)]);
  $("stats").innerHTML = stats.map(([name, value]) => `<div><div class="hint">${name}</div><strong>${value}</strong></div>`).join("");
  const edges = (body.edges_m || []).map((edge, i) => `<li>Edge ${i + 1}: ${meters(edge)} (Z-first ${meters(body.z_first_edges_m[i])})</li>`).join("");
  $("edges").innerHTML = `<ul>${edges}</ul>`;
  const tape = body.tape || {};
  const errors = Object.keys(tape).filter((key) => key.endsWith("_error_m")).map((key) => `${key.replace("_error_m", "")} error ${meters(tape[key])}`);
  $("gap").textContent = `Disparity and triangulation differ by at most ${body.triangulation_gap_m.toExponential(2)} m. ${errors.join(". ")}`;
  msg("Done.", "ok");
}

$("img-left").addEventListener("click", clickPoint);
$("shape").addEventListener("change", () => { showTapeFields(); showGuides(); });
$("btn-undo").addEventListener("click", () => { points.pop(); paint(); });
$("btn-clear").addEventListener("click", () => { points = []; paint(); });
$("btn-measure").addEventListener("click", () => measure().catch((err) => msg(err.message, "error")));
$("btn-sample").addEventListener("click", () => useSample().catch((err) => msg(err.message, "error")));
$("btn-guides").addEventListener("click", () => {
  if (!usingSample || !demoMeta) {
    msg("Guide points belong to the sample pair.", "warn");
    return;
  }
  points = demoMeta.shapes[$("shape").value].map((p) => [p[0], p[1]]);
  paint();
});
$("file-left").addEventListener("change", async () => {
  if (!$("file-left").files[0] || !$("file-right").files[0]) return;
  try {
    leftBlob = await shrink($("file-left").files[0]);
    rightBlob = await shrink($("file-right").files[0]);
    usingSample = false;
    points = [];
    $("img-left").src = URL.createObjectURL(leftBlob);
    $("img-right").src = URL.createObjectURL(rightBlob);
    $("source-note").textContent = "Your photos. Clicks are in the reduced image, long side at most 1600 px.";
  } catch (err) {
    msg("Could not read those photos. Use JPG or PNG.", "error");
  }
});
$("file-right").addEventListener("change", () => $("file-left").dispatchEvent(new Event("change")));
window.addEventListener("resize", paint);
showTapeFields();
useSample().catch((err) => msg(err.message, "error"));
