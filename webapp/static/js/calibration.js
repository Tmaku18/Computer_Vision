// Step 1 page: choose the active calibration or run a new one from uploaded photos.

import { $, UNITS, api, escapeHtml, fmt, initHeader, matrixText, setMsg, store, uploadWithProgress } from "./api.js";

let chart = null;
let files = [];

$("#cal-units").innerHTML = UNITS.map((u) => `<option value="${u}">${u}</option>`).join("");

async function refreshList(selectId = store.calibrationId) {
  const list = await api("/api/calibrations");
  const select = $("#calib-select");
  select.innerHTML = list
    .map((c) => `<option value="${c.id}">${escapeHtml(c.label)} · ${fmt(c.rms_px, 2)} px</option>`)
    .join("");
  select.value = list.some((c) => c.id === selectId) ? selectId : "default";
  await show(select.value);
}

async function show(id) {
  const rec = await api(`/api/calibrations/${id}`);
  store.calibrationId = id;
  initHeader();
  const c = rec.calibration;
  const [rows, cols] = c.image_size;
  const k = c.camera_matrix;
  const d = c.dist_coeffs;
  $("#calib-summary").innerHTML = `
    <dt>Photos used</dt><dd>${c.num_images ?? "—"}</dd>
    <dt>RMS reprojection</dt><dd>${fmt(c.rms_px, 3)} px</dd>
    <dt>Image size</dt><dd>${cols} × ${rows} px</dd>
    <dt>Focal length</dt><dd>f<sub>x</sub> = ${fmt(k[0][0], 1)}, f<sub>y</sub> = ${fmt(k[1][1], 1)} px</dd>
    <dt>Principal point</dt><dd>(${fmt(k[0][2], 1)}, ${fmt(k[1][2], 1)}) px</dd>
    <dt>Distortion</dt><dd>k<sub>1</sub> = ${fmt(d[0], 4)}, k<sub>2</sub> = ${fmt(d[1], 4)}</dd>
    <dt>Board</dt><dd>${c.checkerboard_inner_corners ? c.checkerboard_inner_corners.join(" × ") : "—"} inner corners, ${c.square_size ?? "—"} ${c.world_units ?? ""} squares</dd>
    <dt>Valid lens region</dt><dd>${fmt(rec.valid_radius_px, 0)} px from the principal point</dd>`;
  $("#calib-k").textContent = matrixText(k, 1);
  renderViews(rec.views || []);
}

function renderViews(views) {
  const used = views.filter((v) => v.used);
  $("#views-count").textContent = `${used.length} of ${views.length} photos used`;
  const labels = used.map((v) => v.name);
  const data = used.map((v) => v.reproj_rms_px);
  if (chart) chart.destroy();
  chart = new Chart($("#chart-views"), {
    type: "bar",
    data: { labels, datasets: [{ label: "RMS reprojection error (px)", data, backgroundColor: "#4f74e3" }] },
    options: {
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, title: { display: true, text: "px" } }, x: { ticks: { maxRotation: 70, minRotation: 50, font: { size: 10 } } } },
    },
  });

  const withThumbs = views.filter((v) => v.thumb_url);
  $("#thumbs-card").hidden = !withThumbs.length;
  $("#thumbs").innerHTML = withThumbs
    .map(
      (v) => `<div class="thumb ${v.used ? "" : "unused"}">
        <img src="${v.thumb_url}" alt="${escapeHtml(v.name)}" loading="lazy">
        <div class="cap"><span>${escapeHtml(v.name)}</span>
        <span class="badge ${v.used ? "ok" : "bad"}">${v.used ? fmt(v.reproj_rms_px, 2) + " px" : "skipped"}</span></div>
        ${v.note ? `<div class="cap muted">${escapeHtml(v.note)}</div>` : ""}
      </div>`,
    )
    .join("");
}

function setFiles(list) {
  files = [...list];
  $("#cal-files-label").textContent = files.length
    ? `${files.length} photo${files.length === 1 ? "" : "s"} selected`
    : "Drop 20 or more checkerboard photos here, or click to choose";
  $("#btn-calibrate").disabled = files.length === 0;
}

$("#cal-files").addEventListener("change", (e) => setFiles(e.target.files));
const drop = $("#cal-drop");
drop.addEventListener("dragover", (e) => {
  e.preventDefault();
  drop.classList.add("drag");
});
drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("drag");
  setFiles(e.dataTransfer.files);
});

$("#calib-select").addEventListener("change", (e) => show(e.target.value));

$("#btn-calibrate").addEventListener("click", async () => {
  const msg = $("#cal-msg");
  const min = parseInt($("#cal-min").value, 10);
  if (files.length < min) {
    setMsg(msg, `Select at least ${min} photos (${files.length} selected).`, "error");
    return;
  }
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  form.append("pattern_cols", $("#cal-cols").value);
  form.append("pattern_rows", $("#cal-rows").value);
  form.append("square_size", $("#cal-square").value);
  form.append("units", $("#cal-units").value);
  form.append("min_images", String(min));
  form.append("label", $("#cal-label").value);

  const bar = $("#cal-progress");
  bar.hidden = false;
  $("#btn-calibrate").disabled = true;
  setMsg(msg, "Uploading photos…");
  try {
    const rec = await uploadWithProgress("/api/calibrations", form, (p) => {
      bar.firstElementChild.style.width = `${Math.round(p * 100)}%`;
      if (p >= 1) setMsg(msg, "Detecting corners and calibrating… (a few seconds per 10 photos)");
    });
    setMsg(msg, `Calibrated with ${rec.calibration.num_images} photos · RMS ${fmt(rec.calibration.rms_px, 3)} px. Now active.`, "ok");
    await refreshList(rec.id);
  } catch (err) {
    setMsg(msg, err.message, "error");
  } finally {
    bar.hidden = true;
    $("#btn-calibrate").disabled = files.length === 0;
  }
});

refreshList().catch((err) => setMsg($("#cal-msg"), err.message, "error"));
