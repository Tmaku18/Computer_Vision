// Step 3 page: record 20 ground-truth comparisons and show error statistics.

import { $, api, download, escapeHtml, fmt, setMsg, store } from "./api.js";
import { computeStats, csvToRows, drawCharts, rowsToCsv, statsHtml } from "./stats-view.js";
import { Workspace } from "./workspace.js";

const TARGET = 20;
let current = null; // latest measurement of the clicked segment
let timer = null;

const ws = new Workspace({
  canvasContainer: $("#canvas-wrap"),
  objectLayer: { max: 2, closed: false },
  onUpdate: () => {
    clearTimeout(timer);
    timer = setTimeout(updateSegment, 120);
  },
});

$("#btn-clear").addEventListener("click", () => ws.picker.clear("obj"));
$("#btn-fit").addEventListener("click", () => ws.picker.fit());

async function updateSegment() {
  const pts = ws.picker.points("obj");
  current = null;
  $("#btn-add").disabled = true;
  if (!ws.image || pts.length !== 2) {
    $("#live-estimate").textContent = "Segment: click its two end points";
    ws.picker.setOverlay("labels", null);
    return;
  }
  if (!ws.reference && !ws.distance) {
    $("#live-estimate").textContent = "Segment: set a reference plane or the camera distance";
    return;
  }
  try {
    const res = await ws.measure(pts, false);
    const h = res.homography ? res.homography.edge_lengths[0] : null;
    const p = res.pinhole ? res.pinhole.edge_lengths[0] : null;
    current = { pts, h, p, unit: res.unit };
    $("#live-estimate").innerHTML =
      `Segment: <strong>${fmt(h ?? p, 3)} ${res.unit}</strong>` +
      (h != null && p != null ? ` <span class="muted">(pinhole ${fmt(p, 3)})</span>` : "");
    ws.picker.setOverlay("labels", {
      type: "labels",
      items: [{ x: (pts[0][0] + pts[1][0]) / 2, y: (pts[0][1] + pts[1][1]) / 2, text: `${fmt(h ?? p, 2)} ${res.unit}` }],
    });
    $("#btn-add").disabled = false;
  } catch (err) {
    $("#live-estimate").innerHTML = `<span style="color:var(--bad)">${escapeHtml(err.message)}</span>`;
  }
}

$("#btn-add").addEventListener("click", () => {
  const msg = $("#add-msg");
  const gt = parseFloat($("#gt-value").value);
  if (!current) return;
  if (!(gt > 0)) {
    setMsg(msg, "Enter the tape-measured length first.", "error");
    return;
  }
  const rows = store.rows;
  if (rows.length && rows[0].unit !== current.unit) {
    setMsg(msg, `Existing rows are in ${rows[0].unit}; switch the unit to ${rows[0].unit} (and set the reference again).`, "error");
    return;
  }
  const dist = ws.distance;
  const nextId = rows.reduce((m, r) => Math.max(m, Number(r.measurement_id) || 0), 0) + 1;
  rows.push({
    measurement_id: nextId,
    description: $("#gt-desc").value.trim(),
    image: ws.image.name,
    ground_truth: gt,
    unit: current.unit,
    est_homography: current.h,
    est_pinhole: current.p,
    u1: +current.pts[0][0].toFixed(2),
    v1: +current.pts[0][1].toFixed(2),
    u2: +current.pts[1][0].toFixed(2),
    v2: +current.pts[1][1].toFixed(2),
    width_px: ws.image.width,
    height_px: ws.image.height,
    camera_distance_m: dist ? +(dist.value * { mm: 0.001, cm: 0.01, m: 1, in: 0.0254, ft: 0.3048 }[dist.unit]).toFixed(4) : null,
    reference_kind: ws.reference ? ws.reference.kind : "none",
  });
  store.rows = rows;
  $("#gt-value").value = "";
  $("#gt-desc").value = "";
  ws.picker.clear("obj");
  setMsg(msg, `Added measurement #${nextId}.`, "ok");
  refresh();
});

function refreshTable(rows) {
  const err = (est, gt) => (est == null ? "—" : `${est - gt >= 0 ? "+" : ""}${fmt(est - gt, 3)}`);
  $("#rows-body").innerHTML = rows.length
    ? rows
        .map(
          (r, i) => `<tr>
        <td>${r.measurement_id}</td><td>${escapeHtml(r.description)}</td><td class="small muted">${escapeHtml(r.image)}</td>
        <td class="num">${fmt(r.ground_truth, 3)}</td>
        <td class="num">${fmt(r.est_homography, 3)}</td><td class="num">${err(r.est_homography, r.ground_truth)}</td>
        <td class="num">${fmt(r.est_pinhole, 3)}</td><td class="num">${err(r.est_pinhole, r.ground_truth)}</td>
        <td><button type="button" class="ghost small" data-del="${i}" title="Delete">✕</button></td></tr>`,
        )
        .join("")
    : `<tr><td colspan="9" class="muted">No measurements yet.</td></tr>`;
  const n = rows.length;
  $("#progress-bar").style.width = `${Math.min(100, (n / TARGET) * 100)}%`;
  $("#progress-text").innerHTML =
    `<span class="badge ${n >= TARGET ? "ok" : "warn"}">${n} / ${TARGET}</span>` + (rows[0] ? ` in ${rows[0].unit}` : "");
}

$("#rows-body").addEventListener("click", (e) => {
  const idx = e.target.dataset.del;
  if (idx === undefined) return;
  const rows = store.rows;
  rows.splice(Number(idx), 1);
  store.rows = rows;
  refresh();
});

async function refresh() {
  const rows = store.rows;
  const unit = rows[0] ? rows[0].unit : ws.unit;
  refreshTable(rows);
  try {
    const stats = await computeStats(rows);
    $("#stats-homography").innerHTML = statsHtml(stats.homography, unit);
    $("#stats-pinhole").innerHTML = statsHtml(stats.pinhole, unit);
  } catch (err) {
    $("#stats-homography").innerHTML = `<div class="msg error">${escapeHtml(err.message)}</div>`;
  }
  drawCharts(rows, unit, $("#chart-scatter"), $("#chart-errors"));
}

const METERS = { mm: 0.001, cm: 0.01, m: 1, in: 0.0254, ft: 0.3048 };

$("#btn-apply-distance").addEventListener("click", async () => {
  const msg = $("#add-msg");
  const dist = ws.distance;
  const rows = store.rows;
  if (!rows.length) {
    setMsg(msg, "There are no saved measurements.", "error");
    return;
  }
  if (!dist) {
    setMsg(msg, "Enter the camera distance first.", "error");
    return;
  }
  if (!ws.image) {
    setMsg(msg, "Upload the same photo once. The saved points do not need to be clicked again.", "error");
    return;
  }
  const named = rows.filter((r) => r.image);
  const matches = named.filter((r) => r.image === ws.image.name);
  const useLoadedSize = matches.length === named.length || matches.length === 0;
  if (matches.length === 0 && !confirm(
    `None of the saved rows are labeled ${ws.image.name}. Apply this photo’s size and the distance to all ${rows.length} rows?`,
  )) {
    return;
  }
  $("#btn-apply-distance").disabled = true;
  let updated = 0;
  try {
    for (const row of rows) {
      const samePhoto = !row.image || row.image === ws.image.name || useLoadedSize;
      const width = row.width_px || (samePhoto ? ws.image.width : null);
      const height = row.height_px || (samePhoto ? ws.image.height : null);
      if (!width || !height || row.u1 == null || row.v1 == null || row.u2 == null || row.v2 == null) continue;
      const res = await api("/api/measure", {
        json: {
          width_px: width,
          height_px: height,
          calibration_id: store.calibrationId,
          points: [[row.u1, row.v1], [row.u2, row.v2]],
          closed: false,
          unit: row.unit,
          distance: dist,
        },
      });
      row.est_pinhole = res.pinhole.edge_lengths[0];
      row.camera_distance_m = +(dist.value * METERS[dist.unit]).toFixed(4);
      row.width_px = width;
      row.height_px = height;
      updated += 1;
    }
    store.rows = rows;
    setMsg(msg, `Set the camera distance on ${updated} of ${rows.length} measurements.`, updated ? "ok" : "error");
    refresh();
  } catch (err) {
    setMsg(msg, err.message, "error");
  } finally {
    $("#btn-apply-distance").disabled = false;
  }
});

$("#btn-export-csv").addEventListener("click", () => download("validation_measurements_export.csv", rowsToCsv(store.rows), "text/csv"));
$("#btn-export-json").addEventListener("click", () =>
  download("validation_measurements_export.json", JSON.stringify(store.rows, null, 2), "application/json"),
);
$("#import-file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const text = await file.text();
  try {
    const rows = file.name.endsWith(".json") ? JSON.parse(text) : csvToRows(text);
    if (!Array.isArray(rows) || rows.some((r) => !(r.ground_truth > 0))) throw new Error("Every row needs a positive ground_truth.");
    if (store.rows.length && !confirm(`Replace the ${store.rows.length} current rows with ${rows.length} imported rows?`)) return;
    store.rows = rows;
    setMsg($("#add-msg"), `Imported ${rows.length} rows.`, "ok");
    refresh();
  } catch (err) {
    setMsg($("#add-msg"), `Import failed: ${err.message}`, "error");
  } finally {
    e.target.value = "";
  }
});
$("#btn-clear-rows").addEventListener("click", () => {
  if (store.rows.length && confirm("Delete all validation measurements stored in this browser?")) {
    store.rows = [];
    refresh();
  }
});

refresh();
