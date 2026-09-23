// Printable report assembled from the active calibration and stored validation rows.

import { $, escapeHtml, fmt, getActiveCalibration, matrixText, store } from "./api.js";
import { computeStats, drawCharts, statsTableHtml } from "./stats-view.js";

const DEFAULTS = {
  repo: "https://github.com/Tmaku18/Computer_Vision",
  app: window.location.origin,
  video: "",
};

function applyMeta() {
  const meta = { ...DEFAULTS, ...store.reportMeta };
  $("#repo-url").value = meta.repo;
  $("#app-url").value = meta.app;
  $("#video-url").value = meta.video;
  for (const [key, id] of [["repo", "#r-repo"], ["app", "#r-app"], ["video", "#r-video"]]) {
    $(id).textContent = meta[key] || "";
    $(id).href = meta[key] || "#";
  }
  $("#r-video-dt").hidden = !meta.video;
  $("#r-video").hidden = !meta.video;
}

for (const id of ["#repo-url", "#app-url", "#video-url"]) {
  $(id).addEventListener("input", () => {
    store.reportMeta = { repo: $("#repo-url").value, app: $("#app-url").value, video: $("#video-url").value };
    applyMeta();
  });
}
$("#btn-print").addEventListener("click", () => window.print());
$("#r-date").textContent = new Date().toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" });
applyMeta();

async function renderCalibration() {
  const rec = await getActiveCalibration();
  const c = rec.calibration;
  const k = c.camera_matrix;
  const [rows, cols] = c.image_size;
  $("#r-calib").innerHTML = `
    <dt>Photos used</dt><dd>${c.num_images}</dd>
    <dt>Image size</dt><dd>${cols} × ${rows} px</dd>
    <dt>Board</dt><dd>${(c.checkerboard_inner_corners || []).join(" × ")} inner corners, ${c.square_size} ${c.world_units} squares</dd>
    <dt>RMS reprojection error</dt><dd>${fmt(c.rms_px, 3)} px</dd>
    <dt>Focal length</dt><dd>f<sub>x</sub> = ${fmt(k[0][0], 1)} px, f<sub>y</sub> = ${fmt(k[1][1], 1)} px</dd>
    <dt>Principal point</dt><dd>(${fmt(k[0][2], 1)}, ${fmt(k[1][2], 1)}) px</dd>
    <dt>Radial distortion</dt><dd>k<sub>1</sub> = ${fmt(c.dist_coeffs[0], 4)}, k<sub>2</sub> = ${fmt(c.dist_coeffs[1], 4)}</dd>
    <dt>Valid lens region</dt><dd>within ${fmt(rec.valid_radius_px, 0)} px of the principal point</dd>`;
  $("#r-k").textContent = matrixText(k, 1);
  const used = (rec.views || []).filter((v) => v.used);
  new Chart($("#r-chart-views"), {
    type: "bar",
    data: { labels: used.map((v) => v.name), datasets: [{ label: "Per-photo RMS reprojection error (px)", data: used.map((v) => v.reproj_rms_px), backgroundColor: "#4f74e3" }] },
    options: { animation: false, scales: { y: { beginAtZero: true } }, plugins: { legend: { labels: { boxWidth: 12 } } } },
  });
}

async function renderValidation() {
  const rows = store.rows;
  const unit = rows[0] ? rows[0].unit : "in";
  const distances = [...new Set(rows.map((r) => r.camera_distance_m).filter((d) => d != null))];
  const photos = [...new Set(rows.map((r) => r.image))];
  $("#r-setup").innerHTML = rows.length
    ? `${rows.length} segment lengths were measured with a tape (ground truth) and estimated from ${photos.length} photo(s)
       taken with the camera ${distances.length ? distances.map((d) => `${fmt(d, 2)} m`).join(", ") : "(distance not recorded)"}
       from the object plane. All lengths are in ${unit}; error = estimate − truth.`
    : `<span class="msg warn">No validation rows yet. Record 20 measurements in Step 3.</span>`;
  const err = (e, g) => (e == null ? "—" : `${e - g >= 0 ? "+" : ""}${fmt(e - g, 3)}`);
  $("#r-rows").innerHTML = rows
    .map((r) => `<tr><td>${r.measurement_id}</td><td>${escapeHtml(r.description)}</td><td class="num">${fmt(r.ground_truth, 3)}</td>
      <td class="num">${fmt(r.est_homography, 3)}</td><td class="num">${err(r.est_homography, r.ground_truth)}</td>
      <td class="num">${fmt(r.est_pinhole, 3)}</td><td class="num">${err(r.est_pinhole, r.ground_truth)}</td></tr>`)
    .join("");
  const stats = await computeStats(rows);
  $("#r-stats").innerHTML = statsTableHtml(stats, unit);
  drawCharts(rows, unit, $("#r-chart-scatter"), $("#r-chart-errors"));
}

renderCalibration();
renderValidation();
