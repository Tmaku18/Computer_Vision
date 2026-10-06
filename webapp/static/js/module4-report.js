import { $, api, fmt } from "/static/js/api.js";

const KEY = "m4.reportMeta";

function loadMeta() {
  try {
    return JSON.parse(localStorage.getItem(KEY) || "{}");
  } catch {
    return {};
  }
}

function saveMeta() {
  const meta = {
    repo: $("#repo-url").value.trim(),
    app: $("#app-url").value.trim(),
    video: $("#video-url").value.trim(),
  };
  localStorage.setItem(KEY, JSON.stringify(meta));
  return meta;
}

function link(el, url, empty) {
  el.textContent = url || empty;
  if (url) {
    el.href = url;
  } else {
    el.removeAttribute("href");
  }
}

function paintLinks() {
  const meta = saveMeta();
  link($("#r-repo"), meta.repo, "—");
  link($("#r-app"), meta.app, "—");
  link($("#r-video"), meta.video, "—");
}

function cell(text, numeric = false) {
  const td = document.createElement("td");
  if (numeric) td.className = "num";
  td.textContent = text;
  return td;
}

function fill(report) {
  $("#r-note").textContent = report.note || "";
  const body = $("#r-rows");
  body.innerHTML = "";
  for (const row of report.images || []) {
    const tr = document.createElement("tr");
    const metrics = row.metrics;
    tr.append(
      cell(row.image),
      cell(row.kind),
      cell(row.method),
      cell(metrics ? fmt(metrics.iou, 3) : "—", true),
      cell(metrics ? fmt(metrics.dice, 3) : "—", true),
      cell(metrics ? fmt(metrics.boundary_f, 3) : "—", true),
      cell(metrics && metrics.hausdorff_px != null ? fmt(metrics.hausdorff_px, 2) : "—", true),
    );
    body.appendChild(tr);
  }
  const summary = $("#r-summary");
  summary.innerHTML = "";
  for (const [kind, stats] of Object.entries(report.summary || {})) {
    const tr = document.createElement("tr");
    tr.append(
      cell(kind),
      cell(String(stats.n), true),
      cell(stats.mean_iou == null ? "—" : fmt(stats.mean_iou, 3), true),
      cell(stats.mean_boundary_f == null ? "—" : fmt(stats.mean_boundary_f, 3), true),
    );
    summary.appendChild(tr);
  }
}

const saved = loadMeta();
if (saved.repo) $("#repo-url").value = saved.repo;
if (saved.app) $("#app-url").value = saved.app;
if (saved.video) $("#video-url").value = saved.video;
if (!$("#app-url").value) $("#app-url").value = location.origin;
$("#r-date").textContent = new Date().toLocaleDateString();
paintLinks();
for (const id of ["repo-url", "app-url", "video-url"]) {
  $("#" + id).addEventListener("input", paintLinks);
}
$("#btn-print").addEventListener("click", () => window.print());

api("/api/m4/comparison")
  .then(fill)
  .catch(() => {
    $("#r-note").textContent = "No comparison file yet. Run .venv/bin/python HW3/compare_sam2.py and reload.";
  });
