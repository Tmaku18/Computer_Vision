// Module 3 printable report: links, the 1-D example, and the experiment table.

import { api, fmt } from "./api.js";

const $ = (id) => document.getElementById(id);

const repo = $("repo-url");
const app = $("app-url");
const video = $("video-url");
app.value = location.origin;

function syncLinks() {
  const set = (id, input) => {
    const el = $(id);
    const value = input.value.trim();
    el.textContent = value || "—";
    el.href = value || "#";
  };
  set("r-repo", repo);
  set("r-app", app);
  set("r-video", video);
}
[repo, app, video].forEach((el) => el.addEventListener("input", syncLinks));
syncLinks();
$("r-date").textContent = new Date().toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" });
$("btn-print").addEventListener("click", () => window.print());

function sci(value) {
  if (value == null) return "—";
  if (!Number.isFinite(value)) return "∞";
  return value.toExponential(2);
}

function fillTable(report) {
  const rows = report.rows || [];
  $("r-rows").innerHTML = rows.map((r) => `<tr>
    <td>${r.image}</td><td>${r.kernel}</td><td>${r.boundary}</td>
    <td>${r.expected_match ? "yes" : "no"}</td>
    <td class="num">${sci(r.max_abs)}</td>
    <td class="num">${sci(r.rmse)}</td>
    <td class="num">${sci(r.psnr_db)}</td></tr>`).join("");
  const worst = report.worst_matching_max_abs;
  $("r-verdict").innerHTML = report.passed
    ? `<span class="badge ok">Pass</span> Largest gap on a row that should match: ${sci(worst)}. That is rounding, not a different image.`
    : `<span class="badge bad">Check</span> Largest matching-row gap is ${sci(worst)}, which is more than rounding.`;
  const timing = report.timing || [];
  if (timing.length && window.Chart) {
    new Chart($("r-chart"), {
      type: "bar",
      data: {
        labels: timing.map((t) => `${t.size}×${t.size}`),
        datasets: [
          { label: "Spatial (s)", data: timing.map((t) => t.spatial_s), backgroundColor: "#2952cc" },
          { label: "FFT (s)", data: timing.map((t) => t.fft_s), backgroundColor: "#13a34a" },
        ],
      },
      options: { responsive: true, plugins: { legend: { position: "bottom" }, title: { display: true, text: "Time vs. kernel size" } } },
    });
  }
}

async function loadExperiment() {
  try {
    const saved = await api("/api/m3/report");
    fillTable(saved);
    $("r-fig-diff").src = "/api/m3/report/differences.png";
    $("r-fig-time").src = "/api/m3/report/timing.png";
    $("r-experiment-note").textContent += " Numbers below are from HW2/results/report.json.";
    return;
  } catch {
    /* no saved file yet: compute one on the demo image */
  }
  try {
    const live = await api("/api/m3/verify", { json: {} });
    fillTable(live);
    $("r-fig-diff").alt = "Run HW2/verify_convolution_theorem.py to embed the bar chart.";
    $("r-fig-time").alt = "Run HW2/verify_convolution_theorem.py to embed the timing chart.";
  } catch (err) {
    $("r-rows").innerHTML = `<tr><td colspan="7">${err.message}</td></tr>`;
  }
}

async function loadExample() {
  try {
    const ex = await api("/api/m3/example", { json: {} });
    $("r-example").innerHTML =
      `f = [${ex.signal.join(", ")}], h = [${ex.kernel.join(", ")}]. ` +
      `Spatial sum [${ex.full.map((v) => fmt(v, 4)).join(", ")}]. ` +
      `Inverse DFT gap ${ex.max_abs.toExponential(2)}.`;
  } catch (err) {
    $("r-example").textContent = err.message;
  }
}

loadExample();
loadExperiment();
