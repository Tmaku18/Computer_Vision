// Rendering of validation statistics and charts, shared by Step 3 and the report.

import { api, fmt } from "./api.js";

export const CSV_COLUMNS = [
  "measurement_id", "description", "image", "ground_truth", "unit", "est_homography", "est_pinhole",
  "u1", "v1", "u2", "v2", "camera_distance_m", "reference_kind",
];

export async function computeStats(rows) {
  return api("/api/stats", {
    json: {
      rows: rows.map((r) => ({
        ground_truth: r.ground_truth,
        homography: r.est_homography ?? null,
        pinhole: r.est_pinhole ?? null,
      })),
    },
  });
}

export function statsHtml(s, unit) {
  if (!s || !s.n) return `<p class="muted">No measurements yet.</p>`;
  const ci = s.bias_ci95 ? `[${fmt(s.bias_ci95[0], 3)}, ${fmt(s.bias_ci95[1], 3)}]` : "—";
  const tile = (v, l) => `<div class="stat"><div class="v">${v}</div><div class="l">${l}</div></div>`;
  return `<div class="stat-grid">
    ${tile(s.n, "measurements")}
    ${tile(fmt(s.mae, 3), `MAE (${unit})`)}
    ${tile(fmt(s.rmse, 3), `RMSE (${unit})`)}
    ${tile(fmt(s.mape_percent, 2) + "%", "mean |% error|")}
    ${tile((s.bias >= 0 ? "+" : "") + fmt(s.bias, 3), `bias (${unit})`)}
    ${tile(fmt(s.std, 3), `std of error (${unit})`)}
    ${tile(fmt(s.max_abs_error, 3), `max |error| (${unit})`)}
    ${tile(fmt(s.max_percent_error, 2) + "%", "max |% error|")}
  </div>
  <p class="small muted" style="margin-top:0.5rem">
    95% CI of bias: ${ci} ${unit} · median |error| ${fmt(s.median_abs_error, 3)} ${unit}
    · Pearson r ${fmt(s.pearson_r, 4)} · fitted slope ${fmt(s.fit_slope, 4)}
  </p>`;
}

export function statsTableHtml(stats, unit) {
  const methods = [["homography", "Homography"], ["pinhole", "Pinhole"]].filter(([k]) => stats[k] && stats[k].n);
  if (!methods.length) return `<p class="muted">No measurements yet.</p>`;
  const row = (label, f) => `<tr><td>${label}</td>${methods.map(([k]) => `<td class="num">${f(stats[k])}</td>`).join("")}</tr>`;
  return `<table>
    <thead><tr><th>Statistic</th>${methods.map(([, l]) => `<th class="num">${l}</th>`).join("")}</tr></thead>
    <tbody>
      ${row("Measurements (n)", (s) => s.n)}
      ${row(`Mean absolute error (${unit})`, (s) => fmt(s.mae, 3))}
      ${row(`Root-mean-square error (${unit})`, (s) => fmt(s.rmse, 3))}
      ${row(`Bias, mean signed error (${unit})`, (s) => fmt(s.bias, 3))}
      ${row("95% CI of bias", (s) => (s.bias_ci95 ? `[${fmt(s.bias_ci95[0], 3)}, ${fmt(s.bias_ci95[1], 3)}]` : "—"))}
      ${row(`Std. deviation of error (${unit})`, (s) => fmt(s.std, 3))}
      ${row(`Median absolute error (${unit})`, (s) => fmt(s.median_abs_error, 3))}
      ${row(`Max absolute error (${unit})`, (s) => fmt(s.max_abs_error, 3))}
      ${row("Mean absolute % error", (s) => fmt(s.mape_percent, 2) + "%")}
      ${row("Max absolute % error", (s) => fmt(s.max_percent_error, 2) + "%")}
      ${row("Pearson r (estimate vs truth)", (s) => fmt(s.pearson_r, 4))}
    </tbody></table>`;
}

const charts = {};

export function drawCharts(rows, unit, scatterCanvas, errorCanvas) {
  const withH = rows.filter((r) => r.est_homography != null);
  const withP = rows.filter((r) => r.est_pinhole != null);
  const maxVal = Math.max(1, ...rows.map((r) => Math.max(r.ground_truth, r.est_homography ?? 0, r.est_pinhole ?? 0)));

  if (charts.scatter) charts.scatter.destroy();
  charts.scatter = new Chart(scatterCanvas, {
    type: "scatter",
    data: {
      datasets: [
        { label: "Homography", data: withH.map((r) => ({ x: r.ground_truth, y: r.est_homography })), backgroundColor: "#2952cc" },
        { label: "Pinhole", data: withP.map((r) => ({ x: r.ground_truth, y: r.est_pinhole })), backgroundColor: "#e5484d" },
        { label: "Ideal (y = x)", type: "line", data: [{ x: 0, y: 0 }, { x: maxVal * 1.05, y: maxVal * 1.05 }], borderColor: "#9aa3b8", borderDash: [6, 4], pointRadius: 0 },
      ],
    },
    options: {
      animation: false,
      scales: {
        x: { title: { display: true, text: `Ground truth (${unit})` }, beginAtZero: true },
        y: { title: { display: true, text: `Estimated (${unit})` }, beginAtZero: true },
      },
    },
  });

  if (charts.errors) charts.errors.destroy();
  charts.errors = new Chart(errorCanvas, {
    type: "bar",
    data: {
      labels: rows.map((r) => `#${r.measurement_id}`),
      datasets: [
        { label: "Homography", data: rows.map((r) => (r.est_homography != null ? r.est_homography - r.ground_truth : null)), backgroundColor: "#2952cc" },
        { label: "Pinhole", data: rows.map((r) => (r.est_pinhole != null ? r.est_pinhole - r.ground_truth : null)), backgroundColor: "#e5484d" },
      ],
    },
    options: { animation: false, scales: { y: { title: { display: true, text: `Estimate − truth (${unit})` } } } },
  });
}

export function rowsToCsv(rows) {
  const esc = (v) => {
    const s = v === null || v === undefined ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [CSV_COLUMNS.join(","), ...rows.map((r) => CSV_COLUMNS.map((c) => esc(r[c])).join(","))].join("\n") + "\n";
}

export function csvToRows(text) {
  const lines = [];
  let field = "";
  let record = [];
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') {
        field += '"';
        i++;
      } else if (ch === '"') quoted = false;
      else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") {
      record.push(field);
      field = "";
    } else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      record.push(field);
      lines.push(record);
      record = [];
      field = "";
    } else field += ch;
  }
  if (field || record.length) {
    record.push(field);
    lines.push(record);
  }
  const [header, ...body] = lines.filter((l) => l.some((c) => c.trim() !== ""));
  const numeric = new Set(["ground_truth", "est_homography", "est_pinhole", "u1", "v1", "u2", "v2", "camera_distance_m"]);
  return body.map((cells) => {
    const row = {};
    header.forEach((h, i) => {
      const v = cells[i] ?? "";
      row[h] = numeric.has(h) ? (v === "" ? null : Number(v)) : v;
    });
    return row;
  });
}
