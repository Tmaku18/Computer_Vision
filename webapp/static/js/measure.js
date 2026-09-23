// Step 2 page: click an object's corners and report real-world edge lengths.

import { $, fmt } from "./api.js";
import { Workspace } from "./workspace.js";

let timer = null;
let requestId = 0;

const ws = new Workspace({
  canvasContainer: $("#canvas-wrap"),
  objectLayer: { closed: true },
  onUpdate: () => {
    clearTimeout(timer);
    timer = setTimeout(update, 150);
  },
});

$("#closed").addEventListener("change", (e) => {
  ws.picker.setLayerOption("obj", "closed", e.target.checked);
  update();
});
$("#btn-undo").addEventListener("click", () => ws.picker.undo("obj"));
$("#btn-clear").addEventListener("click", () => ws.picker.clear("obj"));
$("#btn-fit").addEventListener("click", () => ws.picker.fit());

async function update() {
  const pts = ws.picker.points("obj");
  const closed = $("#closed").checked;
  if (!ws.image || pts.length < 2) {
    ws.picker.setOverlay("labels", null);
    $("#results").innerHTML = `<p class="muted">Place at least 2 object points (4 corners for a rectangle).</p>`;
    return;
  }
  if (!ws.reference && !ws.distance) {
    $("#results").innerHTML = `<p class="muted">Set a reference plane or enter the camera distance.</p>`;
    return;
  }
  const id = ++requestId;
  try {
    const res = await ws.measure(pts, closed);
    if (id !== requestId) return;
    render(res, pts, closed);
  } catch (err) {
    if (id !== requestId) return;
    ws.picker.setOverlay("labels", null);
    $("#results").innerHTML = `<div class="msg error">${err.message}</div>`;
  }
}

function render(res, pts, closed) {
  const unit = res.unit;
  const h = res.homography;
  const p = res.pinhole;
  const primary = h || p;
  const n = primary.edge_lengths.length;

  const rows = [];
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % pts.length;
    rows.push(`<tr><td>P${i + 1} → P${j + 1}</td>
      <td class="num">${h ? fmt(h.edge_lengths[i]) : "—"}</td>
      <td class="num">${p ? fmt(p.edge_lengths[i]) : "—"}</td></tr>`);
  }
  if (n > 1) {
    rows.push(`<tr><th>Perimeter</th><td class="num">${h ? fmt(h.perimeter) : "—"}</td><td class="num">${p ? fmt(p.perimeter) : "—"}</td></tr>`);
  }
  if (primary.area != null) {
    rows.push(`<tr><th>Area (${unit}²)</th><td class="num">${h ? fmt(h.area) : "—"}</td><td class="num">${p ? fmt(p.area) : "—"}</td></tr>`);
  }

  const size = closed && pts.length === 4
    ? `<p><strong>Width × height:</strong> ${fmt(primary.edge_lengths[0], 2)} × ${fmt(primary.edge_lengths[1], 2)} ${unit}
       <span class="muted">(${h ? "homography" : "pinhole"})</span></p>`
    : "";
  const depth = p ? `<p class="hint">Pinhole depth Z = ${fmt(p.depth, 3)} ${unit}. It assumes the object plane faces the camera squarely; the homography does not.</p>` : "";

  $("#results").innerHTML = `${size}
    <div class="table-wrap"><table>
      <thead><tr><th>Edge</th><th class="num">Homography (${unit})</th><th class="num">Pinhole (${unit})</th></tr></thead>
      <tbody>${rows.join("")}</tbody>
    </table></div>${depth}`;

  const items = [];
  for (let i = 0; i < n; i++) {
    const a = pts[i];
    const b = pts[(i + 1) % pts.length];
    items.push({ x: (a[0] + b[0]) / 2, y: (a[1] + b[1]) / 2, text: `${fmt(primary.edge_lengths[i], 2)} ${unit}` });
  }
  ws.picker.setOverlay("labels", { type: "labels", items });
}
