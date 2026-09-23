// Interactive two-camera geometry: evaluates equations (1)-(6) of the derivation.

import { $, fmt, getActiveCalibration, matrixText } from "./api.js";

// ----- tiny linear algebra -----------------------------------------------
const mul = (A, B) => A.map((row) => B[0].map((_, j) => row.reduce((s, a, k) => s + a * B[k][j], 0)));
const mv = (A, v) => A.map((row) => row.reduce((s, a, k) => s + a * v[k], 0));
const T = (A) => A[0].map((_, j) => A.map((row) => row[j]));
const add = (a, b) => a.map((x, i) => x + b[i]);
const scale = (a, s) => a.map((x) => x * s);
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
const skew = ([x, y, z]) => [[0, -z, y], [z, 0, -x], [-y, x, 0]];
const rad = (d) => (d * Math.PI) / 180;
const Ry = (t) => [[Math.cos(t), 0, Math.sin(t)], [0, 1, 0], [-Math.sin(t), 0, Math.cos(t)]];
const Rx = (t) => [[1, 0, 0], [0, Math.cos(t), -Math.sin(t)], [0, Math.sin(t), Math.cos(t)]];
function inv3(m) {
  const [[a, b, c], [d, e, f], [g, h, i]] = m;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [
    [A / det, -(b * i - c * h) / det, (b * f - c * e) / det],
    [B / det, (a * i - c * g) / det, -(a * f - c * d) / det],
    [C / det, -(a * h - b * g) / det, (a * e - b * d) / det],
  ];
}
const dehom = (p) => [p[0] / p[2], p[1] / p[2]];

let K = [[3729.9, 0, 2861.9], [0, 3729.8, 2112.2], [0, 0, 1]];
let W = 5712;
let H = 4284;

function compute() {
  const P = ["#px", "#py", "#pz"].map((s) => parseFloat($(s).value) || 0);
  const C2 = ["#cx2", "#cy2", "#cz2"].map((s) => parseFloat($(s).value) || 0);
  const yaw = parseFloat($("#yaw").value);
  const pitch = parseFloat($("#pitch").value);
  $("#yaw-v").textContent = `${yaw}°`;
  $("#pitch-v").textContent = `${pitch}°`;

  const Rc = mul(Ry(rad(yaw)), Rx(rad(pitch)));
  const R = T(Rc);
  const t = scale(mv(R, C2), -1);
  const Kinv = inv3(K);

  // (1) camera 1
  const u1h = mv(K, P);
  const u1 = dehom(u1h);
  const lam1 = u1h[2];
  // (2) direct projection in camera 2
  const P2 = add(mv(R, P), t);
  const u2direct = dehom(mv(K, P2));
  // (3)/(4) transfer through depth
  const Hinf = mul(mul(K, R), Kinv);
  const e2 = mv(K, t);
  const m = mv(Hinf, [u1[0], u1[1], 1]);
  const u2h = add(scale(m, P[2]), e2);
  const u2 = dehom(u2h);
  const lam2 = u2h[2];
  // (5) epipolar constraint
  const E = mul(skew(t), R);
  const F = mul(mul(T(Kinv), E), Kinv);
  const l2 = mv(F, [u1[0], u1[1], 1]);
  const residual = dot([u2[0], u2[1], 1], l2) / Math.hypot(l2[0], l2[1]); // pixel distance to line
  // (6) triangulation
  const x1 = mv(Kinv, [u1[0], u1[1], 1]);
  const x2 = mv(Kinv, [u2[0], u2[1], 1]);
  const a = mv(R, x1);
  const b = scale(x2, -1);
  const [aa, ab, bb] = [dot(a, a), dot(a, b), dot(b, b)];
  const [ra, rb] = [-dot(a, t), -dot(b, t)];
  const det = aa * bb - ab * ab;
  const Ztri = (ra * bb - ab * rb) / det;
  const lam2tri = (aa * rb - ab * ra) / det;
  const Ptri = scale(x1, Ztri);

  const behind = lam1 <= 0 || P2[2] <= 0;
  $("#theory-out").innerHTML = `
    <dt>(1) u₁</dt><dd>(${fmt(u1[0], 2)}, ${fmt(u1[1], 2)}) px, λ₁ = Z = ${fmt(lam1, 4)} m</dd>
    <dt>(4) u₂ via depth</dt><dd>(${fmt(u2[0], 2)}, ${fmt(u2[1], 2)}) px, λ₂ = ${fmt(lam2, 4)} m</dd>
    <dt>(2) u₂ direct</dt><dd>(${fmt(u2direct[0], 2)}, ${fmt(u2direct[1], 2)}) px</dd>
    <dt>(5) epipolar residual</dt><dd>${residual.toExponential(2)} px from l₂ = F ũ₁</dd>
    <dt>(6) triangulated P</dt><dd>(${fmt(Ptri[0], 4)}, ${fmt(Ptri[1], 4)}, ${fmt(Ptri[2], 4)}) m, λ₂ = ${fmt(lam2tri, 4)} m</dd>
    <dt>Baseline B</dt><dd>${fmt(Math.hypot(...t), 3)} m</dd>
    ${behind ? `<dt>Warning</dt><dd style="color:var(--bad)">P is behind a camera: assumption 5 is violated.</dd>` : ""}`;

  $("#m-r").textContent = matrixText(R, 4);
  $("#m-t").textContent = matrixText(t.map((v) => [v]), 4);
  $("#m-e").textContent = matrixText(E, 4);

  drawImages(u1, u2, l2, dehom(e2), e2[2]);
  drawTop(P, C2, Rc);
}

function drawImages(u1, u2, l2, epi, ez) {
  const svg = $("#images-svg");
  const pad = 20;
  const w = 290;
  const s = w / W;
  const h = H * s;
  const inside = (u) => u[0] >= 0 && u[0] <= W && u[1] >= 0 && u[1] <= H;
  const frame = (x0, label) =>
    `<rect x="${x0}" y="${pad + 12}" width="${w}" height="${h}" fill="#fff" stroke="#9aa3b8"/>
     <text x="${x0}" y="${pad + 6}" font-size="12" fill="#1c2233">${label}</text>`;
  const dot = (x0, u, color) =>
    inside(u) ? `<circle cx="${x0 + u[0] * s}" cy="${pad + 12 + u[1] * s}" r="5" fill="${color}"/>` : "";

  // Epipolar line clipped to image 2: a u + b v + c = 0.
  const [a, b, c] = l2;
  const pts = [];
  if (Math.abs(b) > 1e-12) {
    for (const u of [0, W]) {
      const v = -(a * u + c) / b;
      if (v >= 0 && v <= H) pts.push([u, v]);
    }
  }
  if (Math.abs(a) > 1e-12) {
    for (const v of [0, H]) {
      const u = -(b * v + c) / a;
      if (u >= 0 && u <= W) pts.push([u, v]);
    }
  }
  const x2 = 2 * pad + w;
  const line = pts.length >= 2
    ? `<line x1="${x2 + pts[0][0] * s}" y1="${pad + 12 + pts[0][1] * s}" x2="${x2 + pts[1][0] * s}" y2="${pad + 12 + pts[1][1] * s}" stroke="#e5484d" stroke-dasharray="6 4" stroke-width="1.5"/>`
    : "";
  const epiNote = Math.abs(ez) < 1e-9 ? "epipole at infinity" : inside(epi) ? "" : `epipole off-image at (${fmt(epi[0], 0)}, ${fmt(epi[1], 0)})`;
  svg.innerHTML = `${frame(pad, "Camera 1")}${frame(x2, "Camera 2")}${line}
    ${dot(pad, u1, "#2952cc")}${dot(x2, u2, "#2952cc")}
    ${inside(epi) ? `<circle cx="${x2 + epi[0] * s}" cy="${pad + 12 + epi[1] * s}" r="4" fill="none" stroke="#e5484d"/>` : ""}
    <text x="${x2}" y="${pad + 12 + h + 16}" font-size="11" fill="#5e667a">${epiNote}${inside(u2) ? "" : " · P outside camera-2 image"}</text>`;
}

function drawTop(P, C2, Rc) {
  const svg = $("#top-svg");
  const xs = [0, C2[0], P[0]];
  const zs = [0, C2[2], P[2]];
  const minX = Math.min(...xs) - 0.6, maxX = Math.max(...xs) + 0.6;
  const minZ = Math.min(...zs) - 0.5, maxZ = Math.max(...zs) + 0.5;
  const sc = Math.min(600 / (maxX - minX), 260 / (maxZ - minZ));
  const X = (x) => 20 + (x - minX) * sc;
  const Z = (z) => 280 - (z - minZ) * sc;
  const axis = (c, dir, color, label) => {
    const len = 0.6;
    return `<line x1="${X(c[0])}" y1="${Z(c[2])}" x2="${X(c[0] + dir[0] * len)}" y2="${Z(c[2] + dir[2] * len)}" stroke="${color}" stroke-width="2"/>
      <circle cx="${X(c[0])}" cy="${Z(c[2])}" r="6" fill="${color}"/>
      <text x="${X(c[0]) + 8}" y="${Z(c[2]) + 16}" font-size="12" fill="${color}">${label}</text>`;
  };
  const opt2 = [Rc[0][2], Rc[1][2], Rc[2][2]];
  svg.innerHTML = `
    <line x1="${X(0)}" y1="${Z(0)}" x2="${X(P[0])}" y2="${Z(P[2])}" stroke="#9aa3b8" stroke-dasharray="4 3"/>
    <line x1="${X(C2[0])}" y1="${Z(C2[2])}" x2="${X(P[0])}" y2="${Z(P[2])}" stroke="#9aa3b8" stroke-dasharray="4 3"/>
    <line x1="${X(0)}" y1="${Z(0)}" x2="${X(C2[0])}" y2="${Z(C2[2])}" stroke="#c9ced9"/>
    ${axis([0, 0, 0], [0, 0, 1], "#2952cc", "camera 1")}
    ${axis(C2, opt2, "#13a34a", "camera 2")}
    <circle cx="${X(P[0])}" cy="${Z(P[2])}" r="6" fill="#e5484d"/>
    <text x="${X(P[0]) + 8}" y="${Z(P[2]) - 8}" font-size="12" fill="#e5484d">P</text>
    <text x="10" y="16" font-size="11" fill="#5e667a">top view: x → right, z (depth) ↑</text>`;
}

for (const id of ["px", "py", "pz", "cx2", "cy2", "cz2", "yaw", "pitch"]) {
  $(`#${id}`).addEventListener("input", compute);
}

getActiveCalibration()
  .then((c) => {
    K = c.calibration.camera_matrix;
    [H, W] = c.calibration.image_size;
  })
  .catch(() => {})
  .finally(compute);
