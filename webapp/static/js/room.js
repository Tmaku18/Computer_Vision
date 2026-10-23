// Synthetic room, or two still frames from a pause-slide-pause clip.

const $ = (id) => document.getElementById(id);
let points = [];
let leftBlob = null;
let rightBlob = null;

function msg(text, kind = "") {
  const el = $("msg");
  el.className = "msg" + (kind ? " " + kind : "");
  el.textContent = text;
}

function meters(value) {
  return value == null ? "—" : `${Number(value).toFixed(3)} m`;
}

async function shrink(file, maxSide = 960) {
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
  if (!img.naturalWidth || !canvas) return;
  canvas.width = img.clientWidth;
  canvas.height = img.clientHeight;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const sx = img.clientWidth / img.naturalWidth;
  const sy = img.clientHeight / img.naturalHeight;
  ctx.fillStyle = "#2952cc";
  ctx.font = "12px sans-serif";
  points.forEach(([u, v], i) => {
    ctx.beginPath();
    ctx.arc(u * sx, v * sy, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillText(String(i + 1), u * sx + 6, v * sy - 6);
  });
}

function fill(body) {
  const rows = [];
  if (body.length_m != null) rows.push(["Length", meters(body.length_m)]);
  if (body.width_m != null) rows.push(["Width", meters(body.width_m)]);
  if (body.height_m != null) rows.push(["Height", meters(body.height_m)]);
  if (body.median_depth_m != null) rows.push(["Median depth", meters(body.median_depth_m)]);
  if (body.height_from_long_wall_m != null) rows.push(["Height from long wall", meters(body.height_from_long_wall_m)]);
  if (body.height_from_short_wall_m != null) rows.push(["Height from short wall", meters(body.height_from_short_wall_m)]);
  $("stats").innerHTML = rows.map(([name, value]) => `<div><div class="hint">${name}</div><strong>${value}</strong></div>`).join("");
  $("note").textContent = body.note || body.k_note || "";
  if (body.disparity_url) $("img-disp").src = body.disparity_url;
}

$("img-left").addEventListener("click", (event) => {
  const img = event.currentTarget;
  if (!img.naturalWidth) return;
  const rect = img.getBoundingClientRect();
  points.push([
    ((event.clientX - rect.left) * img.naturalWidth) / rect.width,
    ((event.clientY - rect.top) * img.naturalHeight) / rect.height,
  ]);
  if (points.length > 4) points = points.slice(-4);
  paint();
});

$("btn-box").addEventListener("click", async () => {
  msg("Fitting the synthetic room…");
  const form = new FormData();
  form.set("sample", "box");
  const res = await fetch("/api/m7/room", { method: "POST", body: form });
  const body = await res.json();
  if (!res.ok) {
    msg(body.detail || "Failed.", "error");
    return;
  }
  const demo = await (await fetch("/api/m7/demo")).json();
  $("img-left").src = demo.left_url;
  fill(body);
  msg("Synthetic room. Length 4 m, width 3 m, height 2.5 m.", "ok");
});

$("btn-clear").addEventListener("click", () => { points = []; paint(); });

$("file-left").addEventListener("change", async () => {
  if (!$("file-left").files[0] || !$("file-right").files[0]) return;
  leftBlob = await shrink($("file-left").files[0]);
  rightBlob = await shrink($("file-right").files[0]);
  points = [];
  $("img-left").src = URL.createObjectURL(leftBlob);
});
$("file-right").addEventListener("change", () => $("file-left").dispatchEvent(new Event("change")));

$("btn-measure").addEventListener("click", async () => {
  if (!leftBlob || !rightBlob) {
    msg("Choose both still frames.", "error");
    return;
  }
  const form = new FormData();
  form.set("left", leftBlob, "left.jpg");
  form.set("right", rightBlob, "right.jpg");
  form.set("baseline", $("baseline").value);
  form.set("unit", $("unit").value);
  if (points.length) form.set("points", JSON.stringify(points));
  msg("Matching and measuring…");
  const res = await fetch("/api/m7/room", { method: "POST", body: form });
  const body = await res.json();
  if (!res.ok) {
    msg(typeof body.detail === "string" ? body.detail : "Failed.", "error");
    return;
  }
  fill(body);
  msg(`Pose uses ${body.inliers} inliers from ${body.matches} matches.`, "ok");
});

window.addEventListener("resize", paint);
$("btn-box").click();
