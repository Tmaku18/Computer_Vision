import { $, api, setMsg } from "/static/js/api.js";

const form = $("#flow-form");
const fileInput = $("#video");
const label = $("#file-label");

fileInput.addEventListener("change", () => {
  label.textContent = fileInput.files[0] ? fileInput.files[0].name : "Drop a video, or click to choose one";
});

function drawSpeed(values) {
  const canvas = $("#speed-chart");
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!values || !values.length) return;
  const max = Math.max(...values, 0.1);
  ctx.strokeStyle = "#2952cc";
  ctx.lineWidth = 2;
  ctx.beginPath();
  values.forEach((v, i) => {
    const x = 20 + (i / Math.max(values.length - 1, 1)) * (w - 30);
    const y = h - 20 - (v / max) * (h - 40);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = "#5e667a";
  ctx.font = "12px sans-serif";
  ctx.fillText(`0 to ${max.toFixed(2)} px / frame`, 20, 16);
}

function drawHist(hist) {
  const canvas = $("#dir-chart");
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!hist) return;
  const counts = hist.counts;
  const max = Math.max(...counts, 1);
  const bar = (w - 20) / counts.length;
  counts.forEach((c, i) => {
    const bh = (c / max) * (h - 36);
    ctx.fillStyle = `hsl(${(i / counts.length) * 360} 70% 45%)`;
    ctx.fillRect(10 + i * bar, h - 20 - bh, bar - 2, bh);
  });
  ctx.fillStyle = "#5e667a";
  ctx.font = "12px sans-serif";
  ctx.fillText("direction, −180° to 180°", 10, 16);
}

function show(result) {
  const title = result.label ? `${result.label}. ` : "";
  $("#interpretation").textContent = title + (result.interpretation || "");
  $("#flow-note").textContent = result.note || "";
  const ego = result.ego_motion || {};
  const foe = result.focus_of_expansion || {};
  const parallax = result.parallax || {};
  $("#stats").innerHTML = [
    ["Mean speed", `${result.mean_speed_average.toFixed(2)} px`],
    ["Moving fraction", `${(result.moving_fraction_mean * 100).toFixed(1)}%`],
    ["Median flow", `${(ego.median_speed || 0).toFixed(2)} px`],
    ["Homography inliers", `${((ego.homography_inlier_ratio || 0) * 100).toFixed(0)}%`],
    ["Focus of expansion", foe.x == null ? "—" : `${foe.x.toFixed(0)}, ${foe.y.toFixed(0)}`],
    ["Fast / slow speed", parallax.ratio == null ? "—" : parallax.ratio.toFixed(2)],
  ].map(([l, v]) => `<div class="stat"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");

  const grid = $("#video-grid");
  const files = result.files || {};
  const cards = [
    ["Original", files.original],
    ["Color wheel", files.hsv],
    ["Arrows", files.arrows],
    ["Sparse tracks", files.tracks],
    ["Side by side", files.side_by_side],
  ];
  grid.innerHTML = cards.filter(([, src]) => src).map(([title, src]) =>
    `<div><h3>${title}</h3><video class="frame" controls playsinline src="${src}" style="width:100%;background:#000;border-radius:8px"></video></div>`
  ).join("");
  $("#videos").hidden = false;
  if (files.mask) {
    const mask = $("#mask");
    mask.src = files.mask;
    mask.hidden = false;
  }
  drawSpeed(result.mean_speed);
  drawHist(result.direction_histogram);
  if (!result.id) try {
    const slim = { ...result, mean_speed: result.mean_speed };
    localStorage.setItem("m6.flow", JSON.stringify(slim));
  } catch {
    /* a long clip can exceed the browser quota; the page itself still shows it */
  }
}

async function poll(jobId) {
  const job = await api(`/api/m6/jobs/${jobId}`);
  $("#progress").style.width = `${Math.round((job.progress || 0) * 100)}%`;
  if (job.status === "done") {
    setMsg($("#msg"), "Done.", "ok");
    show(job.result);
    $("#go").disabled = false;
    return;
  }
  if (job.status === "error") {
    setMsg($("#msg"), job.error || "Processing failed.", "error");
    $("#go").disabled = false;
    return;
  }
  setMsg($("#msg"), job.message || "Working…");
  setTimeout(() => poll(jobId), 1200);
}

let clips = [];

function fillClips(list) {
  clips = list || [];
  const select = $("#clip");
  if (!clips.length) {
    select.innerHTML = `<option value="">No saved clip yet</option>`;
    return;
  }
  select.innerHTML = clips.map((clip, i) =>
    `<option value="${i}">${clip.label || clip.id}</option>`
  ).join("");
  show(clips[0]);
}

$("#clip").addEventListener("change", () => {
  const clip = clips[Number($("#clip").value)];
  if (clip) show(clip);
});

api("/api/m6/results").then((catalog) => fillClips(catalog.clips)).catch(() => fillClips([]));

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!fileInput.files[0]) {
    setMsg($("#msg"), "Choose a video first.", "error");
    return;
  }
  const body = new FormData();
  body.append("file", fileInput.files[0]);
  body.append("start", $("#start").value || "0");
  body.append("duration", $("#duration").value || "30");
  $("#go").disabled = true;
  $("#progress-wrap").hidden = false;
  setMsg($("#msg"), "Uploading…");
  try {
    const started = await api("/api/m6/flow", { form: body });
    poll(started.job_id);
  } catch (err) {
    setMsg($("#msg"), err.message, "error");
    $("#go").disabled = false;
  }
});
