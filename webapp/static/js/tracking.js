import { $, api, fmt, setMsg } from "/static/js/api.js";
import { PointPicker } from "/static/js/picker.js";

const picker1 = new PointPicker($("#wrap1"));
const picker2 = new PointPicker($("#wrap2"));
picker1.addLayer("corners", { color: "#2952cc", label: "corner" });
picker2.addLayer("truth", { color: "#2952cc", label: "truth" });

let frames = null;
let last = null;

function mean(values) {
  const ok = (values || []).filter((v) => v !== null && Number.isFinite(v));
  if (!ok.length) return null;
  return ok.reduce((s, v) => s + v, 0) / ok.length;
}

function render(data) {
  last = data;
  const rows = data.points.map((_, i) => `<tr>
    <td>${i}</td>
    <td class="num">${fmt(data.scratch_error && data.scratch_error[i], 3)}</td>
    <td class="num">${fmt(data.opencv_error && data.opencv_error[i], 3)}</td>
    <td class="num">${fmt(data.template_error && data.template_error[i], 3)}</td>
    <td class="num">${fmt(data.eigenvalues[i][0], 1)}</td>
    <td class="num">${fmt(data.eigenvalues[i][1], 1)}</td>
  </tr>`);
  $("#rows").innerHTML = rows.join("");
  const parts = [
    ["from-scratch", mean(data.scratch_error)],
    ["OpenCV", mean(data.opencv_error)],
    ["template", mean(data.template_error)],
  ].filter(([, v]) => v !== null).map(([name, v]) => `${name} mean error ${v.toFixed(3)} px`);
  $("#summary").textContent = parts.length
    ? parts.join(" · ")
    : "Click the true locations in frame 2 (same order) and press Predict again.";
  $("#workout").textContent = data.example_text;
  picker2.setOverlay("scratch", { type: "dots", color: "#e5484d", points: data.scratch, radius: 4 });
  picker2.setOverlay("opencv", { type: "dots", color: "#f5a524", points: data.opencv, radius: 3 });
  try {
    localStorage.setItem("m6.tracking", JSON.stringify({
      summary: $("#summary").textContent,
      example_text: data.example_text,
      scratch_error: data.scratch_error,
      opencv_error: data.opencv_error,
      template_error: data.template_error,
    }));
  } catch { /* ignore quota */ }
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Could not read that image."));
    img.src = src;
  });
}

async function shrinkToDataUrl(source, maxEdge) {
  const objectUrl = typeof source === "string" ? null : URL.createObjectURL(source);
  try {
    const img = await loadImage(objectUrl || source);
    const scale = Math.min(1, maxEdge / Math.max(img.naturalWidth, img.naturalHeight));
    const width = Math.max(2, Math.round(img.naturalWidth * scale));
    const height = Math.max(2, Math.round(img.naturalHeight * scale));
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    canvas.getContext("2d").drawImage(img, 0, 0, width, height);
    return canvas.toDataURL("image/jpeg", 0.85);
  } finally {
    if (objectUrl) URL.revokeObjectURL(objectUrl);
  }
}

function grabVideoFrames(file, timeSec) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const video = document.createElement("video");
    video.preload = "auto";
    video.muted = true;
    video.playsInline = true;
    video.src = url;
    const fail = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Could not open that video in the browser."));
    };
    video.onerror = fail;
    video.onloadedmetadata = async () => {
      const scale = Math.min(1, 1280 / Math.max(video.videoWidth, video.videoHeight));
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(2, Math.round(video.videoWidth * scale));
      canvas.height = Math.max(2, Math.round(video.videoHeight * scale));
      const ctx = canvas.getContext("2d");
      const seek = (t) => new Promise((res) => {
        const onSeek = () => {
          video.removeEventListener("seeked", onSeek);
          res();
        };
        video.addEventListener("seeked", onSeek);
        const end = Number.isFinite(video.duration) ? Math.max(0, video.duration - 0.05) : t;
        video.currentTime = Math.min(Math.max(0, t), end);
      });
      try {
        await seek(timeSec);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const frame1 = canvas.toDataURL("image/jpeg", 0.85);
        await seek(timeSec + 1 / 30);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const frame2 = canvas.toDataURL("image/jpeg", 0.85);
        resolve({ frame1, frame2 });
      } catch (err) {
        reject(err);
      } finally {
        URL.revokeObjectURL(url);
      }
    };
  });
}

async function predict() {
  const points = picker1.points("corners");
  if (!frames || points.length < 1) {
    setMsg($("#msg"), "Click at least one corner in frame 1.", "error");
    return;
  }
  const truth = picker2.points("truth");
  setMsg($("#msg"), "Solving the normal equations…");
  const body = {
    points,
    ground_truth: truth.length === points.length ? truth : null,
    example_index: 0,
  };
  if (frames.pair) body.pair = frames.pair;
  else {
    body.frame1_data = frames.frame1;
    body.frame2_data = frames.frame2;
  }
  try {
    const data = await api("/api/m6/track", { json: body });
    render(data);
    setMsg($("#msg"), `Tracked ${points.length} point${points.length === 1 ? "" : "s"}.`, "ok");
  } catch (err) {
    setMsg($("#msg"), err.message, "error");
  }
}

async function showPair(frame1, frame2, pairName) {
  frames = { frame1, frame2, pair: pairName || null };
  await picker1.setImage(frame1);
  await picker2.setImage(frame2);
  $("#predict").disabled = false;
}

let savedPairs = [];

$("#pair").addEventListener("change", async () => {
  const pair = savedPairs.find((item) => item.id === $("#pair").value);
  if (!pair || !pair.frame1 || !pair.frame2) return;
  setMsg($("#msg"), "Loading the saved pair…");
  try {
    await showPair(pair.frame1, pair.frame2, pair.id);
    setMsg($("#msg"), "Click corners in frame 1, then Predict.", "ok");
  } catch (err) {
    setMsg($("#msg"), err.message, "error");
  }
});

api("/api/m6/results").then((catalog) => {
  savedPairs = catalog.tracking || [];
  const select = $("#pair");
  const options = [`<option value="">Choose a saved pair</option>`].concat(
    savedPairs.filter((item) => item.frame1 && item.frame2).map((item) =>
      `<option value="${item.id}">${item.label || item.id}</option>`
    )
  );
  select.innerHTML = options.join("");
}).catch(() => {});

$("#load").addEventListener("click", async () => {
  const video = $("#video").files[0];
  const a = $("#img-a").files[0];
  const b = $("#img-b").files[0];
  setMsg($("#msg"), "Preparing frames in the browser…");
  try {
    let frame1;
    let frame2;
    if (a && b) {
      frame1 = await shrinkToDataUrl(a, 1280);
      frame2 = await shrinkToDataUrl(b, 1280);
    } else if (video) {
      const grabbed = await grabVideoFrames(video, Number($("#time").value || 0));
      frame1 = grabbed.frame1;
      frame2 = grabbed.frame2;
    } else {
      setMsg($("#msg"), "Choose a saved pair, a video, or both frame images.", "error");
      return;
    }
    await showPair(frame1, frame2, null);
    setMsg($("#msg"), "Click corners in frame 1, then Predict.", "ok");
  } catch (err) {
    setMsg($("#msg"), err.message, "error");
  }
});

$("#predict").addEventListener("click", predict);
$("#undo").addEventListener("click", () => {
  picker1.undo();
  picker2.undo();
});
