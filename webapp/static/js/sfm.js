import { $, $$, api, fmt, matrixText, setMsg, store } from "/static/js/api.js";
import { PointPicker } from "/static/js/picker.js";

const picker = new PointPicker($("#wrap"));
picker.addLayer("boundary", { color: "#e5484d", closed: true, label: "boundary" });

const photos = [null, null, null, null];
const clicks = [[], [], [], []];
let active = 0;

function refreshCount() {
  const n = picker.points("boundary").length;
  $("#count").textContent = `${n} point${n === 1 ? "" : "s"} in view ${active + 1}`;
}
picker.onChange = refreshCount;

async function show(index) {
  clicks[active] = picker.points("boundary");
  active = index;
  $$("#view-tabs button").forEach((button) => {
    button.classList.toggle("active", Number(button.dataset.view) === index);
  });
  const photo = photos[index];
  $("#photo-label").textContent = photo ? photo.name : "Choose the photo for this view";
  if (!photo) {
    refreshCount();
    return;
  }
  await picker.setImage(photo.url);
  picker.setPoints("boundary", clicks[index] || []);
  refreshCount();
}

$$("#view-tabs button").forEach((button) => {
  button.addEventListener("click", () => show(Number(button.dataset.view)));
});

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Could not read that image."));
    img.src = src;
  });
}

async function shrinkFile(file, maxEdge) {
  const objectUrl = URL.createObjectURL(file);
  try {
    const img = await loadImage(objectUrl);
    const scale = Math.min(1, maxEdge / Math.max(img.naturalWidth, img.naturalHeight));
    const width = Math.max(2, Math.round(img.naturalWidth * scale));
    const height = Math.max(2, Math.round(img.naturalHeight * scale));
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    canvas.getContext("2d").drawImage(img, 0, 0, width, height);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
    const shrunk = new File([blob], file.name.replace(/\.\w+$/, "") + ".jpg", { type: "image/jpeg" });
    return { url: URL.createObjectURL(shrunk), name: file.name, file: shrunk };
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

$("#photo").addEventListener("change", async () => {
  const file = $("#photo").files[0];
  if (!file) return;
  setMsg($("#msg"), "Preparing the photo…");
  try {
    const photo = await shrinkFile(file, 1600);
    photos[active] = photo;
    $("#photo-label").textContent = file.name;
    clicks[active] = [];
    await picker.setImage(photo.url);
    refreshCount();
    setMsg($("#msg"), `View ${active + 1} ready. Click the boundary, then switch views.`, "ok");
  } catch (err) {
    setMsg($("#msg"), err.message, "error");
  }
});

function parseCameras(text) {
  const rows = text.split("\n").map((line) => line.trim()).filter(Boolean);
  if (!rows.length) return null;
  return rows.map((line, index) => {
    const [x, y, z] = line.split(/[\s,]+/).map(Number);
    if (![x, y, z].every(Number.isFinite)) {
      throw new Error(`Camera line ${index + 1} needs three numbers: x y z`);
    }
    return { view: index + 1, x_m: x, y_m: y, z_m: z };
  });
}

$("#load-example").addEventListener("click", async () => {
  setMsg($("#msg"), "Loading the committed example…");
  try {
    const catalog = await api("/api/m6/results");
    const sfm = catalog.sfm;
    if (!sfm || !sfm.result) {
      setMsg($("#msg"), "No committed reconstruction yet.", "error");
      return;
    }
    const views = sfm.views || [];
    const savedClicks = sfm.result.clicks || [[], [], [], []];
    for (let i = 0; i < 4; i++) {
      if (!views[i]) continue;
      photos[i] = { url: views[i], name: `view ${i + 1}`, file: null };
      clicks[i] = savedClicks[i] || [];
    }
    await show(0);
    for (let i = 0; i < 4; i++) clicks[i] = savedClicks[i] || [];
    picker.setPoints("boundary", clicks[0] || []);
    refreshCount();
    if (Array.isArray(sfm.result.camera_positions)) {
      $("#cameras").value = sfm.result.camera_positions.map((cam) =>
        [cam.x_m, cam.y_m, cam.z_m].map((v) => Number(v).toFixed(3)).join(" ")
      ).join("\n");
    }
    showResult(sfm.result);
    setMsg($("#msg"), "Loaded the committed reconstruction. Uploading new photos still runs only from this browser.", "ok");
  } catch (err) {
    setMsg($("#msg"), err.message, "error");
  }
});

$("#undo").addEventListener("click", () => picker.undo());

function cameraTraces(result) {
  const pts = result.boundary_points;
  const closed = pts.concat([pts[0]]);
  const traces = [
    {
      type: "scatter3d",
      mode: "lines+markers",
      name: "boundary",
      x: closed.map((p) => p[0]),
      y: closed.map((p) => p[1]),
      z: closed.map((p) => p[2]),
      line: { color: "#e5484d", width: 4 },
      marker: { size: 3 },
    },
  ];
  result.cameras.forEach((cam, i) => {
    const c = cam.center;
    const r = cam.R;
    // R maps world to camera, so a camera-frame axis appears in the world as a column of Rᵀ.
    const axis = 0.15;
    const rt = [0, 1, 2].map((row) => [0, 1, 2].map((col) => r[col][row]));
    const colors = ["#e5484d", "#13a34a", "#2952cc"];
    colors.forEach((color, k) => {
      const end = [c[0] + axis * rt[0][k], c[1] + axis * rt[1][k], c[2] + axis * rt[2][k]];
      traces.push({
        type: "scatter3d",
        mode: "lines",
        name: k === 2 ? `view ${i + 1}` : "",
        showlegend: k === 2,
        x: [c[0], end[0]],
        y: [c[1], end[1]],
        z: [c[2], end[2]],
        line: { color, width: 6 },
      });
    });
  });
  return traces;
}

function showResult(result) {
  $("#k-text").textContent = matrixText(result.K, 2);
  const poly = result.polygon_unit;
  const items = poly.edge_lengths.map((length, i) =>
    `<div class="stat"><div class="v">${fmt(length, 3)}</div><div class="l">edge ${i} (${poly.unit})</div></div>`
  );
  items.push(`<div class="stat"><div class="v">${fmt(poly.area, 3)}</div><div class="l">area (${poly.unit}²)</div></div>`);
  if (result.check) {
    items.push(`<div class="stat"><div class="v">${fmt(result.check.error, 3)}</div><div class="l">reconstruction minus tape (${result.check.unit})</div></div>`);
  }
  $("#edges").innerHTML = `<div class="stat-grid">${items.join("")}</div>`;
  $("#poses").innerHTML = result.cameras.map((cam, i) => `<tr>
    <td>${i + 1}</td>
    <td class="num">${fmt(cam.center[0], 3)}</td>
    <td class="num">${fmt(cam.center[1], 3)}</td>
    <td class="num">${fmt(cam.center[2], 3)}</td>
    <td class="num">${fmt(result.reproj_rms_after[i], 3)}</td>
  </tr>`).join("");
  $("#workout").textContent = result.workout;
  $("#view3d-card").hidden = false;
  if (window.Plotly) {
    Plotly.newPlot($("#plot"), cameraTraces(result), {
      margin: { l: 0, r: 0, t: 0, b: 0 },
      scene: { aspectmode: "data", xaxis: { title: "X (m)" }, yaxis: { title: "Y (m)" }, zaxis: { title: "Z (m)" } },
    }, { responsive: true });
  }
  try {
    const slim = {
      K: result.K,
      dist_coeffs: result.dist_coeffs,
      cameras: result.cameras,
      boundary_points: result.boundary_points,
      polygon_unit: result.polygon_unit,
      reproj_rms_after: result.reproj_rms_after,
      workout: result.workout,
      check: result.check || null,
      image_size: result.image_size,
      n_sift_tracks: result.n_sift_tracks,
      n_boundary: result.n_boundary,
      camera_positions: result.camera_positions || null,
      note: result.note || null,
      views: photos.map((photo) => (photo && photo.url && photo.url.startsWith("data:") ? null : photo && photo.url) || null),
    };
    localStorage.setItem("m6.sfm", JSON.stringify(slim));
  } catch { /* ignore quota */ }
}

$("#run").addEventListener("click", async () => {
  clicks[active] = picker.points("boundary");
  if (photos.some((photo) => !photo)) {
    setMsg($("#msg"), "Upload all four views.", "error");
    return;
  }
  const count = clicks[0].length;
  if (count < 3 || clicks.some((view) => view.length !== count)) {
    setMsg($("#msg"), "Click the same number of points (at least 3) in every view.", "error");
    return;
  }
  if (photos.some((photo) => !photo.file)) {
    setMsg($("#msg"), "Upload all four views. The committed example is for viewing; a new reconstruction needs the photo files.", "error");
    return;
  }
  let cameraPositions = null;
  try {
    cameraPositions = parseCameras($("#cameras").value || "");
  } catch (err) {
    setMsg($("#msg"), err.message, "error");
    return;
  }
  const body = new FormData();
  photos.forEach((photo) => body.append("files", photo.file));
  body.append("clicks", JSON.stringify({ boundary: clicks, camera_positions: cameraPositions }));
  body.append("known_length", $("#length").value);
  body.append("unit", $("#unit").value);
  body.append("edge_i", $("#edge-i").value);
  body.append("edge_j", $("#edge-j").value);
  body.append("check_length", $("#check-length").value || "0");
  body.append("check_i", $("#check-i").value || "1");
  body.append("check_j", $("#check-j").value || "2");
  body.append("calibration_id", store.calibrationId || "default");
  setMsg($("#msg"), "Matching features and solving for the cameras…");
  $("#run").disabled = true;
  try {
    const result = await api("/api/m6/sfm", { form: body });
    showResult(result);
    setMsg($("#msg"), "Reconstruction finished.", "ok");
  } catch (err) {
    setMsg($("#msg"), err.message, "error");
  } finally {
    $("#run").disabled = false;
  }
});
