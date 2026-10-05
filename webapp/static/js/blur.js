// Module 3 blur page: upload, run both methods, draw the timing chart.

import { api, setMsg } from "./api.js";

const $ = (id) => document.getElementById(id);
let imageId = null;
let chart = null;

function kernelFields() {
  const name = $("kernel").value;
  $("field-sigma").hidden = name !== "gaussian";
  $("field-angle").hidden = name !== "motion";
}
$("kernel").addEventListener("change", kernelFields);
kernelFields();

$("drop").addEventListener("dragover", (e) => { e.preventDefault(); $("drop").classList.add("drag"); });
$("drop").addEventListener("dragleave", () => $("drop").classList.remove("drag"));
$("drop").addEventListener("drop", (e) => {
  e.preventDefault();
  $("drop").classList.remove("drag");
  if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]);
});
$("photo").addEventListener("change", () => {
  if ($("photo").files[0]) upload($("photo").files[0]);
});

async function upload(file) {
  setMsg($("msg"), `Uploading ${file.name}…`);
  try {
    const body = new FormData();
    body.append("file", file);
    const meta = await api("/api/images", { form: body });
    imageId = meta.id;
    $("photo-name").textContent = `${meta.name} · ${meta.width} × ${meta.height}`;
    setMsg($("msg"), "Uploaded. Press Blur both ways.", "ok");
  } catch (err) {
    setMsg($("msg"), err.message, "error");
  }
}

function show(id, url) {
  const img = $(id);
  img.src = url || "";
  img.style.visibility = url ? "visible" : "hidden";
}

function metricsHtml(body) {
  const m = body.metrics;
  const psnr = Number.isFinite(m.psnr_db) ? m.psnr_db.toFixed(1) + " dB" : "∞";
  const cells = [
    [m.max_abs.toExponential(2), "max |difference|"],
    [m.rmse.toExponential(2), "RMSE"],
    [psnr, "PSNR"],
    [body.timing_s.spatial.toFixed(2) + " s", "spatial time"],
    [body.timing_s.fft.toFixed(2) + " s", "FFT time"],
  ];
  $("metrics").innerHTML = cells.map(([v, l]) => `<div class="stat"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");
  const matched = m.max_abs < 1e-6;
  $("border-note").textContent = matched
    ? `Match. Mode “${body.mode}” uses ${body.boundary} borders, so the theorem applies. ` +
      `The difference map is black because the largest gap is ${m.max_abs.toExponential(2)} (floating-point rounding). ` +
      `Processed at ${body.width}×${body.height}.`
    : `Largest gap is ${m.max_abs.toExponential(2)}. That is bigger than rounding, so the borders do not match the theorem.`;
}

$("btn-blur").addEventListener("click", async () => {
  $("btn-blur").disabled = true;
  setMsg($("msg"), "Blurring… the spatial sum is the slow one for a large kernel.");
  try {
    const body = await api("/api/m3/blur", {
      json: {
        image_id: imageId,
        kernel: $("kernel").value,
        size: Number($("size").value),
        sigma: Number($("sigma").value),
        angle: Number($("angle").value),
        mode: $("mode").value,
      },
    });
    show("img-original", body.original_url);
    show("img-spatial", body.spatial_url);
    show("img-fft", body.fft_url);
    show("img-difference", body.difference_url);
    show("img-spec-image", body.spectrum_image_url);
    show("img-spec-kernel", body.spectrum_kernel_url);
    show("img-spec-product", body.spectrum_product_url);
    show("img-kernel", body.kernel_url);
    metricsHtml(body);
    setMsg($("msg"), "Done. Loading the timing comparison…", "ok");
    const curve = await api("/api/m3/timing", { json: { image_id: imageId } });
    drawTiming(curve.timing);
    setMsg($("msg"), "Done.", "ok");
  } catch (err) {
    setMsg($("msg"), err.message, "error");
  } finally {
    $("btn-blur").disabled = false;
  }
});

function drawTiming(rows) {
  const ctx = $("chart-timing");
  if (chart) chart.destroy();
  chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: rows.map((r) => `${r.size}×${r.size}`),
      datasets: [
        { label: "Spatial sum (s)", data: rows.map((r) => r.spatial_s), backgroundColor: "#2952cc" },
        { label: "FFT multiply (s)", data: rows.map((r) => r.fft_s), backgroundColor: "#13a34a" },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { position: "bottom" } },
      scales: { y: { beginAtZero: true, title: { display: true, text: "seconds" } } },
    },
  });
}

$("btn-blur").click();
