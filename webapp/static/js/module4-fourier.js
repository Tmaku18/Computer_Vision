import { $, api, setMsg, uploadWithProgress } from "/static/js/api.js";

let imageId = null;
let sample = "rgb_wall.png";

const NOTES = {
  ideal: "The ideal cutoff is a rectangle in frequency, so the edge map rings (Gibbs).",
  butterworth: "Butterworth rolls off. A higher order gets closer to the ideal cutoff, and closer to ringing.",
  gaussian: "A Gaussian high-pass has no ringing. Blur is the low-pass complement.",
  gradient: "Each axis is multiplied by j 2π u or j 2π v, then the magnitude is the edge strength.",
  laplacian: "H = −4 + 2 cos(2πu) + 2 cos(2πv). That is exactly the DFT of the 3×3 Laplacian, and about −4π²(u²+v²) at low frequency.",
  dog: "Difference of Gaussians is a band-pass. Edges are its zero crossings; the region map is the sign.",
  gabor: "A bank of oriented Gabors. The region map is Otsu on the total energy, which picks up texture.",
};

function syncFields() {
  const kind = $("#kind").value;
  $("#cutoff-fields").hidden = kind === "dog" || kind === "gabor" || kind === "laplacian" || kind === "gradient";
  $("#field-order").hidden = kind !== "butterworth";
  $("#dog-fields").hidden = kind !== "dog";
  $("#caption").textContent = NOTES[kind] || "";
}

async function loadSamples() {
  const data = await api("/api/m4/samples");
  const select = $("#sample");
  for (const item of data.samples) {
    const option = document.createElement("option");
    option.value = item.name;
    option.textContent = item.name;
    select.appendChild(option);
  }
  select.value = sample;
}

$("#sample").addEventListener("change", () => {
  sample = $("#sample").value;
  imageId = null;
  $("#photo-name").textContent = sample;
});

$("#photo").addEventListener("change", async () => {
  const file = $("#photo").files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  try {
    const meta = await uploadWithProgress("/api/m4/images", form, () => {});
    imageId = meta.id;
    sample = null;
    $("#sample").value = "";
    $("#photo-name").textContent = `${meta.name} · ${meta.width}×${meta.height}`;
  } catch (err) {
    setMsg($("#msg"), err.message, "error");
  }
});

$("#kind").addEventListener("change", syncFields);

$("#btn-run").addEventListener("click", async () => {
  $("#btn-run").disabled = true;
  setMsg($("#msg"), "Filtering…");
  try {
    const body = await api("/api/m4/fourier", {
      json: {
        image_id: imageId,
        sample,
        kind: $("#kind").value,
        cutoff: Number($("#cutoff").value),
        order: Number($("#order").value),
        sigma1: Number($("#sigma1").value),
        sigma2: Number($("#sigma2").value),
      },
    });
    $("#img-filter").src = body.filter_url;
    $("#img-spectrum").src = body.spectrum_url;
    $("#img-edges").src = body.edges_url;
    $("#img-regions").src = body.regions_url;
    setMsg($("#msg"), "Done.", "ok");
  } catch (err) {
    setMsg($("#msg"), err.message, "error");
  } finally {
    $("#btn-run").disabled = false;
  }
});

syncFields();
loadSamples()
  .then(() => $("#btn-run").click())
  .catch((err) => setMsg($("#msg"), err.message, "error"));
