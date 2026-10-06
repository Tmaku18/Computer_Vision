import { $ } from "/static/js/api.js";

function render() {
  const n = Math.max(4, Number($("#ex-n").value) || 8);
  let k = Math.max(0, Number($("#ex-k").value) || 0);
  if (k > n / 2) k = Math.floor(n / 2);
  const u = k / n;
  const discrete = -4 + 2 * Math.cos((2 * Math.PI) / n * k) + 2;
  const continuous = -4 * Math.PI ** 2 * u * u;
  const host = $("#ex-out");
  host.innerHTML = "";
  const rows = [
    ["u = k/N", u.toFixed(4)],
    ["Discrete H", discrete.toFixed(4)],
    ["Continuous −4π²u²", continuous.toFixed(4)],
    ["Gap", Math.abs(discrete - continuous).toFixed(4)],
  ];
  for (const [name, value] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = name;
    const dd = document.createElement("dd");
    dd.textContent = value;
    host.append(dt, dd);
  }
}

$("#ex-n").addEventListener("input", render);
$("#ex-k").addEventListener("input", render);
render();
