// 1-D convolution theorem, computed in the page so editing the signal updates every step.
// DFT matches numpy: F[k] = sum_n x[n] exp(-2 pi i k n / N), inverse divides by N.

import { escapeHtml } from "./api.js";

const $ = (id) => document.getElementById(id);

function parseList(text) {
  const values = text.split(/[,\s]+/).filter(Boolean).map(Number);
  if (!values.length || values.some((v) => Number.isNaN(v))) throw new Error("Enter numbers separated by commas.");
  return values;
}

function dft(x, n) {
  const out = [];
  for (let k = 0; k < n; k++) {
    let re = 0;
    let im = 0;
    for (let t = 0; t < x.length; t++) {
      const angle = (-2 * Math.PI * k * t) / n;
      re += x[t] * Math.cos(angle);
      im += x[t] * Math.sin(angle);
    }
    out.push({ re, im });
  }
  return out;
}

function idft(values) {
  const n = values.length;
  const out = [];
  for (let t = 0; t < n; t++) {
    let re = 0;
    for (let k = 0; k < n; k++) {
      const angle = (2 * Math.PI * k * t) / n;
      re += values[k].re * Math.cos(angle) - values[k].im * Math.sin(angle);
    }
    out.push(re / n);
  }
  return out;
}

function mul(a, b) {
  return { re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re };
}

function fmt(v) {
  return Math.abs(v) < 5e-12 ? "0" : v.toFixed(4);
}

function fmtC(v) {
  if (Math.abs(v.im) < 5e-10) return fmt(v.re);
  const sign = v.im < 0 ? "−" : "+";
  return `${fmt(v.re)} ${sign} ${Math.abs(v.im).toFixed(4)}j`;
}

function render() {
  const out = $("example-out");
  try {
    const f = parseList($("ex-signal").value);
    const h = parseList($("ex-kernel").value);
    if (h.length % 2 === 0) throw new Error("Kernel length must be odd, so it has a center tap.");
    if (f.length > 12 || h.length > 9) throw new Error("Keep the signal to 12 samples and the kernel to 9.");
    const n = f.length + h.length - 1;
    const steps = [];
    const full = [];
    for (let t = 0; t < n; t++) {
      const terms = [];
      let sum = 0;
      for (let k = 0; k < h.length; k++) {
        const i = t - k;
        if (i >= 0 && i < f.length) {
          terms.push(`${fmt(f[i])}×${fmt(h[k])}`);
          sum += f[i] * h[k];
        }
      }
      steps.push(`<tr><td>n = ${t}</td><td>${terms.join(" + ") || "0"}</td><td class="num">${fmt(sum)}</td></tr>`);
      full.push(sum);
    }
    const F = dft(f, n);
    const H = dft(h, n);
    const product = F.map((value, k) => mul(value, H[k]));
    const inverse = idft(product);
    const maxAbs = Math.max(...inverse.map((v, i) => Math.abs(v - full[i])));
    const rows = F.map((value, k) =>
      `<tr><td>${k}</td><td class="num">${fmtC(value)}</td><td class="num">${fmtC(H[k])}</td><td class="num">${fmtC(product[k])}</td><td class="num">${fmt(inverse[k])}</td></tr>`,
    ).join("");
    out.innerHTML = `
      <p>Spatial sum, treating samples outside f as 0. Full result:
        <strong>[${full.map(fmt).join(", ")}]</strong>.
        Same-size crop on the center tap:
        <strong>[${full.slice((h.length / 2) | 0, ((h.length / 2) | 0) + f.length).map(fmt).join(", ")}]</strong>.</p>
      <div class="table-wrap"><table>
        <thead><tr><th>n</th><th>products f[n−k] h[k]</th><th class="num">sum</th></tr></thead>
        <tbody>${steps.join("")}</tbody>
      </table></div>
      <p style="margin-top:0.8rem">DFT of length ${n}. Inverse DFT in the last column; it should copy the spatial sum.</p>
      <div class="table-wrap"><table>
        <thead><tr><th>k</th><th class="num">F[k]</th><th class="num">H[k]</th><th class="num">F[k] H[k]</th><th class="num">inverse</th></tr></thead>
        <tbody>${rows}</tbody>
      </table></div>
      <p class="${maxAbs < 1e-8 ? "msg ok" : "msg error"}" style="margin-top:0.7rem">
        max |inverse DFT − spatial sum| = ${maxAbs.toExponential(2)}
        ${maxAbs < 1e-8 ? " — the two methods agree." : " — they disagree, which should not happen."}
      </p>`;
  } catch (err) {
    out.innerHTML = `<p class="msg error">${escapeHtml(err.message)}</p>`;
  }
}

$("btn-example").addEventListener("click", render);
render();
