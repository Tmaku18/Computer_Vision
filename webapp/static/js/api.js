// Shared helpers: API calls, persisted state, formatting.

export const UNITS = ["in", "cm", "mm", "m", "ft"];

const KEYS = {
  calibration: "m2.calibrationId",
  rows: "m2.validationRows",
  report: "m2.reportMeta",
};

export const store = {
  get calibrationId() {
    return localStorage.getItem(KEYS.calibration) || "default";
  },
  set calibrationId(id) {
    localStorage.setItem(KEYS.calibration, id);
  },
  get rows() {
    try {
      return JSON.parse(localStorage.getItem(KEYS.rows) || "[]");
    } catch {
      return [];
    }
  },
  set rows(rows) {
    localStorage.setItem(KEYS.rows, JSON.stringify(rows));
  },
  get reportMeta() {
    try {
      return JSON.parse(localStorage.getItem(KEYS.report) || "{}");
    } catch {
      return {};
    }
  },
  set reportMeta(meta) {
    localStorage.setItem(KEYS.report, JSON.stringify(meta));
  },
};

export class ApiError extends Error {}

export async function api(path, { method = "GET", json, form } = {}) {
  const opts = { method, headers: {} };
  if (json !== undefined) {
    opts.method = method === "GET" ? "POST" : method;
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(json);
  } else if (form) {
    opts.method = "POST";
    opts.body = form;
  }
  const res = await fetch(path, opts);
  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { detail: text };
  }
  if (!res.ok) {
    const detail = data && data.detail;
    const msg = Array.isArray(detail)
      ? detail.map((d) => `${(d.loc || []).slice(1).join(".")}: ${d.msg}`).join("; ")
      : detail || res.statusText;
    throw new ApiError(msg);
  }
  return data;
}

// Multipart upload with progress callback (fetch has no upload progress).
export function uploadWithProgress(path, form, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", path);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      let data = null;
      try {
        data = JSON.parse(xhr.responseText);
      } catch {
        data = { detail: xhr.responseText };
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data);
      else reject(new ApiError((data && data.detail) || xhr.statusText));
    };
    xhr.onerror = () => reject(new ApiError("Network error during upload."));
    xhr.send(form);
  });
}

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Number(value).toFixed(digits);
}

export function setMsg(el, text, kind = "") {
  if (!el) return;
  el.textContent = text || "";
  el.className = `msg ${kind}`.trim();
}

export function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

export function download(filename, text, type = "text/plain") {
  const blob = new Blob([text], { type });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

export function matrixText(m, digits = 2) {
  const cells = m.map((row) => row.map((v) => Number(v).toFixed(digits)));
  const width = Math.max(...cells.flat().map((c) => c.length));
  return cells.map((row) => "[ " + row.map((c) => c.padStart(width)).join("  ") + " ]").join("\n");
}

export async function getActiveCalibration() {
  try {
    return await api(`/api/calibrations/${store.calibrationId}`);
  } catch (err) {
    if (store.calibrationId !== "default") {
      store.calibrationId = "default";
      return api("/api/calibrations/default");
    }
    throw err;
  }
}

export async function initHeader() {
  const label = document.getElementById("active-calib-label");
  if (!label) return;
  try {
    const calib = await getActiveCalibration();
    label.textContent = calib.label;
  } catch {
    label.textContent = "unavailable";
  }
}
