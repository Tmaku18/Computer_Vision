// Photo + reference plane + camera distance controls shared by Step 2 and Step 3.

import { $, $$, UNITS, api, fmt, getActiveCalibration, setMsg, store } from "./api.js";
import { PointPicker } from "./picker.js";

export class Workspace {
  constructor({ canvasContainer, objectLayer, onUpdate }) {
    this.onUpdate = onUpdate || (() => {});
    this.image = null;
    this.reference = null;
    this.calibration = null;
    this.msg = $("#ws-msg");

    this.picker = new PointPicker(canvasContainer, { onChange: (layer) => this.layerChanged(layer) });
    this.picker.addLayer("obj", { color: "#ff5a5f", ...objectLayer });
    this.picker.addLayer("ref", { color: "#22c55e", max: 4, closed: true });
    this.picker.setActive("obj");

    for (const sel of ["#ref-unit", "#dist-unit"]) {
      $(sel).innerHTML = UNITS.map((u) => `<option value="${u}">${u}</option>`).join("");
    }
    $("#dist-unit").value = "m";

    this.bindPhoto();
    this.bindReference();
    this.bindDistance();
    this.loadCalibration();
  }

  get unit() {
    return this.reference ? this.reference.unit : $("#ref-unit").value;
  }

  get distance() {
    const value = parseFloat($("#dist-value").value);
    return value > 0 ? { value, unit: $("#dist-unit").value } : null;
  }

  message(text, kind = "") {
    setMsg(this.msg, text, kind);
  }

  async loadCalibration() {
    try {
      this.calibration = await getActiveCalibration();
    } catch (err) {
      this.message(`Could not load calibration: ${err.message}`, "error");
    }
  }

  // ----- photo ---------------------------------------------------------

  bindPhoto() {
    const input = $("#photo-input");
    const drop = $("#photo-drop");
    input.addEventListener("change", () => input.files[0] && this.uploadPhoto(input.files[0]));
    drop.addEventListener("dragover", (e) => {
      e.preventDefault();
      drop.classList.add("drag");
    });
    drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
    drop.addEventListener("drop", (e) => {
      e.preventDefault();
      drop.classList.remove("drag");
      if (e.dataTransfer.files[0]) this.uploadPhoto(e.dataTransfer.files[0]);
    });
  }

  async uploadPhoto(file) {
    this.message(`Uploading ${file.name}…`);
    const form = new FormData();
    form.append("file", file);
    try {
      const meta = await api("/api/images", { form });
      await this.picker.setImage(meta.url);
      this.image = meta;
      this.clearReference();
      $("#photo-name").textContent = `${meta.name} · ${meta.width} × ${meta.height}`;
      this.showValidRegion();
      this.message("Photo loaded. Set a reference plane, then click points on the object.", "ok");
      this.onUpdate();
    } catch (err) {
      this.message(err.message, "error");
    }
  }

  showValidRegion() {
    const c = this.calibration;
    if (!c || !this.image) return;
    const [rows, cols] = c.calibration.image_size;
    const s = this.image.width / cols;
    if (Math.abs(this.image.height / rows - s) > 0.01) {
      this.message(
        `This photo is ${this.image.width}×${this.image.height} but the calibration is ${cols}×${rows}. ` +
          "Use the same orientation as the calibration photos.",
        "error",
      );
      return;
    }
    const k = c.calibration.camera_matrix;
    this.picker.setOverlay("valid", {
      type: "circle", cx: k[0][2] * s, cy: k[1][2] * s, r: c.valid_radius_px * s, color: "rgba(255, 196, 0, 0.8)",
    });
  }

  // ----- reference -----------------------------------------------------

  bindReference() {
    $$(".tabs button").forEach((btn) =>
      btn.addEventListener("click", () => {
        $$(".tabs button").forEach((b) => b.classList.toggle("active", b === btn));
        $$(".tab-panel").forEach((p) => (p.hidden = p.id !== btn.dataset.tab));
      }),
    );
    $("#btn-detect-board").addEventListener("click", () => this.detectBoard());
    $("#btn-rect-start").addEventListener("click", () => {
      if (!this.requireImage()) return;
      this.picker.clear("ref");
      this.picker.setActive("ref");
      this.message("Click the rectangle corners: top-left, top-right, bottom-right, bottom-left.");
    });
    $("#btn-set-rect").addEventListener("click", () => this.setRectangle());
    $("#ref-unit").addEventListener("change", () => {
      if (this.reference) {
        this.clearReference();
        this.message("Unit changed: set the reference again.", "warn");
      }
      this.onUpdate();
    });
  }

  requireImage() {
    if (!this.image) {
      this.message("Upload a photo first.", "error");
      return false;
    }
    return true;
  }

  layerChanged(layer) {
    if (layer === "ref") {
      const n = this.picker.points("ref").length;
      $("#btn-set-rect").disabled = n !== 4;
      if (n === 4) {
        this.picker.setActive("obj");
        this.message("4 corners placed (drag to adjust). Press “Use rectangle”.");
      }
      return;
    }
    this.onUpdate();
  }

  async detectBoard() {
    if (!this.requireImage()) return;
    this.message("Detecting checkerboard…");
    try {
      const ref = await api("/api/reference/checkerboard", {
        json: {
          image_id: this.image.id,
          calibration_id: store.calibrationId,
          pattern: [parseInt($("#board-cols").value, 10), parseInt($("#board-rows").value, 10)],
          square_size: parseFloat($("#board-square").value),
          unit: $("#ref-unit").value,
        },
      });
      this.picker.clear("ref");
      this.setReference(ref);
    } catch (err) {
      this.message(err.message, "error");
    }
  }

  async setRectangle() {
    try {
      const ref = await api("/api/reference/rectangle", {
        json: {
          image_id: this.image.id,
          calibration_id: store.calibrationId,
          corners: this.picker.points("ref"),
          width: parseFloat($("#rect-w").value),
          height: parseFloat($("#rect-h").value),
          unit: $("#ref-unit").value,
        },
      });
      this.setReference(ref);
    } catch (err) {
      this.message(err.message, "error");
    }
  }

  setReference(ref) {
    this.reference = ref;
    const pts = ref.image_points;
    if (ref.kind === "checkerboard") {
      const cols = parseInt($("#board-cols").value, 10);
      const rows = pts.length / cols;
      const outline = [pts[0], pts[cols - 1], pts[pts.length - 1], pts[(rows - 1) * cols]];
      this.picker.setOverlay("refdots", { type: "dots", points: pts, color: "#22c55e", radius: 2.5 });
      this.picker.setOverlay("refpoly", { type: "polygon", points: outline, closed: true, color: "#22c55e" });
    } else {
      this.picker.setOverlay("refdots", null);
      this.picker.setOverlay("refpoly", null);
    }
    const dist = ref.distance_to_plane_m;
    $("#ref-info").innerHTML = `
      <dt>Reference</dt><dd>${ref.kind === "checkerboard" ? `checkerboard (${pts.length} corners)` : "rectangle (4 corners)"}</dd>
      <dt>Reprojection</dt><dd>${fmt(ref.reproj_rms_px, 2)} px RMS</dd>
      <dt>Distance to plane</dt><dd>${dist ? `${fmt(dist, 3)} m (from pose)` : "—"}</dd>
      <dt>Plane tilt</dt><dd>${ref.tilt_deg != null ? `${fmt(ref.tilt_deg, 1)}° from fronto-parallel` : "—"}</dd>`;
    $("#btn-dist-from-pnp").disabled = !dist;
    this.picker.setActive("obj");
    this.message("Reference plane set. Click points on the object (drag points to fine-tune).", "ok");
    this.onUpdate();
  }

  clearReference() {
    this.reference = null;
    this.picker.clear("ref");
    this.picker.setOverlay("refdots", null);
    this.picker.setOverlay("refpoly", null);
    $("#ref-info").innerHTML = "";
    $("#btn-dist-from-pnp").disabled = true;
    $("#btn-set-rect").disabled = true;
  }

  // ----- distance ------------------------------------------------------

  bindDistance() {
    for (const sel of ["#dist-value", "#dist-unit"]) $(sel).addEventListener("change", () => this.onUpdate());
    $("#btn-dist-from-pnp").addEventListener("click", () => {
      if (!this.reference) return;
      $("#dist-value").value = this.reference.distance_to_plane_m.toFixed(3);
      $("#dist-unit").value = "m";
      this.onUpdate();
    });
  }

  // ----- measurement ---------------------------------------------------

  async measure(points, closed) {
    return api("/api/measure", {
      json: {
        image_id: this.image.id,
        calibration_id: store.calibrationId,
        points,
        closed,
        unit: this.unit,
        homography: this.reference ? this.reference.homography : null,
        distance: this.distance,
      },
    });
  }
}
