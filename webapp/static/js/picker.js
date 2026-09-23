// Zoomable image canvas for placing measurement points in full-resolution pixels.
//
//   wheel / pinch-trackpad   zoom around the cursor
//   drag on empty area       pan
//   click                    add a point to the active layer
//   drag a point             move it
//   loupe                    magnified view under the cursor for pixel accuracy

export class PointPicker {
  constructor(container, { onChange } = {}) {
    this.container = container;
    this.canvas = document.createElement("canvas");
    this.ctx = this.canvas.getContext("2d");
    this.placeholder = container.querySelector(".placeholder");
    container.appendChild(this.canvas);

    this.onChange = onChange || (() => {});
    this.img = null;
    this.scale = 1;
    this.ox = 0;
    this.oy = 0;
    this.layers = new Map();
    this.activeLayer = null;
    this.overlays = {};
    this.cursor = null;
    this.drag = null;

    new ResizeObserver(() => this.resize()).observe(container);
    this.canvas.addEventListener("wheel", (e) => this.onWheel(e), { passive: false });
    this.canvas.addEventListener("pointerdown", (e) => this.onDown(e));
    this.canvas.addEventListener("pointermove", (e) => this.onMove(e));
    this.canvas.addEventListener("pointerup", (e) => this.onUp(e));
    this.canvas.addEventListener("pointerleave", () => {
      this.cursor = null;
      this.draw();
    });
    this.resize();
  }

  // ----- setup ---------------------------------------------------------

  addLayer(name, { color, max = Infinity, closed = false, label = "" }) {
    this.layers.set(name, { color, max, closed, label, points: [] });
    if (!this.activeLayer) this.activeLayer = name;
  }

  setActive(name) {
    this.activeLayer = name;
    this.draw();
  }

  setLayerOption(name, key, value) {
    this.layers.get(name)[key] = value;
    this.draw();
  }

  points(name) {
    return this.layers.get(name).points.map((p) => [p[0], p[1]]);
  }

  setPoints(name, pts) {
    this.layers.get(name).points = pts.map((p) => [p[0], p[1]]);
    this.draw();
    this.onChange(name);
  }

  clear(name) {
    this.setPoints(name, []);
  }

  undo(name = this.activeLayer) {
    const layer = this.layers.get(name);
    if (layer.points.length) {
      layer.points.pop();
      this.draw();
      this.onChange(name);
    }
  }

  setOverlay(key, overlay) {
    if (overlay) this.overlays[key] = overlay;
    else delete this.overlays[key];
    this.draw();
  }

  async setImage(url) {
    const img = new Image();
    img.decoding = "async";
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = () => reject(new Error("Could not load image."));
      img.src = url;
    });
    this.img = img;
    for (const layer of this.layers.values()) layer.points = [];
    this.overlays = {};
    if (this.placeholder) this.placeholder.hidden = true;
    this.fit();
  }

  // ----- geometry ------------------------------------------------------

  resize() {
    const w = this.container.clientWidth || 800;
    const h = Math.max(360, Math.min(window.innerHeight * 0.72, w * 0.75));
    const dpr = window.devicePixelRatio || 1;
    this.cssW = w;
    this.cssH = h;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.canvas.style.height = `${h}px`;
    this.dpr = dpr;
    if (this.img && !this.userZoomed) this.fit();
    else this.draw();
  }

  fit() {
    if (!this.img) return this.draw();
    this.scale = Math.min(this.cssW / this.img.width, this.cssH / this.img.height);
    this.ox = (this.cssW - this.img.width * this.scale) / 2;
    this.oy = (this.cssH - this.img.height * this.scale) / 2;
    this.userZoomed = false;
    this.draw();
  }

  toImage(sx, sy) {
    return [(sx - this.ox) / this.scale, (sy - this.oy) / this.scale];
  }

  toScreen(x, y) {
    return [x * this.scale + this.ox, y * this.scale + this.oy];
  }

  eventPos(e) {
    const r = this.canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  hitTest(sx, sy) {
    for (const [name, layer] of this.layers) {
      for (let i = layer.points.length - 1; i >= 0; i--) {
        const [px, py] = this.toScreen(...layer.points[i]);
        if (Math.hypot(px - sx, py - sy) < 9) return { name, index: i };
      }
    }
    return null;
  }

  // ----- events --------------------------------------------------------

  onWheel(e) {
    if (!this.img) return;
    e.preventDefault();
    const [sx, sy] = this.eventPos(e);
    const [ix, iy] = this.toImage(sx, sy);
    const factor = Math.exp(-e.deltaY * 0.0015);
    const minScale = Math.min(this.cssW / this.img.width, this.cssH / this.img.height) * 0.5;
    this.scale = Math.min(20, Math.max(minScale, this.scale * factor));
    this.ox = sx - ix * this.scale;
    this.oy = sy - iy * this.scale;
    this.userZoomed = true;
    this.cursor = [sx, sy];
    this.draw();
  }

  onDown(e) {
    if (!this.img) return;
    this.canvas.setPointerCapture(e.pointerId);
    const [sx, sy] = this.eventPos(e);
    const hit = this.hitTest(sx, sy);
    this.drag = { sx, sy, ox: this.ox, oy: this.oy, hit, moved: false };
  }

  onMove(e) {
    const [sx, sy] = this.eventPos(e);
    this.cursor = [sx, sy];
    if (this.drag) {
      const dx = sx - this.drag.sx;
      const dy = sy - this.drag.sy;
      if (Math.hypot(dx, dy) > 3) this.drag.moved = true;
      if (this.drag.hit) {
        const layer = this.layers.get(this.drag.hit.name);
        layer.points[this.drag.hit.index] = this.toImage(sx, sy);
      } else if (this.drag.moved) {
        this.ox = this.drag.ox + dx;
        this.oy = this.drag.oy + dy;
        this.userZoomed = true;
      }
    }
    this.draw();
  }

  onUp(e) {
    if (!this.drag) return;
    const { hit, moved } = this.drag;
    this.drag = null;
    if (hit) {
      if (moved) this.onChange(hit.name);
      return;
    }
    if (moved || !this.activeLayer) return;
    const layer = this.layers.get(this.activeLayer);
    const [sx, sy] = this.eventPos(e);
    const [ix, iy] = this.toImage(sx, sy);
    if (ix < 0 || iy < 0 || ix > this.img.width || iy > this.img.height) return;
    if (layer.points.length >= layer.max) {
      if (layer.max === 1 || layer.max === 2) layer.points = [];
      else return;
    }
    layer.points.push([ix, iy]);
    this.draw();
    this.onChange(this.activeLayer);
  }

  // ----- drawing -------------------------------------------------------

  draw() {
    const ctx = this.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#0f1320";
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    if (!this.img) return;

    ctx.setTransform(this.dpr * this.scale, 0, 0, this.dpr * this.scale, this.dpr * this.ox, this.dpr * this.oy);
    ctx.imageSmoothingEnabled = this.scale < 2;
    ctx.drawImage(this.img, 0, 0);
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);

    for (const ov of Object.values(this.overlays)) this.drawOverlay(ov);
    for (const [name, layer] of this.layers) this.drawLayer(layer, name === this.activeLayer);
    if (this.cursor && !(this.drag && this.drag.moved && !this.drag.hit)) this.drawLoupe();
  }

  drawOverlay(ov) {
    const ctx = this.ctx;
    ctx.save();
    if (ov.type === "dots") {
      ctx.fillStyle = ov.color;
      for (const p of ov.points) {
        const [x, y] = this.toScreen(...p);
        ctx.beginPath();
        ctx.arc(x, y, ov.radius || 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
    } else if (ov.type === "polygon") {
      ctx.strokeStyle = ov.color;
      ctx.lineWidth = ov.width || 2;
      ctx.setLineDash(ov.dash || []);
      ctx.beginPath();
      ov.points.forEach((p, i) => {
        const [x, y] = this.toScreen(...p);
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      if (ov.closed) ctx.closePath();
      ctx.stroke();
    } else if (ov.type === "circle") {
      const [x, y] = this.toScreen(ov.cx, ov.cy);
      ctx.strokeStyle = ov.color;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([8, 6]);
      ctx.beginPath();
      ctx.arc(x, y, ov.r * this.scale, 0, Math.PI * 2);
      ctx.stroke();
    } else if (ov.type === "labels") {
      ctx.font = "600 12px -apple-system, Segoe UI, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      for (const item of ov.items) {
        const [x, y] = this.toScreen(item.x, item.y);
        const w = ctx.measureText(item.text).width + 10;
        ctx.fillStyle = "rgba(15, 19, 32, 0.8)";
        ctx.fillRect(x - w / 2, y - 10, w, 20);
        ctx.fillStyle = item.color || "#fff";
        ctx.fillText(item.text, x, y);
      }
    }
    ctx.restore();
  }

  drawLayer(layer, active) {
    const ctx = this.ctx;
    const pts = layer.points.map((p) => this.toScreen(...p));
    if (!pts.length) return;
    ctx.save();
    ctx.strokeStyle = layer.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    if (layer.closed && pts.length > 2) ctx.closePath();
    ctx.stroke();

    ctx.font = "700 11px -apple-system, Segoe UI, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    pts.forEach(([x, y], i) => {
      ctx.beginPath();
      ctx.arc(x, y, active ? 7 : 6, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(15, 19, 32, 0.55)";
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = layer.color;
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x - 3, y);
      ctx.lineTo(x + 3, y);
      ctx.moveTo(x, y - 3);
      ctx.lineTo(x, y + 3);
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.fillStyle = layer.color;
      ctx.fillText(String(i + 1), x + 12, y - 11);
    });
    ctx.restore();
  }

  drawLoupe() {
    const ctx = this.ctx;
    const [sx, sy] = this.cursor;
    const [ix, iy] = this.toImage(sx, sy);
    const radius = 70;
    const zoom = Math.min(10, Math.max(this.scale * 5, 1.5)); // screen px per image px
    let cx = sx + radius + 20;
    let cy = sy - radius - 20;
    if (cx + radius > this.cssW) cx = sx - radius - 20;
    if (cy - radius < 0) cy = sy + radius + 20;

    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.clip();
    ctx.fillStyle = "#000";
    ctx.fillRect(cx - radius, cy - radius, radius * 2, radius * 2);
    ctx.imageSmoothingEnabled = false;
    const src = radius / zoom;
    ctx.drawImage(this.img, ix - src, iy - src, src * 2, src * 2, cx - radius, cy - radius, radius * 2, radius * 2);

    // Existing points inside the loupe.
    for (const layer of this.layers.values()) {
      ctx.strokeStyle = layer.color;
      ctx.lineWidth = 2;
      for (const [px, py] of layer.points) {
        const lx = cx + (px - ix) * zoom;
        const ly = cy + (py - iy) * zoom;
        ctx.beginPath();
        ctx.arc(lx, ly, 5, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
    ctx.strokeStyle = "rgba(255,255,255,0.9)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - radius, cy);
    ctx.lineTo(cx + radius, cy);
    ctx.moveTo(cx, cy - radius);
    ctx.lineTo(cx, cy + radius);
    ctx.stroke();
    ctx.restore();

    ctx.save();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = "rgba(15,19,32,0.8)";
    ctx.fillRect(cx - 52, cy + radius - 22, 104, 18);
    ctx.fillStyle = "#fff";
    ctx.font = "11px ui-monospace, Menlo, monospace";
    ctx.textAlign = "center";
    ctx.fillText(`${ix.toFixed(1)}, ${iy.toFixed(1)}`, cx, cy + radius - 9);
    ctx.restore();
  }
}
