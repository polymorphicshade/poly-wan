"use strict";

// Canvas editors used by the workspace: a region selection over a video's first frame (video
// inpainting) and motion paths over the first frame (VACE layout control).

function canvasPoint(canvas, e) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (e.clientX - rect.left) * (canvas.width / rect.width),
    y: (e.clientY - rect.top) * (canvas.height / rect.height),
  };
}

function canvasScale(canvas) {
  const rect = canvas.getBoundingClientRect();
  return rect.width ? canvas.width / rect.width : 1;
}

// The same colours the server uses for each path in the control video.
function pathColor(i) {
  const h = (i * 0.31) % 1, s = 0.9, v = 1;
  const f = n => { const k = (n + h * 6) % 6; return Math.round(255 * (v - v * s * Math.max(0, Math.min(k, 4 - k, 1)))); };
  return `rgb(${f(5)}, ${f(3)}, ${f(1)})`;
}

// ----------------------------------------------------------------------------- region selection

class MaskEditor {
  constructor(ws) {
    this.ws = ws;
    this.canvas = ws.r("mask-canvas");
    this.ctx = this.canvas.getContext("2d");
    this.mask = document.createElement("canvas");
    this.tint = document.createElement("canvas");
    this.bitmap = null;
    this.tool = "lasso";
    this.undoStack = [];
    this.bind();
  }

  maskCtx() { return this.mask.getContext("2d", { willReadFrequently: true }); }

  setFrame(bitmap) {
    this.bitmap = bitmap;
    if (bitmap) for (const c of [this.canvas, this.mask, this.tint]) { c.width = bitmap.width; c.height = bitmap.height; }
    this.undoStack = [];
    this.ws.r("mask-empty").hidden = !!bitmap;
    this.canvas.hidden = !bitmap;
    this.render();
  }

  setTool(tool) {
    this.tool = tool;
    $$("[data-tool]", this.ws.el).forEach(b => b.classList.toggle("active", b.dataset.tool === tool));
    this.ws.r("brush-size-row").hidden = !["brush", "erase"].includes(tool);
  }

  pushUndo() {
    if (!this.bitmap) return;
    this.undoStack.push(this.maskCtx().getImageData(0, 0, this.mask.width, this.mask.height));
    if (this.undoStack.length > 20) this.undoStack.shift();
  }

  undo() {
    const prev = this.undoStack.pop();
    if (prev) { this.maskCtx().putImageData(prev, 0, 0); this.render(); }
  }

  clear() {
    if (!this.bitmap) return;
    this.pushUndo();
    this.maskCtx().clearRect(0, 0, this.mask.width, this.mask.height);
    this.render();
  }

  bind() {
    const ws = this.ws, c = this.canvas;
    $$("[data-tool]", ws.el).forEach(b => b.onclick = () => this.setTool(b.dataset.tool));
    ws.r("mask-undo").onclick = () => this.undo();
    ws.r("mask-clear").onclick = () => this.clear();
    let drawing = null;
    const radius = () => (parseInt(ws.r("brush-size").value, 10) / 2) * canvasScale(c);
    const dab = (from, to, erase) => {
      const ctx = this.maskCtx();
      ctx.save();
      ctx.globalCompositeOperation = erase ? "destination-out" : "source-over";
      ctx.strokeStyle = "#fff";
      ctx.lineCap = ctx.lineJoin = "round";
      ctx.lineWidth = radius() * 2;
      ctx.beginPath(); ctx.moveTo(from.x, from.y); ctx.lineTo(to.x, to.y); ctx.stroke();
      ctx.restore();
    };
    c.addEventListener("pointerdown", e => {
      if (!this.bitmap || e.button !== 0) return;
      c.setPointerCapture(e.pointerId);
      const p = canvasPoint(c, e);
      this.pushUndo();
      drawing = { tool: this.tool, points: [p], start: p, last: p, subtract: ws.r("subtract").checked };
      if (drawing.tool === "brush" || drawing.tool === "erase") dab(p, p, drawing.tool === "erase" || drawing.subtract);
      this.render(drawing);
    });
    c.addEventListener("pointermove", e => {
      if (!drawing) return;
      const p = canvasPoint(c, e);
      if (drawing.tool === "brush" || drawing.tool === "erase") dab(drawing.last, p, drawing.tool === "erase" || drawing.subtract);
      else if (drawing.tool === "lasso") drawing.points.push(p);
      drawing.last = p;
      this.render(drawing);
    });
    const finish = () => {
      if (!drawing) return;
      const ctx = this.maskCtx();
      ctx.save();
      ctx.globalCompositeOperation = drawing.subtract ? "destination-out" : "source-over";
      ctx.fillStyle = "#fff";
      if (drawing.tool === "lasso" && drawing.points.length > 2) {
        ctx.beginPath();
        drawing.points.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
        ctx.closePath(); ctx.fill();
      } else if (drawing.tool === "box") {
        const { start: a, last: b } = drawing;
        ctx.fillRect(Math.min(a.x, b.x), Math.min(a.y, b.y), Math.abs(b.x - a.x), Math.abs(b.y - a.y));
      }
      ctx.restore();
      drawing = null;
      this.render();
    };
    c.addEventListener("pointerup", finish);
    c.addEventListener("pointercancel", finish);
  }

  render(drawing = null) {
    if (!this.bitmap) return;
    const { ctx, canvas } = this;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(this.bitmap, 0, 0);
    const t = this.tint.getContext("2d");
    t.globalCompositeOperation = "source-over";
    t.clearRect(0, 0, canvas.width, canvas.height);
    t.drawImage(this.mask, 0, 0);
    t.globalCompositeOperation = "source-in";
    t.fillStyle = "rgb(255, 40, 80)";
    t.fillRect(0, 0, canvas.width, canvas.height);
    ctx.globalAlpha = 0.45;
    ctx.drawImage(this.tint, 0, 0);
    ctx.globalAlpha = 1;
    if (drawing && (drawing.tool === "lasso" || drawing.tool === "box")) {
      const s = canvasScale(canvas);
      ctx.save();
      ctx.lineWidth = Math.max(2, 2 * s);
      ctx.setLineDash([8 * s, 6 * s]);
      ctx.strokeStyle = drawing.subtract ? "#00d0ff" : "#ff2850";
      ctx.beginPath();
      if (drawing.tool === "lasso") drawing.points.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
      else {
        const { start: a, last: b } = drawing;
        ctx.rect(Math.min(a.x, b.x), Math.min(a.y, b.y), Math.abs(b.x - a.x), Math.abs(b.y - a.y));
      }
      ctx.stroke();
      ctx.restore();
    }
  }

  hasSelection() {
    if (!this.bitmap) return false;
    const data = this.maskCtx().getImageData(0, 0, this.mask.width, this.mask.height).data;
    for (let i = 3; i < data.length; i += 4) if (data[i] > 127) return true;
    return false;
  }

  // White-on-black PNG at the video's size.
  maskBlob() {
    const out = document.createElement("canvas");
    out.width = this.mask.width; out.height = this.mask.height;
    const ctx = out.getContext("2d");
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, out.width, out.height);
    ctx.drawImage(this.mask, 0, 0);
    return new Promise(resolve => out.toBlob(resolve, "image/png"));
  }
}

// ----------------------------------------------------------------------------- motion paths

class PathEditor {
  constructor(ws) {
    this.ws = ws;
    this.canvas = ws.r("path-canvas");
    this.ctx = this.canvas.getContext("2d");
    this.paths = [];        // [{ points: [[x, y] in 0..1], radius }]
    this.background = null; // ImageBitmap of the first frame, if any
    this.size = { width: 832, height: 480 };
    this.bind();
    this.resize();
  }

  radius() { return parseInt(this.ws.r("path-radius").value, 10) / 100; }

  setBackground(bitmap) {
    this.background = bitmap;
    this.resize();
  }

  setAspect(w, h) {
    if (this.background) return;
    this.size = { width: w, height: h };
    this.resize();
  }

  resize() {
    const src = this.background || this.size;
    // Paths are stored normalised, so the canvas can use any working size.
    const k = 960 / Math.max(src.width, src.height);
    this.canvas.width = Math.round(src.width * k);
    this.canvas.height = Math.round(src.height * k);
    this.render();
  }

  bind() {
    const c = this.canvas;
    let stroke = null;
    const norm = e => { const p = canvasPoint(c, e); return [p.x / c.width, p.y / c.height]; };
    c.addEventListener("pointerdown", e => {
      if (e.button !== 0) return;
      c.setPointerCapture(e.pointerId);
      stroke = { points: [norm(e)], radius: this.radius() };
      this.render(stroke);
    });
    c.addEventListener("pointermove", e => {
      if (!stroke) return;
      const p = norm(e), last = stroke.points[stroke.points.length - 1];
      if (Math.hypot(p[0] - last[0], p[1] - last[1]) > 0.008) stroke.points.push(p);
      this.render(stroke);
    });
    const finish = () => {
      if (!stroke) return;
      if (stroke.points.length >= 2) this.paths.push(stroke);
      stroke = null;
      this.render();
      persist();
    };
    c.addEventListener("pointerup", finish);
    c.addEventListener("pointercancel", finish);
    this.ws.r("path-undo").onclick = () => this.undo();
    this.ws.r("path-clear").onclick = () => { this.paths = []; this.render(); persist(); };
    this.ws.r("path-radius").oninput = () => this.render();
  }

  undo() { this.paths.pop(); this.render(); persist(); }

  render(active = null) {
    const { ctx, canvas } = this;
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    if (this.background) ctx.drawImage(this.background, 0, 0, W, H);
    else {
      ctx.fillStyle = "#111"; ctx.fillRect(0, 0, W, H);
      ctx.strokeStyle = "rgba(255,255,255,.08)";
      for (let x = 0; x < W; x += W / 8) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
      for (let y = 0; y < H; y += H / 6) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
    }
    const all = active ? [...this.paths, active] : this.paths;
    const s = Math.min(W, H);
    all.forEach((path, i) => {
      const color = pathColor(i);
      const pts = path.points.map(([x, y]) => [x * W, y * H]);
      const r = path.radius * s;
      ctx.save();
      ctx.globalAlpha = 0.35;
      ctx.fillStyle = color;
      ctx.fillRect(pts[0][0] - r, pts[0][1] - r, 2 * r, 2 * r);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = color;
      ctx.lineWidth = Math.max(2, s / 160);
      ctx.lineCap = ctx.lineJoin = "round";
      ctx.beginPath();
      pts.forEach(([x, y], j) => j ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
      ctx.stroke();
      if (pts.length >= 2) {  // arrow head
        const [x1, y1] = pts[pts.length - 1], [x0, y0] = pts[Math.max(0, pts.length - 4)];
        const a = Math.atan2(y1 - y0, x1 - x0), L = Math.max(10, s / 30);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x1 - L * Math.cos(a - 0.45), y1 - L * Math.sin(a - 0.45));
        ctx.lineTo(x1 - L * Math.cos(a + 0.45), y1 - L * Math.sin(a + 0.45));
        ctx.closePath(); ctx.fillStyle = color; ctx.fill();
      }
      ctx.restore();
    });
  }

  toJSON() {
    return this.paths.map(p => ({ points: p.points.map(([x, y]) => [+x.toFixed(4), +y.toFixed(4)]), radius: +p.radius.toFixed(3) }));
  }

  load(paths) {
    this.paths = Array.isArray(paths) ? paths.filter(p => p && Array.isArray(p.points) && p.points.length >= 2) : [];
    this.render();
  }
}
