"use strict";

// ----------------------------------------------------------------------------- API client

const api = {
  async json(path, opts = {}) {
    const res = await fetch(path, opts);
    let body = null;
    try { body = await res.json(); } catch { /* non-JSON */ }
    if (!res.ok) {
      const detail = body && (body.detail || body.error);
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail) || `HTTP ${res.status}`);
    }
    return body;
  },
  get(path) { return this.json(path); },
  post(path, form) { return this.json(path, { method: "POST", body: form }); },
  del(path) { return this.json(path, { method: "DELETE" }); },
  async waitJob(id, onUpdate, signal) {
    for (;;) {
      const job = await this.get(`/api/jobs/${id}`);
      onUpdate(job);
      if (["done", "error", "cancelled"].includes(job.status)) return job;
      if (signal && signal.aborted) return job;
      await new Promise(r => setTimeout(r, 700));
    }
  },
};

const MODES = {
  generate: {
    help: "Create a video from a prompt. With a VACE model, reference images keep a subject or style.",
    promptLabel: "Prompt", placeholder: "A red fox trotting through fresh snow at sunrise, slow tracking shot…",
    imagesLabel: "Subject / style references (optional, up to 4)", run: "Generate", endpoint: "generate", max: 4,
  },
  image: {
    help: "Bring a still image to life: it becomes the first frame. Optionally say how it moves, and where it should end.",
    promptLabel: "Motion (optional)", placeholder: "She turns towards the camera and smiles, hair moving in the wind…",
    imagesLabel: "Subject references (optional, VACE)", run: "Animate image", endpoint: "image", max: 4,
  },
  control: {
    help: "Follow the structure and motion of a video — as depth, edges, line art or pose — or draw how objects move. VACE does the rest from your prompt.",
    promptLabel: "Describe the video to create", placeholder: "A chrome robot dancing on a neon-lit stage…",
    imagesLabel: "Subject references (optional) — keep a character's look", run: "Generate", endpoint: "control", max: 4,
  },
  pose: {
    help: "Copy a person's motion onto your character: poses are tracked in the driving video (DWPose/OpenPose), fitted to the character, and followed frame by frame.",
    promptLabel: "Description (optional)", placeholder: "A knight in silver armour dancing in a castle courtyard…",
    imagesLabel: "Extra references (optional) — outfit, props, background", run: "Transfer motion", endpoint: "pose", max: 3,
  },
  animate: {
    help: "Wan2.2-Animate: the character performs the body motion, gestures and facial expressions of the driving video — or replaces the person in it.",
    promptLabel: "Description (optional)", placeholder: "An anime girl with silver hair…",
    run: "Animate character", endpoint: "animate",
  },
  v2v: {
    help: "Transform a video: restyle it while keeping its motion (e.g. realistic → anime), re-render it by a strength, or regenerate just one region.",
    promptLabel: "Describe the result", placeholder: "Anime style, cel shading, vibrant colours, Studio Ghibli look…",
    imagesLabel: "Style references (optional)", run: "Transform video", endpoint: "v2v", max: 4,
  },
  export: {
    help: "Convert any video to MP4, WebM or GIF with FFmpeg (downloaded automatically the first time).",
    run: "Export", endpoint: "export",
  },
};

// The video slot each mode reads its driving/source video from.
const VIDEO_SLOT = { control: "driving", pose: "driving-pose", animate: "driving-pose", v2v: "source", export: "source" };

let models = { variants: [], default: null, loaded: null, operations: {}, family_defaults: {} };
const workspaces = new Map();
let activeId = null;
let tabCounter = 0;

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

function blobItem(blob, name = "file") {
  return { blob, name, url: URL.createObjectURL(blob) };
}

function isTyping(e) {
  const t = e.target;
  return t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable);
}

function lightbox(url, video = false) {
  const box = document.createElement("div");
  box.className = "lightbox";
  const el = document.createElement(video ? "video" : "img");
  el.src = url;
  if (video) Object.assign(el, { controls: true, autoplay: true, loop: true, muted: true, playsInline: true });
  el.onclick = e => { if (video) e.stopPropagation(); };
  box.appendChild(el);
  box.onclick = () => box.remove();
  document.body.appendChild(box);
}

function floor4k1(n) { return Math.floor((Math.max(1, n) - 1) / 4) * 4 + 1; }

// Metadata and a frame of a video blob, read by the browser.
function videoMeta(url) {
  return new Promise((resolve, reject) => {
    const v = document.createElement("video");
    v.muted = true; v.preload = "metadata"; v.src = url;
    v.onloadedmetadata = () => resolve({ width: v.videoWidth, height: v.videoHeight, duration: v.duration });
    v.onerror = () => reject(new Error("The browser cannot read this video (it can still be sent to the server)"));
  });
}

function videoFrame(url, which = "first") {
  return new Promise((resolve, reject) => {
    const v = document.createElement("video");
    v.muted = true; v.preload = "auto"; v.src = url;
    v.onloadeddata = () => {
      v.onseeked = async () => {
        const c = document.createElement("canvas");
        c.width = v.videoWidth; c.height = v.videoHeight;
        c.getContext("2d").drawImage(v, 0, 0);
        c.toBlob(b => b ? resolve(b) : reject(new Error("Could not read a frame")), "image/png");
      };
      v.currentTime = which === "last" ? Math.max(0, v.duration - 0.05) : 0;
    };
    v.onerror = () => reject(new Error("The browser cannot decode this video"));
  });
}

// ----------------------------------------------------------------------------- workspace

class Workspace {
  constructor(id, saved = {}) {
    this.id = id;
    this.name = saved.name || `Workspace ${id}`;
    this.el = $("#workspace-template").content.firstElementChild.cloneNode(true);
    this.el.id = `ws-${id}`;
    $("#workspaces").appendChild(this.el);

    this.mode = "generate";
    this.multi = [];          // reference images
    this.slots = {};          // named image/video inputs
    this.results = [];
    this.job = null;
    this.lastSeed = null;

    this.bind();
    this.maskEd = new MaskEditor(this);
    this.pathEd = new PathEditor(this);
    this.restore(saved);
    this.fillVariants();
    this.setMode(saved.mode || "generate");
  }

  r(role) { return $(`[data-role="${role}"]`, this.el); }

  // --- persistence (settings only; media lives in memory)
  settingRoles() {
    return ["prompt", "variant", "resolution", "aspect", "seconds", "fps", "steps", "seed", "guidance", "num-videos", "negative",
      "preview", "framepack", "section-frames", "context-frames", "anchor", "cn-type", "cn-invert", "cn-strength", "detector",
      "smooth", "align", "hands", "face", "pose-first", "pose-strength", "anim-mode", "v2v-method", "v2v-type", "v2v-strength",
      "ex-format", "ex-quality", "ex-fps", "ex-width", "path-radius", "brush-size"];
  }
  snapshot() {
    const s = { name: this.name, mode: this.mode, settings: {}, paths: this.pathEd ? this.pathEd.toJSON() : [] };
    for (const role of this.settingRoles()) {
      const el = this.r(role);
      s.settings[role] = el.type === "checkbox" ? el.checked : el.value;
    }
    return s;
  }
  restore(saved) {
    for (const [role, v] of Object.entries(saved.settings || {})) {
      const el = this.r(role);
      if (!el) continue;
      if (el.type === "checkbox") el.checked = !!v; else el.value = v;
      if (role === "variant" && v) el.dataset.pending = v;
    }
    if (saved.paths) this.pathEd.load(saved.paths);
  }

  // --- wiring
  bind() {
    $$(".modes button", this.el).forEach(b => b.onclick = () => { this.setMode(b.dataset.mode); persist(); });
    this.r("run").onclick = () => this.run();
    this.r("cancel").onclick = () => this.cancel();
    this.r("seed-reuse").onclick = () => { if (this.lastSeed != null) { this.r("seed").value = this.lastSeed; persist(); } };
    this.r("clear-results").onclick = () => { this.results = []; this.renderResults(); };
    this.el.addEventListener("change", () => { this.updateVisibility(); persist(); });
    this.el.addEventListener("input", e => {
      if (e.target.matches("input[type=range], input[type=number]")) this.updateVisibility();
      if (e.target === this.r("prompt")) persist();
    });
    const multiInput = this.r("multi-input");
    multiInput.onchange = () => { this.addMulti([...multiInput.files]); multiInput.value = ""; };
    this.dropZone(this.r("multi-drop"), files => this.addMulti(files));
    this.bindSlots();
    this.r("detect-run").onclick = () => this.detect();
  }

  dropZone(el, onFiles, accept = f => f.type.startsWith("image/")) {
    el.addEventListener("dragover", e => { e.preventDefault(); el.classList.add("over"); });
    el.addEventListener("dragleave", () => el.classList.remove("over"));
    el.addEventListener("drop", e => {
      e.preventDefault();
      el.classList.remove("over");
      const files = [...e.dataTransfer.files].filter(accept);
      if (files.length) onFiles(files);
    });
  }

  bindSlots() {
    $$("[data-slot]", this.el).forEach(slot => {
      const name = slot.dataset.slot, video = slot.dataset.kind === "video";
      const input = slot.querySelector("input");
      input.onchange = () => { if (input.files[0]) this.setSlot(name, input.files[0]); input.value = ""; };
      slot.querySelector(".slot-clear").onclick = e => { e.preventDefault(); e.stopPropagation(); this.setSlot(name, null); };
      const media = slot.querySelector(video ? "video" : "img");
      media.onclick = e => { e.preventDefault(); lightbox(media.src, video); };
      const accept = video ? f => f.type.startsWith("video/") || f.type === "image/gif" : f => f.type.startsWith("image/");
      this.dropZone(slot, files => this.setSlot(name, files[0]), accept);
    });
  }

  async setSlot(name, blob) {
    const slot = $(`[data-slot="${name}"]`, this.el);
    const video = slot.dataset.kind === "video";
    const old = this.slots[name];
    if (old) URL.revokeObjectURL(old.url);
    const item = blob ? blobItem(blob, blob.name || `${name}.${video ? "mp4" : "png"}`) : null;
    this.slots[name] = item;
    const media = slot.querySelector(video ? "video" : "img");
    media.hidden = !item;
    if (item) { media.src = item.url; if (video) media.play().catch(() => {}); } else media.removeAttribute("src");
    slot.querySelector(".slot-empty").hidden = !!item;
    slot.querySelector(".slot-clear").hidden = !item;
    if (video && item) {
      try { item.meta = await videoMeta(item.url); } catch (e) { item.meta = null; this.setStatus(e.message, true); }
      if (name === "source") {
        try { this.maskEd.setFrame(await createImageBitmap(await videoFrame(item.url))); } catch { this.maskEd.setFrame(null); }
      }
    } else if (name === "source") this.maskEd.setFrame(null);
    if (name === "first") this.pathEd.setBackground(item ? await createImageBitmap(item.blob) : null);
    if (video) this.resetDetect();
    this.updateVisibility();
  }

  // Slots visible for the current mode, in page order.
  modeSlots(kind) {
    return $$("[data-slot]", this.el).filter(s => s.dataset.kind === kind && !s.closest("[hidden]")).map(s => s.dataset.slot);
  }

  videoItem() { return this.slots[VIDEO_SLOT[this.mode]] || null; }

  families() {
    if (this.mode === "v2v") return this.r("v2v-method").value === "sdedit" ? ["t2v"] : ["vace"];
    return models.operations[this.mode] || [];
  }

  setMode(mode) {
    this.mode = mode;
    const m = MODES[mode];
    $$(".modes button", this.el).forEach(b => b.classList.toggle("active", b.dataset.mode === mode));
    this.r("mode-help").textContent = m.help;
    if (m.promptLabel) this.r("prompt-label").textContent = m.promptLabel;
    if (m.placeholder) this.r("prompt").placeholder = m.placeholder;
    if (m.imagesLabel) this.r("images-label").textContent = m.imagesLabel;
    this.r("run").textContent = m.run;
    this.r("source-label").textContent = mode === "export" ? "Video to convert" : "Source video";
    this.r("char-label").textContent = mode === "animate" ? "Character" : "Character (optional)";
    this.fillVariants();
    this.resetDetect();
    this.updateVisibility();
    this.renderThumbs();
  }

  updateVisibility() {
    const mode = this.mode;
    $$("[data-show]", this.el).forEach(el => el.hidden = !el.dataset.show.split(" ").includes(mode));
    const g = this.r("guidance").value;
    $$("[data-show-cfg]", this.el).forEach(el => el.hidden = !(g === "" ? mode !== "animate" : parseFloat(g) > 1));
    const cnType = this.r("cn-type").value;
    const isPath = mode === "control" && cnType === "path";
    if (mode === "control") {
      this.r("control-video-field").hidden = isPath;
      this.r("pose-group").hidden = cnType !== "pose";
      this.r("cn-invert-row").hidden = !["raw", "lineart", "scribble", "softedge", "canny"].includes(cnType);
    }
    const method = this.r("v2v-method").value;
    this.r("v2v-type-row").hidden = method !== "restyle";
    this.r("v2v-strength-row").hidden = method !== "sdedit";
    $$('[data-show="generate image control pose v2v"]', this.el).forEach(el => {
      if (mode === "v2v" && method === "sdedit") el.hidden = true;
    });
    this.r("cn-strength-val").textContent = (+this.r("cn-strength").value).toFixed(2);
    this.r("pose-strength-val").textContent = (+this.r("pose-strength").value).toFixed(2);
    this.r("smooth-val").textContent = (+this.r("smooth").value).toFixed(2);
    this.r("v2v-strength-val").textContent = (+this.r("v2v-strength").value).toFixed(2);

    // Stage panels
    this.r("path-panel").hidden = !isPath;
    this.r("mask-panel").hidden = !(mode === "v2v" && method === "inpaint");
    const detectable = (mode === "control" && !["path", "raw"].includes(cnType)) || mode === "pose" || mode === "animate"
      || (mode === "v2v" && method === "restyle");
    this.r("detect-panel").hidden = !detectable;
    this.r("detect-title").textContent = mode === "pose" || mode === "animate" ? "Pose video" : "Control video";
    if (isPath) {
      const [w, h] = this.outputAspect();
      this.pathEd.setAspect(w, h);
      requestAnimationFrame(() => this.pathEd.render());
    }
    if (!this.r("mask-panel").hidden) requestAnimationFrame(() => this.maskEd.render());
    const v2vFamilies = this.families().join(",");
    if (this.r("variant").dataset.families !== v2vFamilies) this.fillVariants();
    this.updateInfo();
  }

  outputAspect() {
    const res = this.r("resolution").value === "720p" ? [1280, 720] : [832, 480];
    const aspect = this.r("aspect").value;
    if (aspect !== "auto") { const [a, b] = aspect.split(":").map(Number); return [a, b]; }
    return res;
  }

  defaultFps() {
    return parseFloat(this.r("fps").value) || (this.mode === "animate" ? 30 : 16);
  }

  // Frames, duration and FramePack sections the server will use.
  updateInfo() {
    const vi = this.videoItem();
    const meta = vi && vi.meta;
    this.r("video-info").textContent = meta ? `${meta.width} × ${meta.height} · ${meta.duration.toFixed(1)} s` : "";
    const info = this.r("plan-info");
    if (this.mode === "export") { info.textContent = ""; return; }
    const fps = this.defaultFps();
    const secs = parseFloat(this.r("seconds").value);
    const drives = ["control", "pose", "animate", "v2v"].includes(this.mode) && !(this.mode === "control" && this.r("cn-type").value === "path");
    this.r("seconds").placeholder = drives ? "whole video" : "5";
    let frames;
    if (secs > 0) frames = Math.round(secs * fps) + 1;
    else if (drives && meta && isFinite(meta.duration)) frames = Math.max(1, Math.floor(meta.duration * fps));
    else if (drives) { info.textContent = "Length follows the driving video."; return; }
    else frames = 81;
    if (drives && meta && isFinite(meta.duration)) frames = Math.min(frames, Math.max(1, Math.floor(meta.duration * fps)));
    const section = floor4k1(parseInt(this.r("section-frames").value, 10) || 81);
    const ctx = Math.max(1, Math.min(parseInt(this.r("context-frames").value, 10) || 5, section - 4));
    const fpMode = this.r("framepack").value;
    const sectioned = fpMode === "on" || (fpMode === "auto" && frames > section);
    const n = sectioned ? 1 + Math.ceil(Math.max(0, frames - section) / (section - ctx)) : 1;
    let text = `${frames} frames · ${(frames / fps).toFixed(1)} s at ${fps} fps`;
    if (n > 1) text += ` · ${n} sections (constant VRAM)`;
    if (!sectioned && frames > 161) text += " — too long for one pass; turn on sectioned sampling";
    info.textContent = text;
    info.classList.toggle("error", !sectioned && frames > 161);
  }

  // --- reference images
  addMulti(files) {
    const max = MODES[this.mode].max || 4;
    for (const f of files) {
      if (this.multi.length >= max) { this.setStatus(`At most ${max} images.`, true); break; }
      this.multi.push(blobItem(f, f.name || "reference.png"));
    }
    this.renderThumbs();
  }

  renderThumbs() {
    const box = this.r("thumbs");
    box.innerHTML = "";
    this.multi.forEach((item, i) => {
      const t = document.createElement("div");
      t.className = "thumb";
      t.innerHTML = `<img alt=""><span class="idx">${i + 1}</span><button title="Remove" aria-label="Remove">×</button>`;
      t.querySelector("img").src = item.url;
      t.querySelector("img").onclick = () => lightbox(item.url);
      t.querySelector("button").onclick = () => { URL.revokeObjectURL(item.url); this.multi.splice(i, 1); this.renderThumbs(); };
      box.appendChild(t);
    });
  }

  // --- control / pose preview
  resetDetect() {
    const v = this.r("detect-video");
    v.hidden = true;
    v.removeAttribute("src");
    this.r("detect-empty").hidden = false;
    this.r("detect-status").textContent = "";
  }

  detectType() {
    if (this.mode === "pose" || this.mode === "animate") return "pose";
    if (this.mode === "v2v") return this.r("v2v-type").value;
    return this.r("cn-type").value;
  }

  async detect() {
    const vi = this.videoItem();
    const status = this.r("detect-status");
    if (!vi) { status.textContent = "Add a driving video first."; return; }
    const f = new FormData();
    f.append("video", vi.blob, vi.name);
    f.append("control_type", this.detectType());
    f.append("detector", this.r("detector").value);
    f.append("smooth", this.mode === "control" ? "0.5" : this.r("smooth").value);
    f.append("hands", this.mode === "pose" ? this.r("hands").checked : true);
    f.append("face", this.mode === "pose" ? this.r("face").checked : true);
    f.append("invert", this.mode === "control" && this.r("cn-invert").checked);
    if ((this.mode === "pose" || this.mode === "animate") && this.r("align").checked && this.slots.character) {
      f.append("reference", this.slots.character.blob, this.slots.character.name);
    }
    this.sizeFields(f);
    f.append("preview", "off");
    const btn = this.r("detect-run");
    btn.disabled = true;
    try {
      const job = await api.post("/api/control/detect", f);
      const final = await api.waitJob(job.id, j => {
        status.textContent = j.status === "queued" ? `Queued (position ${j.queue_position})…`
          : j.progress.total ? `${j.message || "Extracting"} — ${j.progress.step}/${j.progress.total}` : (j.message || "Working…");
      });
      if (final.status !== "done") throw new Error(final.error || final.status);
      const v = this.r("detect-video");
      v.src = final.videos[0].url;
      v.hidden = false;
      this.r("detect-empty").hidden = true;
      v.play().catch(() => {});
      status.textContent = `${final.info.frames} frames`;
      refreshModels();
    } catch (e) {
      status.textContent = e.message;
    } finally {
      btn.disabled = false;
    }
  }

  // --- run
  sizeFields(f) {
    const add = (k, v) => { if (v !== "" && v != null) f.append(k, v); };
    add("resolution", this.r("resolution").value);
    add("aspect", this.r("aspect").value);
    add("seconds", this.r("seconds").value);
    add("fps", this.r("fps").value);
  }

  async buildForm() {
    const f = new FormData();
    const prompt = this.r("prompt").value.trim();
    const add = (k, v) => { if (v !== "" && v != null) f.append(k, v); };
    const mode = this.mode;
    const need = (slot, msg) => { const it = this.slots[slot]; if (!it) throw new Error(msg); return it; };
    const file = (field, it) => f.append(field, it.blob, it.name);

    if (mode === "export") {
      file("video", need("source", "Add a video to convert."));
      add("format", this.r("ex-format").value);
      add("quality", this.r("ex-quality").value);
      add("fps", this.r("ex-fps").value);
      add("width", this.r("ex-width").value);
      return f;
    }
    add("variant", this.r("variant").value);
    add("steps", this.r("steps").value);
    add("seed", this.r("seed").value);
    add("guidance", this.r("guidance").value);
    add("num_videos", this.r("num-videos").value);
    add("preview", this.r("preview").value);
    add("negative_prompt", this.r("negative").value);
    add("framepack", this.r("framepack").value);
    add("section_frames", this.r("section-frames").value);
    add("context_frames", this.r("context-frames").value);
    f.append("anchor", this.r("anchor").checked);
    this.sizeFields(f);
    const refs = () => this.multi.forEach(it => f.append("references", it.blob, it.name));

    if (mode === "generate") {
      if (!prompt) throw new Error("Enter a prompt.");
      f.append("prompt", prompt);
      refs();
    } else if (mode === "image") {
      file("image", need("image", "Add the image to animate."));
      if (this.slots.last) file("last_image", this.slots.last);
      f.append("prompt", prompt);
      refs();
    } else if (mode === "control") {
      if (!prompt) throw new Error("Describe the video to create.");
      const type = this.r("cn-type").value;
      f.append("prompt", prompt);
      f.append("control_type", type);
      f.append("strength", this.r("cn-strength").value);
      f.append("invert", this.r("cn-invert").checked);
      f.append("detector", this.r("detector").value);
      if (type === "path") {
        if (!this.pathEd.paths.length) throw new Error("Draw at least one motion path.");
        f.append("paths", JSON.stringify(this.pathEd.toJSON()));
      } else {
        file("video", need("driving", "Add a driving video."));
      }
      if (this.slots.first) file("first_frame", this.slots.first);
      if (this.slots.cnlast) file("last_frame", this.slots.cnlast);
      refs();
    } else if (mode === "pose") {
      file("video", need("driving-pose", "Add a driving video."));
      if (this.slots.character) file("image", this.slots.character);
      f.append("prompt", prompt);
      f.append("detector", this.r("detector").value);
      f.append("align", this.r("align").checked);
      f.append("smooth", this.r("smooth").value);
      f.append("hands", this.r("hands").checked);
      f.append("face", this.r("face").checked);
      f.append("first_frame", this.r("pose-first").checked);
      f.append("strength", this.r("pose-strength").value);
      refs();
    } else if (mode === "animate") {
      file("image", need("character", "Add the character image."));
      file("video", need("driving-pose", "Add a driving video."));
      f.append("prompt", prompt);
      f.append("mode", this.r("anim-mode").value);
      f.append("detector", this.r("detector").value);
      f.append("align", this.r("align").checked);
      f.append("smooth", this.r("smooth").value);
    } else if (mode === "v2v") {
      if (!prompt) throw new Error("Describe the result.");
      file("video", need("source", "Add the video to transform."));
      const method = this.r("v2v-method").value;
      f.append("prompt", prompt);
      f.append("method", method);
      if (method === "restyle") f.append("control_type", this.r("v2v-type").value);
      if (method === "sdedit") f.append("strength", this.r("v2v-strength").value);
      else refs();
      if (method === "inpaint") {
        if (!this.maskEd.hasSelection()) throw new Error("Select the region to regenerate.");
        f.append("mask", await this.maskEd.maskBlob(), "mask.png");
      }
    }
    return f;
  }

  setStatus(text, error = false) {
    const s = this.r("status");
    s.textContent = text;
    s.classList.toggle("error", error);
  }

  setBusy(busy) {
    this.r("run").disabled = busy;
    this.r("cancel").hidden = !busy;
    this.r("progress").hidden = !busy;
    renderTabs();
  }

  async run() {
    let form;
    this.setStatus("Preparing…");
    try { form = await this.buildForm(); } catch (e) { this.setStatus(e.message, true); return; }
    this.setBusy(true);
    this.setStatus("Uploading…");
    const bar = $(".bar", this.r("progress"));
    bar.style.width = "0";
    const mode = this.mode;
    const live = mode !== "export" && this.r("preview").value !== "off";
    try {
      const created = await api.post(`/api/${MODES[mode].endpoint}`, form);
      this.job = created;
      const job = await api.waitJob(created.id, j => {
        this.job = j;
        const { step, total } = j.progress;
        this.r("progress").classList.toggle("indeterminate", !total);
        if (total) bar.style.width = `${(100 * step) / total}%`;
        if (live) this.showLive(j);
        if (j.status === "queued") this.setStatus(`Queued (position ${j.queue_position})…`);
        else if (j.status === "running") this.setStatus(total ? `${j.message || "Sampling"} — step ${step}/${total}` : (j.message || "Working…"));
      });
      if (job.status === "done") {
        if (job.seed >= 0) this.lastSeed = job.seed;
        this.addResults(job);
        const secs = job.finished && job.started ? ` in ${(job.finished - job.started).toFixed(1)}s` : "";
        this.setStatus(`Done${secs}${job.seed >= 0 ? ` · seed ${job.seed}` : ""}`);
      } else if (job.status === "cancelled") {
        this.setStatus("Cancelled.");
      } else {
        this.setStatus(job.error || "Failed.", true);
      }
    } catch (e) {
      this.setStatus(e.message, true);
    } finally {
      this.hideLive();
      this.job = null;
      this.setBusy(false);
      refreshHealth();
    }
  }

  async cancel() {
    if (this.job) {
      try { await api.del(`/api/jobs/${this.job.id}`); } catch { /* already finished */ }
      this.setStatus("Cancelling…");
    }
  }

  // --- live preview
  showLive(job) {
    if (!this.live) {
      const card = document.createElement("div");
      card.className = "result live";
      card.innerHTML = `<div class="img"><img alt="Live preview" hidden><span class="live-wait">Waiting for the first step…</span></div><div class="meta"></div>`;
      card.querySelector("img").onclick = e => lightbox(e.target.src);
      this.live = { card, url: null };
      this.renderResults();
    }
    const { card } = this.live;
    if (job.preview && job.preview !== this.live.url) {
      this.live.url = job.preview;
      const img = card.querySelector("img");
      img.src = job.preview;
      img.hidden = false;
      card.querySelector(".live-wait").hidden = true;
    }
    card.querySelector(".meta").textContent = job.status === "queued" ? "Queued…"
      : job.preview_step ? `Live preview · ${job.message || ""}` : (job.message || "Starting…");
  }

  hideLive() {
    if (!this.live) return;
    this.live.card.remove();
    this.live = null;
    if (!this.results.length) this.renderResults();
  }

  // --- results
  addResults(job) {
    const items = job.videos.map((video, index) => ({ job, video, index }));
    for (const [name, url] of Object.entries(job.files || {})) items.push({ job, file: { name, url } });
    this.results.unshift(...items);
    this.renderResults();
  }

  async useVideo(item, mode, slot) {
    const blob = await (await fetch(item.video.url)).blob();
    const file = new File([blob], `${item.job.kind}-${item.job.id}-${item.index}.mp4`, { type: "video/mp4" });
    if (mode) this.setMode(mode);
    await this.setSlot(slot || VIDEO_SLOT[this.mode] || "source", file);
    persist();
  }

  async continueFrom(item) {
    try {
      const blob = await videoFrame(item.video.url, "last");
      this.setMode("image");
      await this.setSlot("image", new File([blob], "last-frame.png", { type: "image/png" }));
      this.setStatus("The last frame is now the first frame. Describe what happens next.");
      persist();
    } catch (e) { this.setStatus(e.message, true); }
  }

  renderResults() {
    const box = this.r("results");
    box.innerHTML = "";
    if (this.live) box.appendChild(this.live.card);
    if (!this.results.length) {
      if (!this.live) box.innerHTML = `<p class="empty">Results appear here.</p>`;
      return;
    }
    for (const item of this.results) box.appendChild(item.file ? this.fileCard(item) : this.videoCard(item));
  }

  videoCard(item) {
    const { job, video } = item;
    const card = document.createElement("div");
    card.className = "result";
    card.innerHTML = `
      <div class="img"><video muted loop playsinline controls preload="metadata"></video></div>
      <div class="meta"></div>
      <div class="actions">
        <span class="dl-group"><span>Download</span><a data-f="mp4">MP4</a><a data-f="webm">WebM</a><a data-f="gif">GIF</a></span>
        <span class="sep"></span>
        <button data-a="continue" title="Use the last frame as the start of a new video">Continue</button>
        <button data-a="restyle">Video→Video</button>
        <button data-a="input">Use as input</button>
        <button data-a="extra" hidden></button>
        <button data-a="tab">New tab</button>
      </div>`;
    const v = card.querySelector("video");
    v.src = video.url;
    v.onmouseenter = () => v.play().catch(() => {});
    card.querySelector(".img").style.aspectRatio = `${video.width} / ${video.height}`;
    const meta = card.querySelector(".meta");
    const bits = [job.kind, job.variant, `${video.width}×${video.height}`, `${video.duration.toFixed(1)} s`];
    if (job.info && job.info.sections > 1) bits.push(`${job.info.sections} sections`);
    if (job.seed >= 0) bits.push(`seed ${job.seed}`);
    meta.textContent = bits.filter(Boolean).join(" · ");
    meta.title = job.prompt || "";
    for (const a of $$("a[data-f]", card)) {
      a.href = `${video.url}?${a.dataset.f === "mp4" ? "" : `format=${a.dataset.f}&`}download=true`;
      a.title = a.dataset.f === "gif" ? "Encoded on first download; may take a moment" : "";
    }
    card.querySelector('[data-a="continue"]').onclick = () => this.continueFrom(item);
    card.querySelector('[data-a="restyle"]').onclick = () => this.useVideo(item, "v2v", "source");
    card.querySelector('[data-a="input"]').onclick = () => {
      if (this.mode === "image" || this.mode === "generate") return this.continueFrom(item);
      return this.useVideo(item, null);
    };
    const [extraName, extraUrl] = Object.entries(job.extras || {})[0] || [];
    if (extraUrl) {
      const btn = card.querySelector('[data-a="extra"]');
      btn.hidden = false;
      btn.textContent = extraName === "pose" ? "Pose video" : extraName === "mask" ? "Mask video" : "Control video";
      btn.onclick = () => lightbox(extraUrl, true);
    }
    card.querySelector('[data-a="tab"]').onclick = async () => {
      const ws = createTab({ mode: "v2v" });
      await ws.useVideo(item, "v2v", "source");
    };
    return card;
  }

  fileCard(item) {
    const { job, file } = item;
    const fmt = file.name.split(".").pop();
    const card = document.createElement("div");
    card.className = "result";
    card.innerHTML = `<div class="img"></div><div class="meta"></div><div class="actions"><a download>Download ${fmt.toUpperCase()}</a></div>`;
    const media = document.createElement(fmt === "gif" ? "img" : "video");
    media.src = `${file.url}?download=false`;
    if (fmt !== "gif") Object.assign(media, { muted: true, loop: true, controls: true, playsInline: true });
    else media.onclick = () => lightbox(media.src);
    card.querySelector(".img").appendChild(media);
    const mb = job.info && job.info.bytes ? ` · ${(job.info.bytes / 1e6).toFixed(2)} MB` : "";
    card.querySelector(".meta").textContent = `export · ${fmt.toUpperCase()}${mb}`;
    card.querySelector("a").href = file.url;
    return card;
  }

  fillVariants() {
    const sel = this.r("variant");
    const families = this.families();
    sel.dataset.families = families.join(",");
    const current = sel.value || sel.dataset.pending || "";
    sel.innerHTML = "";
    const fits = v => v.family === "any" || families.includes(v.family);
    const def = models.variants.find(v => v.name === models.default);
    const auto = def && fits(def) ? def.name : models.family_defaults[families[0]];
    const o = document.createElement("option");
    o.value = "";
    o.textContent = `Automatic${auto ? ` (${auto})` : ""}`;
    sel.appendChild(o);
    for (const v of models.variants.filter(fits)) {
      const opt = document.createElement("option");
      opt.value = v.name;
      const state = v.loaded ? " — loaded" : v.downloaded ? "" : ` — downloads ~${v.approx_download_gb} GB`;
      opt.textContent = `${v.label}${state}`;
      sel.appendChild(opt);
    }
    sel.value = current;
    if (sel.value !== current) sel.value = "";
    if (models.variants.length) delete sel.dataset.pending;
  }

  render() {
    if (!this.r("mask-panel").hidden) this.maskEd.render();
    if (!this.r("path-panel").hidden) this.pathEd.render();
  }

  destroy() {
    if (this.job) this.cancel();
    for (const item of [...this.multi, ...Object.values(this.slots)].filter(Boolean)) URL.revokeObjectURL(item.url);
    this.el.remove();
  }
}

// ----------------------------------------------------------------------------- tabs

function createTab(saved = {}, activate = true) {
  const id = ++tabCounter;
  const ws = new Workspace(id, saved);
  workspaces.set(id, ws);
  if (activate) activateTab(id);
  persist();
  return ws;
}

function closeTab(id) {
  const ws = workspaces.get(id);
  if (!ws) return;
  if (ws.job && !confirm(`“${ws.name}” has a job running. Close it and cancel the job?`)) return;
  const ids = [...workspaces.keys()];
  const idx = ids.indexOf(id);
  ws.destroy();
  workspaces.delete(id);
  if (!workspaces.size) createTab();
  else if (activeId === id) activateTab([...workspaces.keys()][Math.max(0, idx - 1)]);
  renderTabs();
  persist();
}

function activateTab(id) {
  activeId = id;
  for (const [wid, ws] of workspaces) ws.el.classList.toggle("active", wid === id);
  renderTabs();
  const ws = workspaces.get(id);
  if (ws) requestAnimationFrame(() => ws.render());
  persist();
}

function renderTabs() {
  const bar = $("#tabs");
  bar.innerHTML = "";
  for (const [id, ws] of workspaces) {
    const tab = document.createElement("div");
    tab.className = "tab" + (id === activeId ? " active" : "");
    tab.setAttribute("role", "tab");
    tab.setAttribute("aria-selected", id === activeId);
    tab.title = "Double-click to rename";
    tab.innerHTML = `${ws.job ? '<span class="busy" title="Working"></span>' : ""}<span class="name"></span><button class="close" title="Close tab" aria-label="Close tab">×</button>`;
    tab.querySelector(".name").textContent = ws.name;
    tab.onclick = e => { if (id !== activeId && !e.target.classList.contains("close")) activateTab(id); };
    tab.onauxclick = e => { if (e.button === 1) closeTab(id); };
    tab.querySelector(".close").onclick = () => closeTab(id);
    tab.ondblclick = () => {
      const input = document.createElement("input");
      input.value = ws.name;
      tab.querySelector(".name").replaceWith(input);
      input.focus(); input.select();
      const done = () => { ws.name = input.value.trim() || ws.name; renderTabs(); persist(); };
      input.onblur = done;
      input.onkeydown = e => { if (e.key === "Enter") input.blur(); if (e.key === "Escape") { input.value = ws.name; input.blur(); } };
    };
    bar.appendChild(tab);
  }
}

const STORAGE_KEY = "wan-video-studio.tabs";
let persistTimer = null;
function persist() {
  clearTimeout(persistTimer);
  persistTimer = setTimeout(() => {
    try {
      const ids = [...workspaces.keys()];
      const state = { tabs: [...workspaces.values()].map(ws => ws.snapshot()), active: ids.indexOf(activeId) };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch { /* storage unavailable */ }
  }, 300);
}

function restoreTabs() {
  let state = null;
  try { state = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null"); } catch { /* ignore */ }
  if (state && state.tabs && state.tabs.length) {
    state.tabs.forEach(t => createTab(t, false));
    const ids = [...workspaces.keys()];
    activateTab(ids[Math.min(Math.max(0, state.active), ids.length - 1)]);
  } else {
    createTab();
  }
}

// ----------------------------------------------------------------------------- models

async function refreshModels() {
  try {
    models = await api.get("/api/models");
  } catch (e) {
    $("#models-list").textContent = `Could not load models: ${e.message}`;
    return;
  }
  for (const ws of workspaces.values()) ws.fillVariants();
  renderModels();
  updatePill();
}

function updatePill(text) {
  $("#model-status").textContent = text || (models.loaded ? `Loaded: ${models.loaded}` : `No model loaded · default ${models.default || "?"}`);
  $("#model-dot").className = "dot" + (text ? " busy" : models.loaded ? " ok" : "");
}

async function refreshHealth() {
  try {
    const h = await api.get("/api/health");
    if (h.loaded !== models.loaded) await refreshModels();
  } catch { /* server unreachable */ }
}

const FAMILY_LABELS = {
  vace: "VACE — all-in-one: text, image, control, pose, editing, long videos",
  t2v: "T2V — text-to-video (and SDEdit video-to-video)",
  i2v: "I2V — image-to-video",
  animate: "Animate — Wan2.2 character animation",
  any: "Testing",
};

async function runRowJob(path, form, btn, row) {
  $$("button", row).forEach(b => b.disabled = true);
  try {
    const job = await api.post(path, form);
    const final = await api.waitJob(job.id, j => {
      btn.textContent = j.status === "queued" ? "Queued…" : (j.message || "Working…").slice(0, 40);
      updatePill(`${j.message || j.status}`.slice(0, 60));
    });
    if (final.status === "error") alert(final.error);
  } catch (e) { alert(e.message); }
  await refreshModels();
}

function renderModels() {
  const list = $("#models-list");
  list.innerHTML = "";
  let family = null;
  for (const v of models.variants) {
    if (v.family !== family) {
      family = v.family;
      const h = document.createElement("p");
      h.className = "family";
      h.textContent = FAMILY_LABELS[family] || family;
      list.appendChild(h);
    }
    const row = document.createElement("div");
    row.className = "model-row" + (v.loaded ? " loaded" : "");
    row.innerHTML = `
      <div class="title"></div>
      <div class="desc"></div>
      <div class="btns">
        <button data-a="download" ${v.downloaded ? "hidden" : ""}>Download</button>
        <button data-a="load" class="${v.loaded ? "" : "primary"}" ${v.loaded ? "disabled" : ""}>${v.loaded ? "Loaded" : "Load"}</button>
      </div>`;
    const title = row.querySelector(".title");
    title.textContent = v.label;
    const badge = (text, ok) => { const b = document.createElement("span"); b.className = "badge" + (ok ? " ok" : ""); b.textContent = text; title.appendChild(b); };
    badge(v.downloaded ? "downloaded" : `~${v.approx_download_gb} GB`, v.downloaded);
    if (v.name === models.default) badge("default");
    if (v.offload && v.offload !== "none") badge(`offload: ${v.offload}`);
    row.querySelector(".desc").textContent = `${v.name} — ${v.description}`;
    const form = () => { const f = new FormData(); f.append("variant", v.name); return f; };
    row.querySelector('[data-a="download"]').onclick = e => runRowJob("/api/models/download", form(), e.target, row);
    row.querySelector('[data-a="load"]').onclick = e => runRowJob("/api/models/load", form(), e.target, row);
    list.appendChild(row);
  }
  const h = document.createElement("p");
  h.className = "family";
  h.textContent = "Tools";
  list.appendChild(h);
  const ff = models.ffmpeg || {};
  toolRow(list, "FFmpeg", ff.available, ff.available ? `${ff.version || ""} · ${ff.source}` : "missing",
    ff.available ? ff.path : `Encodes and exports videos. Downloads a static build automatically on first use${ff.download_url ? ` (${ff.download_url})` : ""}.`,
    "/api/ffmpeg/download");
  if (models.dwpose) {
    toolRow(list, "DWPose", models.dwpose.downloaded, models.dwpose.downloaded ? "downloaded" : `~${models.dwpose.approx_download_gb} GB`,
      `${models.dwpose.repo} — whole-body pose estimation (ONNX) for Pose, Animate and pose control.`, "/api/dwpose/download");
  }
}

function toolRow(list, label, ready, badgeText, desc, downloadPath) {
  const row = document.createElement("div");
  row.className = "model-row" + (ready ? " loaded" : "");
  row.innerHTML = `<div class="title"></div><div class="desc"></div><div class="btns"><button data-a="download" ${ready ? "hidden" : ""}>Download</button></div>`;
  row.querySelector(".title").textContent = label;
  const b = document.createElement("span");
  b.className = "badge" + (ready ? " ok" : "");
  b.textContent = badgeText;
  row.querySelector(".title").appendChild(b);
  row.querySelector(".desc").textContent = desc;
  row.querySelector('[data-a="download"]').onclick = e => runRowJob(downloadPath, new FormData(), e.target, row);
  list.appendChild(row);
}

// ----------------------------------------------------------------------------- global events

$("#new-tab").onclick = () => createTab();
$("#model-pill").onclick = () => { $("#models-dialog").showModal(); refreshModels(); };

document.addEventListener("keydown", e => {
  const ws = workspaces.get(activeId);
  if (e.ctrlKey && e.altKey && e.key.toLowerCase() === "t") { e.preventDefault(); createTab(); return; }
  if (e.ctrlKey && e.altKey && e.key.toLowerCase() === "w") { e.preventDefault(); closeTab(activeId); return; }
  if (!ws || isTyping(e)) return;
  const masking = !ws.r("mask-panel").hidden, pathing = !ws.r("path-panel").hidden;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
    if (masking) { e.preventDefault(); ws.maskEd.undo(); } else if (pathing) { e.preventDefault(); ws.pathEd.undo(); }
    return;
  }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const tools = { l: "lasso", r: "box", b: "brush", e: "erase" };
  if (masking && tools[e.key.toLowerCase()]) ws.maskEd.setTool(tools[e.key.toLowerCase()]);
});

document.addEventListener("paste", e => {
  const ws = workspaces.get(activeId);
  if (!ws || isTyping(e)) return;
  const files = [...(e.clipboardData?.files || [])];
  const images = files.filter(f => f.type.startsWith("image/") && f.type !== "image/gif");
  const videos = files.filter(f => f.type.startsWith("video/") || f.type === "image/gif");
  if (!images.length && !videos.length) return;
  e.preventDefault();
  if (videos.length) {
    const slot = ws.modeSlots("video").find(n => !ws.slots[n]) || ws.modeSlots("video")[0];
    if (slot) ws.setSlot(slot, videos[0]);
  }
  if (images.length) {
    const slots = ws.modeSlots("image");
    const empty = slots.find(n => !ws.slots[n]);
    if (empty) ws.setSlot(empty, images[0]);
    else if (MODES[ws.mode].max) ws.addMulti(images);
  }
});

window.addEventListener("resize", () => {
  const ws = workspaces.get(activeId);
  if (ws) ws.render();
});

restoreTabs();
refreshModels();
setInterval(refreshHealth, 5000);
