"""The core operations: text-to-video, image-to-video, VACE control, pose transfer,
character animation, video-to-video and export.

``Engine`` owns at most one loaded pipeline and is not thread-safe; the API serialises access
through the job queue, and the CLI uses it directly. Every operation streams its frames to an
MP4 master in ``Hooks.workdir`` and can render videos of any length in constant memory with
FramePack-style sections (see :mod:`wan_video.framepack`).
"""

from __future__ import annotations

import logging
import math
import random
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, fields
from pathlib import Path

from PIL import Image, ImageOps

from . import annotators as ann
from . import framepack as fp
from . import selection as sel
from .preview import MODES as PREVIEW_MODES
from .preview import Previewer
from .variants import Variant, get_variant, pick_variant
from .video import VideoFile, VideoWriter, probe, read_frames

log = logging.getLogger(__name__)

MAX_REFERENCES = 4
# Wan's recommended negative prompt (English translation of the official one).
DEFAULT_NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly "
    "drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy "
    "background, three legs, many people in the background, walking backwards"
)
GRAY = (128, 128, 128)

# Output sizes for each resolution tier and aspect ratio (multiples of 16, ~the trained pixel count).
SIZES = {
    "480p": {"16:9": (832, 480), "9:16": (480, 832), "1:1": (624, 624), "4:3": (720, 544), "3:4": (544, 720)},
    "720p": {"16:9": (1280, 720), "9:16": (720, 1280), "1:1": (960, 960), "4:3": (1104, 832), "3:4": (832, 1104)},
}
ASPECTS = ("auto", "16:9", "9:16", "1:1", "4:3", "3:4")
CONTROL_KINDS = tuple(ann.CONTROL_TYPES) + ("path",)
V2V_METHODS = ("restyle", "sdedit", "inpaint")
ANIMATE_MODES = ("animate", "replace")


class Cancelled(Exception):
    pass


@dataclass
class Options:
    """Sampling options shared by every operation."""

    resolution: str = "480p"
    aspect: str = "auto"
    width: int | None = None
    height: int | None = None
    # Output length: frames wins over seconds; driving-video operations default to the source length.
    frames: int | None = None
    seconds: float | None = None
    fps: float | None = None
    steps: int | None = None
    seed: int | None = None
    guidance: float | None = None
    negative_prompt: str | None = None
    # Flow-matching shift; default 3.0 at 480p and 5.0 at 720p, as Wan recommends.
    shift: float | None = None
    num_videos: int = 1
    # FramePack-style sectioned sampling: "auto" (when longer than one section), "on" or "off".
    framepack: str = "auto"
    section_frames: int = fp.TRAINED_FRAMES
    context_frames: int = 5
    anchor: bool = True
    preview: str = "off"
    preview_every: int | None = None

    @classmethod
    def from_dict(cls, d: dict) -> Options:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names and v is not None})

    def validate(self) -> Options:
        if self.resolution not in SIZES:
            raise ValueError(f"resolution must be one of {', '.join(SIZES)}")
        if self.aspect not in ASPECTS:
            raise ValueError(f"aspect must be one of {', '.join(ASPECTS)}")
        if self.preview not in PREVIEW_MODES:
            raise ValueError(f"preview must be one of {', '.join(PREVIEW_MODES)}")
        if self.framepack not in ("auto", "on", "off"):
            raise ValueError("framepack must be 'auto', 'on' or 'off'")
        if not 1 <= self.num_videos <= 4:
            raise ValueError("num_videos must be between 1 and 4")
        for name in ("width", "height"):
            v = getattr(self, name)
            if v is not None and not 64 <= v <= 1920:
                raise ValueError(f"{name} must be between 64 and 1920")
        if self.fps is not None and not 1 <= self.fps <= 60:
            raise ValueError("fps must be between 1 and 60")
        if self.frames is not None and not 1 <= self.frames <= fp.MAX_TOTAL_FRAMES:
            raise ValueError(f"frames must be between 1 and {fp.MAX_TOTAL_FRAMES}")
        if not 17 <= self.section_frames <= 161:
            raise ValueError("section_frames must be between 17 and 161")
        if not 1 <= self.context_frames <= 21:
            raise ValueError("context_frames must be between 1 and 21")
        return self


@dataclass
class Hooks:
    progress: Callable[[int, int], None] = lambda step, total: None
    status: Callable[[str], None] = lambda msg: None
    cancelled: Callable[[], bool] = lambda: False
    # Receives (animated PNG bytes, step) when Options.preview is enabled.
    preview: Callable[[bytes, int], None] = lambda data, step: None
    # Where output videos are written; a temporary folder when None.
    workdir: Path | None = None


@dataclass
class Result:
    videos: list[VideoFile]
    seed: int
    prompt: str
    variant: str
    info: dict = field(default_factory=dict)
    # Named side outputs, e.g. the control or pose video a run followed.
    extras: dict[str, VideoFile] = field(default_factory=dict)
    # Other downloadable files by name, e.g. an exported GIF.
    files: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------- helpers


def _round16(v: float) -> int:
    return max(64, int(round(v / 16)) * 16)


def output_size(opts: Options, source: tuple[int, int] | None = None) -> tuple[int, int]:
    """(width, height): explicit, else the aspect preset, else the source aspect at the tier's pixel count."""
    if opts.width and opts.height:
        return _round16(opts.width), _round16(opts.height)
    if opts.aspect != "auto" or source is None:
        aspect = opts.aspect if opts.aspect != "auto" else "16:9"
        w, h = SIZES[opts.resolution][aspect]
    else:
        area = SIZES[opts.resolution]["16:9"][0] * SIZES[opts.resolution]["16:9"][1]
        ratio = source[0] / source[1]
        w, h = math.sqrt(area * ratio), math.sqrt(area / ratio)
    if opts.width:
        return _round16(opts.width), _round16(opts.width * h / w)
    if opts.height:
        return _round16(opts.height * w / h), _round16(opts.height)
    return _round16(w), _round16(h)


def total_frames(opts: Options, fps: float, source_frames: int | None = None, default: int = fp.TRAINED_FRAMES) -> int:
    if opts.frames:
        n = opts.frames
    elif opts.seconds:
        n = int(round(opts.seconds * fps)) + 1
    elif source_frames:
        n = source_frames
    else:
        n = default
    if source_frames:
        n = min(n, source_frames)
    return max(1, min(int(n), fp.MAX_TOTAL_FRAMES))


def fit(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Scale to cover ``size`` and centre-crop (how videos are fitted too)."""
    return ImageOps.fit(img.convert("RGB"), size, Image.Resampling.LANCZOS)


def _pad_to(frames: list[Image.Image], n: int) -> list[Image.Image]:
    if not frames:
        raise ValueError("No frames")
    return frames + [frames[-1]] * (n - len(frames)) if len(frames) < n else frames[:n]


class _Source:
    """Frames of a driving video, fitted to the output size, consumed in order section by section."""

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int], start: float = 0.0, limit: int | None = None):
        self.path, self.fps, self.size = path, fps, size
        info = probe(path)
        self.info = info
        avail = max(1, int((max(0.0, info.duration - start)) * fps)) if info.duration else info.frames
        self.frames = min(avail, limit) if limit else avail
        self.start = start
        self._it: Iterator[Image.Image] | None = None
        self._last: Image.Image | None = None

    def rewind(self) -> None:
        self.close()
        self._it, self._last = None, None

    def take(self, n: int) -> list[Image.Image]:
        if self._it is None:
            self._it = read_frames(self.path, fps=self.fps, size=self.size, start=self.start, max_frames=self.frames, info=self.info)
        out = []
        for _ in range(n):
            try:
                self._last = next(self._it)
            except StopIteration:
                if self._last is None:
                    raise ValueError(f"Could not read frames from {Path(self.path).name}") from None
            out.append(self._last)
        return out

    def close(self):
        if self._it is not None:
            self._it.close()


@dataclass
class _Plan:
    """Everything one rendering pass needs; built by each operation."""

    kind: str
    prompt: str
    variant: Variant
    size: tuple[int, int]
    fps: float
    total: int
    sections: list[fp.Section]
    # Denoising steps per pipeline call.
    steps: int
    # (section, context frames, generator, run, extra writers) -> the section's frames (section.length of them)
    section: Callable
    # Names of side videos the section function writes, e.g. ("control",).
    extras: tuple[str, ...] = ()
    info: dict = field(default_factory=dict)
    # Called before each output video, e.g. to rewind the driving video for the next one.
    on_start: Callable[[], None] | None = None
    # Progress units per output video; defaults to steps x sections.
    work: int | None = None


class _Run:
    """Progress bookkeeping and the pipeline call shared by every section."""

    def __init__(self, engine: Engine, hooks: Hooks, opts: Options, plan: _Plan, videos: int):
        self.engine, self.hooks, self.opts, self.plan = engine, hooks, opts, plan
        self.total_steps = (plan.work or plan.steps * len(plan.sections)) * videos
        self.done = 0
        self.section: fp.Section | None = None

    def call(self, pipe, **kwargs) -> list[Image.Image]:
        hooks, opts = self.hooks, self.opts
        previewer = None
        if opts.preview != "off":
            previewer = Previewer(pipe, opts.preview, opts.preview_every, self.plan.fps, hooks.preview,
                                  mock=self.plan.variant.name == "mock")

        def on_step(p, i, t, kw):
            self.done += 1
            hooks.progress(min(self.done, self.total_steps), self.total_steps)
            if previewer is not None:
                previewer(i, kw.get("latents"))
            if hooks.cancelled():
                p._interrupt = True
            return kw

        if hooks.cancelled():
            raise Cancelled()
        self.engine._set_shift(pipe, self.opts, self.plan.size)
        with self.engine._lock:
            out = pipe(callback_on_step_end=on_step, output_type="pil", **kwargs)
        if hooks.cancelled():
            raise Cancelled()
        frames = out.frames[0]
        if not isinstance(frames[0], Image.Image):  # numpy frames
            import numpy as np

            frames = [Image.fromarray((np.clip(f, 0, 1) * 255).astype("uint8")) for f in frames]
        return frames

    def common(self, num_frames: int | None = None) -> dict:
        """Keyword arguments every Wan pipeline takes."""
        opts, (w, h) = self.opts, self.plan.size
        guidance = opts.guidance if opts.guidance is not None else (1.0 if self.plan.variant.family == "animate" else 5.0)
        kw = dict(prompt=self.plan.prompt, height=h, width=w, num_inference_steps=self.plan.steps, guidance_scale=guidance)
        if guidance > 1:
            kw["negative_prompt"] = opts.negative_prompt or DEFAULT_NEGATIVE
        if num_frames is not None:
            kw["num_frames"] = num_frames
        return kw


# ---------------------------------------------------------------------- engine


class Engine:
    def __init__(self, variant: str | None = None, offload: str = "auto"):
        self.offload = offload
        self.default_variant = variant
        self.variant: Variant | None = None
        self.pipe = None
        self._derived: dict = {}
        self._lock = threading.RLock()
        self.annotators = ann.Annotators(mock=variant == "mock")

    # ------------------------------------------------------------------ model management

    def load(self, variant: str | None = None, hooks: Hooks | None = None) -> Variant:
        from .models import build_pipeline

        hooks = hooks or Hooks()
        v = get_variant(variant or self.default_variant)
        with self._lock:
            if self.variant and self.variant.name == v.name and self.pipe is not None:
                return v
            self.unload()
            # Detectors only run during preprocessing; give their VRAM to the video model.
            self.annotators.release()
            hooks.status(f"Loading {v.label}")
            t0 = time.time()
            self.pipe = build_pipeline(v, self.offload, hooks.status)
            self.variant = v
            hooks.status(f"Loaded {v.label} in {time.time() - t0:.0f}s")
            return v

    def unload(self) -> None:
        with self._lock:
            if self.pipe is None:
                return
            self.pipe, self.variant, self._derived = None, None, {}
            import gc

            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def _pick(self, variant: str | None, families: tuple[str, ...]) -> Variant:
        return pick_variant(variant, self.default_variant, families)

    def _derived_pipe(self, name: str):
        from .models import derived_pipeline

        if name not in self._derived:
            self._derived[name] = derived_pipeline(self.pipe, name)
        return self._derived[name]

    @staticmethod
    def _set_shift(pipe, opts: Options, size: tuple[int, int]) -> None:
        shift = opts.shift if opts.shift is not None else (5.0 if size[0] * size[1] > 600_000 else 3.0)
        sch = pipe.scheduler
        config = getattr(sch, "config", {}) or {}
        if "flow_shift" in config and config["flow_shift"] != shift:
            pipe.scheduler = type(sch).from_config(config, flow_shift=shift)

    def _generator(self, seed: int):
        if self.variant and self.variant.name == "mock":
            import types

            return types.SimpleNamespace(initial_seed=lambda: seed)
        import torch

        return torch.Generator("cpu").manual_seed(seed)

    # ------------------------------------------------------------------ the sectioned renderer

    def _sections(self, opts: Options, total: int, variant: Variant, context: int | None = None) -> list[fp.Section]:
        sectioned = fp.use_framepack(opts.framepack, total, opts.section_frames)
        return fp.plan(total, opts.section_frames, context or opts.context_frames, sectioned)

    def _render(self, plan: _Plan, opts: Options, hooks: Hooks) -> Result:
        workdir = Path(hooks.workdir) if hooks.workdir else Path(tempfile.mkdtemp(prefix="wan-video-"))
        workdir.mkdir(parents=True, exist_ok=True)
        base_seed = opts.seed if opts.seed is not None and opts.seed >= 0 else random.randrange(2**31)
        run = _Run(self, hooks, opts, plan, opts.num_videos)
        n_sec = len(plan.sections)
        if n_sec > 1:
            hooks.status(f"FramePack: {plan.total} frames in {n_sec} sections of up to {plan.sections[0].length}")
        videos: list[VideoFile] = []
        extras: dict[str, VideoFile] = {}
        for k in range(opts.num_videos):
            seed = base_seed + k
            gen = self._generator(seed)
            if plan.on_start:
                plan.on_start()
            writer = VideoWriter(workdir / f"{k}.mp4", plan.fps)
            extra_writers = {name: VideoWriter(workdir / f"{name}.mp4", plan.fps) for name in plan.extras} if k == 0 else {}
            tail: deque[Image.Image] = deque(maxlen=max(s.context for s in plan.sections) or 1)
            try:
                for sec in plan.sections:
                    run.section = sec
                    if n_sec > 1:
                        hooks.status(f"Section {sec.index + 1}/{n_sec}" + (f" · video {k + 1}/{opts.num_videos}" if opts.num_videos > 1 else ""))
                    elif opts.num_videos > 1:
                        hooks.status(f"Sampling video {k + 1}/{opts.num_videos}")
                    else:
                        hooks.status("Sampling")
                    ctx = list(tail)[-sec.context:] if sec.context else []
                    frames = plan.section(sec, ctx, gen, run, extra_writers)
                    new = frames[sec.context:sec.context + sec.keep]
                    for f in new:
                        writer.write(f)
                        tail.append(f)
                videos.append(writer.close())
                for name, w in extra_writers.items():
                    if w.frames:
                        extras[name] = w.close()
            except BaseException:
                writer.abort()
                for w in extra_writers.values():
                    w.abort()
                raise
        info = plan.info | {
            "size": list(plan.size), "fps": plan.fps, "frames": plan.total, "sections": n_sec,
            "framepack": n_sec > 1, "seconds": round(plan.total / plan.fps, 2),
        }
        return Result(videos=videos, seed=base_seed, prompt=plan.prompt, variant=plan.variant.name, info=info, extras=extras)

    def _steps(self, opts: Options, v: Variant) -> int:
        return opts.steps or (20 if v.family == "animate" else 30)

    # ------------------------------------------------------------------ VACE inputs

    @staticmethod
    def _vace_inputs(sec: fp.Section, ctx: list[Image.Image], size: tuple[int, int], control: list[Image.Image] | None = None,
                     fixed: dict[int, Image.Image] | None = None, source: list[Image.Image] | None = None,
                     region: Image.Image | None = None) -> tuple[list | None, list | None]:
        """VACE ``video`` and ``mask`` lists for one section.

        Context frames and ``fixed`` frames (keyed by output frame index) are kept (mask black).
        With ``source`` + ``region`` the source frames are kept outside the region and regenerated
        inside it (video inpainting); with ``control`` the new frames follow the control maps;
        otherwise new frames are free (gray, mask white).
        """
        fixed = fixed or {}
        if not ctx and not control and source is None and not any(sec.start <= i < sec.start + sec.new for i in fixed):
            return None, None
        black, white = Image.new("RGB", size, (0, 0, 0)), Image.new("RGB", size, (255, 255, 255))
        gray = Image.new("RGB", size, GRAY)
        region_rgb = region.convert("RGB") if region is not None else white
        video, mask = [], []
        for j in range(sec.length):
            if j < sec.context:
                video.append(ctx[j]), mask.append(black)
                continue
            k = j - sec.context
            idx = sec.start + k
            if idx in fixed:
                video.append(fixed[idx]), mask.append(black)
            elif source is not None:
                video.append(source[k]), mask.append(region_rgb)
            elif control is not None:
                video.append(control[k]), mask.append(white)
            else:
                video.append(gray), mask.append(white)
        return video, mask

    def _vace_section(self, refs: Callable[[fp.Section], list[Image.Image]], strength: float = 1.0,
                      control: Callable | None = None, fixed: dict[int, Image.Image] | None = None,
                      source: Callable | None = None, region: Image.Image | None = None):
        """A section function for VACE: ``control``/``source`` give (section) -> frames for its new part."""

        def section(sec: fp.Section, ctx, gen, run: _Run, extra_writers):
            size = run.plan.size
            ctrl = control(sec, extra_writers) if control else None
            src = source(sec, extra_writers) if source else None
            video, mask = self._vace_inputs(sec, ctx, size, ctrl, fixed, src, region)
            kw = run.common(sec.length)
            if video is not None:
                kw |= {"video": video, "mask": mask}
            r = refs(sec)
            if r:
                kw["reference_images"] = [fit(x, size) for x in r]
            kw["conditioning_scale"] = float(strength)
            return run.call(self.pipe, generator=gen, **kw)

        return section

    @staticmethod
    def _anchor_refs(opts: Options, user_refs: list[Image.Image], anchor: Image.Image | None = None,
                     first_output: list | None = None) -> Callable[[fp.Section], list[Image.Image]]:
        """Reference images per section: the user's, plus the anchor on every later section (anti-drift)."""

        def refs(sec: fp.Section) -> list[Image.Image]:
            out = list(user_refs)
            if opts.anchor and sec.index > 0 and len(out) < MAX_REFERENCES:
                a = anchor if anchor is not None else (first_output[0] if first_output else None)
                if a is not None:
                    out.append(a)
            return out

        return refs

    @staticmethod
    def _check_refs(refs: list | None) -> list[Image.Image]:
        refs = list(refs or [])
        if len(refs) > MAX_REFERENCES:
            raise ValueError(f"At most {MAX_REFERENCES} reference images are supported")
        return refs

    # ------------------------------------------------------------------ 1. text-to-video

    def generate(self, prompt: str, references: list[Image.Image] | None = None, opts: Options | None = None,
                 hooks: Hooks | None = None, variant: str | None = None) -> Result:
        """Text-to-video. VACE variants also take reference images (subject / style) and support long videos."""
        opts, hooks = (opts or Options()).validate(), hooks or Hooks()
        if not prompt.strip():
            raise ValueError("A prompt is required")
        refs = self._check_refs(references)
        v = self._pick(variant, ("vace", "t2v"))
        fps = opts.fps or v.fps
        size = output_size(opts)
        total = total_frames(opts, fps)
        sections = self._sections(opts, total, v)
        if v.family == "t2v" and (len(sections) > 1 or refs):
            raise ValueError("Long videos and reference images need a VACE variant (e.g. vace-1.3b)")
        self.load(v.name, hooks)
        first: list[Image.Image] = []

        if v.family == "t2v":
            def section(sec, ctx, gen, run, _):
                return run.call(self.pipe, generator=gen, **run.common(sec.length))
        else:
            inner = self._vace_section(self._anchor_refs(opts, refs, first_output=first))

            def section(sec, ctx, gen, run, extra):
                frames = inner(sec, ctx, gen, run, extra)
                if sec.index == 0:
                    first[:] = frames[:1]
                return frames

        plan = _Plan("generate", prompt.strip(), v, size, fps, total, sections, self._steps(opts, v), section,
                     info={"references": len(refs)}, on_start=first.clear)
        return self._render(plan, opts, hooks)

    # ------------------------------------------------------------------ 2. image-to-video

    def image_to_video(self, image: Image.Image, prompt: str = "", last_image: Image.Image | None = None,
                       references: list[Image.Image] | None = None, opts: Options | None = None,
                       hooks: Hooks | None = None, variant: str | None = None) -> Result:
        """Animate a still image (it becomes the first frame), optionally ending on ``last_image``."""
        opts, hooks = (opts or Options()).validate(), hooks or Hooks()
        refs = self._check_refs(references)
        v = self._pick(variant, ("vace", "i2v"))
        fps = opts.fps or v.fps
        size = output_size(opts, image.size)
        total = total_frames(opts, fps)
        start = fit(image, size)
        end = fit(last_image, size) if last_image is not None else None
        text = prompt.strip() or "High quality video with natural, smooth motion."

        if v.family == "i2v":
            if refs:
                raise ValueError("Reference images need a VACE variant")
            sections = self._sections(opts, total, v, context=1)
            self.load(v.name, hooks)

            def section(sec, ctx, gen, run, _):
                kw = run.common(sec.length)
                first_frame = ctx[-1] if ctx else start
                if end is not None and sec.index == len(sections) - 1:
                    kw["last_image"] = end
                return run.call(self.pipe, image=first_frame, generator=gen, **kw)
        else:
            sections = self._sections(opts, total, v)
            self.load(v.name, hooks)
            fixed = {0: start}
            if end is not None:
                fixed[total - 1] = end
            section = self._vace_section(self._anchor_refs(opts, refs, anchor=start), fixed=fixed)

        plan = _Plan("image", text, v, size, fps, total, sections, self._steps(opts, v), section,
                     info={"last_image": end is not None, "references": len(refs)})
        return self._render(plan, opts, hooks)

    # ------------------------------------------------------------------ control maps from a driving video

    def _control_maps(self, kind: str, src: _Source | None, size: tuple[int, int], options: dict,
                      paths: list[dict] | None = None, total: int = 0) -> Callable:
        """(section, extra writers) -> control frames for the section's new part, recorded to the 'control' extra."""
        options = dict(options or {})
        invert = bool(options.pop("invert", False))
        smooth = float(options.pop("smooth", 0.5))
        state = {"smoother": ann.PoseSmoother(smooth), "fn": None, "ready": False, "ref": options.pop("reference_pose", None)}

        def one(frame: Image.Image) -> Image.Image:
            if kind == "raw":
                return ann.prepare_map(frame, invert)
            if kind == "pose":
                pose = state["smoother"](self.annotators.detect_pose(frame, options.get("detector", "dwpose"),
                                                                     options.get("hands", True), options.get("face", True)))
                if state["ref"] is not None and not state["ready"]:
                    state["fn"], state["ready"] = ann.retarget_transform(pose, state["ref"]), True
                if state["fn"]:
                    pose = state["fn"](pose)
                return ann.render_pose(pose, *size, hands=options.get("hands", True), face=options.get("face", True))
            cmap, _ = self.annotators.detect(kind, frame, **options)
            return ImageOps.invert(cmap) if invert else cmap

        def maps(sec: fp.Section, extra_writers) -> list[Image.Image]:
            if kind == "path":
                out = [ann.render_paths(paths, sec.start + k, total, *size) for k in range(sec.new)]
            else:
                frames = src.take(min(sec.new, sec.keep))
                out = _pad_to([one(f) for f in frames], sec.new)
            if "control" in extra_writers:
                for f in out[:sec.keep]:
                    extra_writers["control"].write(f)
            return out

        def reset():
            if src is not None:
                src.rewind()
            state.update(smoother=ann.PoseSmoother(smooth), fn=None, ready=False)

        maps.reset = reset
        return maps

    def detect(self, video: str | Path, control_type: str = "depth", opts: Options | None = None,
               hooks: Hooks | None = None, detector: str = "dwpose", smooth: float = 0.5, hands: bool = True,
               face: bool = True, invert: bool = False, start: float = 0.0, reference: Image.Image | None = None) -> Result:
        """Preview the control video an operation would follow, without loading a Wan model."""
        opts, hooks = (opts or Options()).validate(), hooks or Hooks()
        if control_type not in ann.CONTROL_TYPES:
            raise ValueError(f"control_type must be one of {', '.join(ann.CONTROL_TYPES)}")
        fps = opts.fps or 16
        info = probe(video)
        size = output_size(opts, reference.size if reference is not None else (info.width, info.height))
        src = _Source(video, fps, size, start, total_frames(opts, fps, 10**9) if (opts.frames or opts.seconds) else None)
        total = src.frames
        options = {"detector": detector, "hands": hands, "face": face, "smooth": smooth, "invert": invert}
        if reference is not None and control_type == "pose":
            options["reference_pose"] = self.annotators.detect_pose(fit(reference, size), detector, hands, face)
        maps = self._control_maps(control_type, src, size, options)
        workdir = Path(hooks.workdir) if hooks.workdir else Path(tempfile.mkdtemp(prefix="wan-video-"))
        writer = VideoWriter(workdir / "0.mp4", fps)
        hooks.status(f"Extracting {control_type} from {total} frames")
        try:
            done = 0
            for sec in fp.plan(total, 81, 1, True):
                plain = fp.Section(sec.index, sec.start, 0, sec.keep, sec.keep)
                for f in maps(plain, {}):
                    writer.write(f)
                done += sec.keep
                hooks.progress(done, total)
                if hooks.cancelled():
                    raise Cancelled()
            out = writer.close()
        except BaseException:
            writer.abort()
            raise
        finally:
            src.close()
        return Result(videos=[out], seed=-1, prompt="", variant="", info={"control_type": control_type, "size": list(size), "fps": fps, "frames": total})

    # ------------------------------------------------------------------ 3. VACE motion control

    def control(self, prompt: str, video: str | Path | None = None, control_type: str = "depth",
                references: list[Image.Image] | None = None, first_frame: Image.Image | None = None,
                last_frame: Image.Image | None = None, paths: list[dict] | None = None, strength: float = 1.0,
                detect_options: dict | None = None, start: float = 0.0, opts: Options | None = None,
                hooks: Hooks | None = None, variant: str | None = None) -> Result:
        """Generate guided by a control video (VACE): the structure and motion of ``video`` as depth,
        edges, line art, pose... or a hand-drawn motion layout (``control_type="path"``: each path
        moves one object across the frame). Optional subject references and first/last frames."""
        opts, hooks = (opts or Options()).validate(), hooks or Hooks()
        if control_type not in CONTROL_KINDS:
            raise ValueError(f"control_type must be one of {', '.join(CONTROL_KINDS)}")
        if not prompt.strip():
            raise ValueError("A prompt describing the video is required")
        if not 0 <= strength <= 2:
            raise ValueError("strength must be between 0 and 2")
        refs = self._check_refs(references)
        v = self._pick(variant, ("vace",))
        fps = opts.fps or v.fps
        src = None
        if control_type == "path":
            paths = ann.validate_paths(paths)
            basis = first_frame.size if first_frame is not None else None
            size = output_size(opts, basis)
            total = total_frames(opts, fps)
        else:
            if video is None:
                raise ValueError("A control video is required (or use control_type='path' with motion paths)")
            info = probe(video)
            basis = first_frame.size if first_frame is not None else (info.width, info.height)
            size = output_size(opts, basis)
            src = _Source(video, fps, size, start)
            total = total_frames(opts, fps, src.frames)
        sections = self._sections(opts, total, v)
        fixed = {}
        if first_frame is not None:
            fixed[0] = fit(first_frame, size)
        if last_frame is not None:
            fixed[total - 1] = fit(last_frame, size)
        try:
            self.load(v.name, hooks)
            maps = self._control_maps(control_type, src, size, detect_options or {}, paths, total)
            anchor = fixed.get(0)
            first: list[Image.Image] = []
            inner = self._vace_section(self._anchor_refs(opts, refs, anchor=anchor, first_output=first), strength,
                                       control=maps, fixed=fixed)

            def section(sec, ctx, gen, run, extra):
                frames = inner(sec, ctx, gen, run, extra)
                if sec.index == 0:
                    first[:] = frames[:1]
                return frames

            plan = _Plan("control", prompt.strip(), v, size, fps, total, sections, self._steps(opts, v), section,
                         extras=("control",), info={"control_type": control_type, "strength": strength, "references": len(refs)},
                         on_start=lambda: (maps.reset(), first.clear()))
            return self._render(plan, opts, hooks)
        finally:
            if src:
                src.close()

    # ------------------------------------------------------------------ 4. pose-guided motion transfer

    def pose(self, video: str | Path, image: Image.Image | None = None, prompt: str = "",
             references: list[Image.Image] | None = None, detector: str = "dwpose", align: bool = True,
             smooth: float = 0.5, hands: bool = True, face: bool = True, first_frame: bool = False,
             strength: float = 1.0, start: float = 0.0, opts: Options | None = None, hooks: Hooks | None = None,
             variant: str | None = None) -> Result:
        """Transfer the motion of the person in ``video`` to the character in ``image``.

        Each frame's pose is detected (DWPose or OpenPose), smoothed over time and, with ``align``,
        moved and scaled onto the character's body, then rendered as a skeleton video that VACE
        follows while the character image is given as the subject reference.
        """
        opts, hooks = (opts or Options()).validate(), hooks or Hooks()
        if detector not in ann.POSE_DETECTORS:
            raise ValueError(f"detector must be one of {', '.join(ann.POSE_DETECTORS)}")
        refs = self._check_refs(references)
        if image is not None and len(refs) >= MAX_REFERENCES:
            raise ValueError(f"At most {MAX_REFERENCES - 1} extra references with a character image")
        v = self._pick(variant, ("vace",))
        fps = opts.fps or v.fps
        info = probe(video)
        size = output_size(opts, image.size if image is not None else (info.width, info.height))
        character = fit(image, size) if image is not None else None
        src = _Source(video, fps, size, start)
        total = total_frames(opts, fps, src.frames)
        sections = self._sections(opts, total, v)
        text = prompt.strip() or ("The character from the reference image moving naturally, high quality video."
                                  if character is not None else "A person moving naturally, high quality video.")
        try:
            options = {"detector": detector, "hands": hands, "face": face, "smooth": smooth}
            if character is not None and align:
                hooks.status("Detecting the character's pose")
                options["reference_pose"] = self.annotators.detect_pose(character, detector, hands, face)
            maps = self._control_maps("pose", src, size, options)
            self.load(v.name, hooks)
            all_refs = ([character] if character is not None else []) + refs
            fixed = {0: character} if first_frame and character is not None else {}
            section = self._vace_section(lambda sec: all_refs, strength, control=maps, fixed=fixed)
            plan = _Plan("pose", text, v, size, fps, total, sections, self._steps(opts, v), section, extras=("control",),
                         info={"detector": detector, "aligned": bool(character is not None and align), "strength": strength},
                         on_start=maps.reset)
            return self._render(plan, opts, hooks)
        finally:
            src.close()

    # ------------------------------------------------------------------ 5. Wan-Animate

    def animate(self, image: Image.Image, video: str | Path, prompt: str = "", mode: str = "animate",
                detector: str = "dwpose", align: bool = True, smooth: float = 0.5, start: float = 0.0,
                opts: Options | None = None, hooks: Hooks | None = None, variant: str | None = None) -> Result:
        """Wan2.2-Animate: move the character in ``image`` with the body *and facial* motion of ``video``.

        ``mode="animate"`` renders the character in its own scene; ``mode="replace"`` puts the
        character into the video in place of the person there, keeping the background and
        matching the lighting. The pose video, face crops (for expressions) and, for replace, the
        person mask are extracted from ``video`` automatically.
        """
        opts, hooks = (opts or Options()).validate(), hooks or Hooks()
        if mode not in ANIMATE_MODES:
            raise ValueError(f"mode must be one of {', '.join(ANIMATE_MODES)}")
        if detector not in ann.POSE_DETECTORS:
            raise ValueError(f"detector must be one of {', '.join(ann.POSE_DETECTORS)}")
        v = self._pick(variant, ("animate",))
        fps = opts.fps or v.fps
        info = probe(video)
        size = output_size(opts, image.size if mode == "animate" else (info.width, info.height))
        w, h = size
        character = fit(image, size)
        src = _Source(video, fps, size, start)
        total = total_frames(opts, fps, src.frames)
        seg = fp.floor_4k1(min(opts.section_frames, 81)) if opts.section_frames != fp.TRAINED_FRAMES else 77
        prev = 5 if opts.context_frames >= 5 else 1
        # Wan-Animate segments internally; FramePack bounds memory by feeding it windows of segments.
        sectioned = fp.use_framepack(opts.framepack, total, seg * 4)
        window = (seg - prev) * 4 + prev if sectioned else total
        windows = [fp.Section(i, s, 0, min(window, total - s), min(window, total - s)) for i, s in enumerate(range(0, total, window))]
        steps = self._steps(opts, v)
        segs = sum(max(1, math.ceil((s.length - prev) / (seg - prev))) for s in windows)
        text = prompt.strip() or "The character is moving naturally, high quality video."
        state: dict = {}

        def reset():
            src.rewind()
            state.update(smoother=ann.PoseSmoother(smooth), transform=None, ready=False, box=None)
        ref_pose = None
        try:
            if mode == "animate" and align:
                hooks.status("Detecting the character's pose")
                ref_pose = self.annotators.detect_pose(character, detector)
            self.load(v.name, hooks)

            def section(sec, ctx, gen, run, extra_writers):
                frames = src.take(sec.length)
                poses, faces, bgs, masks = [], [], [], []
                for k, frame in enumerate(frames):
                    if k % 8 == 0:
                        hooks.status(f"Preprocessing frames {sec.start + k + 1}-{min(total, sec.start + k + 8)} of {total}")
                    pose = state["smoother"](self.annotators.detect_pose(frame, detector))
                    person = ann.main_person(pose)
                    box = ann.face_box(person, w, h, state["box"])
                    if state["box"]:  # steady the crop
                        box = tuple(int(0.6 * a + 0.4 * b) for a, b in zip(box, state["box"]))
                    state["box"] = box
                    faces.append(ann.face_crop(frame, box))
                    if mode == "replace":
                        m = ann.person_mask(pose, w, h)
                        masks.append(m.convert("RGB"))
                        bgs.append(Image.composite(Image.new("RGB", size), frame, m))
                    if ref_pose is not None and not state["ready"]:
                        state["transform"], state["ready"] = ann.retarget_transform(pose, ref_pose), True
                    if state["transform"]:
                        pose = state["transform"](pose)
                    poses.append(ann.render_pose(pose, w, h, hands=True, face=False))
                if "pose" in extra_writers:
                    for p in poses:
                        extra_writers["pose"].write(p)
                if "mask" in extra_writers:
                    for m in masks:
                        extra_writers["mask"].write(m)
                kw = run.common()
                kw |= dict(image=character, pose_video=poses, face_video=faces, mode=mode, segment_frame_length=seg,
                           prev_segment_conditioning_frames=prev)
                if mode == "replace":
                    kw |= {"background_video": bgs, "mask_video": masks}
                return run.call(self.pipe, generator=gen, **kw)[:sec.length]

            plan = _Plan("animate", text, v, size, fps, total, windows, steps, section,
                         extras=("pose", "mask") if mode == "replace" else ("pose",),
                         info={"mode": mode, "detector": detector, "segment_frames": seg, "segments": segs,
                               "aligned": ref_pose is not None},
                         on_start=reset, work=steps * segs)  # progress counts every internal segment
            return self._render(plan, opts, hooks)
        finally:
            src.close()

    # ------------------------------------------------------------------ 6. video-to-video

    def v2v(self, video: str | Path, prompt: str, method: str = "restyle", control_type: str = "depth",
            strength: float = 0.7, references: list[Image.Image] | None = None, mask: Image.Image | None = None,
            start: float = 0.0, detect_options: dict | None = None, opts: Options | None = None,
            hooks: Hooks | None = None, variant: str | None = None) -> Result:
        """Transform an existing video.

        ``restyle`` (VACE) keeps the video's structure and motion through a control video (depth by
        default) and repaints everything else from the prompt, e.g. realistic -> anime; optional
        style reference images. ``sdedit`` (T2V) re-noises the video by ``strength`` and denoises
        it towards the prompt: lower keeps more of the original. ``inpaint`` (VACE) regenerates only
        the region in ``mask`` (white) on every frame and keeps the rest.
        """
        opts, hooks = (opts or Options()).validate(), hooks or Hooks()
        if method not in V2V_METHODS:
            raise ValueError(f"method must be one of {', '.join(V2V_METHODS)}")
        if not prompt.strip():
            raise ValueError("A prompt describing the result is required")
        refs = self._check_refs(references)
        families = ("t2v",) if method == "sdedit" else ("vace",)
        v = self._pick(variant, families)
        fps = opts.fps or v.fps
        info = probe(video)
        size = output_size(opts, (info.width, info.height))
        src = _Source(video, fps, size, start)
        total = total_frames(opts, fps, src.frames)
        try:
            if method == "sdedit":
                if not 0.05 <= strength <= 1:
                    raise ValueError("strength must be between 0.05 and 1")
                if refs:
                    raise ValueError("Reference images need the 'restyle' or 'inpaint' method")
                # Independent chunks: the source video itself carries continuity between them.
                chunk = fp.floor_4k1(opts.section_frames) if fp.use_framepack(opts.framepack, total, opts.section_frames) else total
                if chunk > fp.MAX_SINGLE_PASS:
                    raise ValueError(f"{total} frames is too long for a single pass; turn on FramePack for long videos")
                sections = [fp.Section(s.index, s.start, 0, fp.round_4k1(s.keep), s.keep) for s in _chunks(total, chunk)]
                self.load(v.name, hooks)
                v2v_pipe = self._derived_pipe("v2v")

                def section(sec, ctx, gen, run, _):
                    frames = _pad_to(src.take(sec.keep), sec.length)
                    kw = run.common()
                    kw |= {"video": frames, "strength": float(strength)}
                    return run.call(v2v_pipe, generator=gen, **kw)

                steps = self._steps(opts, v)
                # The pipeline skips the first (1 - strength) of the schedule.
                plan = _Plan("v2v", prompt.strip(), v, size, fps, total, sections, steps, section,
                             info={"method": method, "strength": strength}, on_start=src.rewind,
                             work=max(1, int(steps * strength)) * len(sections))
                return self._render(plan, opts, hooks)

            sections = self._sections(opts, total, v)
            self.load(v.name, hooks)
            first: list[Image.Image] = []
            if method == "restyle":
                maps = self._control_maps(control_type, src, size, detect_options or {})
                inner = self._vace_section(self._anchor_refs(opts, refs, first_output=first), strength=1.0, control=maps)
                info_extra = {"control_type": control_type}
                extras = ("control",)
                reset = maps.reset
            else:
                if sel.is_empty(mask):
                    raise ValueError("Inpainting needs a mask or selection over the region to regenerate")
                region = sel.grow(sel.mask_from_image(mask, mask.size), 4)
                region = ImageOps.fit(region, size, Image.Resampling.NEAREST)

                def source(sec, _extra):
                    return _pad_to(src.take(min(sec.new, sec.keep)), sec.new)

                inner = self._vace_section(self._anchor_refs(opts, refs), source=source, region=region)
                info_extra = {"masked": True}
                extras = ()
                reset = src.rewind

            def section(sec, ctx, gen, run, extra):
                frames = inner(sec, ctx, gen, run, extra)
                if sec.index == 0:
                    first[:] = frames[:1]
                return frames

            plan = _Plan("v2v", prompt.strip(), v, size, fps, total, sections, self._steps(opts, v), section,
                         extras=extras, info={"method": method, "references": len(refs)} | info_extra,
                         on_start=lambda: (reset(), first.clear()))
            return self._render(plan, opts, hooks)
        finally:
            src.close()


    # ------------------------------------------------------------------ 7. export

    def export(self, video: str | Path, fmt: str = "mp4", fps: float | None = None, width: int | None = None,
               quality: str = "high", hooks: Hooks | None = None) -> Result:
        """Convert any video to MP4, WebM or GIF with FFmpeg (downloaded automatically if missing)."""
        from .video import EXPORT_FORMATS
        from .video import export as encode

        hooks = hooks or Hooks()
        if fmt not in EXPORT_FORMATS:
            raise ValueError(f"format must be one of {', '.join(EXPORT_FORMATS)}")
        workdir = Path(hooks.workdir) if hooks.workdir else Path(tempfile.mkdtemp(prefix="wan-video-"))
        info = probe(video)
        name = f"export.{fmt}"
        out = encode(video, workdir / name, fmt, fps=fps, width=width, quality=quality, report=hooks.status)
        return Result(videos=[], seed=-1, prompt="", variant="", files={name: str(out)},
                      info={"format": fmt, "source": info.to_dict(), "bytes": out.stat().st_size})


def _chunks(total: int, size: int) -> list[fp.Section]:
    out, start = [], 0
    while start < total:
        keep = min(size, total - start)
        out.append(fp.Section(len(out), start, 0, keep, keep))
        start += keep
    return out

