"""HTTP API (and the static web UI that uses it).

Every operation is a job: ``POST`` returns the job immediately (``202``) and clients poll
``GET /api/jobs/{id}``; pass ``wait=true`` to block until it finishes instead. Images and videos
are sent as multipart file fields so the API is easy to drive with ``curl -F``.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import shutil
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from . import __version__
from . import annotators as ann
from . import ffmpeg
from . import selection as sel
from .engine import ANIMATE_MODES, ASPECTS, CONTROL_KINDS, SIZES, V2V_METHODS, Engine, Options
from .jobs import TERMINAL, Job, JobManager
from .preview import MODES as PREVIEW_MODES
from .variants import DEFAULT_VARIANT, FAMILY_DEFAULTS, get_variant, list_variants, pick_variant
from .video import EXPORT_FORMATS, MEDIA_TYPES, export, probe

log = logging.getLogger(__name__)
WEB_DIR = Path(__file__).parent / "web"

# Which model families can run each operation (the first is the fallback family).
OPERATIONS = {
    "generate": ("vace", "t2v"),
    "image": ("vace", "i2v"),
    "control": ("vace",),
    "pose": ("vace",),
    "animate": ("animate",),
    "v2v": ("vace", "t2v"),
}


def _read_image(upload: UploadFile, name: str) -> Image.Image:
    data = upload.file.read()
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(400, f"'{name}' is not a readable image: {e}") from e
    img = ImageOps.exif_transpose(img)
    return img.convert("RGBA" if "A" in img.getbands() else "RGB")


def _image(upload: UploadFile | None, name: str) -> Image.Image | None:
    return _read_image(upload, name) if upload is not None and upload.filename else None


def _images(uploads: list[UploadFile], name: str) -> list[Image.Image]:
    return [_read_image(u, name) for u in uploads if u.filename]


def _json(text: str | None, name: str, kind=dict):
    if not text:
        return None
    try:
        value = json.loads(text)
    except ValueError as e:
        raise HTTPException(400, f"'{name}' is not valid JSON: {e}") from e
    if not isinstance(value, kind):
        raise HTTPException(400, f"'{name}' must be a JSON {'object' if kind is dict else 'list'}")
    return value


def common_options(
    resolution: str = Form("480p", description="Resolution tier: 480p or 720p"),
    aspect: str = Form("auto", description="auto (follow the input), 16:9, 9:16, 1:1, 4:3 or 3:4"),
    width: int | None = Form(None, description="Explicit width (multiple of 16); overrides resolution/aspect"),
    height: int | None = Form(None),
    frames: int | None = Form(None, description="Output length in frames (default 81, or the driving video's length)"),
    seconds: float | None = Form(None, description="Output length in seconds (used when frames is not given)"),
    fps: float | None = Form(None, description="Frame rate (default: the model's, 16 for Wan 2.1 and 30 for Animate)"),
    steps: int | None = Form(None, description="Denoising steps (default 30; Animate 20)"),
    seed: int | None = Form(None, description="Random seed; omit or -1 for random"),
    guidance: float | None = Form(None, description="CFG scale (default 5.0; Animate 1.0)"),
    negative_prompt: str | None = Form(None),
    shift: float | None = Form(None, description="Flow shift (default 3.0 at 480p, 5.0 at 720p)"),
    num_videos: int = Form(1, description="Videos to generate (1-4), with consecutive seeds"),
    framepack: str = Form("auto", description="Constant-VRAM sectioned sampling for long videos: auto, on or off"),
    section_frames: int = Form(81, description="FramePack: frames per section (4k+1)"),
    context_frames: int = Form(5, description="FramePack: frames carried over between sections"),
    anchor: bool = Form(True, description="FramePack: anchor every section on the first frame/reference (anti-drift)"),
    preview: str = Form("fast", description="Live previews: off, fast (approximate) or full (VAE decode, slower)"),
    preview_every: int | None = Form(None, description="Preview every N steps (default 1 for fast, 5 for full)"),
) -> Options:
    opts = Options.from_dict(dict(locals()))
    try:
        return opts.validate()
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


def create_app(engine: Engine | None = None, output_dir: str | Path = "outputs") -> FastAPI:
    engine = engine or Engine(DEFAULT_VARIANT)
    jobs = JobManager(engine, output_dir)
    uploads = Path(output_dir) / "_uploads"
    app = FastAPI(title="poly-wan", version=__version__,
                  description="Wan video generation: text/image-to-video, VACE control, pose transfer, Wan-Animate, video-to-video and export.")
    app.state.engine, app.state.jobs = engine, jobs

    async def respond(job: Job, wait: bool) -> JSONResponse:
        if wait:
            await asyncio.to_thread(job.done_event.wait)
        body = job.to_dict() | {"queue_position": jobs.queue_position(job)}
        return JSONResponse(body, status_code=200 if job.status in TERMINAL else 202)

    async def save_video(upload: UploadFile | None, name: str, required: bool = True) -> Path | None:
        """Store an uploaded video so FFmpeg can read it; checks that it decodes."""
        if upload is None or not upload.filename:
            if required:
                raise HTTPException(400, f"'{name}' (a video file) is required")
            return None
        uploads.mkdir(parents=True, exist_ok=True)
        path = uploads / f"{uuid.uuid4().hex[:12]}{Path(upload.filename).suffix.lower() or '.mp4'}"
        with open(path, "wb") as fh:
            shutil.copyfileobj(upload.file, fh)
        try:
            await asyncio.to_thread(probe, path)
        except ValueError as e:
            path.unlink(missing_ok=True)
            raise HTTPException(400, f"'{name}' is not a readable video: {e}") from e
        except ffmpeg.FFmpegMissing as e:
            path.unlink(missing_ok=True)
            raise HTTPException(503, str(e)) from e
        return path

    def check_variant(variant: str | None, op: str) -> str | None:
        try:
            return pick_variant(variant or None, engine.default_variant, OPERATIONS[op]).name
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e.args[0])) from e

    def submit(kind: str, params: dict, fn, cleanup: list[Path | None] = ()) -> Job:
        def run(hooks):
            try:
                return fn(hooks)
            finally:
                for p in cleanup:
                    if p:
                        Path(p).unlink(missing_ok=True)

        return jobs.submit(kind, params, run)

    def opts_params(opts: Options) -> dict:
        return {k: v for k, v in opts.__dict__.items() if v is not None}

    # ------------------------------------------------------------------ meta

    @app.get("/api/health")
    def health():
        return {"ok": True, "version": __version__, "loaded": engine.variant.name if engine.variant else None}

    @app.get("/api/models")
    async def models():
        from . import dwpose
        from .models import is_downloaded, resolve_offload

        async def row(v):
            downloaded = await asyncio.to_thread(is_downloaded, v)
            try:
                offload = resolve_offload(engine.offload, v)
            except Exception:
                offload = engine.offload
            return v.to_dict() | {"downloaded": downloaded, "offload": offload,
                                  "loaded": bool(engine.variant and engine.variant.name == v.name)}

        rows = await asyncio.gather(*(row(v) for v in list_variants()))
        return {
            "default": engine.default_variant or DEFAULT_VARIANT,
            "family_defaults": FAMILY_DEFAULTS,
            "loaded": engine.variant.name if engine.variant else None,
            "operations": {k: list(v) for k, v in OPERATIONS.items()},
            "variants": rows,
            "dwpose": {"repo": dwpose.DWPOSE_REPO, "approx_download_gb": dwpose.DWPOSE_GB,
                       "downloaded": engine.annotators.mock or await asyncio.to_thread(dwpose.is_downloaded)},
            "ffmpeg": await asyncio.to_thread(ffmpeg.status),
        }

    @app.post("/api/models/load", status_code=202)
    async def load_model(variant: str = Form(...), wait: bool = Form(False)):
        try:
            v = get_variant(variant).name
        except KeyError as e:
            raise HTTPException(400, str(e.args[0])) from e

        def run(hooks):
            engine.load(v, hooks)
            engine.default_variant = v

        return await respond(jobs.submit("load", {"variant": v}, run), wait)

    @app.post("/api/models/download", status_code=202)
    async def download_model(variant: str = Form(...), wait: bool = Form(False)):
        from .models import download

        try:
            v = get_variant(variant)
        except KeyError as e:
            raise HTTPException(400, str(e.args[0])) from e
        return await respond(jobs.submit("download", {"variant": v.name}, lambda hooks: download(v, hooks.status) and None), wait)

    @app.post("/api/models/unload")
    def unload_model():
        engine.unload()
        return {"ok": True}

    @app.post("/api/dwpose/download", status_code=202, summary="Download the DWPose pose estimator (0.35 GB)")
    async def dwpose_download(wait: bool = Form(False)):
        from . import dwpose

        return await respond(jobs.submit("download", {"variant": "dwpose"}, lambda hooks: dwpose.download(hooks.status) and None), wait)

    @app.get("/api/ffmpeg", summary="Where FFmpeg is, or whether it still has to be downloaded")
    async def ffmpeg_status():
        return await asyncio.to_thread(ffmpeg.status)

    @app.post("/api/ffmpeg/download", status_code=202, summary="Download a static FFmpeg build into the cache")
    async def ffmpeg_download(wait: bool = Form(False), force: bool = Form(False, description="Download even if one is found")):
        def run(hooks):
            (ffmpeg.download if force else ffmpeg.ffmpeg_path)(report=hooks.status)

        return await respond(jobs.submit("download", {"variant": "ffmpeg"}, run), wait)

    @app.get("/api/options", summary="Choices for the operation fields")
    def options():
        return {
            "resolutions": list(SIZES), "aspects": list(ASPECTS), "sizes": SIZES,
            "control_types": ann.CONTROL_TYPES | {"path": "Motion paths (draw how objects move)"},
            "pose_detectors": ann.POSE_DETECTORS, "v2v_methods": list(V2V_METHODS), "animate_modes": list(ANIMATE_MODES),
            "export_formats": list(EXPORT_FORMATS), "preview_modes": list(PREVIEW_MODES), "template_pose": ann.template_pose(),
        }

    # ------------------------------------------------------------------ 1. text-to-video

    @app.post("/api/generate", status_code=202, summary="Text-to-video")
    async def generate(
        prompt: str = Form(...),
        references: list[UploadFile] = File(default=[], description="Optional subject/style reference images (VACE, up to 4)"),
        variant: str | None = Form(None), wait: bool = Form(False), opts: Options = Depends(common_options),
    ):
        refs = _images(references, "references")
        v = check_variant(variant, "generate")
        params = {"prompt": prompt, "references": len(refs)} | opts_params(opts)
        return await respond(submit("generate", params, lambda hooks: engine.generate(prompt, refs, opts, hooks, v)), wait)

    # ------------------------------------------------------------------ 2. image-to-video

    @app.post("/api/image", status_code=202, summary="Image-to-video: animate a still image, optional text prompt")
    async def image_to_video(
        image: UploadFile = File(..., description="The first frame"),
        prompt: str = Form("", description="Optional description of the motion"),
        last_image: UploadFile | None = File(None, description="Optional last frame (VACE or FLF2V)"),
        references: list[UploadFile] = File(default=[], description="Optional subject references (VACE)"),
        variant: str | None = Form(None), wait: bool = Form(False), opts: Options = Depends(common_options),
    ):
        img, last, refs = _read_image(image, "image"), _image(last_image, "last_image"), _images(references, "references")
        v = check_variant(variant, "image")
        params = {"prompt": prompt, "last_image": last is not None, "references": len(refs)} | opts_params(opts)
        job = submit("image", params, lambda hooks: engine.image_to_video(img, prompt, last, refs, opts, hooks, v))
        return await respond(job, wait)

    # ------------------------------------------------------------------ 3. VACE control

    @app.post("/api/control", status_code=202, summary="Motion control with VACE: follow a control video or motion paths")
    async def control(
        prompt: str = Form(..., description="Describe the video to create"),
        video: UploadFile | None = File(None, description="Driving video whose structure/motion to follow"),
        control_type: str = Form("depth", description=f"One of: {', '.join(CONTROL_KINDS)}"),
        paths: str | None = Form(None, description='path type: JSON [{"points": [[x, y], ...], "radius": 0.08}], 0..1 coordinates'),
        references: list[UploadFile] = File(default=[], description="Optional subject references (up to 4)"),
        first_frame: UploadFile | None = File(None, description="Optional fixed first frame"),
        last_frame: UploadFile | None = File(None, description="Optional fixed last frame"),
        strength: float = Form(1.0, description="Control strength 0-2"),
        invert: bool = Form(False, description="Invert the control maps (black-on-white drawings)"),
        detector: str = Form("dwpose", description="pose: dwpose or openpose"),
        detect_options: str | None = Form(None, description='JSON detector options, e.g. {"low_threshold": 50}'),
        start: float = Form(0.0, description="Start time in the driving video (seconds)"),
        variant: str | None = Form(None), wait: bool = Form(False), opts: Options = Depends(common_options),
    ):
        if control_type not in CONTROL_KINDS:
            raise HTTPException(400, f"control_type must be one of {', '.join(CONTROL_KINDS)}")
        path_data = None
        if control_type == "path":
            try:
                path_data = ann.validate_paths(_json(paths, "paths", list))
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
        refs = _images(references, "references")
        first, last = _image(first_frame, "first_frame"), _image(last_frame, "last_frame")
        det = (_json(detect_options, "detect_options") or {}) | {"detector": detector, "invert": invert}
        v = check_variant(variant, "control")
        src = await save_video(video, "video", required=control_type != "path")
        params = {"prompt": prompt, "control_type": control_type, "strength": strength, "references": len(refs),
                  "first_frame": first is not None, "last_frame": last is not None} | opts_params(opts)
        job = submit("control", params, lambda hooks: engine.control(
            prompt, src, control_type, refs, first, last, path_data, strength, det, start, opts, hooks, v), [src])
        return await respond(job, wait)

    @app.post("/api/control/detect", status_code=202, summary="Preview the control video extracted from a video")
    async def control_detect(
        video: UploadFile = File(...),
        control_type: str = Form("depth"),
        detector: str = Form("dwpose"),
        smooth: float = Form(0.5, description="pose: temporal smoothing 0-0.95"),
        hands: bool = Form(True), face: bool = Form(True), invert: bool = Form(False),
        reference: UploadFile | None = File(None, description="pose: retarget the skeleton onto this character"),
        start: float = Form(0.0),
        wait: bool = Form(False), opts: Options = Depends(common_options),
    ):
        if control_type not in ann.CONTROL_TYPES:
            raise HTTPException(400, f"control_type must be one of {', '.join(ann.CONTROL_TYPES)}")
        if detector not in ann.POSE_DETECTORS:
            raise HTTPException(400, f"detector must be one of {', '.join(ann.POSE_DETECTORS)}")
        ref = _image(reference, "reference")
        src = await save_video(video, "video")
        params = {"control_type": control_type, "detector": detector} | opts_params(opts)
        job = submit("detect", params, lambda hooks: engine.detect(
            src, control_type, opts, hooks, detector, smooth, hands, face, invert, start, ref), [src])
        return await respond(job, wait)

    # ------------------------------------------------------------------ 4. pose transfer

    @app.post("/api/pose", status_code=202, summary="Pose-guided motion transfer (DWPose/OpenPose + VACE)")
    async def pose(
        video: UploadFile = File(..., description="Driving video: the motion to copy"),
        image: UploadFile | None = File(None, description="The character to move (subject reference)"),
        prompt: str = Form("", description="Optional description of the result"),
        references: list[UploadFile] = File(default=[], description="Extra references (e.g. an outfit or a background)"),
        detector: str = Form("dwpose", description="dwpose or openpose"),
        align: bool = Form(True, description="Move and scale the skeleton onto the character's body"),
        smooth: float = Form(0.5, description="Temporal smoothing of the keypoints, 0-0.95"),
        hands: bool = Form(True), face: bool = Form(True),
        first_frame: bool = Form(False, description="Also use the character image as the exact first frame"),
        strength: float = Form(1.0, description="How closely to follow the pose, 0-2"),
        start: float = Form(0.0),
        variant: str | None = Form(None), wait: bool = Form(False), opts: Options = Depends(common_options),
    ):
        if detector not in ann.POSE_DETECTORS:
            raise HTTPException(400, f"detector must be one of {', '.join(ann.POSE_DETECTORS)}")
        char, refs = _image(image, "image"), _images(references, "references")
        v = check_variant(variant, "pose")
        src = await save_video(video, "video")
        params = {"prompt": prompt, "detector": detector, "align": align, "character": char is not None,
                  "references": len(refs)} | opts_params(opts)
        job = submit("pose", params, lambda hooks: engine.pose(
            src, char, prompt, refs, detector, align, smooth, hands, face, first_frame, strength, start, opts, hooks, v), [src])
        return await respond(job, wait)

    # ------------------------------------------------------------------ 5. Wan-Animate

    @app.post("/api/animate", status_code=202, summary="Wan-Animate: character animation (body, face, gestures) or replacement")
    async def animate(
        image: UploadFile = File(..., description="The character to animate"),
        video: UploadFile = File(..., description="Driving video: body motion and facial expressions"),
        prompt: str = Form(""),
        mode: str = Form("animate", description="animate (character in its own scene) or replace (into the video)"),
        detector: str = Form("dwpose"),
        align: bool = Form(True, description="animate: fit the skeleton to the character's proportions"),
        smooth: float = Form(0.5),
        start: float = Form(0.0),
        variant: str | None = Form(None), wait: bool = Form(False), opts: Options = Depends(common_options),
    ):
        if mode not in ANIMATE_MODES:
            raise HTTPException(400, f"mode must be one of {', '.join(ANIMATE_MODES)}")
        if detector not in ann.POSE_DETECTORS:
            raise HTTPException(400, f"detector must be one of {', '.join(ann.POSE_DETECTORS)}")
        char = _read_image(image, "image")
        v = check_variant(variant, "animate")
        src = await save_video(video, "video")
        params = {"prompt": prompt, "mode": mode, "detector": detector, "align": align} | opts_params(opts)
        job = submit("animate", params, lambda hooks: engine.animate(
            char, src, prompt, mode, detector, align, smooth, start, opts, hooks, v), [src])
        return await respond(job, wait)

    # ------------------------------------------------------------------ 6. video-to-video

    @app.post("/api/v2v", status_code=202, summary="Video-to-video: restyle, SDEdit or inpaint a region")
    async def v2v(
        video: UploadFile = File(...),
        prompt: str = Form(..., description="Describe the result, e.g. 'anime style, cel shading'"),
        method: str = Form("restyle", description="restyle (VACE, keeps structure), sdedit (T2V, strength) or inpaint (VACE, mask)"),
        control_type: str = Form("depth", description="restyle: the structure to keep (depth, lineart, softedge, canny, pose, ...)"),
        strength: float = Form(0.7, description="sdedit: how much to change, 0.05-1"),
        references: list[UploadFile] = File(default=[], description="Style or subject references (restyle/inpaint)"),
        mask: UploadFile | None = File(None, description="inpaint: region to regenerate (white), at the video's size"),
        selection: str | None = Form(None, description='inpaint: JSON {"polygons": [[[x,y],...]], "boxes": [[x0,y0,x1,y1]]} in video pixels'),
        start: float = Form(0.0),
        variant: str | None = Form(None), wait: bool = Form(False), opts: Options = Depends(common_options),
    ):
        if method not in V2V_METHODS:
            raise HTTPException(400, f"method must be one of {', '.join(V2V_METHODS)}")
        if method == "restyle" and control_type not in ann.CONTROL_TYPES:
            raise HTTPException(400, f"control_type must be one of {', '.join(ann.CONTROL_TYPES)}")
        if method == "sdedit" and not 0.05 <= strength <= 1:
            raise HTTPException(400, "strength must be between 0.05 and 1")
        refs = _images(references, "references")
        try:
            v = pick_variant(variant or None, engine.default_variant, ("t2v",) if method == "sdedit" else ("vace",)).name
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e.args[0])) from e
        mask_img = _image(mask, "mask")
        src = await save_video(video, "video")
        region = None
        if method == "inpaint":
            info = await asyncio.to_thread(probe, src)
            parsed = {"polygons": [], "boxes": []}
            if selection:
                try:
                    parsed = sel.parse_selection_json(selection)
                except ValueError as e:
                    src.unlink(missing_ok=True)
                    raise HTTPException(400, f"Invalid selection: {e}") from e
            region = sel.build_mask((info.width, info.height), mask_img, parsed["polygons"], parsed["boxes"])
            if region is None:
                src.unlink(missing_ok=True)
                raise HTTPException(400, "inpaint needs a 'mask' or 'selection' over the region to regenerate")
        params = {"prompt": prompt, "method": method, "control_type": control_type if method == "restyle" else None,
                  "strength": strength if method == "sdedit" else None, "references": len(refs)} | opts_params(opts)
        job = submit("v2v", params, lambda hooks: engine.v2v(
            src, prompt, method, control_type, strength, refs, region, start, None, opts, hooks, v), [src])
        return await respond(job, wait)

    # ------------------------------------------------------------------ 7. export

    @app.post("/api/export", status_code=202, summary="Convert a video to MP4, WebM or GIF (FFmpeg)")
    async def export_video(
        video: UploadFile = File(...),
        format: str = Form("mp4", description="mp4, webm or gif"),
        fps: float | None = Form(None, description="Resample the frame rate"),
        width: int | None = Form(None, description="Scale to this width (height keeps the aspect ratio)"),
        quality: str = Form("high", description="high, medium or small"),
        wait: bool = Form(False),
    ):
        if format not in EXPORT_FORMATS:
            raise HTTPException(400, f"format must be one of {', '.join(EXPORT_FORMATS)}")
        if quality not in ("high", "medium", "small"):
            raise HTTPException(400, "quality must be high, medium or small")
        src = await save_video(video, "video")
        params = {"format": format, "fps": fps, "width": width, "quality": quality}
        job = submit("export", params, lambda hooks: engine.export(src, format, fps, width, quality, hooks), [src])
        return await respond(job, wait)

    # ------------------------------------------------------------------ jobs

    @app.get("/api/jobs")
    def list_jobs(limit: int = 50):
        return [j.to_dict() | {"queue_position": jobs.queue_position(j)} for j in jobs.list()[:limit]]

    def _job(job_id: str) -> Job:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "No such job")
        return job

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str, wait: bool = False):
        return await respond(_job(job_id), wait)

    @app.delete("/api/jobs/{job_id}")
    def cancel_job(job_id: str):
        _job(job_id)
        return jobs.cancel(job_id).to_dict()

    @app.get("/api/jobs/{job_id}/preview", summary="Latest live preview of a running job (animated PNG)")
    def job_preview(job_id: str):
        job = _job(job_id)
        if not job.preview_png:
            raise HTTPException(404, "No preview yet")
        return Response(job.preview_png, media_type="image/png",
                        headers={"Cache-Control": "no-store", "X-Preview-Step": str(job.preview_step)})

    @app.get("/api/jobs/{job_id}/videos/{index}", summary="A result video; ?format=webm|gif converts it")
    async def job_video(job_id: str, index: int, format: str = "mp4", fps: float | None = None, width: int | None = None,
                        quality: str = "high", download: bool = False):
        job = _job(job_id)
        if not 0 <= index < len(job.videos):
            raise HTTPException(404, "No such video")
        if format not in EXPORT_FORMATS:
            raise HTTPException(400, f"format must be one of {', '.join(EXPORT_FORMATS)}")
        if quality not in ("high", "medium", "small"):
            raise HTTPException(400, "quality must be high, medium or small")
        if fps is not None and not 1 <= fps <= 60:
            raise HTTPException(400, "fps must be between 1 and 60")
        if width is not None and not 16 <= width <= 3840:
            raise HTTPException(400, "width must be between 16 and 3840")
        master = Path(job.videos[index].path)
        path = master
        if format != "mp4" or fps or width or quality != "high":
            tag = f"{index}-{fps or 'src'}-{width or 'src'}-{quality}.{format}"
            path = jobs.folder(job) / "exports" / tag
            if not path.exists():
                try:
                    await asyncio.to_thread(export, master, path, format, fps, width, quality)
                except (RuntimeError, ffmpeg.FFmpegMissing) as e:
                    raise HTTPException(500, str(e)) from e
        name = f"{job.kind}-{job.id}-{index}.{format}"
        return FileResponse(path, media_type=MEDIA_TYPES[format], filename=name if download else None)

    @app.get("/api/jobs/{job_id}/extras/{name}", summary="A side output, e.g. the control or pose video used")
    def job_extra(job_id: str, name: str):
        job = _job(job_id)
        if name not in job.extras:
            raise HTTPException(404, "No such output")
        return FileResponse(job.extras[name].path, media_type="video/mp4")

    @app.get("/api/jobs/{job_id}/files/{name}", summary="Another job output, e.g. an exported file")
    def job_file(job_id: str, name: str, download: bool = True):
        job = _job(job_id)
        if name not in job.files:
            raise HTTPException(404, "No such file")
        fmt = Path(name).suffix.lstrip(".")
        return FileResponse(job.files[name], media_type=MEDIA_TYPES.get(fmt, "application/octet-stream"),
                            filename=f"{job.kind}-{job.id}.{fmt}" if download else None)

    # ------------------------------------------------------------------ web UI

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app
