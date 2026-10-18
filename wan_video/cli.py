"""Command-line interface.

    wan-video generate -p "a red fox running through snow" -o fox.mp4
    wan-video image -i portrait.png -p "she turns and smiles" -o smile.webm
    wan-video control -p "a robot dancing" -v dancer.mp4 --type depth -o robot.mp4
    wan-video control -p "a ball rolling" --type path --path "0.1,0.5 0.9,0.5" --first-frame scene.png
    wan-video detect -v dancer.mp4 --type pose -o pose.mp4
    wan-video pose -v dancer.mp4 -i knight.png -o knight_dancing.mp4
    wan-video animate -i character.png -v performance.mp4 [--mode replace] -o animated.mp4
    wan-video v2v -v street.mp4 -p "anime style, cel shading" -o anime.mp4
    wan-video export -v clip.mp4 -o clip.gif --fps 12 --width 480
    wan-video models list | download VARIANT
    wan-video ffmpeg status | download
    wan-video serve --port 7860

Operations run in-process by default (the model loads once per invocation). Pass
``--api http://host:port`` to run them on a server instead, which keeps the model warm. The
output format follows the extension of ``-o``: .mp4, .webm or .gif.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

from .variants import DEFAULT_VARIANT, list_variant_names

CONTROL_CHOICES = ["pose", "depth", "canny", "lineart", "softedge", "scribble", "gray", "raw", "path"]


def _floats(text: str, n: int | None = None) -> list[float]:
    vals = [float(v) for v in text.replace(";", ",").split(",") if v.strip()]
    if n is not None and len(vals) != n:
        raise argparse.ArgumentTypeError(f"expected {n} comma-separated numbers, got '{text}'")
    return vals


def _polygon(text: str) -> list[list[float]]:
    pts = [_floats(p, 2) for p in text.split()]
    if len(pts) < 3:
        raise argparse.ArgumentTypeError("a polygon needs at least 3 points: 'x,y x,y x,y'")
    return pts


def _path(text: str) -> list[list[float]]:
    pts = [_floats(p, 2) for p in text.split()]
    if len(pts) < 2:
        raise argparse.ArgumentTypeError("a path needs at least 2 points: 'x,y x,y' (0..1 coordinates)")
    return pts


def _add_common(p: argparse.ArgumentParser):
    g = p.add_argument_group("video")
    g.add_argument("--variant", "-m", help=f"model variant (default: {DEFAULT_VARIANT}); see 'models list'")
    g.add_argument("--resolution", choices=["480p", "720p"], default="480p")
    g.add_argument("--aspect", choices=["auto", "16:9", "9:16", "1:1", "4:3", "3:4"], default="auto")
    g.add_argument("--width", type=int, help="explicit width (overrides resolution/aspect)")
    g.add_argument("--height", type=int)
    g.add_argument("--frames", type=int, help="output length in frames (default 81, or the driving video's length)")
    g.add_argument("--seconds", type=float, help="output length in seconds")
    g.add_argument("--fps", type=float, help="frame rate (default: the model's)")
    g.add_argument("--steps", type=int, help="denoising steps (default 30; Animate 20)")
    g.add_argument("--seed", type=int, help="random seed (default: random)")
    g.add_argument("--guidance", "--cfg", type=float, help="CFG scale (default 5.0; Animate 1.0)")
    g.add_argument("--negative", dest="negative_prompt", help="negative prompt (used when guidance > 1)")
    g.add_argument("--shift", type=float, help="flow shift (default 3 at 480p, 5 at 720p)")
    g.add_argument("--num-videos", "-n", type=int, default=1)
    g.add_argument("--offload", choices=["auto", "none", "model", "group", "sequential"], default="auto",
                   help="where the weights live; 'group' fits 14B models in a few GB of VRAM")
    g = p.add_argument_group("long videos (FramePack-style constant VRAM)")
    g.add_argument("--framepack", choices=["auto", "on", "off"], default="auto", help="auto: on when longer than one section")
    g.add_argument("--section-frames", type=int, default=81, help="frames per section (4k+1)")
    g.add_argument("--context-frames", type=int, default=5, help="frames carried over between sections")
    g.add_argument("--no-anchor", dest="anchor", action="store_false", help="don't re-anchor each section on the first frame")
    g = p.add_argument_group("live previews")
    g.add_argument("--preview", choices=["off", "fast", "full"], help="preview mode (default: fast when --preview-dir is set)")
    g.add_argument("--preview-every", type=int)
    g.add_argument("--preview-dir", help="save each step's preview here as step_NN.png (animated)")
    _add_output(p)


def _add_output(p: argparse.ArgumentParser):
    p.add_argument("--output", "-o", default=None, help="output file (.mp4, .webm or .gif); several videos get _1, _2 ...")
    p.add_argument("--export-fps", type=float, help="resample the frame rate of the saved file")
    p.add_argument("--export-width", type=int, help="scale the saved file to this width")
    p.add_argument("--quality", choices=["high", "medium", "small"], default="high", help="export quality")
    p.add_argument("--api", help="run on a server, e.g. http://127.0.0.1:7860")
    p.add_argument("--json", action="store_true", help="print a JSON summary to stdout")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="wan-video", description="Wan video generation, control, animation and editing.")
    ap.add_argument("-V", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="text-to-video")
    p.add_argument("--prompt", "-p", required=True)
    p.add_argument("--reference", "-r", action="append", default=[], help="subject/style reference image (VACE, repeatable)")
    _add_common(p)

    p = sub.add_parser("image", help="image-to-video: animate a still image")
    p.add_argument("--image", "-i", required=True, help="the first frame")
    p.add_argument("--prompt", "-p", default="", help="optional description of the motion")
    p.add_argument("--last-image", help="optional last frame (VACE or FLF2V variants)")
    p.add_argument("--reference", "-r", action="append", default=[], help="subject reference (VACE, repeatable)")
    _add_common(p)

    p = sub.add_parser("control", help="motion control with VACE: follow a control video or motion paths")
    p.add_argument("--prompt", "-p", required=True)
    p.add_argument("--video", "-v", help="driving video")
    p.add_argument("--type", dest="control_type", default="depth", choices=CONTROL_CHOICES)
    p.add_argument("--path", dest="paths", action="append", type=_path, default=[],
                   help="--type path: one moving object as 'x,y x,y ...' in 0..1 coordinates (repeatable)")
    p.add_argument("--path-radius", type=float, default=0.08, help="size of each moving object (fraction of the frame)")
    p.add_argument("--reference", "-r", action="append", default=[])
    p.add_argument("--first-frame", help="fixed first frame")
    p.add_argument("--last-frame", help="fixed last frame")
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--invert", action="store_true", help="invert the control maps")
    p.add_argument("--detector", choices=["dwpose", "openpose"], default="dwpose", help="--type pose: pose estimator")
    p.add_argument("--start", type=float, default=0.0, help="start time in the driving video (seconds)")
    p.add_argument("--save-control", help="also save the control video that was used")
    _add_common(p)

    p = sub.add_parser("detect", help="extract the control video (pose, depth, edges, ...) from a video")
    p.add_argument("--video", "-v", required=True)
    p.add_argument("--type", dest="control_type", default="pose", choices=CONTROL_CHOICES[:-1])
    p.add_argument("--detector", choices=["dwpose", "openpose"], default="dwpose")
    p.add_argument("--smooth", type=float, default=0.5, help="pose: temporal smoothing 0-0.95")
    p.add_argument("--reference", help="pose: retarget the skeleton onto the character in this image")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--resolution", choices=["480p", "720p"], default="480p")
    p.add_argument("--aspect", choices=["auto", "16:9", "9:16", "1:1", "4:3", "3:4"], default="auto")
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--fps", type=float, default=16)
    p.add_argument("--frames", type=int)
    p.add_argument("--seconds", type=float)
    _add_output(p)

    p = sub.add_parser("pose", help="pose-guided motion transfer: move a character like the person in a video")
    p.add_argument("--video", "-v", required=True, help="driving video")
    p.add_argument("--image", "-i", help="the character to move")
    p.add_argument("--prompt", "-p", default="")
    p.add_argument("--reference", "-r", action="append", default=[], help="extra reference images")
    p.add_argument("--detector", choices=["dwpose", "openpose"], default="dwpose")
    p.add_argument("--no-align", dest="align", action="store_false", help="don't fit the skeleton to the character")
    p.add_argument("--smooth", type=float, default=0.5)
    p.add_argument("--no-hands", dest="hands", action="store_false")
    p.add_argument("--no-face", dest="face", action="store_false")
    p.add_argument("--first-frame", action="store_true", help="use the character image as the exact first frame")
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--save-control", help="also save the pose video that was used")
    _add_common(p)

    p = sub.add_parser("animate", help="Wan-Animate: animate a character with the body and face motion of a video")
    p.add_argument("--image", "-i", required=True, help="the character")
    p.add_argument("--video", "-v", required=True, help="driving video")
    p.add_argument("--prompt", "-p", default="")
    p.add_argument("--mode", choices=["animate", "replace"], default="animate",
                   help="animate: character in its own scene; replace: into the video in place of the person")
    p.add_argument("--detector", choices=["dwpose", "openpose"], default="dwpose")
    p.add_argument("--no-align", dest="align", action="store_false")
    p.add_argument("--smooth", type=float, default=0.5)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--save-control", help="also save the pose video that was used")
    _add_common(p)

    p = sub.add_parser("v2v", help="video-to-video: restyle, SDEdit or inpaint a region")
    p.add_argument("--video", "-v", required=True)
    p.add_argument("--prompt", "-p", required=True)
    p.add_argument("--method", choices=["restyle", "sdedit", "inpaint"], default="restyle")
    p.add_argument("--type", dest="control_type", default="depth", choices=CONTROL_CHOICES[:-2],
                   help="restyle: the structure to keep")
    p.add_argument("--strength", type=float, default=0.7, help="sdedit: how much to change (0.05-1)")
    p.add_argument("--reference", "-r", action="append", default=[], help="style/subject reference image")
    p.add_argument("--mask", help="inpaint: mask image (white = regenerate)")
    p.add_argument("--polygon", action="append", type=_polygon, default=[], help="inpaint: region 'x,y x,y x,y' in video pixels")
    p.add_argument("--box", action="append", type=lambda t: _floats(t, 4), default=[], help="inpaint: region 'x0,y0,x1,y1'")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--save-control", help="also save the control video that was used")
    _add_common(p)

    p = sub.add_parser("export", help="convert a video to MP4, WebM or GIF")
    p.add_argument("--video", "-v", required=True)
    p.add_argument("--fps", type=float)
    p.add_argument("--width", type=int)
    p.add_argument("--quality", choices=["high", "medium", "small"], default="high")
    p.add_argument("--output", "-o", required=True, help="output file; the extension picks the format")
    p.add_argument("--api", help="run on a server")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("models", help="list or download model variants")
    msub = p.add_subparsers(dest="models_command", required=True)
    m = msub.add_parser("list")
    m.add_argument("--json", action="store_true")
    m = msub.add_parser("download")
    m.add_argument("variant", choices=list_variant_names() + ["dwpose"])

    p = sub.add_parser("ffmpeg", help="show where FFmpeg is, or download it")
    p.add_argument("action", choices=["status", "download"])

    p = sub.add_parser("serve", help="start the web UI and API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--variant", "-m", help="default variant")
    p.add_argument("--preload", action="store_true", help="load the default variant at startup")
    p.add_argument("--offload", choices=["auto", "none", "model", "group", "sequential"], default="auto")
    p.add_argument("--outputs", default="outputs", help="directory for generated videos")
    return ap


# ---------------------------------------------------------------------- helpers


def _open(path: str):
    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(Image.open(path))
    return img.convert("RGBA" if "A" in img.getbands() else "RGB")


def _output_paths(args, count: int, default_ext: str = ".mp4") -> list[Path]:
    base = Path(args.output) if args.output else Path(f"{args.command}-{time.strftime('%Y%m%d-%H%M%S')}{default_ext}")
    if base.suffix == "":
        base = base.with_suffix(default_ext)
    if count == 1:
        return [base]
    return [base.with_name(f"{base.stem}_{i + 1}{base.suffix}") for i in range(count)]


def _save(src: Path, dst: Path, args) -> None:
    """Copy a master MP4 to ``dst``, re-encoding when the format, fps, width or quality asks for it."""
    from .video import export

    dst.parent.mkdir(parents=True, exist_ok=True)
    fmt = dst.suffix.lstrip(".").lower()
    fps, width, quality = getattr(args, "export_fps", None), getattr(args, "export_width", None), getattr(args, "quality", "high")
    if fmt == "mp4" and not fps and not width and quality == "high":
        shutil.copyfile(src, dst)
    else:
        export(src, dst, fmt, fps=fps, width=width, quality=quality)


def _options(args):
    from .engine import Options

    return Options.from_dict(vars(args)).validate()


def _progress_hooks(args=None):
    from .engine import Hooks

    state = {"mid_bar": False}

    def progress(step, total):
        width = 30
        done = int(width * step / max(1, total))
        sys.stderr.write(f"\r[{'#' * done}{'.' * (width - done)}] {step}/{total}")
        sys.stderr.flush()
        state["mid_bar"] = step < total
        if step >= total:
            sys.stderr.write("\n")

    def status(msg):
        if state["mid_bar"]:
            sys.stderr.write("\n")
            state["mid_bar"] = False
        print(msg, file=sys.stderr)

    hooks = Hooks(progress=progress, status=status)
    if args is not None and getattr(args, "preview_dir", None):
        folder = Path(args.preview_dir)

        def preview(data, step):
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"step_{step:02d}.png").write_bytes(data)

        hooks.preview = preview
    return hooks


def _region(args, video: str):
    from . import selection as sel
    from .video import probe

    if not (args.mask or args.polygon or args.box):
        return None
    info = probe(video)
    return sel.build_mask((info.width, info.height), _open(args.mask) if args.mask else None, args.polygon, args.box)


def _paths(args) -> list[dict]:
    return [{"points": p, "radius": args.path_radius} for p in args.paths]


# ---------------------------------------------------------------------- local execution


def run_local(args, workdir: Path):
    from .engine import Engine

    engine = Engine(getattr(args, "variant", None), offload=getattr(args, "offload", "auto"))
    hooks = _progress_hooks(args)
    hooks.workdir = workdir
    if args.command == "export":
        fmt = Path(args.output).suffix.lstrip(".").lower() or "mp4"
        return engine.export(args.video, fmt, args.fps, args.width, args.quality, hooks)
    opts = _options(args)
    c = args.command
    if c == "generate":
        return engine.generate(args.prompt, [_open(p) for p in args.reference], opts, hooks)
    if c == "image":
        last = _open(args.last_image) if args.last_image else None
        return engine.image_to_video(_open(args.image), args.prompt, last, [_open(p) for p in args.reference], opts, hooks)
    if c == "control":
        return engine.control(
            args.prompt, args.video, args.control_type, [_open(p) for p in args.reference],
            _open(args.first_frame) if args.first_frame else None, _open(args.last_frame) if args.last_frame else None,
            _paths(args) if args.control_type == "path" else None, args.strength,
            {"detector": args.detector, "invert": args.invert}, args.start, opts, hooks,
        )
    if c == "detect":
        ref = _open(args.reference) if args.reference else None
        return engine.detect(args.video, args.control_type, opts, hooks, args.detector, args.smooth, start=args.start, reference=ref)
    if c == "pose":
        return engine.pose(
            args.video, _open(args.image) if args.image else None, args.prompt, [_open(p) for p in args.reference],
            args.detector, args.align, args.smooth, args.hands, args.face, args.first_frame, args.strength, args.start, opts, hooks,
        )
    if c == "animate":
        return engine.animate(_open(args.image), args.video, args.prompt, args.mode, args.detector, args.align, args.smooth,
                              args.start, opts, hooks)
    if c == "v2v":
        region = _region(args, args.video) if args.method == "inpaint" else None
        if args.method == "inpaint" and region is None:
            raise SystemExit("inpaint needs a region: --mask, --polygon or --box")
        return engine.v2v(args.video, args.prompt, args.method, args.control_type, args.strength,
                          [_open(p) for p in args.reference], region, args.start, None, opts, hooks)
    raise AssertionError(c)


# ---------------------------------------------------------------------- remote execution


_OPTION_KEYS = ("variant", "resolution", "aspect", "width", "height", "frames", "seconds", "fps", "steps", "seed", "guidance",
                "negative_prompt", "shift", "num_videos", "framepack", "section_frames", "context_frames", "anchor",
                "preview", "preview_every")
_ENDPOINTS = {"generate": "generate", "image": "image", "control": "control", "detect": "control/detect", "pose": "pose",
              "animate": "animate", "v2v": "v2v", "export": "export"}


def run_remote(args, workdir: Path) -> dict:
    """Submit the command to a server, follow the job, and download its outputs into ``workdir``."""
    import httpx

    base = args.api.rstrip("/")
    c = args.command
    data: dict = {}
    for key in _OPTION_KEYS:
        val = getattr(args, key, None)
        if val is not None:
            data[key] = str(val).lower() if isinstance(val, bool) else str(val)
    files: list = []

    def attach(field, path):
        files.append((field, (Path(path).name, open(path, "rb"), "application/octet-stream")))

    def put(**kw):
        for k, v in kw.items():
            if v is not None:
                data[k] = str(v).lower() if isinstance(v, bool) else str(v)

    for ref in getattr(args, "reference", None) or [] if c != "detect" else []:
        attach("references", ref)
    if c == "generate":
        put(prompt=args.prompt)
    elif c == "image":
        attach("image", args.image)
        put(prompt=args.prompt)
        if args.last_image:
            attach("last_image", args.last_image)
    elif c == "control":
        put(prompt=args.prompt, control_type=args.control_type, strength=args.strength, invert=args.invert,
            detector=args.detector, start=args.start)
        if args.video:
            attach("video", args.video)
        if args.control_type == "path":
            data["paths"] = json.dumps(_paths(args))
        for field in ("first_frame", "last_frame"):
            if getattr(args, field):
                attach(field, getattr(args, field))
    elif c == "detect":
        attach("video", args.video)
        put(control_type=args.control_type, detector=args.detector, smooth=args.smooth, start=args.start)
        if args.reference:
            attach("reference", args.reference)
    elif c == "pose":
        attach("video", args.video)
        if args.image:
            attach("image", args.image)
        put(prompt=args.prompt, detector=args.detector, align=args.align, smooth=args.smooth, hands=args.hands,
            face=args.face, first_frame=args.first_frame, strength=args.strength, start=args.start)
    elif c == "animate":
        attach("image", args.image)
        attach("video", args.video)
        put(prompt=args.prompt, mode=args.mode, detector=args.detector, align=args.align, smooth=args.smooth, start=args.start)
    elif c == "v2v":
        attach("video", args.video)
        put(prompt=args.prompt, method=args.method, control_type=args.control_type, strength=args.strength, start=args.start)
        if args.mask:
            attach("mask", args.mask)
        if args.polygon or args.box:
            data["selection"] = json.dumps({"polygons": args.polygon, "boxes": args.box})
    elif c == "export":
        attach("video", args.video)
        put(format=Path(args.output).suffix.lstrip(".").lower() or "mp4", fps=args.fps, width=args.width, quality=args.quality)

    hooks = _progress_hooks(args)
    with httpx.Client(timeout=httpx.Timeout(60.0, read=None)) as client:
        resp = client.post(f"{base}/api/{_ENDPOINTS[c]}", data=data, files=files or None)
        if resp.status_code >= 400:
            raise SystemExit(f"Server error {resp.status_code}: {resp.text}")
        job = resp.json()
        last_msg, last_preview = None, 0
        while job["status"] not in ("done", "error", "cancelled"):
            time.sleep(0.5)
            job = client.get(f"{base}/api/jobs/{job['id']}").json()
            if job["message"] and job["message"] != last_msg:
                hooks.status(job["message"])
                last_msg = job["message"]
            if job["progress"]["total"]:
                hooks.progress(job["progress"]["step"], job["progress"]["total"])
            if getattr(args, "preview_dir", None) and job.get("preview") and job["preview_step"] != last_preview:
                last_preview = job["preview_step"]
                hooks.preview(client.get(base + job["preview"]).content, last_preview)
        if job["status"] != "done":
            raise SystemExit(f"Job {job['status']}: {job.get('error') or ''}")
        videos = []
        for i, v in enumerate(job["videos"]):
            path = workdir / f"{i}.mp4"
            path.write_bytes(client.get(base + v["url"]).content)
            videos.append(path)
        extras = {}
        for name, url in job.get("extras", {}).items():
            extras[name] = workdir / f"{name}.mp4"
            extras[name].write_bytes(client.get(base + url).content)
        files_out = {}
        for name, url in job.get("files", {}).items():
            files_out[name] = workdir / name
            files_out[name].write_bytes(client.get(base + url).content)
    return {"videos": videos, "extras": extras, "files": files_out, "seed": job["seed"], "prompt": job["prompt"],
            "variant": job["variant"], "info": job["info"], "job": job["id"]}


# ---------------------------------------------------------------------- commands


def cmd_models(args) -> int:
    from .models import download, is_downloaded
    from .variants import get_variant, list_variants

    if args.models_command == "list":
        rows = [v.to_dict() | {"downloaded": is_downloaded(v)} for v in list_variants()]
        if args.json:
            print(json.dumps(rows, indent=2))
            return 0
        print(f"{'variant':<22} {'family':<8} {'dl':<3} {'~GB':>5}  description")
        for r in rows:
            mark = "yes" if r["downloaded"] else "-"
            default = " (default)" if r["name"] == DEFAULT_VARIANT else ""
            print(f"{r['name']:<22} {r['family']:<8} {mark:<3} {r['approx_download_gb']:>5.1f}  {r['description']}{default}")
        return 0
    if args.variant == "dwpose":
        from . import dwpose

        dwpose.download(lambda msg: print(msg, file=sys.stderr))
        return 0
    download(get_variant(args.variant), lambda msg: print(msg, file=sys.stderr))
    return 0


def cmd_ffmpeg(args) -> int:
    from . import ffmpeg

    if args.action == "download":
        print(ffmpeg.ffmpeg_path(report=lambda msg: print(msg, file=sys.stderr)))
        return 0
    print(json.dumps(ffmpeg.status(), indent=2))
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from .api import create_app
    from .engine import Engine

    engine = Engine(args.variant or DEFAULT_VARIANT, offload=args.offload)
    if args.preload:
        engine.load(hooks=_progress_hooks())
    app = create_app(engine, args.outputs)
    print(f"Web UI: http://{args.host}:{args.port}/   API docs: http://{args.host}:{args.port}/docs", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if args.command == "models":
        return cmd_models(args)
    if args.command == "ffmpeg":
        return cmd_ffmpeg(args)
    if args.command == "serve":
        return cmd_serve(args)
    if getattr(args, "preview", "off") is None:
        args.preview = "fast" if getattr(args, "preview_dir", None) else "off"
    if args.command == "control" and args.control_type == "path" and not args.paths:
        raise SystemExit("--type path needs at least one --path")
    if args.command == "control" and args.control_type != "path" and not args.video:
        raise SystemExit("control needs --video (or --type path with --path)")
    if args.command == "detect":
        args.variant = None

    with tempfile.TemporaryDirectory(prefix="wan-video-") as tmp:
        workdir = Path(tmp)
        if args.api:
            r = run_remote(args, workdir)
        else:
            res = run_local(args, workdir)
            r = {"videos": [Path(v.path) for v in res.videos], "extras": {k: Path(v.path) for k, v in res.extras.items()},
                 "files": {k: Path(v) for k, v in res.files.items()}, "seed": res.seed, "prompt": res.prompt,
                 "variant": res.variant, "info": res.info}
        if args.command == "export":
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(next(iter(r["files"].values())), out)
            paths = [out]
        else:
            paths = _output_paths(args, len(r["videos"]))
            for src, dst in zip(r["videos"], paths):
                _save(src, dst, args)
        save_control = getattr(args, "save_control", None)
        if save_control and r["extras"]:
            extra = r["extras"].get("control") or r["extras"].get("pose")
            if extra:
                _save(extra, Path(save_control), args)
                r["info"]["control_video"] = save_control
    meta = {k: v for k, v in r.items() if k not in ("videos", "extras", "files")} | {"outputs": [str(p) for p in paths]}
    if args.json:
        print(json.dumps(meta, indent=2, default=str))
    else:
        for p in paths:
            print(p)
        if meta.get("seed", -1) >= 0:
            print(f"seed: {meta['seed']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
