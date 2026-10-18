"""Video I/O through FFmpeg: probing, streaming frames in and out, and exporting.

Frames are piped as raw RGB, so nothing here holds a whole video in memory: ``read_frames``
yields one frame at a time and ``VideoWriter`` encodes frames as they arrive. That is what lets
long (FramePack) runs keep RAM use constant as well as VRAM.

Every job writes a high-quality H.264 *master* MP4. ``export`` turns a master (or any video)
into MP4, WebM or GIF for download.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from .ffmpeg import ffmpeg_path

log = logging.getLogger(__name__)

EXPORT_FORMATS = ("mp4", "webm", "gif")
MEDIA_TYPES = {"mp4": "video/mp4", "webm": "video/webm", "gif": "image/gif"}
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int
    duration: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VideoFile:
    """A finished video on disk (a job's master MP4)."""

    path: str
    fps: float
    frames: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.frames / self.fps if self.fps else 0.0

    def to_dict(self) -> dict:
        return asdict(self) | {"duration": round(self.duration, 3)}


def _run(args: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, timeout=timeout, creationflags=_NO_WINDOW)


def probe(path: str | Path) -> VideoInfo:
    """Size (after rotation), frame rate, duration and an estimated frame count."""
    proc = _run([ffmpeg_path(), "-hide_banner", "-i", str(path)], timeout=60)
    text = proc.stderr.decode("utf-8", "replace")
    stream = re.search(r"Stream #\d+:\d+.*?: Video: (.*)", text)
    if not stream:
        raise ValueError(f"'{Path(path).name}' has no video stream FFmpeg can read")
    line = stream.group(1)
    size = re.search(r"[ ,](\d{1,5})x(\d{1,5})[\s,\[]", line + " ")
    if not size:
        raise ValueError(f"Could not read the frame size of '{Path(path).name}'")
    width, height = int(size.group(1)), int(size.group(2))
    rate = re.search(r"([\d.]+) fps", line) or re.search(r"([\d.]+)k? tbr", line)
    fps = float(rate.group(1)) if rate else 25.0
    rot = re.search(r"rotation of (-?[\d.]+) degrees", text) or re.search(r"rotate\s*:\s*(-?\d+)", text)
    if rot and round(abs(float(rot.group(1)))) % 180 == 90:
        width, height = height, width
    dur = re.search(r"Duration: (\d+):(\d+):([\d.]+)", text)
    duration = int(dur.group(1)) * 3600 + int(dur.group(2)) * 60 + float(dur.group(3)) if dur else 0.0
    frames = max(1, round(duration * fps)) if duration else 1
    return VideoInfo(width, height, fps, frames, duration)


def _fit_filter(size: tuple[int, int] | None, fit: str) -> list[str]:
    if size is None:
        return []
    w, h = size
    if fit == "stretch":
        return [f"scale={w}:{h}:flags=lanczos"]
    if fit == "pad":
        return [f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos", f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"]
    return [f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos", f"crop={w}:{h}"]


def read_frames(
    path: str | Path,
    fps: float | None = None,
    size: tuple[int, int] | None = None,
    start: float = 0.0,
    max_frames: int | None = None,
    fit: str = "crop",
    info: VideoInfo | None = None,
) -> Iterator[Image.Image]:
    """Yield RGB frames one at a time, resampled to ``fps`` and fitted to ``size`` (w, h).

    ``fit`` is ``crop`` (fill and centre-crop), ``pad`` (letterbox) or ``stretch``.
    """
    info = info or probe(path)
    w, h = size or (info.width, info.height)
    filters = ([f"fps={fps:g}"] if fps else []) + _fit_filter(size, fit)
    args = [ffmpeg_path(), "-hide_banner", "-loglevel", "error"]
    if start:
        args += ["-ss", f"{start:.3f}"]
    args += ["-i", str(path)]
    if filters:
        args += ["-vf", ",".join(filters)]
    if max_frames:
        args += ["-frames:v", str(int(max_frames))]
    args += ["-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    frame_bytes = w * h * 3
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=err, creationflags=_NO_WINDOW)
        count = 0
        try:
            while True:
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                count += 1
                yield Image.frombytes("RGB", (w, h), buf)
        finally:
            proc.stdout.close()
            if proc.poll() is None:
                proc.kill()
            proc.wait()
        if count == 0:
            err.seek(0)
            msg = err.read().decode("utf-8", "replace").strip()
            raise ValueError(f"Could not decode '{Path(path).name}': {msg or 'no frames'}")


def load_frames(path: str | Path, **kwargs) -> list[Image.Image]:
    return list(read_frames(path, **kwargs))


class VideoWriter:
    """Encode frames to an H.264 MP4 as they arrive (constant memory)."""

    def __init__(self, path: str | Path, fps: float, crf: int = 14, preset: str = "fast"):
        self.path = Path(path)
        self.fps = fps
        self.crf, self.preset = crf, preset
        self.size: tuple[int, int] | None = None
        self.frames = 0
        self._proc: subprocess.Popen | None = None
        self._err = None

    def _start(self, size: tuple[int, int]):
        self.size = size
        self.path.parent.mkdir(parents=True, exist_ok=True)
        w, h = size
        self._err = tempfile.TemporaryFile()
        args = [
            ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{self.fps:g}", "-i", "-",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-preset", self.preset, "-crf", str(self.crf), "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(self.path),
        ]
        self._proc = subprocess.Popen(args, stdin=subprocess.PIPE, stderr=self._err, creationflags=_NO_WINDOW)

    def write(self, frame: Image.Image) -> None:
        frame = frame.convert("RGB")
        if self._proc is None:
            self._start(frame.size)
        elif frame.size != self.size:
            frame = frame.resize(self.size, Image.Resampling.LANCZOS)
        try:
            self._proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError) as e:
            raise RuntimeError(f"FFmpeg stopped while encoding: {self._error_text()}") from e
        self.frames += 1

    def write_all(self, frames: Iterable[Image.Image]) -> None:
        for f in frames:
            self.write(f)

    def _error_text(self) -> str:
        if self._err is None:
            return ""
        self._err.seek(0)
        return self._err.read().decode("utf-8", "replace").strip()

    def close(self) -> VideoFile:
        if self._proc is None:
            raise ValueError("No frames were written")
        self._proc.stdin.close()
        code = self._proc.wait()
        msg = self._error_text()
        self._err.close()
        if code != 0:
            raise RuntimeError(f"FFmpeg failed to encode {self.path.name}: {msg}")
        return VideoFile(str(self.path), self.fps, self.frames, *self.size)

    def abort(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
            self._proc.kill()
            self._proc.wait()
        if self._err is not None:
            self._err.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.abort()


def write_video(frames: Iterable[Image.Image], path: str | Path, fps: float) -> VideoFile:
    writer = VideoWriter(path, fps)
    with writer:
        writer.write_all(frames)
        return writer.close()


def export(
    src: str | Path,
    dst: str | Path,
    fmt: str | None = None,
    fps: float | None = None,
    width: int | None = None,
    quality: str = "high",
    loop: bool = True,
    report: Callable[[str], None] | None = None,
) -> Path:
    """Re-encode ``src`` as MP4 (H.264), WebM (VP9) or GIF (palette-optimised).

    ``fps`` resamples the frame rate and ``width`` scales (height follows the aspect ratio).
    ``quality`` is ``high``, ``medium`` or ``small``.
    """
    dst = Path(dst)
    fmt = (fmt or dst.suffix.lstrip(".")).lower()
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"format must be one of {', '.join(EXPORT_FORMATS)}")
    if quality not in ("high", "medium", "small"):
        raise ValueError("quality must be 'high', 'medium' or 'small'")
    dst.parent.mkdir(parents=True, exist_ok=True)
    base = []
    if fps:
        base.append(f"fps={fps:g}")
    if width:
        base.append(f"scale={int(width)}:-2:flags=lanczos")
    args = [ffmpeg_path(report=report), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-an"]
    if fmt == "gif":
        colors = {"high": 256, "medium": 192, "small": 96}[quality]
        chain = ",".join(base or ["null"])
        args += [
            "-filter_complex",
            f"[0:v]{chain},split[a][b];[a]palettegen=max_colors={colors}:stats_mode=diff[p];"
            f"[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
            "-loop", "0" if loop else "-1",
        ]
    else:
        vf = base + ["pad=ceil(iw/2)*2:ceil(ih/2)*2"]
        args += ["-vf", ",".join(vf), "-pix_fmt", "yuv420p"]
        if fmt == "mp4":
            crf = {"high": 17, "medium": 22, "small": 28}[quality]
            args += ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-movflags", "+faststart"]
        else:
            crf = {"high": 24, "medium": 32, "small": 40}[quality]
            args += ["-c:v", "libvpx-vp9", "-crf", str(crf), "-b:v", "0", "-row-mt", "1", "-deadline", "good", "-cpu-used", "4"]
    args.append(str(dst))
    if report:
        report(f"Encoding {fmt.upper()}")
    proc = _run(args)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg export failed: {proc.stderr.decode('utf-8', 'replace').strip()}")
    return dst


def first_frame(path: str | Path) -> Image.Image:
    return next(read_frames(path, max_frames=1))
