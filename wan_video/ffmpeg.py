"""Finding FFmpeg, and downloading a static build when it is missing.

Lookup order:

1. ``WAN_VIDEO_FFMPEG``: an explicit path to the ffmpeg binary.
2. ``ffmpeg`` on ``PATH``.
3. A build downloaded earlier into the cache folder (``WAN_VIDEO_FFMPEG_DIR``, default
   ``~/.cache/wan-video/ffmpeg``).
4. Otherwise a static build is downloaded into that folder (BtbN builds for Windows and Linux,
   evermeet.cx for macOS). Set ``WAN_VIDEO_FFMPEG_AUTO_DOWNLOAD=0`` to turn this off.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import threading
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

log = logging.getLogger(__name__)

_BTBN = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
_lock = threading.Lock()
_cached: str | None = None


class FFmpegMissing(RuntimeError):
    pass


def cache_dir() -> Path:
    if os.environ.get("WAN_VIDEO_FFMPEG_DIR"):
        return Path(os.environ["WAN_VIDEO_FFMPEG_DIR"])
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "wan-video" / "ffmpeg"


def _exe_name() -> str:
    return "ffmpeg.exe" if os.name == "nt" else "ffmpeg"


def download_url() -> str:
    system, machine = platform.system(), platform.machine().lower()
    arm = machine in ("arm64", "aarch64")
    if system == "Windows":
        return _BTBN + ("ffmpeg-master-latest-winarm64-gpl.zip" if arm else "ffmpeg-master-latest-win64-gpl.zip")
    if system == "Linux":
        return _BTBN + ("ffmpeg-master-latest-linuxarm64-gpl.tar.xz" if arm else "ffmpeg-master-latest-linux64-gpl.tar.xz")
    if system == "Darwin":
        # Intel build; Apple silicon runs it through Rosetta.
        return "https://evermeet.cx/ffmpeg/getrelease/zip"
    raise FFmpegMissing(f"No automatic FFmpeg download for {system}; install ffmpeg and put it on PATH")


def _auto_download_enabled() -> bool:
    return os.environ.get("WAN_VIDEO_FFMPEG_AUTO_DOWNLOAD", "1").lower() not in ("0", "false", "no")


def find() -> tuple[str | None, str | None]:
    """(path, source) of an available ffmpeg without downloading; source is env, path or cache."""
    env = os.environ.get("WAN_VIDEO_FFMPEG")
    if env and Path(env).is_file():
        return env, "env"
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path, "path"
    cached = cache_dir() / _exe_name()
    if cached.is_file():
        return str(cached), "cache"
    return None, None


def ffmpeg_path(download: bool = True, report: Callable[[str], None] | None = None) -> str:
    """The ffmpeg binary to use, downloading it first if needed and allowed."""
    global _cached
    if _cached and Path(_cached).is_file():
        return _cached
    with _lock:
        path, _ = find()
        if path is None:
            if not (download and _auto_download_enabled()):
                raise FFmpegMissing(
                    "FFmpeg was not found. Install it, set WAN_VIDEO_FFMPEG, or allow the automatic download "
                    "(POST /api/ffmpeg/download or `wan-video ffmpeg download`)."
                )
            path = _download(report)
        _cached = path
        return path


def download(report: Callable[[str], None] | None = None) -> str:
    """Download a static build into the cache folder even if another ffmpeg exists."""
    global _cached
    with _lock:
        _cached = _download(report)
        return _cached


def _download(report: Callable[[str], None] | None = None) -> str:
    report = report or (lambda msg: log.info(msg))
    url = download_url()
    dest = cache_dir()
    dest.mkdir(parents=True, exist_ok=True)
    report(f"Downloading FFmpeg from {url}")
    with tempfile.TemporaryDirectory(dir=dest) as tmp:
        archive = Path(tmp) / ("ffmpeg.zip" if url.endswith("zip") else "ffmpeg.tar.xz")
        req = urllib.request.Request(url, headers={"User-Agent": "wan-video"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(archive, "wb") as fh:
            total = int(resp.headers.get("Content-Length") or 0)
            done, last = 0, -1
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                pct = int(100 * done / total) if total else -1
                if pct // 10 != last // 10:
                    last = pct
                    report(f"Downloading FFmpeg: {done / 1e6:.0f} MB" + (f" ({pct}%)" if total else ""))
        report("Extracting FFmpeg")
        wanted = {_exe_name(), "ffprobe.exe" if os.name == "nt" else "ffprobe"}
        found = False
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as zf:
                for member in zf.namelist():
                    name = member.rsplit("/", 1)[-1]
                    if name in wanted:
                        with zf.open(member) as src, open(dest / name, "wb") as out:
                            shutil.copyfileobj(src, out)
                        found |= name == _exe_name()
        else:
            with tarfile.open(archive) as tf:
                for member in tf.getmembers():
                    name = member.name.rsplit("/", 1)[-1]
                    if member.isfile() and name in wanted:
                        with tf.extractfile(member) as src, open(dest / name, "wb") as out:
                            shutil.copyfileobj(src, out)
                        found |= name == _exe_name()
    exe = dest / _exe_name()
    if not found or not exe.is_file():
        raise FFmpegMissing(f"The download from {url} did not contain {_exe_name()}")
    if os.name != "nt":
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    report(f"FFmpeg ready: {exe}")
    return str(exe)


def version(path: str | None = None) -> str | None:
    path = path or find()[0]
    if not path:
        return None
    try:
        out = subprocess.run([path, "-hide_banner", "-version"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"ffmpeg version (\S+)", out)
    return m.group(1) if m else None


def status() -> dict:
    path, source = find()
    return {
        "available": path is not None,
        "path": path,
        "source": source,
        "version": version(path) if path else None,
        "auto_download": _auto_download_enabled(),
        "download_url": _safe_url(),
    }


def _safe_url() -> str | None:
    try:
        return download_url()
    except FFmpegMissing:
        return None
