"""Live previews of a video while it is being sampled, as small animated PNGs.

Wan samples with flow matching, so two consecutive latents give back the model's velocity and
with it its current estimate of the *finished* video (``x0 = x - sigma * v``). That estimate is
what gets shown, which reads far better than the raw noisy latents.

Two ways to turn latents into pixels:

``fast``
    A linear map from Wan's 16 latent channels to RGB, fitted once per loaded model by
    VAE-encoding a synthetic calibration image. Costs almost nothing per step; low resolution,
    approximate colours, one frame per latent frame (every 4th video frame).
``full``
    The real VAE decode, every few steps. Exact but slow, so it only runs every
    ``preview_every`` steps (5 by default).
"""

from __future__ import annotations

import io
import logging
import random
from collections.abc import Callable

import numpy as np
from PIL import Image, ImageDraw

log = logging.getLogger(__name__)

MODES = ("off", "fast", "full")
DEFAULT_EVERY = {"fast": 1, "full": 5}
MAX_SIDE = 320
_maps: dict[int, np.ndarray] = {}


def to_apng(frames: list[Image.Image], fps: float) -> bytes:
    buf = io.BytesIO()
    first, rest = frames[0], frames[1:]
    first.save(buf, "PNG", save_all=bool(rest), append_images=rest, duration=max(20, int(1000 / max(1, fps))), loop=0)
    return buf.getvalue()


def _shrink(img: Image.Image) -> Image.Image:
    k = MAX_SIDE / max(img.size)
    if k >= 1:
        return img
    return img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))), Image.Resampling.BILINEAR)


def _calibration_image(size: int = 256) -> Image.Image:
    rng = random.Random(1234)
    grad = Image.linear_gradient("L").resize((size, size))
    img = Image.merge("RGB", (grad, grad.rotate(90), grad.rotate(180)))
    draw = ImageDraw.Draw(img)
    for _ in range(80):
        x0, y0 = rng.randrange(size), rng.randrange(size)
        x1, y1 = x0 + rng.randrange(8, size // 3), y0 + rng.randrange(8, size // 3)
        (draw.ellipse if rng.random() < 0.5 else draw.rectangle)([x0, y0, x1, y1], fill=tuple(rng.randrange(256) for _ in range(3)))
    return img


def _norm_stats(vae):
    import torch

    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    return mean, std


def _run_vae(pipe, fn, x):
    """Run ``fn(x)`` on the device the VAE computes on.

    With CPU offload the parameters report "cpu" but hooks move them to the GPU when the VAE
    runs, so the pipeline's execution device is tried first, then the parameters' device.
    """
    import torch

    param = next(pipe.vae.parameters())
    devices = []
    for dev in (getattr(pipe, "_execution_device", None), param.device):
        if dev is not None and torch.device(dev).type != "meta" and torch.device(dev) not in devices:
            devices.append(torch.device(dev))
    error = None
    for dev in devices or [torch.device("cpu")]:
        try:
            with torch.no_grad():
                return fn(x.to(dev, param.dtype))
        except (RuntimeError, NotImplementedError) as e:
            error = e
    raise error


def _fit_map(pipe) -> np.ndarray:
    """Least-squares map from normalised latent channels (+ bias) to RGB in [-1, 1]."""
    import torch

    vae = pipe.vae
    key = id(vae)
    if key in _maps:
        return _maps[key]
    img = _calibration_image()
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)[None, :, None]
    z = _run_vae(pipe, lambda t: vae.encode(t).latent_dist.mode(), x).float().cpu()
    mean, std = _norm_stats(vae)
    z = ((z - mean) / std)[0, :, 0]  # (C, h, w)
    c, h, w = z.shape
    target = np.asarray(img.resize((w, h), Image.Resampling.BOX), dtype=np.float32) / 127.5 - 1
    a = np.concatenate([z.reshape(c, -1).T.numpy(), np.ones((h * w, 1), np.float32)], 1)
    m, *_ = np.linalg.lstsq(a, target.reshape(-1, 3), rcond=None)
    _maps[key] = m
    return m


class Previewer:
    """Called after each sampling step with the latents; emits (APNG bytes, step) via ``emit``."""

    def __init__(self, pipe, mode: str, every: int | None, fps: float, emit: Callable[[bytes, int], None], mock: bool = False):
        self.pipe, self.mode, self.fps, self.emit, self.mock = pipe, mode, fps, emit, mock
        self.every = max(1, every or DEFAULT_EVERY.get(mode, 1))
        self.prev = None
        self.map = None
        self.failed = False
        if mode == "fast" and not mock:
            try:
                self.map = _fit_map(pipe)
            except Exception as e:  # previews are best-effort
                log.warning("Fast preview calibration failed (%s); showing raw latents", e)

    def _x0(self, step: int, latents):
        sigmas = getattr(self.pipe.scheduler, "sigmas", None)
        prev, self.prev = self.prev, latents.detach().float().cpu()
        if prev is None or sigmas is None or prev.shape != self.prev.shape or step + 1 >= len(sigmas):
            return self.prev
        s0, s1 = float(sigmas[step]), float(sigmas[step + 1])
        if abs(s1 - s0) < 1e-6:
            return self.prev
        v = (self.prev - prev) / (s1 - s0)
        return self.prev - s1 * v

    def _fast(self, x0) -> list[Image.Image]:
        z = x0[0].numpy()  # (C, T, h, w)
        c, t, h, w = z.shape
        if self.map is not None and self.map.shape[0] == c + 1:
            flat = np.concatenate([z.reshape(c, -1).T, np.ones((t * h * w, 1), np.float32)], 1) @ self.map
            rgb = flat.reshape(t, h, w, 3)
        else:
            rgb = z[:3].transpose(1, 2, 3, 0)
            rgb = rgb / (np.abs(rgb).max() + 1e-6)
        rgb = ((np.clip(rgb, -1, 1) + 1) * 127.5).astype(np.uint8)
        k = max(1, MAX_SIDE // max(h, w))
        return [Image.fromarray(f).resize((w * k, h * k), Image.Resampling.BILINEAR) for f in rgb]

    def _full(self, x0) -> list[Image.Image]:
        vae = self.pipe.vae
        mean, std = _norm_stats(vae)
        video = _run_vae(self.pipe, lambda t: vae.decode(t, return_dict=False)[0], x0 * std + mean)
        frames = ((video[0].float().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 3, 0).cpu().numpy()
        return [_shrink(Image.fromarray(f)) for f in frames[::2]]

    def __call__(self, step: int, latents) -> None:
        if self.failed or latents is None:
            return
        if self.mock:
            if step % self.every == 0:
                self.emit(to_apng(self._mock_frames(step, latents), self.fps / 4), step + 1)
            return
        try:
            x0 = self._x0(step, latents)
            if step % self.every and step != 0:
                return
            frames = self._fast(x0) if self.mode == "fast" else self._full(x0)
            fps = self.fps / 4 if self.mode == "fast" else self.fps / 2
            self.emit(to_apng(frames, fps), step + 1)
        except Exception as e:
            log.warning("Preview failed (%s); turning previews off for this run", e)
            self.failed = True

    @staticmethod
    def _mock_frames(step: int, latents) -> list[Image.Image]:
        w, h = latents.get("width", 160), latents.get("height", 96)
        frames = []
        for i in range(4):
            img = Image.new("RGB", (w, h), (30 + 20 * i, 30, 60))
            ImageDraw.Draw(img).text((6, 6), f"step {step + 1} f{i}", fill=(255, 255, 255))
            frames.append(img)
        return frames
