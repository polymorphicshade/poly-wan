"""A stand-in for every Wan pipeline that draws placeholder videos.

It accepts the call arguments of ``WanPipeline``, ``WanImageToVideoPipeline``,
``WanVACEPipeline``, ``WanVideoToVideoPipeline`` and ``WanAnimatePipeline``, and honours the
inputs in a way tests can check: frames VACE is told to keep (mask black) are returned
unchanged, I2V starts on the given image, and so on. That lets every code path above the model
(sectioning, control extraction, jobs, API, CLI, web UI) run without weights. Enable it with
``WAN_VIDEO_ENABLE_MOCK=1`` and the ``mock`` variant.
"""

from __future__ import annotations

import math
import os
import random
import textwrap
import time
from types import SimpleNamespace

from PIL import Image, ImageDraw, ImageOps


class MockPipeline:
    def __init__(self):
        self.delay = float(os.environ.get("WAN_VIDEO_MOCK_DELAY", "0.01"))
        self._interrupt = False
        self.scheduler = SimpleNamespace(sigmas=None, config={})
        self.calls: list[dict] = []

    @property
    def interrupt(self):
        return self._interrupt

    def _generated(self, rng_seed: int, j: int, n: int, w: int, h: int, prompt: str) -> Image.Image:
        rng = random.Random(rng_seed)
        c1 = tuple(rng.randrange(256) for _ in range(3))
        c2 = tuple(rng.randrange(256) for _ in range(3))
        grad = Image.linear_gradient("L").resize((w, h))
        img = Image.composite(Image.new("RGB", (w, h), c1), Image.new("RGB", (w, h), c2), grad)
        draw = ImageDraw.Draw(img)
        t = j / max(1, n - 1)
        r = min(w, h) // 8
        cx, cy = int(w * (0.15 + 0.7 * t)), int(h / 2 + h / 4 * math.sin(t * math.pi * 2))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255))
        label = textwrap.fill(f"MOCK seed={rng_seed} f{j}\n{prompt or ''}"[:200], 50)
        draw.multiline_text((8, 8), label, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        return img

    def __call__(
        self,
        prompt=None,
        negative_prompt=None,
        video=None,
        mask=None,
        reference_images=None,
        image=None,
        last_image=None,
        pose_video=None,
        face_video=None,
        background_video=None,
        mask_video=None,
        mode=None,
        height=480,
        width=832,
        num_frames=81,
        num_inference_steps=30,
        generator=None,
        callback_on_step_end=None,
        strength=None,
        segment_frame_length=77,
        **kwargs,
    ):
        self._interrupt = False
        w, h = width, height
        self.calls.append({
            "prompt": prompt, "num_frames": num_frames, "width": w, "height": h, "video": video is not None,
            "mask": mask is not None, "references": len(reference_images or []), "image": image is not None,
            "last_image": last_image is not None, "pose_video": len(pose_video or []), "mode": mode,
            "strength": strength, **{k: v for k, v in kwargs.items() if isinstance(v, (int, float, str, bool))},
        })
        seed = generator.initial_seed() if generator is not None else 0
        if pose_video is not None:
            n = len(pose_video)
        elif strength is not None and video is not None:
            n = len(video)
        else:
            n = num_frames
        steps = num_inference_steps or 30
        segments = 1
        if pose_video is not None:
            eff = segment_frame_length - kwargs.get("prev_segment_conditioning_frames", 1)
            segments = max(1, math.ceil((n - kwargs.get("prev_segment_conditioning_frames", 1)) / eff))
        for _ in range(segments):
            for i in range(steps):
                if self._interrupt:
                    break
                time.sleep(self.delay)
                if callback_on_step_end:
                    callback_on_step_end(self, i, 1000 - i * 1000 // steps, {"latents": {"width": w // 4, "height": h // 4}})

        frames = []
        for j in range(n):
            gen = self._generated(seed, j, n, w, h, prompt)
            if pose_video is not None:
                pose = pose_video[j].convert("RGB").resize((w, h))
                if mode == "replace" and background_video is not None and mask_video is not None:
                    bg = background_video[j].convert("RGB").resize((w, h))
                    gen = Image.composite(gen, bg, mask_video[j].convert("L").resize((w, h)))
                frames.append(Image.blend(gen, pose, 0.5))
            elif strength is not None and video is not None:
                src = video[j].convert("RGB").resize((w, h))
                frames.append(Image.blend(ImageOps.posterize(src, 3), gen, min(1.0, strength) * 0.5))
            elif image is not None and j == 0:
                frames.append(image.convert("RGB").resize((w, h)))
            elif image is not None and last_image is not None and j == n - 1:
                frames.append(last_image.convert("RGB").resize((w, h)))
            elif video is not None:
                src = video[j].convert("RGB").resize((w, h))
                m = mask[j].convert("L").resize((w, h)) if mask is not None else Image.new("L", (w, h), 255)
                # Generated where the mask is white, the given frame where it is black.
                frames.append(Image.composite(Image.blend(gen, src, 0.3), src, m))
            else:
                frames.append(gen)
        return SimpleNamespace(frames=[frames])

    def set_progress_bar_config(self, **_):
        pass
