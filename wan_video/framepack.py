"""FramePack-style sectioned sampling: videos of any length in constant VRAM.

A Wan denoiser attends over every latent frame at once, so memory grows with video length and
anything much past the trained 81 frames runs out of VRAM (and degrades). FramePack's idea is to
generate the video as a sequence of fixed-size sections, each conditioned on a fixed-size
*packed context* of what came before, so the cost of every step is the same no matter how long
the video gets: VRAM is O(1) in the number of frames.

This module implements that scheme on top of the Wan models (it does not use lllyasviel's
HunyuanVideo FramePack weights):

* Each section samples ``section_frames`` frames (4k+1, the Wan VAE's temporal grid).
* With VACE models, the first ``context_frames`` of a section are the last frames already
  generated, given to VACE as fixed frames (mask = keep). The model continues from them, so
  motion carries over seamlessly. The rest of the section is new.
* With ``anchor`` on, an anchor image (the first frame, the input image or the character
  reference) is passed to every section as a VACE reference image. This is FramePack's
  anti-drifting trick: identity and colours stay tied to the start instead of slowly drifting
  over many sections.
* With I2V models the context is a single frame: each section starts from the last frame of
  the one before.
* Frames are streamed to the encoder as they are produced, so RAM stays constant as well.

The context never grows, so the 10th section costs exactly what the 2nd did.
"""

from __future__ import annotations

from dataclasses import dataclass

# The length Wan 2.1 was trained on; single passes beyond it degrade and use far more memory.
TRAINED_FRAMES = 81
MAX_SINGLE_PASS = 161
MAX_TOTAL_FRAMES = 16 * 60 * 15  # 15 minutes at 16 fps


def round_4k1(n: int) -> int:
    """The smallest 4k+1 >= n (Wan's VAE compresses time 4x around a first frame)."""
    n = max(1, int(n))
    return (n - 1 + 3) // 4 * 4 + 1


def floor_4k1(n: int) -> int:
    n = max(1, int(n))
    return (n - 1) // 4 * 4 + 1


@dataclass(frozen=True)
class Section:
    index: int
    # Output frame index of this section's first *new* frame.
    start: int
    # Frames carried over from the previous section (fixed, not generated).
    context: int
    # Frames sampled in total (context + new, including padding past the end), always 4k+1.
    length: int
    # New frames that are kept in the output (the tail past the requested length is dropped).
    keep: int

    @property
    def new(self) -> int:
        return self.length - self.context

    def to_dict(self) -> dict:
        return {"index": self.index, "start": self.start, "context": self.context, "length": self.length, "keep": self.keep}


def plan(total: int, section_frames: int = TRAINED_FRAMES, context_frames: int = 5, sectioned: bool = True) -> list[Section]:
    """Split ``total`` output frames into sections.

    Without ``sectioned`` the whole video is one pass (rounded up to 4k+1 and trimmed).
    """
    total = int(total)
    if total < 1:
        raise ValueError("A video needs at least one frame")
    if total > MAX_TOTAL_FRAMES:
        raise ValueError(f"At most {MAX_TOTAL_FRAMES} frames are supported")
    if not sectioned:
        if total > MAX_SINGLE_PASS:
            raise ValueError(
                f"{total} frames is too long for a single pass (max {MAX_SINGLE_PASS}); turn on FramePack for long videos"
            )
        return [Section(0, 0, 0, round_4k1(total), total)]
    section_frames = floor_4k1(max(9, section_frames))
    context_frames = max(1, min(int(context_frames), section_frames - 4))
    first = min(section_frames, round_4k1(total))
    sections = [Section(0, 0, 0, first, min(first, total))]
    done = sections[0].keep
    while done < total:
        remaining = total - done
        length = min(section_frames, round_4k1(context_frames + remaining))
        keep = min(length - context_frames, remaining)
        sections.append(Section(len(sections), done, context_frames, length, keep))
        done += keep
    return sections


def use_framepack(mode: str, total: int, section_frames: int = TRAINED_FRAMES) -> bool:
    """``mode`` is ``on``, ``off`` or ``auto`` (on when the video is longer than one section)."""
    if mode not in ("auto", "on", "off"):
        raise ValueError("framepack must be 'auto', 'on' or 'off'")
    if mode == "auto":
        return total > floor_4k1(max(9, section_frames))
    return mode == "on"
