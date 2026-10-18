"""Registry of selectable Wan model variants.

Wan is not one pipeline but a family of checkpoints, each trained for a different job:

``t2v``      text-to-video (``WanPipeline``); also drives SDEdit-style video-to-video.
``i2v``      image-to-video (``WanImageToVideoPipeline``); FLF2V also takes a last frame.
``vace``     VACE all-in-one (``WanVACEPipeline``): text, first frame, reference images,
             control videos (pose, depth, edges, ...), masked video editing and extension.
``animate``  Wan2.2-Animate (``WanAnimatePipeline``): character animation and replacement.

Every operation lists the families that can run it. A variant is a family plus a checkpoint
and a quantization. All weights download from the Hugging Face Hub on first use and are cached
in the standard HF cache (``HF_HOME``).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field

FAMILIES = ("t2v", "i2v", "vace", "animate")


@dataclass(frozen=True)
class Variant:
    name: str
    label: str
    description: str
    family: str
    # Diffusers-format repo: configs, text encoder, VAE (and the denoiser unless a GGUF replaces it).
    repo: str
    # "none": bf16 weights; "bnb4": bitsandbytes NF4 applied on load; "gguf": denoiser from a GGUF file.
    quant: str = "none"
    gguf_repo: str | None = None
    gguf_file: str | None = None
    # Quantize the UMT5-XXL text encoder to 4-bit too (saves ~8 GB of RAM/VRAM).
    quantize_text_encoder: bool = False
    params_b: float = 1.3
    # Native frame rate and the resolution tiers the checkpoint was trained for.
    fps: int = 16
    resolutions: tuple[str, ...] = ("480p",)
    # Rough peak VRAM (GB) with everything on the GPU at 480p (denoiser + text encoder + VAE and
    # activations), and the denoiser's weights alone; together they decide offload="auto".
    vram_gb: float = 0.0
    denoiser_gb: float = 0.0
    approx_download_gb: float = 0.0
    hidden: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tags"] = list(self.tags)
        d["resolutions"] = list(self.resolutions)
        return d


def _gguf(base: Variant, gguf_repo: str, gguf_file: str, quant: str, size_gb: float) -> Variant:
    return Variant(
        name=f"{base.name}-{quant.lower()}",
        label=f"{base.label} GGUF {quant}",
        description=f"{base.description.split('.')[0]}. Denoiser from {gguf_repo} (~{size_gb:g} GB), text encoder in 4-bit.",
        family=base.family,
        repo=base.repo,
        quant="gguf",
        gguf_repo=gguf_repo,
        gguf_file=gguf_file,
        quantize_text_encoder=True,
        params_b=base.params_b,
        fps=base.fps,
        resolutions=base.resolutions,
        vram_gb=round(size_gb + 9, 1),  # + 4-bit text encoder (~3 GB) + VAE/activations
        denoiser_gb=size_gb,
        approx_download_gb=round(size_gb + 12, 1),
        tags=("quantized", "gguf"),
    )


_VACE_13 = Variant(
    name="vace-1.3b",
    label="VACE 1.3B",
    description="All-in-one 1.3B model: text, first frame, references, control videos, editing and long videos. Runs on 8 GB GPUs.",
    family="vace",
    repo="Wan-AI/Wan2.1-VACE-1.3B-diffusers",
    params_b=1.3,
    vram_gb=19,
    denoiser_gb=3.5,
    approx_download_gb=17.5,
    tags=("official", "default"),
)
_VACE_14 = Variant(
    name="vace-14b",
    label="VACE 14B",
    description="All-in-one 14B model at 480p or 720p. Best control and editing quality.",
    family="vace",
    repo="Wan-AI/Wan2.1-VACE-14B-diffusers",
    params_b=14,
    resolutions=("480p", "720p"),
    vram_gb=46,
    denoiser_gb=30,
    approx_download_gb=75,
    tags=("official",),
)
_T2V_13 = Variant(
    name="t2v-1.3b",
    label="T2V 1.3B",
    description="Text-to-video 1.3B. Fast, runs on 8 GB GPUs; also used for SDEdit video-to-video.",
    family="t2v",
    repo="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    params_b=1.3,
    vram_gb=18,
    denoiser_gb=2.8,
    approx_download_gb=17,
    tags=("official",),
)
_T2V_14 = Variant(
    name="t2v-14b",
    label="T2V 14B",
    description="Text-to-video 14B at 480p or 720p.",
    family="t2v",
    repo="Wan-AI/Wan2.1-T2V-14B-Diffusers",
    params_b=14,
    resolutions=("480p", "720p"),
    vram_gb=45,
    denoiser_gb=28.6,
    approx_download_gb=69,
    tags=("official",),
)
_I2V_480 = Variant(
    name="i2v-14b-480p",
    label="I2V 14B 480p",
    description="Image-to-video 14B, trained at 480p.",
    family="i2v",
    repo="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    params_b=14,
    vram_gb=47,
    denoiser_gb=32.8,
    approx_download_gb=82,
    tags=("official",),
)
_I2V_720 = Variant(
    name="i2v-14b-720p",
    label="I2V 14B 720p",
    description="Image-to-video 14B, trained at 720p.",
    family="i2v",
    repo="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
    params_b=14,
    resolutions=("720p", "480p"),
    vram_gb=50,
    denoiser_gb=32.8,
    approx_download_gb=82,
    tags=("official",),
)
_ANIMATE = Variant(
    name="animate-14b",
    label="Wan2.2 Animate 14B",
    description="Character animation: moves a character image with the body and face motion of a video, or replaces a person in it.",
    family="animate",
    repo="Wan-AI/Wan2.2-Animate-14B-Diffusers",
    params_b=14,
    fps=30,
    resolutions=("480p", "720p"),
    vram_gb=50,
    denoiser_gb=34.5,
    approx_download_gb=60,
    tags=("official",),
)

VARIANTS: dict[str, Variant] = {
    v.name: v
    for v in [
        _VACE_13,
        _VACE_14,
        _gguf(_VACE_14, "QuantStack/Wan2.1_14B_VACE-GGUF", "Wan2.1_14B_VACE-Q4_K_M.gguf", "Q4_K_M", 11.64),
        _gguf(_VACE_14, "QuantStack/Wan2.1_14B_VACE-GGUF", "Wan2.1_14B_VACE-Q8_0.gguf", "Q8_0", 18.66),
        _T2V_13,
        _T2V_14,
        _gguf(_T2V_14, "city96/Wan2.1-T2V-14B-gguf", "wan2.1-t2v-14b-Q4_K_M.gguf", "Q4_K_M", 10.12),
        _I2V_480,
        _gguf(_I2V_480, "city96/Wan2.1-I2V-14B-480P-gguf", "wan2.1-i2v-14b-480p-Q4_K_M.gguf", "Q4_K_M", 11.34),
        _I2V_720,
        _gguf(_I2V_720, "city96/Wan2.1-I2V-14B-720P-gguf", "wan2.1-i2v-14b-720p-Q4_K_M.gguf", "Q4_K_M", 11.34),
        Variant(
            name="flf2v-14b-720p",
            label="FLF2V 14B 720p",
            description="Image-to-video 14B that can also end on a given last frame.",
            family="i2v",
            repo="Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
            params_b=14,
            resolutions=("720p", "480p"),
            vram_gb=50,
            denoiser_gb=32.8,
            approx_download_gb=82,
            tags=("official", "last-frame"),
        ),
        _ANIMATE,
        _gguf(_ANIMATE, "QuantStack/Wan2.2-Animate-14B-GGUF", "Wan2.2-Animate-14B-Q4_K_M.gguf", "Q4_K_M", 11.5),
        Variant(
            name="animate-14b-nf4",
            label="Wan2.2 Animate 14B NF4",
            description="Wan2.2-Animate quantized to 4-bit NF4 on load (denoiser and text encoder).",
            family="animate",
            repo="Wan-AI/Wan2.2-Animate-14B-Diffusers",
            quant="bnb4",
            quantize_text_encoder=True,
            params_b=14,
            fps=30,
            resolutions=("480p", "720p"),
            vram_gb=23,
            denoiser_gb=9.5,
            approx_download_gb=60,
            tags=("quantized", "bitsandbytes"),
        ),
        Variant(
            name="mock",
            label="Mock (no model)",
            description="Procedural stand-in for every family that needs no download. For UI/API/CLI testing.",
            family="any",
            repo="",
            resolutions=("480p", "720p"),
            hidden=True,
            tags=("testing",),
        ),
    ]
}

DEFAULT_VARIANT = os.environ.get("WAN_VIDEO_VARIANT", "vace-1.3b")
# Used when an operation needs a family the chosen default does not provide.
FAMILY_DEFAULTS = {
    "vace": os.environ.get("WAN_VIDEO_VACE_VARIANT", "vace-1.3b"),
    "t2v": os.environ.get("WAN_VIDEO_T2V_VARIANT", "t2v-1.3b"),
    "i2v": os.environ.get("WAN_VIDEO_I2V_VARIANT", "i2v-14b-480p-q4_k_m"),
    "animate": os.environ.get("WAN_VIDEO_ANIMATE_VARIANT", "animate-14b-q4_k_m"),
}


def mock_enabled() -> bool:
    return os.environ.get("WAN_VIDEO_ENABLE_MOCK", "").lower() in ("1", "true", "yes")


def get_variant(name: str | None) -> Variant:
    name = (name or DEFAULT_VARIANT).lower()
    if name not in VARIANTS:
        raise KeyError(f"Unknown variant '{name}'. Choose one of: {', '.join(list_variant_names())}")
    v = VARIANTS[name]
    if v.hidden and not (v.name == "mock" and mock_enabled()):
        raise KeyError(f"Variant '{name}' is not enabled")
    return v


def list_variant_names() -> list[str]:
    return [v.name for v in VARIANTS.values() if not v.hidden or (v.name == "mock" and mock_enabled())]


def list_variants() -> list[Variant]:
    return [VARIANTS[n] for n in list_variant_names()]


def supports(variant: Variant, families: tuple[str, ...]) -> bool:
    return variant.family == "any" or variant.family in families


def pick_variant(requested: str | None, default: str | None, families: tuple[str, ...]) -> Variant:
    """The variant an operation runs on.

    An explicitly requested variant must belong to one of ``families``. Otherwise the server
    default is used when it fits, else the per-family default of the first family listed.
    """
    if requested:
        v = get_variant(requested)
        if not supports(v, families):
            raise ValueError(
                f"Variant '{v.name}' is a {v.family} model; this operation needs one of: {', '.join(families)}"
            )
        return v
    v = get_variant(default)
    if supports(v, families):
        return v
    return get_variant(FAMILY_DEFAULTS[families[0]])
