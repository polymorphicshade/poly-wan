"""Downloading variants from the Hugging Face Hub and turning them into pipelines."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

from .variants import Variant

log = logging.getLogger(__name__)

OFFLOAD_MODES = ("auto", "none", "model", "group", "sequential")
# Never needed for inference (the repos also carry sample media and original-format weights).
_IGNORE_ALWAYS = ["assets/*", "examples/*", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.mp4", "*.webp", "*.md", "*.pth"]
_TRANSFORMER_WEIGHTS = ["transformer/*.safetensors", "transformer/*.bin"]


def _ignore(variant: Variant) -> list[str]:
    ignore = list(_IGNORE_ALWAYS)
    if variant.quant == "gguf":
        # The denoiser comes from the GGUF file; only its config is needed from the base repo.
        ignore += _TRANSFORMER_WEIGHTS
    return ignore


def is_downloaded(variant: Variant) -> bool:
    """True when every file the variant needs is already in the local HF cache."""
    if variant.name == "mock":
        return True
    from huggingface_hub import snapshot_download, try_to_load_from_cache

    try:
        snapshot_download(variant.repo, ignore_patterns=_ignore(variant), local_files_only=True)
    except Exception:
        return False
    if variant.gguf_file:
        return isinstance(try_to_load_from_cache(variant.gguf_repo, variant.gguf_file), str)
    return True


def download(variant: Variant, report: Callable[[str], None] | None = None) -> tuple[str, str | None]:
    """Fetch (or reuse cached) files for ``variant``; returns (pipeline dir, gguf path or None)."""
    from huggingface_hub import hf_hub_download, snapshot_download

    report = report or (lambda msg: log.info(msg))
    if variant.name == "mock":
        return "", None
    report(f"Downloading {variant.repo} (cached files are reused)")
    path = snapshot_download(variant.repo, ignore_patterns=_ignore(variant))
    gguf_path = None
    if variant.gguf_file:
        report(f"Downloading {variant.gguf_repo}/{variant.gguf_file}")
        gguf_path = hf_hub_download(variant.gguf_repo, variant.gguf_file)
    report("Download complete")
    return path, gguf_path


def vram_gb() -> float:
    try:
        import torch
    except ImportError:
        return 0.0
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


def denoiser_gb(variant: Variant) -> float:
    """Approximate VRAM the denoiser's weights need on the GPU."""
    return variant.denoiser_gb or variant.params_b * 2  # bf16


def resolve_offload(offload: str, variant: Variant) -> str:
    """Pick where the weights live.

    ``none``        everything on the GPU (fastest; needs the most VRAM).
    ``model``       each component moves to the GPU only while it runs (the denoiser must fit).
    ``group``       layers stream to the GPU as they run, overlapped with compute (fits in a few GB).
    ``sequential``  like group, without overlap (slowest, smallest).

    ``auto`` chooses ``none`` when the variant fits, else ``model`` when the denoiser fits with
    headroom for activations, else ``group``.
    """
    if offload not in OFFLOAD_MODES:
        raise ValueError(f"offload must be one of {', '.join(OFFLOAD_MODES)}")
    if offload != "auto":
        return offload
    total = vram_gb()
    if total == 0:
        return "none"
    if variant.vram_gb and total >= variant.vram_gb:
        return "none"
    if denoiser_gb(variant) + 3.5 <= total:
        return "model"
    return "group"


def _text_encoder_4bit(variant: Variant) -> bool:
    """UMT5-XXL is ~11 GB in bf16; on small GPUs it would not fit even on its own, so quantize it."""
    env = os.environ.get("WAN_VIDEO_TEXT_ENCODER_4BIT", "auto").lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    total = vram_gb()
    return variant.quantize_text_encoder or (0 < total < 14)


def _dequantize_non_linear_gguf_params(model, dtype) -> None:
    """Turn GGUF-typed parameters outside linear layers (e.g. norm weights) into plain tensors."""
    import torch
    from diffusers.quantizers.gguf.utils import GGUFLinear, GGUFParameter, dequantize_gguf_tensor

    for module in model.modules():
        if isinstance(module, GGUFLinear):
            continue
        for name, param in list(module.named_parameters(recurse=False)):
            if isinstance(param, GGUFParameter):
                plain = dequantize_gguf_tensor(param).to(dtype)
                setattr(module, name, torch.nn.Parameter(plain.as_subclass(torch.Tensor), requires_grad=False))


def _classes(family: str):
    import diffusers

    pipe = {
        "t2v": "WanPipeline",
        "i2v": "WanImageToVideoPipeline",
        "vace": "WanVACEPipeline",
        "animate": "WanAnimatePipeline",
    }[family]
    transformer = {
        "t2v": "WanTransformer3DModel",
        "i2v": "WanTransformer3DModel",
        "vace": "WanVACETransformer3DModel",
        "animate": "WanAnimateTransformer3DModel",
    }[family]
    try:
        return getattr(diffusers, pipe), getattr(diffusers, transformer)
    except AttributeError as e:
        raise RuntimeError(f"This diffusers version has no {pipe}; upgrade diffusers (pip install -U diffusers)") from e


def place(pipe, mode: str, report: Callable[[str], None]) -> None:
    import torch

    report(f"Placing model (offload={mode})")
    if mode == "model":
        pipe.enable_model_cpu_offload()
    elif mode == "group":
        pipe.enable_group_offload(
            onload_device=torch.device("cuda"), offload_device=torch.device("cpu"), offload_type="leaf_level", use_stream=True
        )
    elif mode == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif torch.cuda.is_available():
        pipe.to("cuda")
    if mode != "none" and hasattr(pipe.vae, "enable_tiling"):
        # Decode in spatial tiles so the VAE's peak memory does not grow with resolution.
        pipe.vae.enable_tiling()


def build_pipeline(variant: Variant, offload: str = "auto", report: Callable[[str], None] | None = None):
    report = report or (lambda msg: log.info(msg))
    if variant.name == "mock":
        from .mock import MockPipeline

        return MockPipeline()

    import torch
    from diffusers import AutoencoderKLWan, PipelineQuantizationConfig

    pipe_cls, transformer_cls = _classes(variant.family)
    path, gguf_path = download(variant, report)
    dtype = torch.bfloat16
    components: dict = {}
    quant_mapping: dict = {}

    if variant.quant == "gguf":
        from diffusers import GGUFQuantizationConfig

        report(f"Loading GGUF denoiser {variant.gguf_file}")
        components["transformer"] = transformer_cls.from_single_file(
            gguf_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
            config=path,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        _dequantize_non_linear_gguf_params(components["transformer"], dtype)
    elif variant.quant == "bnb4":
        from diffusers import BitsAndBytesConfig as DiffusersBnbConfig

        quant_mapping["transformer"] = DiffusersBnbConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)

    if _text_encoder_4bit(variant):
        from transformers import BitsAndBytesConfig as TransformersBnbConfig

        report("Text encoder will be quantized to 4-bit")
        quant_mapping["text_encoder"] = TransformersBnbConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)

    # The Wan VAE (and CLIP image encoder) are small and noticeably better in float32.
    components["vae"] = AutoencoderKLWan.from_pretrained(path, subfolder="vae", torch_dtype=torch.float32)
    if variant.family in ("i2v", "animate"):
        from transformers import CLIPVisionModel

        components["image_encoder"] = CLIPVisionModel.from_pretrained(path, subfolder="image_encoder", torch_dtype=torch.float32)

    report(f"Loading pipeline from {variant.repo}")
    pipe = pipe_cls.from_pretrained(
        path,
        torch_dtype=dtype,
        quantization_config=PipelineQuantizationConfig(quant_mapping=quant_mapping) if quant_mapping else None,
        **components,
    )
    place(pipe, resolve_offload(offload, variant), report)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def derived_pipeline(pipe, name: str):
    """Another pipeline over the same loaded weights (e.g. video-to-video from a T2V model)."""
    if name != "v2v":
        raise ValueError(name)
    if pipe.__class__.__name__ == "MockPipeline":
        return pipe
    from diffusers import WanVideoToVideoPipeline

    v2v = WanVideoToVideoPipeline.from_pipe(pipe)
    v2v.set_progress_bar_config(disable=True)
    return v2v
