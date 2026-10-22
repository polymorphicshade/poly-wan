# poly-wan: web UI + HTTP API on an NVIDIA GPU.
#
# The PyTorch CUDA wheels bundle the CUDA runtime, so a slim Python base is enough. The host
# only needs the NVIDIA driver and the NVIDIA Container Toolkit. Model weights are not baked in:
# they download on first use into /data/cache (mount a volume there).
#
#   docker build -t poly-wan .
#   docker run --gpus all -p 7860:7860 -v wan-cache:/data/cache -v wan-outputs:/data/outputs poly-wan

ARG PYTHON_VERSION=3.12
# CUDA 12.8 wheels run on driver >= 570 and support every GPU from Turing to Blackwell.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128

# ---------------------------------------------------------------------------- build
FROM python:${PYTHON_VERSION}-slim-bookworm AS build
ARG TORCH_INDEX

# git: diffusers is installed from its main branch (WanAnimatePipeline is not released yet).
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Torch first, from the CUDA index, in its own layer: it is by far the largest and changes least.
RUN pip install torch torchvision --index-url ${TORCH_INDEX}

WORKDIR /src
COPY pyproject.toml README.md ./
COPY wan_video ./wan_video
RUN pip install . \
    && find /opt/venv -name "__pycache__" -type d -prune -exec rm -rf {} +

# ---------------------------------------------------------------------------- runtime
FROM python:${PYTHON_VERSION}-slim-bookworm

# ffmpeg: video encoding and export (so it is never downloaded at runtime); libgl1 and
# libglib2.0-0: OpenCV, used by the preprocessors.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /data/cache /data/outputs \
    && chown -R app:app /data

COPY --from=build /opt/venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/cache/huggingface \
    TORCH_HOME=/data/cache/torch \
    XDG_CACHE_HOME=/data/cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    WAN_VIDEO_FFMPEG_AUTO_DOWNLOAD=0 \
    WAN_VIDEO_SERVE_ARGS=""

USER app
WORKDIR /data
VOLUME ["/data/cache", "/data/outputs"]
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/health', timeout=4)"

# Extra `serve` flags (e.g. "--preload --offload model") go in WAN_VIDEO_SERVE_ARGS; the default
# variant comes from WAN_VIDEO_VARIANT.
CMD ["sh", "-c", "exec wan-video serve --host 0.0.0.0 --port 7860 --outputs /data/outputs $WAN_VIDEO_SERVE_ARGS"]
