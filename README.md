# poly-wan

A custom front end for [Wan 2.1](https://github.com/Wan-Video/Wan2.1) and Wan2.2-Animate, built on PyTorch, diffusers and FastAPI. Every operation is available from the Python API, the HTTP API, the web UI and the CLI:

| Operation | What it does |
|---|---|
| **generate** | Text-to-video. With a VACE model, reference images keep a subject or style. |
| **image** | Image-to-video: a still image becomes the first frame, with an optional prompt for the motion and an optional last frame. |
| **control** | Advanced motion control with VACE. The video follows the structure and motion of a driving video, extracted as depth, pose, Canny edges, line art, soft edges, scribble or grayscale, or a ready-made control video. You can also draw **motion paths** that say how objects move. Optional subject references and fixed first/last frames. |
| **pose** | Pose-guided motion transfer. Poses are tracked through a driving video with **DWPose** or **OpenPose**, smoothed over time, fitted to your character's body, and rendered as a skeleton video that VACE follows while it keeps the character's look. |
| **animate** | Character animation with Wan2.2-Animate. The character copies the driving video's body motion, gestures *and facial expressions*. In **replace** mode it takes the place of the person in the video instead, and the scene and lighting are kept. |
| **v2v** | Video-to-video and editing. **Restyle** keeps the structure and motion and repaints the look, for example realistic → anime. **Re-render** (SDEdit) changes the video by a chosen strength. **Inpaint** regenerates one region of every frame and keeps the rest. |
| **export** | Converts any video to MP4 (H.264), WebM (VP9) or GIF with FFmpeg. FFmpeg is downloaded automatically when it is missing. |

Any operation can produce **long videos** (tens of seconds to minutes) in **constant VRAM** with FramePack-style sectioned sampling. See [Long videos](#long-videos-framepack).

## Install

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate elsewhere)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e .[dev]
```

`WanAnimatePipeline` is newer than the last diffusers release at the time of writing, so `pyproject.toml` installs diffusers from git.

## Docker (Linux server)

There are two compose files. Both run the app on an NVIDIA GPU. The host needs Docker, the NVIDIA driver (570 or newer) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). On Ubuntu or Debian, `sudo ./NVIDIA.sh` installs all three and checks that a container can see the GPU.

**Plain HTTP** (`docker-compose.yml`):

```bash
cp .env.example .env          # optional: port, variant and serve flags
docker compose up -d --build
docker compose logs -f app
```

The UI is then at `http://<host>:7860`.

**HTTPS** (`docker-compose.https.yml`) runs the same app plus an nginx sidecar that terminates TLS:

```bash
cp .env.https.example .env    # list the host's LAN address in TLS_SAN
docker compose -f docker-compose.https.yml up -d --build
```

The UI is then at `https://<host>:8443`.

- **Certificate:** a self-signed certificate is generated into `./certs` on first boot, so the browser warns once until you accept it. Every address you type in the browser must be in `TLS_SAN`, e.g. `IP:192.168.1.99`. After changing it, delete `./certs` and restart to mint a new one. To use a CA-issued certificate, put `fullchain.pem` and `privkey.pem` in `./certs`.
- **Ports:** in the HTTPS stack, 7860 is the app's plain HTTP listener and is published on `127.0.0.1` only. `https://<host>:7860` fails with `SSL_ERROR_RX_RECORD_TOO_LONG` because nothing there speaks TLS. Change either port with `HTTPS_PORT` and `PORT` in `.env`.
- **Storage:** weights download on first use into the `models` volume. Generated videos go to the `outputs` volume. Both stacks share these volumes.
- **Settings:** `WAN_VIDEO_VARIANT` sets the default variant. `WAN_VIDEO_SERVE_ARGS` passes extra `wan-video serve` flags, for example `--preload --offload model`. `HF_TOKEN` speeds up downloads. FFmpeg is installed in the image, so it is never downloaded at runtime.

The image runs on its own too: `docker build -t poly-wan .`, then `docker run --gpus all -p 7860:7860 -v wan-cache:/data/cache -v wan-outputs:/data/outputs poly-wan`.

## Model variants

Wan is a family of checkpoints, and each one is trained for a different job. Each operation runs on one of these families:

| Family | Pipeline | Operations |
|---|---|---|
| **VACE** | `WanVACEPipeline` | generate, image, control, pose, v2v restyle / inpaint, long videos |
| **T2V** | `WanPipeline` | generate (short), v2v re-render (SDEdit) |
| **I2V** | `WanImageToVideoPipeline` | image (FLF2V also takes a last frame) |
| **Animate** | `WanAnimatePipeline` | animate |

```bash
wan-video models list               # shows the variants, their family and what is downloaded
wan-video models download vace-1.3b
```

| Variant | Download | Notes |
|---|---|---|
| `vace-1.3b` | ~17.5 GB | **Default.** Handles everything except Animate, and runs on 8 GB GPUs. |
| `vace-14b`, `vace-14b-q4_k_m`, `vace-14b-q8_0` | 75 / 24 / 31 GB | Best control quality, at 480p or 720p. GGUF Q4 fits 12–16 GB cards. |
| `t2v-1.3b`, `t2v-14b`, `t2v-14b-q4_k_m` | 17 / 69 / 22 GB | Text-to-video and SDEdit. |
| `i2v-14b-480p(-q4_k_m)`, `i2v-14b-720p(-q4_k_m)`, `flf2v-14b-720p` | 23–90 GB | Dedicated image-to-video. |
| `animate-14b`, `animate-14b-q4_k_m`, `animate-14b-nf4` | 60 / 24 / 60 GB | Wan2.2-Animate. |

The GGUF denoisers come from [city96](https://huggingface.co/city96) and [QuantStack](https://huggingface.co/QuantStack). The configs, VAE and text encoder come from the official `Wan-AI` diffusers repos. When an operation needs a family the selected variant doesn't provide, it uses that family's default instead (`WAN_VIDEO_{VACE,T2V,I2V,ANIMATE}_VARIANT`). The overall default is `WAN_VIDEO_VARIANT`.

**Memory.** `--offload auto|none|model|group|sequential` picks where the weights live:

- `auto` keeps everything on the GPU when the variant fits.
- If it doesn't fit, `auto` moves each component to the GPU only while it runs (`model`), as long as the denoiser fits with headroom.
- If even the denoiser doesn't fit, `auto` streams layers to the GPU as they run (`group`), which needs only a few GB.
- The VAE decodes in tiles whenever offloading is on.
- On GPUs under 14 GB, the UMT5-XXL text encoder (11 GB in bf16) is quantized to 4-bit. Set `WAN_VIDEO_TEXT_ENCODER_4BIT=1/0` to force it on or off.

## Web UI and HTTP API

```bash
wan-video serve --port 7860 [--variant vace-1.3b] [--preload] [--offload auto]
```

Open http://127.0.0.1:7860. Each **tab** is an independent workspace with all the controls. Use **+** or Ctrl+Alt+T to open a tab, × or middle-click to close one, and double-click a tab to rename it. Tab settings are kept across reloads.

- Drop or paste (Ctrl+V) images and videos onto the inputs.
- **Preview** shows the control or pose video a run will follow, before you generate.
- On a result, use **Download MP4 / WebM / GIF**. **Continue** turns the last frame into the start of a new video. **Video→Video** sends the result to the v2v tab.
- In Video→Video inpainting, select the region with the lasso, box or brush (L/R/B/E, Ctrl+Z to undo).
- In Control, *Motion paths* lets you drag one stroke per moving object.

The UI uses only the HTTP API below. Interactive API docs are at `/docs`.

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/generate` | `prompt`, optional `references[]` |
| POST | `/api/image` | `image`, optional `prompt`, `last_image`, `references[]` |
| POST | `/api/control` | `prompt`, `control_type`, plus `video`, or `paths` JSON for `control_type=path`; optional `references[]`, `first_frame`, `last_frame`, `strength`, `invert`, `detector`, `start` |
| POST | `/api/control/detect` | `video`, `control_type`: returns the extracted control/pose video (`reference` retargets a pose onto a character) |
| POST | `/api/pose` | `video`, optional `image` (character), `prompt`, `detector` (`dwpose`/`openpose`), `align`, `smooth`, `hands`, `face`, `first_frame`, `strength` |
| POST | `/api/animate` | `image`, `video`, `mode` (`animate`/`replace`), `prompt`, `detector`, `align`, `smooth` |
| POST | `/api/v2v` | `video`, `prompt`, `method` (`restyle`/`sdedit`/`inpaint`), `control_type`, `strength`, `references[]`, `mask` or `selection` JSON |
| POST | `/api/export` | `video`, `format` (`mp4`/`webm`/`gif`), `fps`, `width`, `quality` |
| GET | `/api/jobs/{id}/videos/{n}` | a result video; `?format=webm\|gif&fps=&width=&quality=&download=true` converts it |
| GET | `/api/jobs/{id}/extras/{name}` | the control / pose / mask video a run followed |
| GET | `/api/jobs/{id}/preview` | latest live preview (animated PNG) while the job runs |
| GET / DELETE | `/api/jobs/{id}` | poll or cancel a job |
| GET | `/api/models` · `/api/options` · `/api/ffmpeg` | variants and their state, field choices, FFmpeg status |
| POST | `/api/models/load` · `/download` · `/unload`, `/api/dwpose/download`, `/api/ffmpeg/download` | model and tool management |

All generation endpoints take multipart form fields and share these options: `variant`, `resolution` (`480p`/`720p`), `aspect`, `width`, `height`, `frames` or `seconds`, `fps`, `steps`, `seed`, `guidance`, `negative_prompt`, `shift`, `num_videos`, `framepack` (`auto`/`on`/`off`), `section_frames`, `context_frames`, `anchor`, `preview`, `preview_every` and `wait`. Each call returns a job (`202`), which you poll; send `wait=true` to block until the job finishes instead. Jobs share a single GPU worker queue. For operations driven by a video, the length defaults to the driving video's length.

```bash
curl -F prompt="a red fox running through snow" -F seconds=5 -F wait=true http://127.0.0.1:7860/api/generate
curl -F video=@dancer.mp4 -F image=@knight.png -F wait=true http://127.0.0.1:7860/api/pose
curl -o fox.gif "http://127.0.0.1:7860/api/jobs/<id>/videos/0?format=gif&fps=12&width=480"
```

## CLI

```bash
wan-video generate -p "a red fox running through snow" --seconds 5 -o fox.mp4
wan-video image -i portrait.png -p "she turns and smiles" [--last-image end.png] -o smile.webm
wan-video control -p "a chrome robot dancing" -v dancer.mp4 --type depth [-r robot.png] -o robot.mp4
wan-video control -p "a red ball rolls across the table" --type path --path "0.1,0.6 0.9,0.6" --first-frame table.png -o ball.mp4
wan-video detect -v dancer.mp4 --type pose -o pose.mp4                   # preview the control video
wan-video pose -v dancer.mp4 -i knight.png [--detector openpose] -o knight_dancing.mp4
wan-video animate -i character.png -v performance.mp4 [--mode replace] -o animated.mp4
wan-video v2v -v street.mp4 -p "anime style, cel shading" [--type lineart] -o anime.mp4
wan-video v2v -v street.mp4 -p "a red sports car" --method inpaint --box 100,200,400,380 -o edit.mp4
wan-video generate -p "a timelapse of a city from dawn to night" --seconds 60 -o city.mp4   # FramePack, constant VRAM
wan-video export -v clip.mp4 -o clip.gif --fps 12 --width 480
wan-video ffmpeg status | download
```

The output format follows the extension of `-o` (`.mp4`, `.webm` or `.gif`). `--export-fps`, `--export-width` and `--quality` control that encode. By default each command loads the model in-process. Add `--api http://127.0.0.1:7860` to run the command on a server that already has the model loaded. Use `--json` for machine-readable output, and `wan-video <command> -h` to list every option.

## Long videos (FramePack)

A Wan denoiser attends over every latent frame at once, so memory grows with video length. Much past the trained 81 frames, a consumer GPU runs out of VRAM, and quality drops anyway. [FramePack](https://github.com/lllyasviel/FramePack) avoids this by generating fixed-size sections, each conditioned on a fixed-size context of what came before. Each step then costs the same however long the video is: VRAM is **O(1) in video length**.

This app applies that scheme to the Wan models (`wan_video/framepack.py`). Note that it does not use lllyasviel's HunyuanVideo FramePack weights.

- Each section samples `section_frames` frames (81 by default).
- With VACE models, the first `context_frames` (5 by default) of each section are the last frames already generated. They are passed to VACE as fixed frames, so motion carries straight over.
- With **anchor** on, the first frame, input image or character is passed to every section as a reference image. This is FramePack's anti-drift idea: identity and colours stay tied to the start.
- With I2V models, each section starts from the previous section's last frame.
- With Wan-Animate, the driving video is fed in windows of four internal segments.
- Control, pose and source videos are decoded and preprocessed section by section. Output frames are streamed to the encoder as they are produced. RAM therefore stays constant too.

`framepack=auto` (the default) switches it on when the video is longer than one section. A 60-second clip at 16 fps is 13 sections of 81 frames, and on an 8 GB card with `vace-1.3b` every one of them uses the same memory as the first.

## Pose, Animate and preprocessing

- **DWPose** runs directly on the official ONNX exports ([yzd-v/DWPose](https://huggingface.co/yzd-v/DWPose), 0.35 GB, with onnxruntime). It uses a YOLOX person detector and a whole-body keypoint model for body, hands, face and feet, so there is no mmpose dependency. **OpenPose**, HED, line art (`controlnet_aux`) and Depth Anything V2 download on first use.
- Keypoints are tracked across frames and smoothed (`smooth`, 0–0.95) to remove detector jitter.
- **Align** moves and scales the whole sequence so the first frame's neck and torso land on the character's. The motion is kept, while the size and placement follow your character.
- **Animate** also crops the driving face every frame (512×512, stabilised) for Wan-Animate's expression encoder. **Replace** mode builds a loose person mask from the keypoints and blanks that region of the background video.
- Motion paths are rendered as coloured boxes moving along each stroke, which is VACE's layout control signal. Give a first frame so the model knows which object each box is.

## Export and FFmpeg

Every job writes a high-quality H.264 master MP4. WebM (VP9) and GIF are encoded from it the first time they are requested, then cached. GIFs use a two-pass palette for clean colours. FFmpeg is found through `WAN_VIDEO_FFMPEG`, then `PATH`, then a previous download. If none of those has it, a static build is downloaded into `~/.cache/wan-video/ffmpeg`: BtbN builds on Windows and Linux, evermeet.cx on macOS. Set `WAN_VIDEO_FFMPEG_AUTO_DOWNLOAD=0` to turn this off.

## Live previews

`preview=fast` (the default) shows an animated PNG of the model's current estimate of the *finished* video, recovered from consecutive steps (`x0 = x - sigma * v`). It is projected from latents to colours with a linear map that is calibrated once per model, so each preview costs almost nothing. `preview=full` runs the real VAE decode every `preview_every` steps (5 by default). `preview=off` turns previews off. From the CLI, `--preview-dir steps/` saves each step's preview.

## Development

```bash
pytest           # runs against a mock pipeline, so no model download or GPU is needed (FFmpeg is fetched if missing)
WAN_VIDEO_ENABLE_MOCK=1 wan-video serve --variant mock   # UI/API without weights
```
