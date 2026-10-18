"""End-to-end tests over the mock variant: engine, FramePack, video I/O, HTTP API and CLI.

No model weights are needed. FFmpeg is used for real (downloaded on first use if missing).
"""

import io
import json
import os

os.environ["WAN_VIDEO_ENABLE_MOCK"] = "1"
os.environ["WAN_VIDEO_MOCK_DELAY"] = "0"

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from wan_video import annotators as ann
from wan_video import framepack as fp
from wan_video.api import create_app
from wan_video.cli import main as cli_main
from wan_video.engine import Engine, Options, output_size, total_frames
from wan_video.variants import get_variant, pick_variant
from wan_video.video import VideoWriter, export, load_frames, probe

W, H = 160, 96


def make_image(size=(W, H), color=(40, 120, 200)):
    img = Image.new("RGB", size, color)
    ImageDraw.Draw(img).ellipse([size[0] // 3, size[1] // 4, size[0] * 2 // 3, size[1] * 3 // 4], fill=(250, 200, 30))
    return img


def png(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def clip(tmp_path_factory):
    """A 2-second 16 fps clip of a bright square moving left to right."""
    path = tmp_path_factory.mktemp("media") / "clip.mp4"
    w = VideoWriter(path, 16)
    for i in range(32):
        img = Image.new("RGB", (W, H), (20, 30, 40))
        x = 10 + i * 4
        ImageDraw.Draw(img).rectangle([x, 30, x + 30, 70], fill=(240, 240, 240))
        w.write(img)
    w.close()
    return path


def mean_diff(a, b):
    return float(np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b.resize(a.size), dtype=np.float32)).mean())


@pytest.fixture
def engine():
    return Engine("mock")


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(Engine("mock"), tmp_path / "outputs"))


def small(**kw):
    return Options(width=W, height=H, seed=1, **kw)


# ---------------------------------------------------------------- FramePack planning


def test_round_4k1():
    assert [fp.round_4k1(n) for n in (1, 2, 5, 6, 81, 82)] == [1, 5, 5, 9, 81, 85]
    assert fp.floor_4k1(84) == 81


def test_plan_single_pass_and_sections():
    one = fp.plan(50, sectioned=False)
    assert len(one) == 1 and one[0].length == 53 and one[0].keep == 50
    with pytest.raises(ValueError):
        fp.plan(400, sectioned=False)

    secs = fp.plan(300, section_frames=81, context_frames=5)
    assert sum(s.keep for s in secs) == 300
    assert all(s.length <= 81 and (s.length - 1) % 4 == 0 for s in secs)
    assert secs[0].context == 0 and all(s.context == 5 for s in secs[1:])
    # Each section starts where the previous one's kept frames end.
    for a, b in zip(secs, secs[1:]):
        assert b.start == a.start + a.keep


def test_use_framepack_auto():
    assert not fp.use_framepack("auto", 81)
    assert fp.use_framepack("auto", 82)
    assert fp.use_framepack("on", 10) and not fp.use_framepack("off", 1000)


# ---------------------------------------------------------------- sizes and variants


def test_output_size_and_length():
    assert output_size(Options()) == (832, 480)
    assert output_size(Options(resolution="720p", aspect="9:16")) == (720, 1280)
    w, h = output_size(Options(), source=(1000, 1000))
    assert w == h and w % 16 == 0
    assert output_size(Options(width=400)) == (400, 224)
    assert total_frames(Options(seconds=5), 16) == 81
    assert total_frames(Options(), 16, source_frames=40) == 40
    assert total_frames(Options(frames=100), 16, source_frames=40) == 40


def test_pick_variant_by_family():
    assert pick_variant(None, "vace-1.3b", ("vace", "t2v")).name == "vace-1.3b"
    assert pick_variant(None, "vace-1.3b", ("t2v",)).name == "t2v-1.3b"
    assert pick_variant(None, "vace-1.3b", ("animate",)).family == "animate"
    with pytest.raises(ValueError):
        pick_variant("t2v-1.3b", None, ("vace",))
    assert get_variant("vace-14b-q4_k_m").gguf_file == "Wan2.1_14B_VACE-Q4_K_M.gguf"


# ---------------------------------------------------------------- annotators


def test_render_pose_and_mask():
    pose = ann.template_pose(W, H)
    img = ann.render_pose(pose, W, H)
    assert img.size == (W, H) and img.getbbox() is not None and img.getpixel((0, 0)) == (0, 0, 0)
    mask = ann.person_mask(pose, W, H)
    assert mask.getpixel((W // 2, H // 2)) == 255 and mask.getpixel((2, 2)) == 0
    x0, y0, x1, y1 = ann.face_box(ann.main_person(pose), W, H)
    assert x1 - x0 == y1 - y0 and 0 <= x0 and x1 <= W


def test_retarget_moves_and_scales_onto_reference():
    drive = ann.template_pose()
    small_ref = {"people": [{"body": [[0.25 + (x - 0.5) * 0.5, 0.3 + (y - 0.5) * 0.5] for x, y in ann.TEMPLATE_BODY]}]}
    fn = ann.retarget_transform(drive, small_ref)
    out = fn(drive)["people"][0]["body"]
    ref = small_ref["people"][0]["body"]
    assert abs(out[1][0] - ref[1][0]) < 1e-3 and abs(out[1][1] - ref[1][1]) < 1e-3  # necks coincide
    assert abs((out[13][1] - out[0][1]) - (ref[13][1] - ref[0][1])) < 1e-3  # same height
    assert ann.retarget_transform({"people": []}, small_ref) is None


def test_pose_smoother_damps_jitter():
    smoother = ann.PoseSmoother(0.5)
    a = ann.template_pose()
    b = {"people": [{"body": [[x + 0.02, y] for x, y in ann.TEMPLATE_BODY]}]}
    smoother(a)
    out = smoother(b)["people"][0]["body"][0]
    assert abs(out[0] - (ann.TEMPLATE_BODY[0][0] + 0.01)) < 1e-4


def test_render_paths_moves_objects():
    paths = ann.validate_paths([{"points": [[0.1, 0.5], [0.9, 0.5]], "radius": 0.1}])
    first, last = ann.render_paths(paths, 0, 10, W, H), ann.render_paths(paths, 9, 10, W, H)
    assert first.getbbox()[0] < W // 3 and last.getbbox()[2] > W * 2 // 3
    with pytest.raises(ValueError):
        ann.validate_paths([{"points": [[0, 0]]}])


# ---------------------------------------------------------------- video I/O


def test_video_roundtrip_and_export(clip, tmp_path):
    info = probe(clip)
    assert (info.width, info.height) == (W, H) and info.fps == 16 and abs(info.frames - 32) <= 1
    frames = load_frames(clip, fps=8, size=(80, 48))
    assert 15 <= len(frames) <= 17 and frames[0].size == (80, 48)
    for fmt in ("webm", "gif", "mp4"):
        out = export(clip, tmp_path / f"o.{fmt}", fps=8, width=80, quality="small")
        assert out.stat().st_size > 0
        assert probe(out).width == 80
    assert Image.open(tmp_path / "o.gif").n_frames > 1


# ---------------------------------------------------------------- engine


def test_generate_short(engine):
    r = engine.generate("a fox", opts=small(frames=17))
    v = r.videos[0]
    assert (v.width, v.height, v.frames) == (W, H, 17) and r.seed == 1
    assert probe(v.path).width == W and r.info["sections"] == 1


def test_generate_long_uses_constant_size_sections(engine):
    r = engine.generate("a long walk", opts=small(frames=150, section_frames=33, context_frames=5))
    assert r.videos[0].frames == 150 and r.info["framepack"] and r.info["sections"] == 6
    calls = engine.pipe.calls
    # Every section samples at most one fixed-size window (only the last may be shorter).
    assert all(c["num_frames"] == 33 for c in calls[:-1]) and calls[-1]["num_frames"] <= 33
    assert calls[0]["references"] == 0 and all(c["references"] == 1 for c in calls[1:])  # anchored
    assert all(c["mask"] for c in calls[1:])  # context frames are passed as fixed frames


def test_framepack_context_carries_over(engine, tmp_path):
    from wan_video.engine import Hooks

    r = engine.generate("x", opts=small(frames=40, section_frames=17, context_frames=4, anchor=False),
                        hooks=Hooks(workdir=tmp_path))
    frames = load_frames(r.videos[0].path)
    assert len(frames) == 40
    # The mock keeps context frames unchanged, so there is no jump at the section boundary (frame 17).
    assert mean_diff(frames[16], frames[17]) < 40


def test_t2v_variant_cannot_do_long_videos(engine):
    e = Engine("mock")
    e.pipe = None
    with pytest.raises(ValueError):
        # A real T2V variant is rejected before anything loads.
        e.generate("x", opts=small(frames=300), variant="t2v-1.3b")


def test_image_to_video_starts_on_image(engine):
    img = make_image()
    r = engine.image_to_video(img, "it moves", last_image=make_image(color=(200, 20, 20)), opts=small(frames=21))
    frames = load_frames(r.videos[0].path)
    assert len(frames) == 21
    assert mean_diff(frames[0], img) < 12
    assert mean_diff(frames[-1], make_image(color=(200, 20, 20))) < 12


def test_control_depth_and_paths(engine, clip):
    r = engine.control("a robot", clip, "depth", opts=small())
    assert r.videos[0].frames == 32 and "control" in r.extras and r.extras["control"].frames == 32
    paths = [{"points": [[0.1, 0.5], [0.9, 0.5]]}]
    r = engine.control("a ball rolls", None, "path", paths=paths, first_frame=make_image(), opts=small(frames=17))
    assert r.videos[0].frames == 17 and r.info["control_type"] == "path"
    with pytest.raises(ValueError):
        engine.control("x", None, "depth", opts=small())


def test_pose_transfer(engine, clip):
    r = engine.pose(clip, make_image(), "a knight", opts=small(frames=20))
    assert r.videos[0].frames == 20 and r.info["aligned"]
    assert engine.pipe.calls[-1]["references"] == 1
    assert r.extras["control"].frames == 20


def test_animate_modes(engine, clip):
    r = engine.animate(make_image(), clip, mode="animate", opts=small(frames=24))
    assert r.videos[0].frames == 24 and engine.pipe.calls[-1]["pose_video"] == 24
    r = engine.animate(make_image(), clip, mode="replace", opts=small(frames=24))
    assert set(r.extras) == {"pose", "mask"}

def test_animate_long_video_in_windows(engine, tmp_path):
    # Long driving videos are fed to Wan-Animate in bounded windows of 4 internal segments.
    path = tmp_path / "long.mp4"
    w = VideoWriter(path, 16)
    for i in range(80):
        w.write(make_image(color=(i * 3, 40, 80)))
    w.close()
    r = engine.animate(make_image(), path, opts=small(framepack="on", section_frames=17, context_frames=1))
    assert r.info["sections"] == 2 and r.videos[0].frames == 80
    assert [c["pose_video"] for c in engine.pipe.calls[-2:]] == [65, 15]


def test_v2v_methods(engine, clip):
    r = engine.v2v(clip, "anime style", "restyle", "lineart", opts=small())
    assert r.videos[0].frames == 32 and r.info["control_type"] == "lineart"
    r = engine.v2v(clip, "anime style", "sdedit", strength=0.5, opts=small())
    assert r.videos[0].frames == 32 and engine.pipe.calls[-1]["strength"] == 0.5

    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).rectangle([0, 0, 40, 40], fill=255)
    r = engine.v2v(clip, "a red box", "inpaint", mask=mask, opts=small())
    src, out = load_frames(clip)[5], load_frames(r.videos[0].path)[5]
    # Outside the mask the source is kept.
    assert mean_diff(out.crop((60, 50, 160, 96)), src.crop((60, 50, 160, 96))) < 8
    with pytest.raises(ValueError):
        engine.v2v(clip, "x", "inpaint", opts=small())


def test_num_videos_rewinds_driving_video(engine, clip):
    r = engine.pose(clip, None, opts=small(frames=16, num_videos=2))
    assert [v.frames for v in r.videos] == [16, 16]


def test_export_op(engine, clip, tmp_path):
    from wan_video.engine import Hooks

    r = engine.export(clip, "gif", fps=8, width=80, hooks=Hooks(workdir=tmp_path))
    assert Image.open(r.files["export.gif"]).size[0] == 80


# ---------------------------------------------------------------- API


def _done(resp):
    assert resp.status_code == 200, resp.text
    job = resp.json()
    assert job["status"] == "done", job
    return job


def _video(client, url):
    return client.get(url)


def test_api_generate_and_exports(client):
    job = _done(client.post("/api/generate", data={"prompt": "a fox", "width": W, "height": H, "frames": 9, "wait": "true"}))
    v = job["videos"][0]
    assert (v["width"], v["height"], v["frames"]) == (W, H, 9)
    mp4 = client.get(v["url"])
    assert mp4.status_code == 200 and mp4.headers["content-type"] == "video/mp4"
    gif = client.get(v["exports"]["gif"] + "&download=true")
    assert gif.status_code == 200 and gif.content[:3] == b"GIF" and "attachment" in gif.headers["content-disposition"]
    webm = client.get(f"{v['url']}?format=webm&width=80")
    assert webm.status_code == 200 and webm.content[:4] == b"\x1a\x45\xdf\xa3"
    assert client.get(f"{v['url']}?format=avi").status_code == 400


def test_api_image_and_control(client, clip):
    files = {"image": ("in.png", png(make_image()), "image/png")}
    _done(client.post("/api/image", data={"width": W, "height": H, "frames": 9, "wait": "true"}, files=files))
    with open(clip, "rb") as fh:
        job = _done(client.post("/api/control", data={"prompt": "robot", "control_type": "canny", "width": W, "height": H,
                                                      "wait": "true"}, files={"video": ("c.mp4", fh, "video/mp4")}))
    assert "control" in job["extras"] and client.get(job["extras"]["control"]).status_code == 200
    paths = json.dumps([{"points": [[0.1, 0.2], [0.8, 0.8]]}])
    _done(client.post("/api/control", data={"prompt": "ball", "control_type": "path", "paths": paths, "width": W,
                                            "height": H, "frames": 9, "wait": "true"}))
    assert client.post("/api/control", data={"prompt": "x", "control_type": "path", "paths": "[]"}).status_code == 400
    assert client.post("/api/control", data={"prompt": "x", "control_type": "depth"}).status_code == 400


def test_api_pose_animate_v2v(client, clip):
    data = {"width": W, "height": H, "frames": 12, "wait": "true"}
    with open(clip, "rb") as fh:
        vid = fh.read()
    img = ("c.png", png(make_image()), "image/png")
    _done(client.post("/api/pose", data=data, files={"video": ("v.mp4", vid, "video/mp4"), "image": img}))
    _done(client.post("/api/animate", data=data | {"mode": "replace"}, files={"video": ("v.mp4", vid, "video/mp4"), "image": img}))
    sel = json.dumps({"boxes": [[0, 0, 40, 40]]})
    job = _done(client.post("/api/v2v", data=data | {"prompt": "anime", "method": "inpaint", "selection": sel},
                            files={"video": ("v.mp4", vid, "video/mp4")}))
    assert job["info"]["masked"]
    bad = client.post("/api/v2v", data={"prompt": "x", "method": "inpaint"}, files={"video": ("v.mp4", vid, "video/mp4")})
    assert bad.status_code == 400
    wrong = client.post("/api/v2v", data={"prompt": "x", "method": "sdedit", "variant": "vace-1.3b"},
                        files={"video": ("v.mp4", vid, "video/mp4")})
    assert wrong.status_code == 400  # sdedit needs a T2V model


def test_api_detect_and_export(client, clip):
    with open(clip, "rb") as fh:
        vid = fh.read()
    job = _done(client.post("/api/control/detect", data={"control_type": "pose", "width": W, "height": H, "wait": "true"},
                            files={"video": ("v.mp4", vid, "video/mp4")}))
    assert job["videos"][0]["frames"] == 32
    job = _done(client.post("/api/export", data={"format": "gif", "fps": "8", "wait": "true"},
                            files={"video": ("v.mp4", vid, "video/mp4")}))
    gif = client.get(job["files"]["export.gif"])
    assert gif.content[:3] == b"GIF"
    assert client.post("/api/export", files={"video": ("v.txt", b"not a video", "text/plain")}).status_code == 400


def test_api_rejects_bad_options(client):
    assert client.post("/api/generate", data={"prompt": "x", "variant": "nope"}).status_code == 400
    assert client.post("/api/generate", data={"prompt": "x", "resolution": "4k"}).status_code == 400
    assert client.post("/api/generate", data={"prompt": "x", "framepack": "maybe"}).status_code == 400
    assert client.post("/api/animate", data={"variant": "vace-1.3b"},
                       files={"image": ("c.png", png(make_image()), "image/png"),
                              "video": ("v.mp4", b"x", "video/mp4")}).status_code == 400


def test_api_cancel_queued_job(tmp_path):
    os.environ["WAN_VIDEO_MOCK_DELAY"] = "0.05"
    try:
        c = TestClient(create_app(Engine("mock"), tmp_path))
        first = c.post("/api/generate", data={"prompt": "slow", "steps": "40", "width": W, "height": H, "frames": 5}).json()
        second = c.post("/api/generate", data={"prompt": "queued", "width": W, "height": H, "frames": 5}).json()
        assert c.delete(f"/api/jobs/{second['id']}").json()["status"] == "cancelled"
        c.delete(f"/api/jobs/{first['id']}")
        assert c.get(f"/api/jobs/{first['id']}?wait=true").json()["status"] == "cancelled"
    finally:
        os.environ["WAN_VIDEO_MOCK_DELAY"] = "0"


def test_api_meta_and_web_ui(client):
    m = client.get("/api/models").json()
    assert any(v["name"] == "mock" for v in m["variants"]) and m["operations"]["animate"] == ["animate"]
    assert m["ffmpeg"]["available"]
    assert "path" in client.get("/api/options").json()["control_types"]
    assert "Wan Video Studio" in client.get("/").text
    for f in ("app.js", "editors.js", "style.css"):
        assert client.get(f"/static/{f}").status_code == 200


def test_api_live_preview(client):
    job = _done(client.post("/api/generate", data={"prompt": "x", "width": W, "height": H, "frames": 5, "wait": "true"}))
    assert job["preview"] is None  # cleared once finished


# ---------------------------------------------------------------- CLI


def test_cli_generate_formats(tmp_path):
    out = tmp_path / "fox.gif"
    assert cli_main(["generate", "-p", "a fox", "-m", "mock", "--width", str(W), "--height", str(H), "--frames", "9", "-o", str(out)]) == 0
    assert Image.open(out).n_frames == 9
    out = tmp_path / "fox.mp4"
    assert cli_main(["generate", "-p", "a fox", "-m", "mock", "--width", str(W), "--height", str(H), "--frames", "9", "-n", "2", "-o", str(out)]) == 0
    assert (tmp_path / "fox_1.mp4").exists() and (tmp_path / "fox_2.mp4").exists()


def test_cli_ops(tmp_path, clip):
    src = tmp_path / "img.png"
    make_image().save(src)
    common = ["-m", "mock", "--width", str(W), "--height", str(H)]
    assert cli_main(["image", "-i", str(src), "--frames", "9", *common, "-o", str(tmp_path / "i.webm")]) == 0
    ctrl = tmp_path / "ctrl.mp4"
    assert cli_main(["control", "-p", "robot", "-v", str(clip), "--type", "canny", "--save-control", str(ctrl), *common,
                     "-o", str(tmp_path / "c.mp4")]) == 0
    assert probe(ctrl).frames > 1
    assert cli_main(["control", "-p", "ball", "--type", "path", "--path", "0.1,0.5 0.9,0.5", "--frames", "9", *common,
                     "-o", str(tmp_path / "p.mp4")]) == 0
    assert cli_main(["pose", "-v", str(clip), "-i", str(src), "--frames", "12", *common, "-o", str(tmp_path / "pose.mp4")]) == 0
    assert cli_main(["animate", "-v", str(clip), "-i", str(src), "--frames", "12", *common, "-o", str(tmp_path / "a.mp4")]) == 0
    assert cli_main(["v2v", "-v", str(clip), "-p", "anime", "--method", "inpaint", "--box", "0,0,40,40", *common,
                     "-o", str(tmp_path / "v.mp4")]) == 0
    assert cli_main(["detect", "-v", str(clip), "--type", "canny", "--width", str(W), "--height", str(H), "-o", str(tmp_path / "d.mp4")]) == 0
    assert cli_main(["export", "-v", str(clip), "--fps", "8", "--width", "80", "-o", str(tmp_path / "e.gif")]) == 0
    assert Image.open(tmp_path / "e.gif").size[0] == 80
