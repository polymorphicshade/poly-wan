"""Control-video preprocessors ("annotators") for VACE, pose transfer and Wan-Animate.

Each control map follows the convention VACE was trained on: features on black (depth: near is
bright). Poses are OpenPose-style skeletons rendered from keypoints, so detected poses can be
smoothed over time, retargeted onto another body and re-rendered.

Pose keypoint JSON (coordinates normalised to 0..1, ``null`` = not visible)::

    {"width": W, "height": H, "people": [
        {"body": [[x, y] | null] * 18, "left_hand": [[x, y] | null] * 21 | null,
         "right_hand": ... | null, "face": [[x, y] | null] * 68 | null}
    ]}

Body order is OpenPose COCO-18: nose, neck, r-shoulder, r-elbow, r-wrist, l-shoulder, l-elbow,
l-wrist, r-hip, r-knee, r-ankle, l-hip, l-knee, l-ankle, r-eye, l-eye, r-ear, l-ear.
"""

from __future__ import annotations

import colorsys
import logging
import math
import threading

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

log = logging.getLogger(__name__)

ANNOTATOR_REPO = "lllyasviel/Annotators"
DEPTH_REPO = "depth-anything/Depth-Anything-V2-Small-hf"

CONTROL_TYPES = {
    "pose": "Pose (skeleton)",
    "depth": "Depth",
    "canny": "Canny edges",
    "lineart": "Line art",
    "softedge": "Soft edges (HED)",
    "scribble": "Scribble",
    "gray": "Grayscale (recolour)",
    "raw": "Already a control video",
}
POSE_DETECTORS = {"dwpose": "DWPose (whole body, more accurate)", "openpose": "OpenPose"}

LIMBS = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12], [12, 13],
         [13, 14], [2, 1], [1, 15], [15, 17], [1, 16], [16, 18]]
COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
HAND_EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11],
              [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

TEMPLATE_BODY = [
    [0.50, 0.14], [0.50, 0.24], [0.42, 0.24], [0.39, 0.38], [0.37, 0.51], [0.58, 0.24], [0.61, 0.38],
    [0.63, 0.51], [0.45, 0.52], [0.45, 0.71], [0.45, 0.90], [0.55, 0.52], [0.55, 0.71], [0.55, 0.90],
    [0.48, 0.12], [0.52, 0.12], [0.46, 0.13], [0.54, 0.13],
]


def template_pose(width: int = 832, height: int = 480) -> dict:
    return {"width": width, "height": height, "people": [{"body": [list(p) for p in TEMPLATE_BODY]}]}


# ---------------------------------------------------------------------- pose rendering


def _pt(p, w, h):
    if p is None or len(p) < 2 or p[0] is None or p[0] < 0 or p[1] < 0:
        return None
    return float(p[0]) * w, float(p[1]) * h


def render_pose(pose: dict, width: int, height: int, hands: bool = True, face: bool = True) -> Image.Image:
    """Draw OpenPose-style skeletons (limbs as ellipses, joints as dots) on black."""
    import cv2

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    s = max(width, height) / 512.0
    stick, joint_r = max(2, round(4 * s)), max(2, round(4 * s))
    for person in pose.get("people", []):
        body = (list(person.get("body") or []) + [None] * 18)[:18]
        for (a, b), color in zip(LIMBS, COLORS):
            p1, p2 = _pt(body[a - 1], width, height), _pt(body[b - 1], width, height)
            if p1 is None or p2 is None:
                continue
            mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            length = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
            angle = math.degrees(math.atan2(p1[1] - p2[1], p1[0] - p2[0]))
            poly = cv2.ellipse2Poly((int(mx), int(my)), (int(length / 2), stick), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, poly, [int(c * 0.6) for c in color])
        for p, color in zip(body, COLORS):
            q = _pt(p, width, height)
            if q is not None:
                cv2.circle(canvas, (int(q[0]), int(q[1])), joint_r, color, thickness=-1)
        if hands:
            for key in ("left_hand", "right_hand"):
                pts = person.get(key)
                if not pts:
                    continue
                pts = [_pt(p, width, height) for p in pts]
                for i, (a, b) in enumerate(HAND_EDGES):
                    if a < len(pts) and b < len(pts) and pts[a] and pts[b]:
                        rgb = colorsys.hsv_to_rgb(i / len(HAND_EDGES), 1.0, 1.0)
                        cv2.line(canvas, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]),
                                 [int(c * 255) for c in rgb], thickness=max(1, round(2 * s)))
                for q in pts:
                    if q:
                        cv2.circle(canvas, (int(q[0]), int(q[1])), max(1, round(3 * s)), (0, 0, 255), thickness=-1)
        if face and person.get("face"):
            for p in person["face"]:
                q = _pt(p, width, height)
                if q:
                    cv2.circle(canvas, (int(q[0]), int(q[1])), max(1, round(2.5 * s)), (255, 255, 255), thickness=-1)
    return Image.fromarray(canvas)


def validate_pose(pose: dict) -> dict:
    if not isinstance(pose, dict) or not isinstance(pose.get("people"), list):
        raise ValueError("pose must be an object with a 'people' list")
    for person in pose["people"]:
        if not isinstance(person, dict) or not isinstance(person.get("body"), list):
            raise ValueError("each person needs a 'body' list of 18 [x, y] points (or null)")
    return pose


# ---------------------------------------------------------------------- pose geometry


def _all_points(person: dict, keys=("body", "left_hand", "right_hand", "face")) -> list[list[float]]:
    out = []
    for key in keys:
        out += [p for p in (person.get(key) or []) if p is not None]
    return out


def person_bbox(person: dict, keys=("body",)) -> tuple[float, float, float, float] | None:
    pts = _all_points(person, keys)
    if not pts:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def main_person(pose: dict) -> dict | None:
    """The largest person in the frame."""
    people = pose.get("people") or []

    def area(p):
        b = person_bbox(p)
        return 0 if b is None else (b[2] - b[0]) * (b[3] - b[1])

    return max(people, key=area) if people else None


def _map_person(person: dict, fn) -> dict:
    out = {}
    for key, pts in person.items():
        out[key] = None if pts is None else [None if p is None else list(fn(p)) for p in pts]
    return out


def _torso(person: dict) -> tuple[list[float], float] | None:
    """(neck point, torso length): neck to the middle of the hips."""
    body = (person.get("body") or []) + [None] * 18
    neck, rh, lh = body[1], body[8], body[11]
    hips = [p for p in (rh, lh) if p is not None]
    if neck is None or not hips:
        return None
    mid = [sum(p[0] for p in hips) / len(hips), sum(p[1] for p in hips) / len(hips)]
    return neck, math.hypot(mid[0] - neck[0], mid[1] - neck[1])


def retarget_transform(first: dict, reference: dict):
    """The function that moves and scales the driving pose ``first`` onto the body in ``reference``.

    The main people are matched by neck position and torso length (or bounding-box height).
    Both poses are normalised to the same output canvas, so one factor for both axes is uniform
    in pixels. Returns None when either pose has no usable person.
    """
    ref, src = main_person(reference), main_person(first)
    if ref is None or src is None:
        return None
    a, b = _torso(src), _torso(ref)
    if a and b and a[1] > 1e-3:
        anchor_src, anchor_dst, k = a[0], b[0], b[1] / a[1]
    else:
        bs, bd = person_bbox(src), person_bbox(ref)
        if bs is None or bd is None or bs[3] - bs[1] < 1e-3:
            return None
        k = (bd[3] - bd[1]) / (bs[3] - bs[1])
        anchor_src, anchor_dst = [(bs[0] + bs[2]) / 2, bs[1]], [(bd[0] + bd[2]) / 2, bd[1]]
    k = min(4.0, max(0.25, k))

    def fn(p):
        return round(anchor_dst[0] + (p[0] - anchor_src[0]) * k, 5), round(anchor_dst[1] + (p[1] - anchor_src[1]) * k, 5)

    def apply(pose: dict) -> dict:
        return {**pose, "people": [_map_person(person, fn) for person in pose.get("people", [])]}

    return apply


class PoseSmoother:
    """Tracks people across frames by position and smooths their keypoints (exponential moving average).

    Per-frame detectors jitter; smoothing makes the skeleton video far steadier, which the model
    then follows more cleanly. ``strength`` 0 disables it, values near 1 smooth heavily.
    """

    def __init__(self, strength: float = 0.5):
        self.alpha = 1.0 - max(0.0, min(0.95, strength))
        self.tracks: list[dict] = []

    @staticmethod
    def _center(person):
        b = person_bbox(person)
        return None if b is None else ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

    def __call__(self, pose: dict) -> dict:
        if self.alpha >= 1.0:
            return pose
        out_people, used = [], set()
        for person in pose.get("people", []):
            c = self._center(person)
            best, best_d = None, 0.15
            for i, t in enumerate(self.tracks):
                tc = self._center(t)
                if i in used or c is None or tc is None:
                    continue
                d = math.hypot(c[0] - tc[0], c[1] - tc[1])
                if d < best_d:
                    best, best_d = i, d
            if best is None:
                self.tracks.append(person)
                used.add(len(self.tracks) - 1)
                out_people.append(person)
                continue
            used.add(best)
            prev = self.tracks[best]
            smoothed = {}
            for key, pts in person.items():
                old = prev.get(key)
                if pts is None or old is None or len(old) != len(pts):
                    smoothed[key] = pts
                    continue
                smoothed[key] = [
                    None if p is None else (p if q is None else [
                        round(q[0] + self.alpha * (p[0] - q[0]), 5), round(q[1] + self.alpha * (p[1] - q[1]), 5)
                    ])
                    for p, q in zip(pts, old)
                ]
            self.tracks[best] = smoothed
            out_people.append(smoothed)
        self.tracks = [t for i, t in enumerate(self.tracks) if i in used]
        return {**pose, "people": out_people}


def face_box(person: dict | None, width: int, height: int, prev: tuple | None = None) -> tuple[int, int, int, int]:
    """A square pixel box around the face: from face keypoints, else from the head joints."""
    pts = []
    if person:
        pts = [p for p in (person.get("face") or []) if p is not None]
        if len(pts) < 5:
            body = (person.get("body") or []) + [None] * 18
            pts = [body[i] for i in (0, 14, 15, 16, 17) if body[i] is not None]
    if not pts:
        if prev:
            return prev
        side = min(width, height) // 3
        return (width - side) // 2, height // 10, (width + side) // 2, height // 10 + side
    xs, ys = [p[0] * width for p in pts], [p[1] * height for p in pts]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    side = max(max(xs) - min(xs), max(ys) - min(ys)) * (1.6 if len(pts) >= 5 else 2.6)
    side = max(32.0, min(side, min(width, height)))
    x0 = int(min(max(0, cx - side / 2), width - side))
    y0 = int(min(max(0, cy - side / 2), height - side))
    return x0, y0, x0 + int(side), y0 + int(side)


def face_crop(frame: Image.Image, box: tuple[int, int, int, int], size: int = 512) -> Image.Image:
    return frame.crop(box).resize((size, size), Image.Resampling.LANCZOS)


def person_mask(pose: dict, width: int, height: int, grow: float = 0.06) -> Image.Image:
    """A coarse mask around every person: the convex hull of their keypoints, dilated.

    Wan-Animate's replacement mode expects a loose mask, so the hull is grown by ``grow`` of
    the frame's short side.
    """
    import cv2

    mask = np.zeros((height, width), dtype=np.uint8)
    for person in pose.get("people", []):
        pts = np.array([[p[0] * width, p[1] * height] for p in _all_points(person)], dtype=np.float32)
        if len(pts) >= 3:
            cv2.fillConvexPoly(mask, cv2.convexHull(pts).astype(np.int32), 255)
        elif len(pts):
            for x, y in pts:
                cv2.circle(mask, (int(x), int(y)), int(0.05 * min(width, height)), 255, -1)
    r = max(1, int(grow * min(width, height)))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return Image.fromarray(mask)


# ---------------------------------------------------------------------- motion paths


def validate_paths(paths) -> list[dict]:
    """``[{"points": [[x, y], ...], "radius": r}]`` in 0..1 coordinates; each path is one moving object."""
    if not isinstance(paths, list) or not paths:
        raise ValueError("paths must be a non-empty list of {'points': [[x, y], ...]}")
    out = []
    for p in paths:
        pts = p.get("points") if isinstance(p, dict) else p
        if not isinstance(pts, list) or len(pts) < 2:
            raise ValueError("each path needs at least two points")
        clean = [[float(x), float(y)] for x, y in pts]
        radius = float(p.get("radius", 0.08)) if isinstance(p, dict) else 0.08
        out.append({"points": clean, "radius": max(0.01, min(0.5, radius))})
    return out


def _along(points: list[list[float]], t: float) -> tuple[float, float]:
    """The point a fraction ``t`` of the way along a polyline, by arc length."""
    seg = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:])]
    total = sum(seg) or 1.0
    want = t * total
    for (a, b), d in zip(zip(points, points[1:]), seg):
        if want <= d or d == seg[-1]:
            k = 0 if d == 0 else min(1.0, want / d)
            return a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k
        want -= d
    return points[-1][0], points[-1][1]


def render_paths(paths: list[dict], index: int, count: int, width: int, height: int) -> Image.Image:
    """Frame ``index`` of a layout control video: each object drawn as a coloured box on its path."""
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    t = index / max(1, count - 1)
    for i, path in enumerate(paths):
        x, y = _along(path["points"], t)
        r = path["radius"] * min(width, height)
        rgb = colorsys.hsv_to_rgb((i * 0.31) % 1.0, 0.9, 1.0)
        draw.rectangle([x * width - r, y * height - r, x * width + r, y * height + r], fill=tuple(int(c * 255) for c in rgb))
    return img


def path_mask(paths: list[dict], width: int, height: int) -> Image.Image:
    """Where the objects start, as a mask (white = the moving objects on the first frame)."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for path in paths:
        x, y = path["points"][0]
        r = path["radius"] * min(width, height)
        draw.rectangle([x * width - r, y * height - r, x * width + r, y * height + r], fill=255)
    return mask


# ---------------------------------------------------------------------- detectors


def _np_rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def _short_side(image: Image.Image, target: int) -> Image.Image:
    w, h = image.size
    k = target / min(w, h)
    return image.resize((max(64, round(w * k)), max(64, round(h * k))), Image.Resampling.LANCZOS)


def _kp_list(points, n):
    out = []
    for p in (points or [])[:n]:
        if p is None or getattr(p, "x", -1) < 0 or getattr(p, "y", -1) < 0:
            out.append(None)
        else:
            out.append([round(float(p.x), 5), round(float(p.y), 5)])
    return out or None


def _canny(image: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    import cv2

    edges = cv2.Canny(cv2.cvtColor(_np_rgb(image), cv2.COLOR_RGB2GRAY), low, high)
    return Image.fromarray(edges).convert("RGB")


class Annotators:
    """Lazily loaded detectors; weights download from the Hub on first use.

    ``mock=True`` replaces every model with a cheap stand-in (used by tests and the mock variant):
    the "pose" follows the brightest region of the frame, depth is a blurred grayscale, and line
    detectors are Canny.
    """

    def __init__(self, mock: bool = False):
        self.mock = mock
        self._models: dict = {}
        self._lock = threading.Lock()

    @staticmethod
    def _device():
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _get(self, name, factory):
        with self._lock:
            if name not in self._models:
                log.info("Loading annotator %s", name)
                self._models[name] = factory()
            return self._models[name]

    def release(self) -> None:
        """Drop every loaded detector (they are cheap to reload and only needed during preprocessing)."""
        with self._lock:
            self._models.clear()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def detect(self, kind: str, image: Image.Image, **opt) -> tuple[Image.Image, dict]:
        """(control map at the image's size, info); info["pose"] holds keypoints for pose."""
        if kind not in CONTROL_TYPES:
            raise ValueError(f"control type must be one of {', '.join(CONTROL_TYPES)}")
        image = image.convert("RGB")
        size = image.size
        info: dict = {}
        if kind == "raw":
            out = image
        elif kind == "gray":
            out = ImageOps.grayscale(image).convert("RGB")
        elif kind == "canny":
            out = _canny(image, int(opt.get("low_threshold", 100)), int(opt.get("high_threshold", 200)))
        elif kind == "pose":
            pose = self.detect_pose(image, opt.get("detector", "dwpose"), opt.get("hands", True), opt.get("face", True))
            info["pose"] = pose
            out = render_pose(pose, *size)
        elif self.mock:
            if kind == "depth":
                out = ImageOps.autocontrast(ImageOps.grayscale(image).filter(ImageFilter.GaussianBlur(4))).convert("RGB")
            else:
                out = _canny(image, 60, 150)
        elif kind == "depth":
            out = self._depth(image)
        else:
            out = self._aux(kind, image, opt)
        if out.size != size:
            out = out.resize(size, Image.Resampling.BILINEAR)
        return out.convert("RGB"), info

    def detect_pose(self, image: Image.Image, detector: str = "dwpose", hands: bool = True, face: bool = True) -> dict:
        if detector not in POSE_DETECTORS:
            raise ValueError(f"pose detector must be one of {', '.join(POSE_DETECTORS)}")
        if self.mock:
            return self._mock_pose(image)
        if detector == "dwpose":
            from .dwpose import DWPose

            model = self._get("dwpose", DWPose)
            return model(image, hands=hands, face=face)
        from controlnet_aux import OpenposeDetector

        model = self._get("openpose", lambda: OpenposeDetector.from_pretrained(ANNOTATOR_REPO))
        small = _short_side(image.convert("RGB"), 512)
        dev = self._device()
        model.to(dev)
        try:
            results = model.detect_poses(_np_rgb(small), include_hand=hands, include_face=face)
        finally:
            model.to("cpu")
        people = []
        for r in results:
            person = {"body": _kp_list(r.body.keypoints, 18) or [None] * 18}
            if hands:
                person["left_hand"] = _kp_list(r.left_hand, 21)
                person["right_hand"] = _kp_list(r.right_hand, 21)
            if face:
                person["face"] = _kp_list(r.face, 68)
            people.append(person)
        return {"width": image.width, "height": image.height, "people": people}

    @staticmethod
    def _mock_pose(image: Image.Image) -> dict:
        """The template figure, centred on the frame's brightest region (so it follows moving content)."""
        g = np.asarray(ImageOps.grayscale(image).resize((64, 64)), dtype=np.float32)
        g = np.clip(g - g.mean(), 0, None)
        if g.sum() > 0:
            ys, xs = np.mgrid[0:64, 0:64]
            cx, cy = float((xs * g).sum() / g.sum() / 63), float((ys * g).sum() / g.sum() / 63)
        else:
            cx, cy = 0.5, 0.5
        dx, dy = cx - 0.5, (cy - 0.5) * 0.3
        body = [[round(min(1, max(0, x + dx)), 5), round(min(1, max(0, y + dy)), 5)] for x, y in TEMPLATE_BODY]
        face = [[body[0][0] + 0.02 * math.cos(a), body[0][1] + 0.02 * math.sin(a)] for a in np.linspace(0, 2 * math.pi, 68)]
        return {"width": image.width, "height": image.height, "people": [{"body": body, "face": face}]}

    def _depth(self, image: Image.Image) -> Image.Image:
        from transformers import pipeline

        dev = 0 if self._device() == "cuda" else -1
        pipe = self._get("depth", lambda: pipeline("depth-estimation", model=DEPTH_REPO, device=dev))
        depth = pipe(_short_side(image, 518))["depth"]
        return ImageOps.autocontrast(depth.convert("L")).convert("RGB")

    def _aux(self, kind: str, image: Image.Image, opt: dict) -> Image.Image:
        import controlnet_aux as aux

        res = int(opt.get("detect_resolution", 512))
        if kind in ("softedge", "scribble"):
            model = self._get("hed", lambda: aux.HEDdetector.from_pretrained(ANNOTATOR_REPO))
            kwargs = {"scribble": kind == "scribble", "safe": kind == "softedge"}
        elif kind == "lineart":
            model = self._get("lineart", lambda: aux.LineartDetector.from_pretrained(ANNOTATOR_REPO))
            kwargs = {"coarse": bool(opt.get("coarse", False))}
        else:
            raise ValueError(kind)
        dev = self._device()
        model.to(dev)
        try:
            return model(image, detect_resolution=res, image_resolution=res, output_type="pil", **kwargs)
        finally:
            model.to("cpu")


def prepare_map(image: Image.Image, invert: bool = False) -> Image.Image:
    """Normalise a user-supplied control frame; ``invert`` turns black-on-white drawings into white-on-black."""
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        bg.alpha_composite(rgba)
        image = bg
    image = image.convert("RGB")
    return ImageOps.invert(image) if invert else image
