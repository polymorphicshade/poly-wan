"""DWPose whole-body pose estimation over the official ONNX exports (``yzd-v/DWPose``).

Two models run with onnxruntime, so there is no mmpose/mmcv dependency:

1. YOLOX-L finds people (``yolox_l.onnx``, 640x640 input).
2. DWPose-L (an RTMPose SimCC model, ``dw-ll_ucoco_384.onnx``, 288x384 input) predicts the 133
   COCO-WholeBody keypoints of each person: 17 body, 6 feet, 68 face and 2x21 hand points.

The result is converted to the OpenPose layout the rest of the app uses (18 body points with a
synthesised neck, 21 points per hand, 68 face points), normalised to 0..1.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

DWPOSE_REPO = "yzd-v/DWPose"
DET_FILE = "yolox_l.onnx"
POSE_FILE = "dw-ll_ucoco_384.onnx"
DWPOSE_GB = 0.35

_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
# COCO-WholeBody body order -> OpenPose COCO-18 (index 17 is the synthesised neck).
_MMPOSE_IDX = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
_OPENPOSE_IDX = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
VISIBLE = 0.3


def is_downloaded() -> bool:
    from huggingface_hub import try_to_load_from_cache

    return all(isinstance(try_to_load_from_cache(DWPOSE_REPO, f), str) for f in (DET_FILE, POSE_FILE))


def download(report=None) -> tuple[str, str]:
    from huggingface_hub import hf_hub_download

    if report:
        report(f"Downloading DWPose from {DWPOSE_REPO} (~{DWPOSE_GB} GB, once)")
    return hf_hub_download(DWPOSE_REPO, DET_FILE), hf_hub_download(DWPOSE_REPO, POSE_FILE)


def _nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> list[int]:
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= thr]
    return keep


class DWPose:
    def __init__(self, report=None):
        import onnxruntime as ort

        det_path, pose_path = download(report)
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.det = ort.InferenceSession(det_path, providers=providers)
        self.pose = ort.InferenceSession(pose_path, providers=providers)
        _, _, ph, pw = self.pose.get_inputs()[0].shape
        self.pose_size = (int(pw), int(ph))  # (288, 384)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ detection

    def _detect_people(self, img: np.ndarray, score_thr: float = 0.3) -> np.ndarray:
        size = 640
        h, w = img.shape[:2]
        r = min(size / h, size / w)
        import cv2

        resized = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
        padded = np.full((size, size, 3), 114, dtype=np.uint8)
        padded[: resized.shape[0], : resized.shape[1]] = resized
        inp = padded.transpose(2, 0, 1)[None].astype(np.float32)
        out = self.det.run(None, {self.det.get_inputs()[0].name: inp})[0][0]

        grids, strides = [], []
        for stride in (8, 16, 32):
            n = size // stride
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), stride))
        grids, strides = np.concatenate(grids), np.concatenate(strides)
        out[:, :2] = (out[:, :2] + grids) * strides
        out[:, 2:4] = np.exp(out[:, 2:4]) * strides

        cx, cy, bw, bh = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1) / r
        scores = out[:, 4] * out[:, 5]  # objectness x person-class score
        keep = scores > score_thr
        boxes, scores = boxes[keep], scores[keep]
        if not len(boxes):
            return np.zeros((0, 4), dtype=np.float32)
        boxes = boxes[_nms(boxes, scores, 0.45)]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        return boxes

    # ------------------------------------------------------------------ keypoints

    def _keypoints(self, img: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import cv2

        pw, ph = self.pose_size
        x0, y0, x1, y1 = box
        center = np.array([(x0 + x1) / 2, (y0 + y1) / 2], dtype=np.float32)
        scale = np.array([x1 - x0, y1 - y0], dtype=np.float32) * 1.25
        aspect = pw / ph
        if scale[0] > scale[1] * aspect:
            scale[1] = scale[0] / aspect
        else:
            scale[0] = scale[1] * aspect
        src = np.array([center, center + [0, -scale[1] / 2], center + [-scale[0] / 2, -scale[1] / 2]], dtype=np.float32)
        dst = np.array([[pw / 2, ph / 2], [pw / 2, 0], [0, 0]], dtype=np.float32)
        warp = cv2.getAffineTransform(src, dst)
        crop = cv2.warpAffine(img, warp, (pw, ph), flags=cv2.INTER_LINEAR)
        inp = ((crop.astype(np.float32) - _MEAN) / _STD).transpose(2, 0, 1)[None]
        simcc_x, simcc_y = self.pose.run(None, {self.pose.get_inputs()[0].name: inp})
        x_locs = simcc_x[0].argmax(1)
        y_locs = simcc_y[0].argmax(1)
        scores = np.minimum(simcc_x[0].max(1), simcc_y[0].max(1))
        locs = np.stack([x_locs, y_locs], 1).astype(np.float32) / 2.0  # SimCC split ratio 2
        kpts = locs / np.array([pw, ph]) * scale + center - scale / 2
        return kpts, scores

    def __call__(self, image: Image.Image, hands: bool = True, face: bool = True) -> dict:
        img = np.asarray(image.convert("RGB"))
        h, w = img.shape[:2]
        with self._lock:
            boxes = self._detect_people(img)
            results = [self._keypoints(img, b) for b in boxes]
        people = []
        for kpts, scores in results:
            neck = kpts[[5, 6]].mean(0)
            neck_score = scores[5] if scores[5] > VISIBLE and scores[6] > VISIBLE else 0.0
            kpts = np.insert(kpts, 17, neck, axis=0)
            scores = np.insert(scores, 17, neck_score)
            kpts[_OPENPOSE_IDX] = kpts[_MMPOSE_IDX]
            scores[_OPENPOSE_IDX] = scores[_MMPOSE_IDX]

            def pts(lo, hi):
                return [
                    [round(float(kpts[i, 0] / w), 5), round(float(kpts[i, 1] / h), 5)] if scores[i] > VISIBLE else None
                    for i in range(lo, hi)
                ]

            person = {"body": pts(0, 18)}
            if face:
                person["face"] = pts(24, 92)
            if hands:
                person["left_hand"] = pts(92, 113)
                person["right_hand"] = pts(113, 134)
            if any(p is not None for p in person["body"]):
                people.append(person)
        return {"width": w, "height": h, "people": people}
