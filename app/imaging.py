"""图像读写：EXIF 方向修正、长边限制、blob 构造。全部用 OpenCV，避免额外依赖。"""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image, ImageOps

EXIF_OK = {270: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 90: cv2.ROTATE_90_COUNTERCLOCKWISE}


def decode(b: bytes, max_side: int = 1600) -> np.ndarray:
    """字节流 -> BGR ndarray。优先 Pillow 处理 EXIF 旋转，失败回退.imdecode。"""
    try:
        im = Image.open(io.BytesIO(b))
        im = ImageOps.exif_transpose(im)
        if im.mode != "RGB":
            im = im.convert("RGB")
        arr = np.asarray(im)[:, :, ::-1].copy()  # RGB -> BGR
    except Exception:
        arr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("无法解码图片，可能已损坏或不是图片")
    return limit_side(arr, max_side)


def limit_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if max_side and m > max_side:
        s = max_side / m
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    return img


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """等比缩放后右下贴零填充到 size x size，返回 (画布, 缩放比, 左偏移, 上偏移)。"""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = max(1, int(round(h * r))), max(1, int(round(w * r)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 127.5 if img.dtype == np.float32 else 0, dtype=img.dtype)
    canvas[:nh, :nw] = resized
    return canvas, r, 0, 0


def to_blob(img: np.ndarray, size: int, mean: float, std: float) -> np.ndarray:
    """BGR uint8 -> NCHW float32，BGR->RGB 交换，与 InsightFace 预处理一致。"""
    if isinstance(size, int):
        size = (size, size)
    return cv2.dnn.blobFromImage(
        img, 1.0 / std, size, (mean, mean, mean), swapRB=True, crop=False
    )


def resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    return limit_side(img, max_side)


def encode_jpg(img: np.ndarray, quality: int = 82) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else b""
