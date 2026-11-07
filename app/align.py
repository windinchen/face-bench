"""5 点关键点对齐到 112x112，与 InsightFace norm_crop 等价。"""
from __future__ import annotations

import cv2
import numpy as np

from .config import ARC_REF_LMK

_REF = np.asarray(ARC_REF_LMK, dtype=np.float32)


def align(img: np.ndarray, kps: np.ndarray, size: int = 112) -> np.ndarray:
    """按 5 点做相似变换裁剪对齐，输出 size x size BGR。"""
    kps = np.asarray(kps, dtype=np.float32).reshape(-1, 2)
    if kps.shape != (5, 2):
        raise ValueError(f"关键点需要 5x2，实际 {kps.shape}")
    m, _ = cv2.estimateAffinePartial2D(kps, _REF, method=cv2.LMEDS)
    if m is None:
        raise ValueError("相似变换拟合失败")
    return cv2.warpAffine(img, m, (size, size), borderValue=0.0, flags=cv2.INTER_LINEAR)


def crop_square(img: np.ndarray, box, pad: float = 0.25) -> np.ndarray:
    """检测框外扩成正方形裁剪，用于关键点缺失时的兜底对齐。"""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = int(round(max(x2 - x1, y2 - y1) * (1.0 + pad) / 2))
    a, b = max(0, int(cx) - half), min(w, int(cx) + half)
    c, d = max(0, int(cy) - half), min(h, int(cy) + half)
    if b - a < 8 or d - c < 8:
        return img
    return img[c:d, a:b]


def sharpness(img: np.ndarray) -> float:
    if img is None or img.size == 0:
        return 0.0
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def quality_report(img: np.ndarray, box, kps) -> dict:
    """底库图质量检查：尺寸、贴边、关键点合理性、清晰度。"""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    fw, fh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    reasons: list[str] = []
    if min(fw, fh) < 48:
        reasons.append("人脸过小(<48px)")
    if x1 <= 1 or y1 <= 1 or x2 >= w - 2 or y2 >= h - 2:
        reasons.append("人脸贴边或未完整入框")
    if kps is not None:
        eyes_y = float(np.mean(kps[0:2, 1]))
        mouth_y = float(np.mean(kps[3:5, 1]))
        if mouth_y - eyes_y < fh * 0.18:
            reasons.append("关键点分布异常")
        ex = abs(float(kps[0, 0]) - float(kps[1, 0]))
        if ex < fw * 0.3:
            reasons.append("眼球间距异常(疑似侧脸)")
    crop = crop_square(img, box)
    sharp = sharpness(crop)
    if sharp < 25.0:
        reasons.append(f"清晰度偏低(lap={sharp:.0f})")
    return {
        "face_w": round(fw, 1),
        "face_h": round(fh, 1),
        "sharpness": round(sharp, 1),
        "reasons": reasons,
        "ok": not reasons,
        "aligned": True,
    }
