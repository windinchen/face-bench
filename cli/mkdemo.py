#!/usr/bin/env python3
"""把一堆散图按人员切成 底库/ + 探针/ 两个目录，用于自检。

命名规则：名字_序号.jpg（LFW 风格自动识别）。每人取前 N 张进底库，其余作探针。
用法：python -m cli.mkdemo /Users/windinchen/work/项目任务/人脸识别/_rp/face-bench/_tmp/lfw_raw /Users/windinchen/work/项目任务/人脸识别/_rp/face-bench/_tmp/demo --per-gallery 2
"""
from __future__ import annotations

import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

IMG = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def augment(img, rng):
    """模拟不同机位/光照：轻微旋转、缩放、位移、亮度对比度、模糊、噪点。"""
    h, w = img.shape[:2]
    ang = rng.uniform(-18, 18)
    sc = rng.uniform(0.82, 1.12)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    M[0, 2] += rng.uniform(-w * 0.08, w * 0.08)
    M[1, 2] += rng.uniform(-h * 0.08, h * 0.08)
    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    out = cv2.convertScaleAbs(out, alpha=rng.uniform(0.75, 1.3), beta=rng.uniform(-22, 22))
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (3, 3), rng.uniform(0.4, 1.1))
    if rng.random() < 0.4:
        n = np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, rng.uniform(3, 9), out.shape)
        out = np.clip(out.astype(np.float32) + n, 0, 255).astype(np.uint8)
    return out


def person_of(stem: str) -> str:
    m = re.match(r"^(.*?)_0*\d{1,6}$", stem)
    return m.group(1) if m else stem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--per-gallery", type=int, default=1)
    ap.add_argument("--augment", type=int, default=0, help="每张底库图额外生成的增强探针数")
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    by_person: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(src.rglob("*")):
        if p.suffix.lower() in IMG and p.is_file():
            by_person[person_of(p.stem)].append(p)

    for sub in ("gallery", "probe"):
        d = dst / sub
        if d.exists():
            shutil.rmtree(d)
    rng = random.Random(20261106)
    n_g = n_p = aug_p = 0
    for person, files in sorted(by_person.items()):
        gd, pd = dst / "gallery" / person, dst / "probe" / person
        gd.mkdir(parents=True, exist_ok=True)
        pd.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(files):
            if i < a.per_gallery:
                shutil.copy2(f, gd / f.name)
                n_g += 1
            else:
                shutil.copy2(f, pd / f.name)
                n_p += 1
            if a.augment:
                img = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                for k in range(a.augment):
                    a_img = augment(img, rng)
                    ext = ".jpg"
                    ok, buf = cv2.imencode(ext, a_img, [cv2.IMWRITE_JPEG_QUALITY, 82])
                    if ok:
                        buf.tofile(str(pd / f"{f.stem}_aug{k}{ext}"))
                        aug_p += 1
    print(f"人员 {len(by_person)} · 底库 {n_g} 张 · 真实探针 {n_p} 张 · 增强探针 {aug_p} 张")
    print("提示：增强探针是同人不同姿态/光照的模拟，识别率会偏乐观，只用于验证流程通畅")


if __name__ == "__main__":
    main()
