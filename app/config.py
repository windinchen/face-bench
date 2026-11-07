"""全局配置：模型清单、阈值先验、路径。"""
from __future__ import annotations

import hashlib
import math
import os
import re
import json
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get("FACE_MODEL_DIR") or ROOT / "models")


def _env_dir(key: str, default: Path) -> Path:
    """底库默认放在工程目录里，但板子上工程可能在只读分区、或底库要放外置盘，
    所以留一个环境变量开关；并行跑一套测试库也不用复制整个工程。"""
    v = os.environ.get(key)
    return Path(v).expanduser().resolve() if v else default


DATA_DIR = _env_dir("FACE_DATA_DIR", ROOT / "data")
REPORT_DIR = _env_dir("FACE_REPORT_DIR", ROOT / "reports")

for _d in (MODEL_DIR, DATA_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# 识别模型 112x112 对齐模板（InsightFace 通用 5 点参考）
ARC_REF_LMK = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
]

# 模型登记表：单个 .onnx 直链写在 url（走 hf-mirror，国内直连可用）；
# 官方只以 zip 发行的成员写在 zip + member，由 models/fetch.py 按 range 只抽那一段。
MODELS = {
    "w600k_r50": {
        "file": "w600k_r50.onnx",
        "kind": "recognition",
        "dim": 512,
        "size": 112,
        "sha256": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
        "far1e6_threshold": 0.4078,
        "mrall_far1e6": 91.249,
        "note": "buffalo_l 主力识别器，R50，公开榜 FAR=1e-6 时 91.25%",
        "url": "https://hf-mirror.com/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx",
    },
    "glintr100": {
        "file": "glintr100.onnx",
        "kind": "recognition",
        "dim": 512,
        "size": 112,
        "sha256": "4ab1d6435d639628a6f3e5008dd4f929edf4c4124b1a7169e1048f9fef534cdf",
        "far1e6_threshold": None,
        "mrall_far1e6": 90.659,
        "note": "antelopev2 主力识别器，ResNet100@Glint360K，算力约 w600k_r50 的 1.6 倍，作高算力对照；阈值需本地标定",
        "zip": "https://github.com/deepinsight/insightface/releases/download/model-zoo/antelopev2.zip",
        "member": "antelopev2/glintr100.onnx",
    },
    "det_10g": {
        "file": "det_10g.onnx",
        "kind": "detection",
        "strides": [8, 16, 32],
        "sha256": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",  # 实测自 HF 官方仓库文件
        "note": "SCRFD-10GF，精度优先",
        "url": "https://hf-mirror.com/public-data/insightface/resolve/main/models/buffalo_l/det_10g.onnx",
    },
    "det_2.5g": {
        "file": "det_2.5g.onnx",
        "kind": "detection",
        "strides": [8, 16, 32],
        "sha256": "041f73f47371333d1d17a6fee6c8ab4e6aecabefe398ff32cca4e2d5eaee0af9",  # 实测自官方 buffalo_m.zip
        "note": "SCRFD-2.5GF，RK3588 Gundam 方案同款，速度优先",
        "zip": "https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_m.zip",
        "member": "buffalo_m/det_2.5g.onnx",
    },
}

# 探针参数默认值（Web UI 可调）
DEFAULTS = {
    "det_size": 320,
    "det_thresh": 0.5,
    "det_nms": 0.4,
    "threshold": 0.4078,
    "topk": 5,
    "probe_mode": "auto",  # auto=先检测后对齐, det=强制检测, raw=直接缩放
    "max_side": 1280,
    # 底库（单人大头照）：一张图默认只取最大的一张脸，避免把旁人存成模板
    "max_faces": 1,            # 0 = 同图多脸全收
    "max_templates": 0,        # 每人模板数上限，0 = 不限
    "enroll_fallback_whole": 0,  # 1 = 底库检不到脸时允许按整图入库（未对齐）
    # 探针（默认也是每张图一个人，合影为可选场景）：默认逐张人脸独立出结果
    "all_faces": 1,            # 0 = 只识别最大的一张脸
    "min_face": 0,             # 忽略短边小于该像素的人脸，0 = 不过滤
    # 同时跑几张图。onnxruntime 的 run() 会放开 GIL，多核机器上多路并行是真加速
    # （本机 10 核实测 1 路 23.9 张/秒 → 5 路 41.6 张/秒），但单图延迟会涨到约 3 倍，
    # 所以吞吐一律按整批墙钟算。1 = 串行，单张响应的场景用这个。
    "workers": 1,
}

DET_SIZES = [160, 320, 640]

# 特殊真值：这张脸/这张图的人不在底库里，期望结果是「不放行」而不是「认成谁」
UNKNOWN = "__unknown__"
UNKNOWN_LABEL = "非底库人员"


def infer_person(relpath: str | None, filename: str, mode: str = "path",
                 delim: str = "_") -> str | None:
    """从目录名或文件名推断人员 ID。

    name   : 每张图一个人，文件名（去扩展名）就是人员 ID —— 底库最常见的放法
    path   : <root>/<人员ID>/<图片>，取父目录名（每人一个文件夹时用）
    id_num : 摄像头命名 张三_0001.jpg / li4-0012.jpg，去掉尾部序号
    prefix : 按自定义分隔符取前缀（分隔符在用户界面上可自定义）
    none   : 不推断（纯速度测试）

    模式名写错（比如把 prefix 打成 delim）以前会静默走 prefix 分支，
    整批真值悄悄变形却没人报错；现在直接抛 ValueError 让人一眼看见。
    """
    if mode == "none":
        return None
    if mode == "path":
        parts = [p for p in str(relpath or "").replace("\\", "/").split("/") if p]
        return parts[-2] if len(parts) >= 2 else None
    stem = Path(filename).stem.strip()
    if mode == "name":
        return stem or None
    if mode == "id_num":
        # 序号位数不限（原来只剥 6 位以内，张三20261106 这种日期后缀根本不剥），
        # 也接受没有分隔符直接跟数字的形式；只剥尾部那一段数字，人名里的数字不动
        m = re.match(r"^(.*?)[_\-\.\s]*(\d{1,})$", stem)
        if m and m.group(1).strip():
            return m.group(1).strip()
        # 纯数字文件名（12345.jpg、0007.jpg）正则同样匹配，只是剥完序号什么都不剩。
        # 原来这里落到 `or None` 判成「推不出真值」，这批图就静默退出 Rank-1/Recall 分母，
        # 而 detect_rate 的分母还是全部图 —— 两个指标的分母悄悄打架，n_truth_known 比图片数
        # 少且界面上没有任何提示，换个命名规则 Rank-1 就变还没法解释。跟下面 prefix 模式
        # 同一个口径：交回整个文件名让上层去比对底库，命中才算数，不命中也不会造假真值。
        return stem or None
    if mode == "delim":
        mode = "prefix"          # 早期文档把这个模式叫 delim，老报告和脚本还在传，收下来当别名
    if mode != "prefix":
        raise ValueError(f"未知的人员推断模式：{mode}（可选 name/path/id_num/prefix/none）")
    d = delim if delim is not None else "__"
    if d and d in stem:
        return stem.split(d)[0].strip() or None
    # 分隔符不在文件名里时退回整个文件名，而不是直接判「推不出来」：
    # 一批探针里混了几张没有分隔符的图，原来这些图整批退出真值口径（分母悄悄变小），
    # 界面却毫无提示。交回整名让上层去比对底库，命中才算数，不命中照样不会造假真值。
    return stem or None


def model_path(key: str) -> Path:
    cfg = MODELS[key]
    return MODEL_DIR / cfg["file"]


# 路径 -> (mtime_ns, 大小, 摘要)。按路径分槽，见 sha256_of 的说明
_SHA_CACHE: dict[str, tuple[int, int, str]] = {}


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    """算文件 SHA-256，按 (路径, mtime_ns, 大小) 缓存。

    识别模型 170 MB、检测模型 16 MB，而 /api/env 前端每 20 秒轮询一次、
    每次注册和评测结束还会各调一次 —— 每次都把 455 MB 重算一遍，
    本机 0.32 秒纯 CPU，RK3588 的 A55 上要 1~3 秒，还正好和被测的推理抢核。
    文件没动过就不必重算；下模型会改 mtime 和大小，缓存自动失效。

    缓存按**路径**分槽，不能算完一个就 clear()：model_status() 一次要连问
    四个模型，原先问第二个时就把第一个的结果扔了，实测轮询三次
    12 个哈希全部落空，等于这个缓存从来没生效过。
    """
    try:
        st = path.stat()
    except OSError:
        return _hash_file(path, chunk)
    key = str(path)
    hit = _SHA_CACHE.get(key)
    if hit is not None and hit[0] == st.st_mtime_ns and hit[1] == st.st_size:
        return hit[2]
    digest = _hash_file(path, chunk)
    if len(_SHA_CACHE) > 32:    # 登记表里只有 4 个模型，真涨到这个数说明有人拿它算大文件
        _SHA_CACHE.clear()
    _SHA_CACHE[key] = (st.st_mtime_ns, st.st_size, digest)
    return digest


def _hash_file(path: Path, chunk: int) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def model_status() -> list[dict]:
    out = []
    for key, cfg in MODELS.items():
        p = model_path(key)
        exists = p.exists()
        out.append(
            {
                "key": key,
                "kind": cfg["kind"],
                "note": cfg["note"],
                "file": cfg["file"],
                "exists": exists,
                "bytes": p.stat().st_size if exists else 0,
                "sha256_ok": (
                    (cfg["sha256"] is None or sha256_of(p) == cfg["sha256"]) if exists else None
                ),
                "expected_sha256": cfg["sha256"],
                "threshold": cfg.get("far1e6_threshold"),
                "mrall_far1e6": cfg.get("mrall_far1e6"),
            }
        )
    return out


def _drop_nonfinite(o):
    """把 NaN / Infinity 换成 None。

    json 默认 allow_nan=True，会把它们写成裸的 NaN / Infinity —— 那不是合法 JSON，
    浏览器 JSON.parse 整个文件直接失败，界面上「批次报告读不出来」，
    而服务端的 HTTP 层反倒会因为 Out of range float values 先抛 500。
    只在兜底路径上调它，正常批次一个 NaN 都不该有（有就是上游算错了）。
    """
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _drop_nonfinite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_drop_nonfinite(v) for v in o]
    return o


def write_json(path: Path, obj) -> None:
    """原子写 JSON：先写同目录临时文件，再 os.replace 换上去。

    同一个报告文件可能有两个写者（后台落盘线程 + 导出时同步冲一次），
    write_text 是「先截断再逐段写」，两边撞上就会留下半截文件；
    读的一侧对坏 JSON 是静默 continue，于是重启后这个批次凭空从历史里消失。
    同目录的 rename 在同一文件系统上是原子的，读者只会看到旧的那份或新的那一份。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)
    except ValueError:
        text = json.dumps(_drop_nonfinite(obj), ensure_ascii=False, indent=2)
    # 临时名里必须带进程号以外的唯一值：同一进程里就有两个写者会撞在一起
    # （后台 _flush_loop 落盘 与 /api/export/<批次> 导出同一份报告）。
    # 只用 os.getpid() 时两边写到同一个 tmp 上，A 写完 rename 走了，
    # B 的 fh 还在往那个已经不在的路径里追加，最后 os.replace 搬过去的
    # 可能是半截 JSON —— 而读的一侧 load_runs_from_disk 对坏 JSON 是静默
    # continue，这个批次就从历史里凭空消失了。
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)      # 别在报告目录里留一堆写坏一半的 .tmp
        raise


def ts_name(prefix: str = "run") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
