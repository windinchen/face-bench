"""推理后端抽象层。

PC 阶段用 OnnxBackend（CPU/CoreML）；移植到 RK3588-Ubuntu 时换成
RknnBackend（rknnlite），Android 侧只要提供 HTTP 推理服务就能用
RemoteBackend，上层检索与评测代码完全不用改。
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import cv2
import numpy as np

from .align import align, crop_square, quality_report
from .config import DEFAULTS, MODELS, model_path
from .imaging import to_blob


@dataclass
class Stage:
    """单阶段耗时记录（毫秒）。"""

    name: str
    ms: float


class BackendError(RuntimeError):
    pass


@dataclass
class Face:
    box: np.ndarray
    score: float
    kps: np.ndarray | None = None
    crop: np.ndarray | None = None
    quality: dict = field(default_factory=dict)


class Backend(ABC):
    key = "base"

    @abstractmethod
    def detect(self, img, det_size: int, thresh: float, nms: float) -> list[Face]: ...

    @abstractmethod
    def embed(self, crop: np.ndarray) -> tuple[np.ndarray, float]: ...

    @abstractmethod
    def info(self) -> dict: ...

    # ---- 通用管线：检测 -> 对齐 -> 提特征 ----
    def faces_from_image(
        self, img: np.ndarray, mode: str = "auto", det_size: int = 320, thresh: float = 0.5,
        nms: float = 0.4, min_face: int = 0, retry: bool = True,
    ) -> tuple[list[Face], dict[str, float]]:
        """返回 (人脸列表, 分阶段耗时)。

        mode="auto" 先检测；一张脸都没有时，先用更大输入+更低阈值重试一次
        （紧贴的大头照常被首检漏掉），仍失败才把整图当作已裁好的头像。
        mode="det" 只检测（底库注册用，检不到就算失败）；mode="raw" 直接整图缩放。
        检到几张脸就返回几张，因此多人合影会返回多个 Face。
        """
        t = {"detect": 0.0, "align": 0.0}
        if img is None:
            raise BackendError("faces_from_image 需要图像输入")
        faces: list[Face] = []
        if mode in ("auto", "det"):
            t0 = time.perf_counter()
            found = self.detect(img, det_size, thresh, nms)
            t["detect"] = (time.perf_counter() - t0) * 1000
            retried = False
            if not found and retry and mode == "auto" and det_size < 640:
                t1 = time.perf_counter()
                found = self.detect(img, 640, max(0.05, thresh * 0.6), nms)
                t["detect"] += (time.perf_counter() - t1) * 1000
                retried = bool(found)
            for f in found:
                box = np.asarray(f.box, np.float32).reshape(-1)
                if min_face and min(box[2] - box[0], box[3] - box[1]) < min_face:
                    continue
                t0 = time.perf_counter()
                f.crop = (
                    align(img, f.kps)
                    if f.kps is not None
                    else cv2.resize(crop_square(img, box), (112, 112), interpolation=cv2.INTER_LINEAR)
                )
                t["align"] += (time.perf_counter() - t0) * 1000
                f.box = box
                q = quality_report(img, box, f.kps)
                if retried:
                    q["retried_640"] = True
                    q["reasons"] = (q.get("reasons") or []) + ["首检失败，已用更大输入重试检出"]
                f.quality = q
                faces.append(f)
            if faces or mode == "det":
                return faces, t
        # raw / auto 兜底：整图当作已裁好的人脸头像
        t0 = time.perf_counter()
        crop = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
        t["align"] += (time.perf_counter() - t0) * 1000
        h, w = img.shape[:2]
        small = min(h, w) < 48
        faces.append(
            Face(box=np.array([0, 0, w, h], np.float32), score=1.0, kps=None, crop=crop,
                 quality={"face_w": w, "face_h": h, "sharpness": 0.0,
                          "aligned": False,
                          "reasons": (["图像过小"] if small else [])
                                     + ["未检出人脸，按整图缩放（未经关键点对齐）"],
                          "ok": not small})
        )
        return faces, t


def _session_opts(threads: int | None):
    import onnxruntime as ort

    o = ort.SessionOptions()
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    o.log_severity_level = 3
    # 关掉推理线程池的忙等：否则每次推理结束后线程会空转霸占 CPU，
    # 拖慢紧随其后的对齐/检索，并把它们的耗时统计放大两个数量级
    # （实测本机 embed 58.8ms -> 33.7ms，search 4.7ms -> 0.05ms）。
    # RK3588 上同样重要，可避免和 NPU/RGA 线程抢 A55/A76 核。
    for k, v in (("session.intra_op.allow_spinning", "0"),
                 ("session.inter_op.allow_spinning", "0")):
        try:
            o.add_session_config_entry(k, v)
        except Exception:
            pass
    # inter-op（算子级并行）无条件锁成 1，包括调用方没指定线程数的时候。
    # 检测和识别这两个图都是纯串联的卷积栈，没有可以并行的分支，开 inter-op
    # 只会多一层线程池调度；而 ORT 的默认值会给每个 session 各起一个 inter 池，
    # 识别 + 检测两个模型再乘以并发路数，线程数成倍膨胀，
    # RK3588 的 A55 小核会被上下文切换吃满，吞吐反而比单池更低。
    # 留给用户调的是 intra（单张图内部的分核），也就是下面这一行。
    o.inter_op_num_threads = 1
    if threads:
        o.intra_op_num_threads = int(threads)
    return o


class OnnxBackend(Backend):
    key = "onnx"

    def __init__(self, rec_key: str = "w600k_r50", det_key: str = "det_10g",
                 providers: list[str] | None = None, threads: int | None = None):
        import onnxruntime as ort

        from .scrfd import Scrfd

        self.rec_key, self.det_key = rec_key, det_key
        avail = ort.get_available_providers()
        if providers is None:
            providers = [p for p in ["CPUExecutionProvider"] if p in avail] or avail
        bad = [p for p in providers if p not in avail]
        if bad:
            raise BackendError(f"不支持的执行提供者: {bad}，可用: {avail}")
        self.providers = providers
        opts = _session_opts(threads)

        rp = model_path(rec_key)
        if not rp.exists():
            raise BackendError(f"识别模型缺失: {rp}")
        self.rec = ort.InferenceSession(str(rp), sess_options=opts, providers=providers)
        self.rec_input = self.rec.get_inputs()[0].name
        self.rec_output = self.rec.get_outputs()[0].name
        shp = self.rec.get_inputs()[0].shape
        self.rec_size = int(shp[3]) if len(shp) == 4 else 112
        self.rec_dim = MODELS.get(rec_key, {}).get("dim", 512)

        self.scrfd = None
        dp = model_path(det_key)
        if dp.exists():
            self.scrfd = Scrfd(
                ort.InferenceSession(str(dp), sess_options=opts, providers=providers)
            )

    def detect(self, img, det_size: int = 320, thresh: float = 0.5, nms: float = 0.4) -> list[Face]:
        if self.scrfd is None:
            raise BackendError(f"检测模型缺失: {self.det_key}")
        det, kps = self.scrfd.detect(img, det_size, thresh, nms)
        return [Face(box=d[:4], score=float(d[4]), kps=None if kps is None else kps[i])
                for i, d in enumerate(det)]

    def embed(self, crop: np.ndarray) -> tuple[np.ndarray, float]:
        blob = to_blob(crop, self.rec_size, 127.5, 127.5)
        t0 = time.perf_counter()
        out = np.asarray(self.rec.run([self.rec_output], {self.rec_input: blob})[0], np.float32)
        ms = (time.perf_counter() - t0) * 1000
        v = out.reshape(-1)
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            raise BackendError("特征范数为 0，检查模型输入尺寸")
        return v / n, ms

    def info(self) -> dict:
        import onnxruntime as ort

        return {
            "key": self.key,
            "recognition": self.rec_key,
            "detection": self.det_key if self.scrfd else None,
            "providers": self.providers,
            "available_providers": ort.get_available_providers(),
            "onnxruntime": ort.__version__,
            "rec_input_size": self.rec_size,
            "det_ready": self.scrfd is not None,
        }


class RemoteBackend(Backend):
    """指向远端推理服务（RK3588 上的 RKNN 服务或 Android App 内 HTTP 服务）。

    约定接口:
      POST /v1/detect  {image_b64, det_size, thresh, nms}
            -> {faces:[{box:[4], score, kps:[[2]x5]}], ms}
      POST /v1/embed   {image_b64(crop 112), model}
            -> {vector:[512], ms}
    """

    key = "remote"

    def __init__(self, base_url: str, timeout: float = 20.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict):
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def detect(self, img, det_size: int = 320, thresh: float = 0.5, nms: float = 0.4) -> list[Face]:
        import base64

        from .imaging import encode_jpg

        b64 = base64.b64encode(encode_jpg(img)).decode()
        res = self._post("/v1/detect",
                         {"image_b64": b64, "det_size": det_size, "thresh": thresh, "nms": nms})
        out = []
        for f in res.get("faces", []):
            kps = np.asarray(f["kps"], np.float32) if f.get("kps") else None
            out.append(Face(box=np.asarray(f["box"], np.float32), score=float(f["score"]), kps=kps))
        return out

    def embed(self, crop: np.ndarray) -> tuple[np.ndarray, float]:
        import base64

        from .imaging import encode_jpg

        b64 = base64.b64encode(encode_jpg(crop, 95)).decode()
        res = self._post("/v1/embed", {"image_b64": b64, "model": "rec"})
        v = np.asarray(res["vector"], np.float32)
        n = float(np.linalg.norm(v))
        # 和 OnnxBackend.embed 同一道闸：远端模型输出全零时 n=0，直接除得到
        # 一串 NaN 向量 —— 它过不了任何阈值，于是每张脸都算「拒绝」，
        # 落盘时又被 _drop_nonfinite 静默换成 null，报告里看着像识别率崩了，
        # 其实是远端那台服务出问题了。报错至少能说清是谁坏了。
        if n < 1e-9:
            raise BackendError("远端返回的特征范数为 0，检查远端模型与输入尺寸")
        return v / n, float(res.get("ms", 0.0))

    def info(self) -> dict:
        return {"key": self.key, "base_url": self.base_url}


def build(kind: str = "onnx", **kw) -> Backend:
    if kind == "remote":
        return RemoteBackend(kw["base_url"])
    return OnnxBackend(**{k: v for k, v in kw.items() if v is not None})


def cpu_count() -> int:
    return os.cpu_count() or 4
