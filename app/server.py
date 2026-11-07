"""FastAPI 服务：静态 UI + 评测 API。

前端是纯静态页（无构建、无外链），因此整体可以原样拷到 RK3588-Ubuntu 上
跑 `python -m app.server`，也可以只把 static/ 部署到别处，指向板子上的 API。
"""
from __future__ import annotations

import atexit
import json
import logging
import math
import os
import platform
import queue
import re
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect

from . import bench, config
from .backend import BackendError, OnnxBackend, build
from .config import DATA_DIR, DEFAULTS, REPORT_DIR, UNKNOWN, infer_person, model_status
from .gallery import Gallery
from .imaging import decode, encode_jpg, limit_side

app = FastAPI(title="Face Bench", version="0.1.0")
log = logging.getLogger("facebench")


@app.exception_handler(Exception)
async def on_unhandled(request, exc: Exception):
    """未捕获异常也要返回带原因的 JSON，并留下 traceback。

    不然前端只拿到一句空的 Internal Server Error，用户只能说「点开始评测失败」，
    服务端日志里却什么都没有（uvicorn 默认 warning 级）。
    """
    log.exception("请求失败 %s %s", request.method, request.url.path)
    return JSONResponse(
        {"detail": f"{type(exc).__name__}: {str(exc)[:300]}"}, status_code=500)


@app.exception_handler(RequestValidationError)
async def on_validation(request, exc: RequestValidationError):
    """参数校验失败要说人话。

    上传请求体没收全时，FastAPI 只会报 `missing body files`，前端原样显示成
    「HTTP 422：[{'type': 'missing'...}]」，几千张底库传到一半断掉时根本看不出
    发生了什么。这里按请求体大小把「没传完」和「真的没选文件」分开说。
    """
    errs = exc.errors()
    missing = [e for e in errs if e.get("type") == "missing" and "body" in (e.get("loc") or ())]
    raw_len = (request.headers.get("content-length") or "").strip()
    # 带 FormData 的请求一定带 multipart content-type；它和 content-length 一起缺失，
    # 说明浏览器根本没把请求体发出来（多半是所在窗口读不到磁盘上的图片）。
    ctype = (request.headers.get("content-type") or "").split(";")[0].strip() or "无"
    ua = (request.headers.get("user-agent") or "无")[:60]
    log.warning("参数校验失败 %s %s content-length=%s content-type=%s ua=%s -> %s",
                request.method, request.url.path, raw_len or "?", ctype, ua,
                [(list(e.get("loc") or []), e.get("msg")) for e in errs][:6])
    if missing and raw_len.isdigit() and int(raw_len) > 4_000_000:
        detail = (f"这一批上传的数据在服务端没收全（请求体约 {int(raw_len) // 1048576} MB），"
                  "已自动重试。反复失败通常是图片没真正读到：iCloud 或外接硬盘上的照片"
                  "要先「下载保留」到本地，或者把底库分两三次拖进来，每次一千张以内。")
    elif missing:
        empty = not raw_len.isdigit() or int(raw_len) == 0
        detail = ("浏览器这次没把图片发出来（请求体是空的）。这多半不是照片坏了，"
                  "而是当前窗口没权限读磁盘上的图片：" if empty else
                  "图片到了服务端但没带上文件：") + (
            "请改用 Chrome 或 Safari 直接打开 http://127.0.0.1:8617 重新拖入，"
            "别在应用内置的预览窗口里上传；仍不行的话按 Cmd+Shift+R 强制刷新一次。")
    else:
        detail = "；".join(
            f"{'.'.join(str(x) for x in (e.get('loc') or []))}: {e.get('msg')}" for e in errs[:4])
    return JSONResponse({"detail": detail}, status_code=422)


@app.post("/api/preflight")
async def api_preflight(request: Request) -> dict:
    """上传自检：把请求体原样读一遍，只回答「服务端到了多少字节」。

    内置预览窗口这类环境里，浏览器把 File 句柄交给网络进程后读不到磁盘，
    请求体会整个丢掉（Content-Length: 0），几千张底库换来几十屏同一个
    422「没收到 files」。客户端在传底库之前先打这里发一个小包，
    就能在第一秒发现并说清楚该换哪个浏览器。
    """
    try:
        raw = await request.body()
    except ClientDisconnect:
        # 浏览器中途掐断和「Content-Length: 0」是同一回事，一律按 0 字节回答：
        # 页面本来就认这条，能照原话讲清该换哪个浏览器，而不是被一个 500 打断。
        raw = b""
    return {"bytes": len(raw),
            "ctype": (request.headers.get("content-type") or "").split(";")[0].strip() or "无"}


@app.post("/api/clientlog")
async def api_clientlog(request: Request) -> dict:
    """把浏览器页面里 catch 到的失败原文记进服务端日志。

    页面里的异常只会写进 DOM，用户常常只回一句「评测失败」，日志里
    却只有 preflight/open 两个 200，看不出断在哪一步。有了这条日志，
    是谁的浏览器、哪一步、报的哪句话，一眼就能定位。
    """
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    log.warning("浏览器报错 [%s] %s | UA=%s", payload.get("stage") or "?",
                str(payload.get("msg") or "")[:500],
                (request.headers.get("user-agent") or "-")[:200])
    if payload.get("stack"):
        log.warning("浏览器堆栈：%s", str(payload["stack"])[:600])
    return {"ok": True}


ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
PROBE_DIR = DATA_DIR / "probes"
PROBE_DIR.mkdir(parents=True, exist_ok=True)


def build_id() -> str:
    """前端资源版本号：三个静态文件任一改动就变，用来根治「新版服务端 + 浏览器缓存的旧页面」。

    几千张底库报「没收到图片文件」，服务端却一条上传日志都没收到时，
    九成是浏览器还在用缓存里的老 app.js：它拼的是 multipart 表单，
    而服务端已经换成逐张原始字节。版本号写进 index.html 里的资源链接，
    浏览器就只能整套（HTML/JS/CSS）一起换，不会再拼出半新半旧的页面。
    """
    try:
        parts = [int((STATIC / f).stat().st_mtime) for f in ("index.html", "app.js", "style.css")]
    except OSError:
        parts = [0, 0, 0]
    return "-".join(f"{x:x}" for x in parts)

STATE = {
    "backend_kind": "onnx",
    "rec": "w600k_r50",
    "det": "det_10g",
    "providers": None,
    "threads": None,
    "remote_url": os.environ.get("FACE_REMOTE_URL", ""),
}
BACKEND = None
GALLERY: Gallery | None = None
JOBS: dict[str, dict] = {}
RUNS: dict[str, dict] = {}
# 注册在后台线程、评测在请求线程池、前端每几百毫秒轮询一次进度，
# 三边都会读改这几个全局字典。普通 Lock 一旦有函数「持锁再取锁」就自己锁死自己，
# 这里用可重入锁，换来 get_backend() 可以安心被别处带锁调用。
LOCK = threading.RLock()
# 任务进度（done/ok/bad_total）的计数锁。注册并行以后同一个任务的多个 worker
# 会同时改这几个字段：int 的 += 是「读-改-写」三步，在 CPython 里也不是原子的，
# 几千张里少掉两三条计数正好落在「到底成功几张」这种要给人看的数字上。
# 用可重入锁是因为 _finish_enroll 持锁期间还会再调 _note_bad。
_JOB_LOCK = threading.RLock()
# 一次会话在 JOBS 里最多留几个已结束的任务对象（运行中的不参与淘汰）。
MAX_JOBS = 50
# 正在往 data/probes/<批次>/ 写截图的评测（批次号 -> 目录）。
# 清空/删除历史批次要靠它认出「这个目录正被一次跑着的评测往里写」：
# 一万六千张的目录评测要跑十几分钟，中途点「清空历史」会把 out_dir 整个 rmtree，
# 正在跑的 worker 下一块开 meta.jsonl 就是 FileNotFoundError，几分钟推理全丢、
# 整批报 error。只登记真的建了目录的批次（扫描表用 bare，不建目录），
# 于是释放点全覆盖在 _probe_finish / _discard_probe_dir 两处，漏不掉也不涨。
_ACTIVE_PROBE: dict[str, Path] = {}
_ACTIVE_LOCK = threading.Lock()


# ---------------- 报告延迟落盘 ----------------
FLUSH_DELAY_S = 5.0
DIRTY: dict[str, float] = {}          # 批次号 -> 首次标脏时间
_DIRTY_REV: dict[str, int] = {}       # 批次号 -> 标脏次数，用来发现「写盘期间又改了」
_DIRTY_LOCK = threading.Lock()
_flush_thread: threading.Thread | None = None


def _flush_run(run_id: str) -> bool:
    """把一个批次立刻写回 reports/，返回这一版是否已经落干净。

    导出、清历史、进程退出前都要先冲干净。返回 False 表示写的这几毫秒里又被改了
    （手还压在阈值滑杆上），脏标记原样留着等下一轮，不能当成写完。

    写盘本身在两把锁之外：报告 JSON 有几十 MB，攥着 LOCK 写磁盘会让前端
    轮询历史批次的请求陪着等。这里每段只拿一把锁、拿完就放，全程不嵌套，
    文档里那条锁序（LOCK -> _DIRTY_LOCK）根本没机会出现，也就不会互相咬住。
    """
    with LOCK:
        run = RUNS.get(run_id)
    if run is None or run.get("_stub"):
        with _DIRTY_LOCK:
            DIRTY.pop(run_id, None)
            _DIRTY_REV.pop(run_id, None)
        return True
    with _DIRTY_LOCK:
        rev = _DIRTY_REV.get(run_id, 0)
    config.write_json(REPORT_DIR / f"probe_{run_id}.json", run)
    # 「删批次」的请求可能正好插在取 run 和写完文件之间：它删报告、撤 RUNS、撤 DIRTY，
    # 然后这里把文件又落回磁盘 —— 重启时 load_runs_from_disk 把用户已经删掉的批次
    # 原样读回来，下拉里它复活了。写完认一次账：表里没这批了就当没写过，把文件撤掉。
    with LOCK:
        gone = RUNS.get(run_id) is None
    with _DIRTY_LOCK:
        if gone:
            # 批次已经被删掉了：这一版落不落地都无所谓，撤掉脏标记就行。
            # 留着的话后台线程会一直替一个不存在的批次反复写盘。
            DIRTY.pop(run_id, None)
            _DIRTY_REV.pop(run_id, None)
        elif _DIRTY_REV.get(run_id, 0) != rev:
            return False            # 写盘期间又被标脏，留着下一轮重来
        else:
            DIRTY.pop(run_id, None)
            _DIRTY_REV.pop(run_id, None)
    if gone:
        try:
            (REPORT_DIR / f"probe_{run_id}.json").unlink(missing_ok=True)
        except OSError:
            log.warning("批次 %s 已删除，但刚写回的报告没撤掉", run_id, exc_info=True)
    return True


def _flush_loop() -> None:
    while True:
        with _DIRTY_LOCK:
            due = [r for r, t in list(DIRTY.items()) if time.time() - t >= FLUSH_DELAY_S]
        for r in due:
            try:
                _flush_run(r)
            except Exception:
                # 写失败（磁盘满、瞬时 IO 错误）必须留在 DIRTY 里等下一轮，
                # 原来先 pop 再写、异常只 log，那一次改标就永久没了。
                # 重新盖时间戳是为了退避：不然每秒重试一次，磁盘满时能刷爆日志。
                log.warning("批次 %s 落盘失败，%.0fs 后重试", r, FLUSH_DELAY_S, exc_info=True)
                with _DIRTY_LOCK:
                    DIRTY[r] = time.time()
        time.sleep(1.0)


def _mark_dirty(run_id: str) -> None:
    """标脏，保证有个后台线程在收口。

    拖一次阈值滑杆、改一张真值，都要把整份报告重写一遍：一万张的批次是几十 MB，
    手连着拖十几下就是几百 MB 的无谓写盘。开发机 SSD 上只是慢一点，
    RK3588 常见的 eMMC 上会直接把界面拖到卡住。攒一个静默窗口，
    手停下来了才真写一次；任何一次改动最迟 6 秒后一定落盘（时间戳只记首次，
    不会被人一直拖就一直推迟），导出时再强制冲一次。
    """
    global _flush_thread
    with _DIRTY_LOCK:
        DIRTY.setdefault(run_id, time.time())
        # 每标一次脏加一代：写盘要几十毫秒，期间完全可以又被改了一次，
        # 只看「在不在 DIRTY 里」分不出落盘的是不是最新那一版。
        _DIRTY_REV[run_id] = _DIRTY_REV.get(run_id, 0) + 1
        if _flush_thread is None or not _flush_thread.is_alive():
            _flush_thread = threading.Thread(target=_flush_loop, name="report-flush", daemon=True)
            _flush_thread.start()


def _flush_all() -> None:
    """退出前把没写完的都写完，别让「刚改完就关服务」丢掉最后一次判定。

    写完一轮再扫一轮：落盘那几毫秒里可能又被改了（`_flush_run` 会把脏标记留着），
    进程只跑这一回的场合不补这一下，最后一次改标就真丢了。
    """
    for _attempt in range(3):
        with _DIRTY_LOCK:
            ids = list(DIRTY)
        if not ids:
            return
        for r in ids:
            try:
                _flush_run(r)
            except Exception:
                log.exception("退出前落盘 %s 失败", r)


atexit.register(_flush_all)


def get_backend(force: bool = False):
    global BACKEND
    with LOCK:
        if BACKEND is None or force:
            if STATE["backend_kind"] == "remote":
                BACKEND = build("remote", base_url=STATE["remote_url"])
            else:
                BACKEND = build(
                    "onnx", rec_key=STATE["rec"], det_key=STATE["det"],
                    providers=STATE["providers"], threads=STATE["threads"],
                )
        return BACKEND


def get_gallery() -> Gallery:
    global GALLERY
    # 以前这三行不带锁：两个线程同时判定「模型换了」就各建一个 Gallery，
    # 各读一遍整库（几千张向量、几百 MB），而后台注册线程还在往老对象里写，
    # 界面已经指向新对象 —— 表现就是「明明注册了几千张，人数不对」。
    with LOCK:
        key = STATE["rec"] if STATE["backend_kind"] == "onnx" else f"remote_{STATE['remote_url']}"
        if GALLERY is None or GALLERY.model_key != key:
            GALLERY = Gallery(key)
        return GALLERY


@app.get("/api/env")
def api_env():
    st = model_status()
    info = {
        "machine": {
            "system": f"{platform.system()} {platform.release()}",
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "models": st,
        "state": dict(STATE),
        "defaults": DEFAULTS,
        "probe_max": PROBE_MAX_IMAGES,
        "probe_dir_max": PROBE_DIR_MAX,
        # 扫描的几道上限一并下发：前端以前抄了一份 SWEEP_MAX_CELLS，
        # 服务端改上限（比如给板子降到 12 格）时前端不知道，
        # 会在本地把一格都没超的选择判成「超过上限」，或者反过来放行一个必被 400 的请求。
        "sweep": {
            "max_cells": SWEEP_MAX_CELLS,
            "workers_max": SWEEP_WORKERS_MAX,
            "grid_workers": list(SWEEP_GRID_WORKERS),
            "grid_threads": list(SWEEP_GRID_THREADS),
            "default_images": SWEEP_DEFAULT_IMAGES,
            "min_images": SWEEP_MIN_IMAGES,
            "max_images": SWEEP_MAX_IMAGES,
            "latency_ms": SWEEP_LATENCY_MS,
        },
        "build": build_id(),
        "gallery": get_gallery().stats(),
        "ready": any(m["exists"] and m["kind"] == "recognition" for m in st),
    }
    try:
        info["backend"] = get_backend().info()
    except Exception as e:
        info["backend"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    return info


@app.post("/api/backend")
def api_backend(payload: dict = Body(...)):
    with LOCK:
        _apply_backend_state(payload)
    global BACKEND, GALLERY
    BACKEND = None
    try:
        bk = get_backend(force=True)
    except BackendError as e:
        # 其余路由一律抛 HTTPException（错误信息在 detail 里），这里以前返回 {"error": ...}，
        # 前端按统一口径取 detail 就永远显示「请求失败」，看不出是模型文件没了还是路径写错
        raise HTTPException(400, str(e)) from e
    with LOCK:
        GALLERY = get_gallery()
    return {"backend": bk.info(), "gallery": get_gallery().stats()}


def _apply_backend_state(payload: dict) -> None:
    """校验 /api/backend 的入参，全部合格才一次性写进 STATE。

    原来是一路裸赋值 `STATE[k] = payload[k]`，两个后果都很实在：
      · rec 填错时 MODELS[key] 抛的是 KeyError，不是 BackendError，躲过了 400 分支，
        界面收到 500；而 STATE 已经被改成那个不存在的 key，之后每次 /api/env
        都显示底库为空、且没有任何入口能变回来，只能重启。
      · get_gallery() 拿它拼 DATA_DIR / f"gallery_{key}"，于是
        "../../../../Users/windinchen/work/项目任务/人脸识别/_rp/face-bench/_tmp/x" 会把缩略图目录一路建到工程外面去。
    所以先全部验完再赋值：要么整份生效，要么原样不动。
    """
    known = {k: v.get("kind") for k, v in config.MODELS.items()}
    for key, want in (("rec", "recognition"), ("det", "detection")):
        if key not in payload:
            continue
        v = payload[key]
        if not isinstance(v, str) or known.get(v) != want:
            raise HTTPException(
                400, f"{key} 只能是已登记的{('识别' if want == 'recognition' else '检测')}模型："
                     + "、".join(k for k, kind in known.items() if kind == want)
                     + f"（收到 {v!r}）")
    if "backend_kind" in payload and payload["backend_kind"] not in ("onnx", "remote"):
        raise HTTPException(400, "backend_kind 只能是 onnx 或 remote")
    if "remote_url" in payload:
        url = str(payload["remote_url"] or "").strip()
        # 只放行 http/https：RemoteBackend 会真去请求这个地址，
        # file:// 或 gopher:// 之类的会把服务变成别人的 SSRF 跳板
        if url and urlparse(url).scheme not in ("http", "https"):
            raise HTTPException(400, f"remote_url 必须是 http(s) 地址（收到 {url!r}）")
        payload = {**payload, "remote_url": url}
    if "threads" in payload:
        raw = payload["threads"]
        if raw in (None, "", 0, "0"):
            payload = {**payload, "threads": None}      # 空 = 交给 onnxruntime 自己定
        else:
            try:
                n = int(raw)
            except (TypeError, ValueError):
                raise HTTPException(400, f"threads 得是整数（收到 {raw!r}）") from None
            if not 0 <= n <= 256:
                raise HTTPException(400, f"threads 要在 0~256 之间（收到 {n}）")
            payload = {**payload, "threads": n or None}
    if "providers" in payload:
        pv = payload["providers"]
        if pv is not None and (not isinstance(pv, list)
                               or not all(isinstance(x, str) and x.strip() for x in pv)):
            raise HTTPException(400, 'providers 要写成 ["CPUExecutionProvider"] 这样的列表')
    for k in ("backend_kind", "rec", "det", "providers", "threads", "remote_url"):
        if k in payload:
            STATE[k] = payload[k]


def _clean_threshold(raw) -> float:
    """把界面传来的判定阈值收成 0~1 之间的有限小数，不合法直接 400。

    NaN / Infinity 会一路漂到报告里：config.write_json 以前按默认 allow_nan=True
    落盘，写出的是裸 NaN 字面量 —— 那不是合法 JSON，浏览器 JSON.parse 整份失败；
    而 HTTP 响应侧更早，JSONResponse 遇到非有限浮点直接抛 500。
    阈值本来就是余弦相似度的判定线，只有 [0,1] 这一段有意义。
    """
    try:
        thr = float(raw)
    except (TypeError, ValueError):
        raise HTTPException(400, f"阈值得是个数字（收到 {raw!r}）") from None
    # 两种失败分开报：把 0.6 敲成 60 是数字错，NaN 是 JSON 字面量，
    # 处置办法不一样。以前共用一句话，对着 1.5 会困惑「我没写 NaN 啊」。
    if not math.isfinite(thr):
        raise HTTPException(400, f"阈值 {raw!r} 不是有限数字。"
                                 "NaN / Infinity 会把报告写成浏览器解析不了的 JSON")
    if not 0.0 <= thr <= 1.0:
        raise HTTPException(400, f"阈值必须是 0~1 之间的数（收到 {thr:g}）。"
                                 "它是余弦相似度的判定线，只有这一段有意义")
    return thr


def _clean_num(raw, label: str, default, lo, hi, cast=int):
    """把界面传来的数字收成 [lo,hi] 之间的数，不合法一律 400。

    以前这几处是直接 int(...) / float(...)：扫描张数那个框手滑打了「200 张」，
    FastAPI 抛 ValueError 变成 500，用户对着「服务器内部错误」完全不知道该改哪个框，
    而服务日志里只有一行 int() 的 traceback。阈值那边早就有 _clean_threshold 报 400，
    其余几个能由用户在界面上填的数按同一口径收口。
    顺带把上限也改成报错而不是悄悄夹掉：填 5000 张的人是真想跑 5000 张，
    静默夹到 800 只会让他以为机器变慢了，不如直接说清楚。
    """
    if raw is None or raw == "":
        return default
    try:
        v = cast(raw)
    except (TypeError, ValueError, OverflowError):
        # OverflowError 不是可选项：数字框里填 Infinity 或 1e309，cast 是 int 时
        # int(float('inf')) 抛的是 OverflowError，不在原来那两个 except 里，
        # 于是「填错一个数」又变回一句看不懂的 500。
        raise HTTPException(400, f"{label}得是个数字（收到 {raw!r}）") from None
    if not math.isfinite(v):
        raise HTTPException(400, f"{label} {raw!r} 不是有限数字，"
                                 "NaN / Infinity 会把报告写成浏览器解析不了的 JSON")
    if not lo <= v <= hi:
        raise HTTPException(400, f"{label}必须在 {lo}~{hi} 之间（收到 {v:g}）")
    return v


# ---------------- 底库 ----------------
ENROLL_STREAM_IDLE_S = 180
ENROLL_MAX_TOTAL = 30000
STREAMS: dict[str, dict] = {}
# 注册任务的并行路数：默认 4 路上限 8。取 4 是因为本机实测吞吐到 4 路就基本饱和
# （240 张真图：1 路 33.9 张/秒 → 4 路 58.3 → 8 路 61.2），再加路数只是排队；
# 上限 8 是给 16 核服务器留的余量，再高就会开始抢走同机评测的核，两个数一起变难看。
# 三路结果逐条一致（ok=212 / bad=28 / 人数 212 / 模板 212），并行不改变判定。
ENROLL_WORKERS = 4
ENROLL_WORKERS_MAX = 8


def _enroll_opts(opts: dict) -> dict:
    """归一化注册参数。

    浏览器逐张上传、本机目录导入、multipart 补传三条路共用一套默认值，
    写在一处才不会以后改了一边忘了另一边（比如「只取主角」的缺省）。
    """
    return {
        "person_mode": str(opts.get("person_mode") or "name"),
        "delimiter": str(opts.get("delimiter") or "__"),
        "max_side": int(opts.get("max_side") or 1280),
        # 缺省取 1（只取主角）：大头照里经常拍到路人，
        # 全部取会把路人的脸存成这个人的模板，把底库污染掉。
        "max_faces": int(1 if opts.get("max_faces") is None else opts["max_faces"]),
        "max_templates": int(opts.get("max_templates") or 0),
        # 注册是「一批丢进去、只看进度条」的批处理，没人等单张图的延迟，
        # 所以默认按吞吐取，而不是跟评测参数（DEFAULTS["workers"]=1）保持一致：
        # 推理走 onnxruntime 的 C++ 调用不持 GIL，多路真能吃到核，
        # 本机实测 1 路 33.9 张/秒 → 4 路 58.3 张/秒。
        # 同时还在跑评测想省 CPU 的话，请求里显式传 workers=1 就退回原来的串行。
        "workers": max(1, min(int(opts.get("workers") or ENROLL_WORKERS), ENROLL_WORKERS_MAX)),
    }


def _new_job(total: int, kind: str = "enroll") -> tuple[str, dict]:
    job_id = uuid.uuid4().hex[:8]
    j = {"kind": kind, "total": total, "done": 0, "ok": 0, "bad": [],
         "bad_total": 0, "status": "running", "started": time.time()}
    # 插新任务和清理旧任务要在同一把锁里：浏览器一边连发几次注册/评测就会在多个
    # 请求线程里调到这里，而下面那句列表推导是裸遍历 JOBS —— 别的线程正好插入
    # 就是「dictionary changed size during iteration」，报出来的 500 和评测本身毫无关系。
    # 所有对 JOBS 的插入/删除/遍历都收在这把锁里，读取用 .get() 不用遍历，才留得住这个前提。
    with _JOB_LOCK:
        JOBS[job_id] = j
        # 一次会话能积累上千个任务对象，不清理的话内存和 /api/jobs 的序列化都白涨
        # [:-MAX_JOBS] 而不是 [50:]：JOBS 按插入顺序遍历，列表推导出来是最旧在前，
        # 写 [50:] 等于留下最旧 50 个、把刚跑完那批删掉 —— 脚本正要去轮询
        # /api/jobs/<刚提交的 id> 拿到 404，而界面上结果明明白白摆着。
        for stale in [k for k in JOBS
                      if k != job_id and JOBS[k].get("status") in ("done", "error")][:-MAX_JOBS]:
            JOBS.pop(stale, None)
            STREAMS.pop(stale, None)
    return job_id, j


def _probe_hold(run_id: str, out_dir: Path) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE_PROBE[run_id] = out_dir


def _probe_release(ctx: dict) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE_PROBE.pop(ctx.get("run_id"), None)


def _active_probe_ids() -> set[str]:
    """正在评测中的批次号快照。取完就放锁，不让调用方攥着锁做别的事。"""
    with _ACTIVE_LOCK:
        return set(_ACTIVE_PROBE)


def _guarded(fn, j: dict) -> None:
    """后台线程的兜底：没接住的异常会让任务永远停在 running。

    worker 里漏出来的一句会让线程直接死掉，status 还是 running，
    前端每 600ms 问一次问到天荒地老 —— 界面上是「转圈不停」而不是「失败了」。
    """
    try:
        fn()
    except Exception as e:
        log.exception("后台任务 %s 抛出未捕获异常", j.get("kind"))
        j.update(status="error", error=f"{type(e).__name__}: {str(e)[:300]}")


def _note_bad(j: dict, file: str, reason: str) -> None:
    """记一条「需复查」。计数和列表要在同一把锁里改，见 _JOB_LOCK 的说明。"""
    with _JOB_LOCK:
        j["bad_total"] += 1
        if len(j["bad"]) < 200:
            j["bad"].append({"file": file, "reason": reason})


async def _read_upload(request: Request, label: str, dir_hint: str) -> bytes:
    """读一张图的请求体，并把「浏览器中途把请求掐了」讲清楚。

    Safari 拿到读不动的本机原图时，请求头发出去了、正文一个字节都没来，
    然后自己把连接断了：starlette 在 request.body() 抛 ClientDisconnect，
    默认按 500 处理，界面上只剩一句「Load failed」加服务端一屏堆栈，
    完全看不出是浏览器读不到文件。这里接住它，和「0 字节」归成同一类
    能照着做的提示。
    """
    try:
        return await request.body()
    except ClientDisconnect as e:
        # 请求都断了，这句 400 多半送不回浏览器（页面那边只能收到 Load failed），
        # 所以同时记进日志：哪张图被浏览器掐掉、该改走哪条路，一眼能看到。
        log.warning("浏览器掐断上传：%s（正文一个字节都没到）→ 建议改用「%s」", label, dir_hint)
        raise HTTPException(
            400, f"「{label}」这张浏览器发到一半把请求断了（正文一个字节都没到，"
                 f"是浏览器读不到这个文件）。照片就在这台机器上，改用「{dir_hint}」"
                 "填目录完全不走浏览器上传；要继续拖文件的话请换 Chrome / Edge "
                 "直接打开 http://127.0.0.1:8617") from e


def _enroll_one(bk, gallery, o: dict, rel: str, name: str, read_raw,
                j: dict, seen: set) -> None:
    """注册一张图：取字节 → 解码 → 认人 → 提特征 → 入库，结果记进任务。

    单张图的任何失败只记一条「需复查」，绝不抛出：几千张底库里混一两张
    坏图（0 字节、非图片、iCloud 没下下来）是常态，为一整批重传不划算。
    """
    try:
        raw = read_raw()
        if not raw:
            raise ValueError("图片内容是空的（0 字节）")
        img = limit_side(decode(raw, max_side=o["max_side"]), max_side=o["max_side"])
        person = infer_person(rel, name, o["person_mode"], o["delimiter"])
        if not person:
            _note_bad(j, rel or name, "无法判定人员ID")
            return
        with _JOB_LOCK:
            seen.add(person)
        recs = bench.enroll_image(
            bk, gallery, img, person, rel or name,
            {"max_faces": o["max_faces"], "max_templates": o["max_templates"]},
        )
        n_ok = len([r for r in recs if "tid" in r])
        if n_ok:
            with _JOB_LOCK:
                j["ok"] += n_ok
        for r in recs:
            if "skipped" in r:
                _note_bad(j, rel or name, "、".join(r["skipped"]))
    except Exception as e:
        _note_bad(j, rel or name, f"{type(e).__name__}: {e}")


def _finish_enroll(j: dict, gallery, o: dict, n: int, seen: set) -> None:
    """收尾：存盘 + 出统计，顺带抓「人数塌成 1」这个最常见的配错。

    按文件夹认人时最容易踩的坑：底库其实是「一张图一个人」平铺的，
    结果整批全被归到根目录那一个人名下，人数从几千塌成 1，还不报错。
    """
    if o["person_mode"] == "path" and n >= 5 and len(seen) <= 1:
        who = seen.pop() if seen else "无"
        _note_bad(
            j, f"本批 {n} 张",
            f"整批只认出 1 个人员 ID「{who}」，疑似把底库根目录当成了人名。"
            "若底库是每张图一个人，请把「人员 ID 来源」改成「文件名即人名」再注册",
        )
    try:
        gallery.save()
    except Exception as e:      # 落盘失败必须显式报错，否则前端以为注册成功了
        log.exception("底库落盘失败")
        j.update(status="error", error=f"底库保存失败 {type(e).__name__}: {e}")
        return
    j.update(status="done", finished=time.time(), gallery=gallery.stats())


def _start_enroll(batch: list[tuple[str, str, object]], opts: dict) -> str:
    """起一个后台注册任务，返回 job_id。

    batch 每项是 (相对路径, 文件名, 取字节的函数)。字节写成函数而不是直接塞
    bytes：从本机目录导入几千张原图时按需读盘，不会一次占住几 GB 内存
    （RK3588 上只有几 GB）；浏览器上传的字节本来已经在内存里，直接复用。

    并行度来自 workers。原来是老老实实一张接一张跑，导入 3000 张底库要一分多钟，
    而这段时间 CPU 有大半是闲着的（本机实测并行能换到 1.7 倍吞吐，见 ENROLL_WORKERS）。
    每张图之间的耦合只有三处，都在 _JOB_LOCK 里：任务计数、seen 集合、
    以及每人模板上限（已挪到 Gallery.add 的锁内判定）。
    """
    o = _enroll_opts(opts)
    job_id, j = _new_job(len(batch))

    def worker():
        try:
            bk = get_backend()
        except BackendError as e:
            j.update(status="error", error=str(e))
            return
        gallery = get_gallery()
        seen: set[str] = set()

        def one(item) -> None:
            rel, name, read_raw = item
            _enroll_one(bk, gallery, o, rel, name, read_raw, j, seen)
            with _JOB_LOCK:
                j["done"] += 1        # 并行完成没有先后，只能累加，不能再写 i+1

        n = max(1, min(o["workers"], len(batch)))
        if n == 1:
            for item in batch:
                one(item)
        else:
            with ThreadPoolExecutor(max_workers=n, thread_name_prefix="enroll") as pool:
                list(pool.map(one, batch))     # _enroll_one 不抛异常，map 不会中途散场
        _finish_enroll(j, gallery, o, len(batch), seen)

    threading.Thread(target=worker, daemon=True).start()
    return job_id


def _stream_worker(job_id: str, q, o: dict) -> None:
    """逐张注册的服务端侧：从队列取浏览器推来的图片，结束后统一落盘。

    浏览器一张一个请求、请求体就是图片原始字节（而不是几十张拼一个
    multipart 包）是有意为之：multipart 的 body 一旦被发出去过就不可重放，
    重试时经常发出 0 字节的空包，服务端只能回 422「没收到 files」，
    几千张底库换来满屏同一个错。原始字节没有表单可解析，
    重试永远发得出去，查询参数放不下才需要的那点元数据走 URL。
    """
    j = JOBS.get(job_id)
    if not j:
        STREAMS.pop(job_id, None)
        return
    try:
        bk = get_backend()
    except BackendError as e:
        j.update(status="error", error=str(e))
        STREAMS.pop(job_id, None)
        return
    gallery = get_gallery()
    seen: set[str] = set()
    n = j["done"]
    while True:
        try:
            item = q.get(timeout=ENROLL_STREAM_IDLE_S)
        except queue.Empty:
            # 页面刷新、断网、浏览器被关：已经收到的先存好，
            # 别把线程和几百 MB 字节一直挂在半路上。
            _note_bad(j, "上传中断",
                      f"{ENROLL_STREAM_IDLE_S} 秒没有新图片进来，先把已收到的存好了；"
                      "剩下的重新点「开始注册」补传")
            break
        if item is None:
            break
        rel, name, raw = item
        _enroll_one(bk, gallery, o, rel, name, (lambda r=raw: r), j, seen)
        n += 1
        with _JOB_LOCK:
            j["done"] = n
    _finish_enroll(j, gallery, o, n, seen)
    STREAMS.pop(job_id, None)


@app.post("/api/gallery/enroll")
def api_gallery_enroll(
    files: list[UploadFile] = File(...),
    relpaths: str = Form(""),
    person_mode: str = Form("name"),
    delimiter: str = Form("__"),
    max_templates: int = Form(0),
    max_faces: int = Form(1),
    max_side: int = Form(1280),
):
    """/api/gallery/enroll 表单：relpaths 用 "||" 连接，与 files 同序。

    底库默认 person_mode=name：每张图一个人，文件名（去扩展名）就是人员 ID。
    每人一个文件夹的旧结构传 person_mode=path。
    """
    if not files:
        raise HTTPException(400, "请求里一张图片都没有，请重新选择图片")
    rels = relpaths.split("||") if relpaths else []
    batch: list[tuple[str, str, object]] = []
    bad: list[tuple[str, str]] = []
    # UploadFile 的临时文件在请求一结束就被关闭，字节必须在这里全部读走再交给后台
    # 线程；否则从第二张开始就是 "I/O operation on closed file"，几千张底库只成一张。
    for i, up in enumerate(files):
        rel = rels[i] if i < len(rels) else ""
        name = up.filename or f"item{i}"
        try:
            raw = up.file.read()
            if not raw:
                raise ValueError("请求体里这张图是空的（浏览器没读到文件内容）")
            batch.append((rel, name, lambda raw=raw: raw))
        except Exception as e:
            bad.append((rel or name, f"读取失败 {type(e).__name__}: {e}"))
    job_id = _start_enroll(batch, {"person_mode": person_mode, "delimiter": delimiter,
                                  "max_side": max_side, "max_faces": max_faces,
                                  "max_templates": max_templates})
    for nm, why in bad:
        # 走 _note_bad 而不是自己 += ：这一句和后台 worker 里的失败计数改的是同一个字段，
        # int 的 += 是「读-改-写」三步，绕开 _JOB_LOCK 就会少计，
        # 少掉的正好落在「到底有几张需要复查」这种要给人看的数字上。
        if (jj := JOBS.get(job_id)) is not None:
            _note_bad(jj, nm, why)
    return {"job_id": job_id}


IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_DIR_FILES = 30000


@app.post("/api/gallery/enroll_dir")
def api_gallery_enroll_dir(payload: dict = Body(...)):
    """直接注册服务端本机某个目录里的底库，不经过浏览器上传。

    几千张图走浏览器上传要把字节在内存里过两遍，还容易被「当前窗口读不到磁盘」
    这类环境问题整批打回 422。底库和评测机常常就在同一台机器上（部署到 RK3588
    时更是如此），让服务端自己读目录最快也最稳。relpath 用相对该目录的路径，
    「文件名即人名」和「文件夹名即人名」两种规则都和上传时一致。
    """
    raw_path = str(payload.get("path") or "").strip().rstrip(chr(92) + "/")
    if not raw_path:
        raise HTTPException(400, "请先填底库目录的完整路径")
    root = Path(raw_path).expanduser()
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    if not root.exists():
        raise HTTPException(400, f"路径不存在：{root}")
    if not root.is_dir():
        raise HTTPException(400, f"这是个文件不是文件夹：{root}")
    found = sorted(p for p in root.rglob("*")
                   if p.suffix.lower() in IMG_EXT and p.is_file())
    if not found:
        raise HTTPException(400, f"{root} 里没找到图片（支持 jpg / png / webp / bmp，含子目录）")
    if len(found) > MAX_DIR_FILES:
        raise HTTPException(400, f"这个目录有 {len(found)} 张图片，超过单次 {MAX_DIR_FILES} 张上限，"
                                "请把底库分成两三个目录分批导入")
    batch = [(str(p.relative_to(root)), p.name, (lambda p=p: p.read_bytes())) for p in found]
    job_id = _start_enroll(batch, payload)
    return {"job_id": job_id, "found": len(batch), "path": str(root)}


@app.post("/api/gallery/open")
def api_gallery_open(payload: dict = Body(...)):
    """开一个逐张注册会话，返回的 job_id 也是轮询进度用的任务号。

    底库注册拆成 open → push（每张一次）→ close。任何一张失败只影响那一张，
    「传到一半断掉」不再让整批重来，也不会再出现「成功 0 个模板」。
    """
    total = int(payload.get("total") or 0)
    if total <= 0:
        raise HTTPException(400, "这次一张图片都没选中，请重新拖入底库图片")
    if total > ENROLL_MAX_TOTAL:
        raise HTTPException(400, f"一次最多 {ENROLL_MAX_TOTAL} 张，当前 {total} 张，"
                                "请分两次拖进来")
    o = _enroll_opts(payload)
    job_id, j = _new_job(total)
    j["status"] = "uploading"
    # 有界队列 = 天然的节流阀：推理比上传慢时让浏览器等着，
    # 免得几千张原图先在服务端内存里堆成几百 MB（板子上只有几 GB）。
    q: queue.Queue = queue.Queue(maxsize=48)
    STREAMS[job_id] = {"q": q, "opts": o, "n": 0, "last": time.time()}
    threading.Thread(target=_stream_worker, args=(job_id, q, o), daemon=True).start()
    return {"job_id": job_id, "total": total}


@app.post("/api/gallery/push")
async def api_gallery_push(request: Request, job: str = "", rel: str = "", name: str = ""):
    """收一张图片的原始字节，交给注册任务排队处理。

    图片字节整个占满请求体，所以文件名/相对路径走查询参数。
    字节为空当场报错：「浏览器根本没读到文件」这类环境问题在第一张就暴露，
    而不是几千张之后换来几十屏同一个 422。
    """
    st = STREAMS.get(job)
    if not st:
        raise HTTPException(404, f"注册会话 {job or '(空)'} 已经结束了"
                                 "（页面刷新过或太久没有新图片）。请重新点「开始注册」")
    j = JOBS.get(job) or {}
    if j.get("status") == "error":
        raise HTTPException(400, j.get("error") or "注册任务已出错")
    raw = await _read_upload(request, name or (rel.rsplit("/", 1)[-1] if rel else "图片"),
                             "导入本机目录")
    if not raw:
        nm = name or (rel.rsplit("/", 1)[-1] if rel else "图片")
        raise HTTPException(400, f"「{nm}」这张浏览器没发出内容（0 字节）。"
                                 "请重新选择图片；如果是在应用内置的预览窗口里上传，"
                                 "改用 Safari 直接打开 http://127.0.0.1:8617")
    nm = name or (rel.rsplit("/", 1)[-1] if rel else "")
    if not nm:
        raise HTTPException(400, f"「{rel or '图片'}」没带文件名，无法判定人员ID")
    st["last"] = time.time()
    # 上面读请求体是 await，中途收尾线程可能已经下班并把会话从 STREAMS 摘掉了
    # （空闲超时、或者别的窗口点了「结束上传」）。这时候再计数就会报出
    # 「recved 300/300」而实际只注册了两百多张，字节还塞进一个没人读的队列里。
    with _JOB_LOCK:
        if STREAMS.get(job) is not st:
            raise HTTPException(404, f"注册会话 {job} 的收尾已经开始了，这张没排进去。"
                                     "看一下进度里的「需复查」，剩下的重新点「开始注册」补传")
        st["n"] += 1
    try:
        await run_in_threadpool(st["q"].put, (rel, nm, raw), True, 120)
    except queue.Full:
        raise HTTPException(503, "服务端处理不过来（排队已满），请稍后重试这一张")
    return {"recved": st["n"], "total": j.get("total", st["n"]),
            "done": j.get("done", 0), "ok": j.get("ok", 0)}


@app.post("/api/gallery/close")
def api_gallery_close(job: str = ""):
    """告诉服务端「这次没有更多图片了」，让它落盘并汇总统计。"""
    st = STREAMS.get(job)
    if not st:
        raise HTTPException(404, f"注册会话 {job or '(空)'} 已结束，进度和结果按任务号查")
    try:
        # 排队满时最多等 20 秒：收尾线程要是已经自己下班了（空闲超时），
        # 这里无限等下去会让「结束上传」这个请求一直挂着，前端只显示「评测失败」。
        st["q"].put(None, True, 20)
    except queue.Full:
        log.warning("注册会话 %s 收尾信号排队满，交给空闲超时兜底", job)
    j = JOBS.get(job) or {}
    return {"job_id": job, "recved": st["n"], "status": j.get("status", "uploading")}



@app.get("/api/gallery/persons")
def api_gallery_persons(names: int = 0):
    """底库人员。names=1 只回人名清单（探针真值下拉要用，几千人不带质量/缩略图，几十 KB）。"""
    g = get_gallery()
    if names:
        return {"stats": g.stats(), "names": g.names()}
    return {"stats": g.stats(), "persons": g.person_list()}


@app.get("/api/gallery/list")
def api_gallery_list(page: int = 1, size: int = 80, q: str = ""):
    """底库分页：取第 page 页（从 1 开始）的人员，配合搜索框按人名过滤。

    几千张底库以前是一次把所有人塞进一个页面渲染，浏览器要卡好几秒；
    分页后每屏只画 size 个格子，翻页时服务端只做一次 SQL LIMIT。
    """
    g = get_gallery()
    size = max(1, min(int(size or 80), 500))
    page = max(1, int(page or 1))
    out = g.page(q=q, offset=(page - 1) * size, size=size)
    total = out["total"]
    return {
        "page": page, "size": size, "q": out["q"],
        "total": total, "total_pages": (total + size - 1) // size if total else 0,
        "items": out["items"], "stats": g.stats(),
    }


@app.delete("/api/gallery")
def api_gallery_clear():
    """清空底库：内存模板、数据库（gallery.db 含 -wal/-shm）、缩略图一起删。

    不留下一个空数据库——磁盘上留个空库没意义，而 thumbs/ 里那几千张 JPG
    才是真占地方的。删干净后重启才是真·空库，不会读回一个半套状态。
    """
    g = get_gallery()
    purged = g.clear()              # 必须先清再取 stats，反过来会把清空前的人数报出去
    return {**g.stats(), **purged}


@app.delete("/api/gallery/person/{person}")
def api_gallery_del(person: str):
    """删掉一个人（模板 + 数据库行 + 他的缩略图），只动他一个。

    以前只有整库清空：底库里混进一个错标的人（把路人注册成了张三），或者测试时
    注册了几个假名字，只能把几千张全删了重注册。现在按人删，
    和界面上「删除本批」是同一个思路 —— 坏的那一条自己走，别连累好的。
    """
    g = get_gallery()
    if not person:
        raise HTTPException(400, "人名不能是空的")
    if not g.has_person(person):
        # 不能省这一步：remove_person 对不存在的人返回 persons_removed=0 而不报错，
        # 照这返回 200 的话界面弹「已删除」，可那个人其实还在库里。
        raise HTTPException(404, f"底库里没有「{person}」，可能已经被删过了")
    try:
        # 连同这个人的缩略图一起删，别留孤儿。删不动数据库现在会抛出来：
        # 静默记一条 integrity 的话，界面上人没了、重启他又带着一堆 404 头像回来
        removed = g.remove_person(person)
        g.save()
    except (RuntimeError, OSError) as e:
        raise HTTPException(500, f"删除「{person}」没成功：{e}") from e
    return {**g.stats(), **removed}


def _inside(root: Path, p: Path) -> bool:
    """p 是不是真的落在 root 目录**里面**，而不是名字以 root 打头的隔壁目录。

    只比字符串前缀的话，thumbs/../thumbs_old/a.jpg 解析成隔壁的 thumbs_old/
    也算「在 thumbs 里」，底库缩略图接口就变成能读工程内任意 jpg 的小口子。
    gallery._drop_thumb 早就是带分隔符的写法，三处取图的地方统一到这里。
    """
    return str(p).startswith(str(root.resolve()) + os.sep)


@app.get("/api/thumb/{name}")
def api_thumb(name: str):
    g = get_gallery()
    p = (g.thumbs_dir / name).resolve()
    if not _inside(g.thumbs_dir, p) or not p.exists():
        raise HTTPException(404, "缩略图不存在")
    return FileResponse(p, media_type="image/jpeg")


@app.get("/api/person/thumb")
def api_person_thumb(person: str = Query(...)):
    """按人名取一张底库留档照。

    标注时只看摄像头截图猜不出是谁，把系统认为的那个人在库里的原照并排摆出来，
    人是是错一眼就能定，一键标注才敢真点。
    """
    g = get_gallery()
    hit = None
    try:
        hit = next((x for x in g.page(q=person, size=300)["items"]
                    if x["person"] == person), None)
    except Exception:
        hit = None
    fname = (hit or {}).get("thumb") or ""
    p = (g.thumbs_dir / fname).resolve()
    if not fname or not _inside(g.thumbs_dir, p) or not p.exists():
        raise HTTPException(404, "这个人没有留档缩略图")
    return FileResponse(p, media_type="image/jpeg")


# ---------------- 探针评测 ----------------

# 相机自动编号不是人名：IMG_0001 / DSC_0123 / 1792052059096 / 去序号后剩下的 IMG
_AUTO_WORDS = {"img", "dsc", "dscn", "pxl", "vid", "pic", "photo", "cam", "camera",
               "screenshot", "mmexport", "微信图片", "截图"}
_AUTO_NAME = re.compile(
    r"^(?:(?:img|dsc|dscn|pxl|vid|pic|photo|cam|camera|screenshot|mmexport|微信图片|截图)"
    r"[_\-\s\.]*)?\d{3,}$", re.I)
# 纯数字加分隔符也是机器编号：摄像头按时间戳命名（081013_203_7、20261107_080530）
# 或按毫秒时间戳命名（1792052059096）。这类名字命中不了底库时不能当成「非底库人员」，
# 否则整批抓拍会被自动写成假真值，误识率直接算出 30% 这种没法看的数。
_AUTO_STAMP = re.compile(r"^\d[\d_\-\s.:]{2,}$")


def _auto_named(cand: str) -> bool:
    """看着像相机自己编的号，而不是人起的名字。"""
    c = str(cand or "").strip()
    return bool(c) and (c.lower() in _AUTO_WORDS or _AUTO_NAME.match(c) is not None
                        or _AUTO_STAMP.match(c) is not None)


def _probe_truth(rel: str, name: str, mode: str, delim: str,
                 persons: set[str]) -> tuple[str | None, str | None]:
    """推断探针图真值，返回 (真值, 原始名)。

    探针默认也是「每张图一个人」，所以文件名排第一位：先拿文件名（连去掉尾部序号的
    形式一起试）去底库比对，命中不了才退回父目录名，谁真正命中底库就用谁——父目录
    经常是 `cam1`、`20261106` 这种批次目录，命中不了人就不能拿它当人名。
    谁都没命中时，父目录名照旧标成「非底库人员」并把原名留给界面复核（`路人甲/`
    这种批次就是专门测误识的样本）；只有文件名可用时，`IMG_0001`、`1792052059096`、
    `081013_203_7` 这类相机自动编号不当人名也不当「库外人员」，真值留空交人工标注，
    这类相机自动编号不当人名，真值留空交人工标注，免得一次选错规则就把整批算成
    库外样本。一个候选都推不出来同样留空，绝不造假真值。
    """
    if mode == "none":
        return None, None
    folder = infer_person(rel, name, "path", delim)
    if mode == "name":
        # 每张图一个人（默认）：文件名优先，顺带试去掉尾部序号的形式
        cands = [Path(name).stem.strip(), infer_person(rel, name, "id_num", delim), folder]
    elif mode == "path":
        # 摄像头批次常是 cam1/、20261106/ 这种目录，底库却是「文件名即人名」平铺的；
        # 只认文件夹会让整批探针都标成「无真值」，逼人手点几千次。
        # 文件名只有在正好命中底库时才会被采用，所以这一步不会造假真值。
        # 顺带也试「去掉尾部序号」的形式：cam1/王五_0007.jpg 这种命名，
        # 底库里的人叫王五，只试完整文件名 王五_0007 就命中不了，白丢一条真值。
        cands = [folder, infer_person(rel, name, "name", delim),
                 infer_person(rel, name, "id_num", delim)]
    else:
        by_name = infer_person(rel, name, mode, delim)
        cands = [folder, by_name] if folder and by_name and folder != by_name else [folder or by_name]
    cands = [c for c in dict.fromkeys(cands) if c]
    for c in cands:
        if c in persons:
            return c, None
    for c in cands:
        if c == folder or not _auto_named(c):
            return UNKNOWN, c
    return None, None


RUNS_KEEP = 3          # 内存里最多留几个「整份带明细」的批次，万张一批的记录很占内存
# error 单独立一类：坏图（解码失败 / 推理报错）和「检不到脸」的原因完全不同，
# 前者要去查文件是否损坏，后者才是检测器的问题。混在 noface 里会让人白改检测阈值。
_FACETS = ("hit", "miss", "all", "bad", "risk", "unlabeled", "multi", "noface", "error", "ok")


def _trim_runs(keep: str | None = None) -> None:
    """只把最近几个批次的明细整份留在内存，其余收成「只剩汇总」的壳。

    一万张探针的明细有几十 MB，全压在内存里评两三次就该 OOM 了；壳里保留
    summary/params，历史批次下拉照样列得出来，点开来再按需从磁盘读回。
    """
    # 整个函数持锁：_probe_finish 在 worker 线程里往 RUNS 塞新批次，
    # 这里裸遍历就是「dictionary changed size during iteration」，
    # 报错点随机落在评测收尾或历史列表上，看着像批次自己坏了。
    with LOCK:
        full = [(k, v) for k, v in RUNS.items() if not v.get("_stub")]
        if len(full) <= RUNS_KEEP:
            return
        full.sort(key=lambda kv: kv[1].get("created_at") or 0, reverse=True)
        # 取一份待压壳名单时要和 _mark_dirty 互斥：改完真值刚标脏、这边正好把它压成壳，
        # 那一版判定就永久丢了（壳里没有 records，落盘时又会跳过 _stub）。
        # 锁序统一为 LOCK -> _DIRTY_LOCK，_flush_run 也是这个顺序，不会互相咬住。
        with _DIRTY_LOCK:
            for rid, run in full[RUNS_KEEP:]:
                # keep = 调用方正要返回给用户的那一批。不排掉它的话，
                # 「点开一个旧批次」会当场把刚读回来的明细又压回空壳再交出去，
                # 界面永远显示 total: 0，刷新重试都一样。
                if rid == keep or rid in DIRTY:
                    continue
                if (REPORT_DIR / f"probe_{rid}.json").is_file():
                    RUNS[rid] = {**{k: v for k, v in run.items() if k != "records"},
                                 "records": [], "_stub": True}


def _ensure_run(run_id: str) -> dict:
    """取批次：内存里没有（或只是壳）就从 reports/ 读回来。"""
    run = RUNS.get(run_id)
    if run and not run.get("_stub"):
        return run
    f = REPORT_DIR / f"probe_{run_id}.json"
    if not f.is_file():
        raise HTTPException(404, "评测批次不存在")
    try:
        run = json.loads(f.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(500, f"批次报告读不出来：{type(e).__name__} {str(e)[:160]}")
    if not run.get("records"):
        raise HTTPException(404, "评测批次里没有明细记录")
    # 读回来的明细要持锁塞回表里：api_runs 正拿着 LOCK 遍历 RUNS 列历史批次，
    # 这里裸插入就会让那边冒出「dictionary changed size during iteration」，
    # 用户看到的是「点开一个旧批次，历史下拉反而 500 了」。
    with LOCK:
        RUNS[run_id] = run
    # keep=run_id：本批次是 created_at 最旧的那个，不点名保护就会被当场压回壳再返回
    _trim_runs(keep=run_id)
    # 交自己手里那份，别再回表里查一次：keep=run_id 挡住的是「本批次这次不压壳」，
    # 但并发的那次 _trim_runs 可能已经先一步把它换成了 _stub（records 是空的），
    # 那时 RUNS.get(run_id) or run 取出的是那个空壳而不是刚读回来的 run ——
    # 用户点开一个几千张的旧批次，屏幕上写着「本批共 0 张」。
    return run


def _is_hit(f: dict) -> bool:
    """这张脸算「比对成功」：分数过了阈值、并且给出了底库里的人名。"""
    return bool(f.get("accepted") and f.get("top1_person"))


def _facet_hit(r: dict, only: str) -> bool:
    """明细筛选。合影/检不出单独归类，「判错」只算有真值却没认对的。"""
    nface = int(r.get("faces_found") or 0)
    truth = r.get("truth")
    if only in ("", "all"):
        return True
    faces = r.get("faces") or []
    if only == "hit":
        return any(_is_hit(f) for f in faces)
    if only == "miss":
        return not any(_is_hit(f) for f in faces)
    if only == "noface":
        # 坏图也是一张脸都没有，但它有自己的一类，别重复计进「检不到脸」
        return (nface == 0 or bool(r.get("fallback"))) and r.get("verdict") != "error"
    if only == "error":
        return r.get("verdict") == "error"
    if only == "multi":
        return nface > 1
    if only == "unlabeled":
        return nface > 0 and not truth
    if only == "ok":
        return bool(r.get("rank1_correct"))
    if only == "bad":
        return bool(truth) and truth != UNKNOWN and not r.get("rank1_correct")
    if only == "risk":
        # 没真值却被放行：底库里可能压根没这个人，也可能是认错，需要人工看一眼
        return nface > 0 and not truth and bool(r.get("accepted"))
    return True


def _facets(recs: list[dict]) -> dict:
    return {k: sum(1 for r in recs if _facet_hit(r, k)) for k in _FACETS}


def _clean_probe_params(extra: dict) -> dict:
    """挡掉评测参数里的非法值，尤其是 NaN / Infinity。

    这条路径比 /api/rethreshold 更容易中招：Python 的 json 默认把裸 NaN 当合法字面量
    解析，所以 params='{"threshold": NaN}' 会一路漂进报告、再漂进落盘文件，
    最后表现为「这个批次的报告浏览器打不开」。范围之外不会算错得离谱，
    但会把一次评测变成几分钟白跑（det_size=100000）或者一个样本都不剩（topk=0）。
    """
    limits = {          # 键 -> (下界, 上界)；闭区间，None 表示该侧不限
        "det_size": (64, 2048), "det_thresh": (0.0, 1.0), "det_nms": (0.0, 1.0),
        "threshold": (0.0, 1.0), "topk": (1, 50), "max_side": (128, 8192),
        "max_faces": (0, 64), "max_templates": (0, 100000),
        "min_face": (0, 4096), "workers": (1, 16),
        "enroll_fallback_whole": (0, 1), "all_faces": (0, 1),
    }
    out = dict(extra)
    for k, (lo, hi) in limits.items():
        if k not in out or out[k] is None or out[k] == "":
            continue
        try:
            v = float(out[k])
        except (TypeError, ValueError):
            raise HTTPException(400, f"参数 {k} 得是数字（收到 {out[k]!r}）") from None
        # 同上：「不是数字」「超出可用区间」是两回事，分开报才知道该改哪个
        if not math.isfinite(v):
            raise HTTPException(400, f"参数 {k}={out[k]!r} 不是有限数字。"
                                     "NaN / Infinity 会把批次报告写成浏览器解析不了的 JSON")
        if v < lo or v > hi:
            raise HTTPException(400, f"参数 {k}={out[k]!r} 不在 {lo}~{hi} 范围内")
        cur = DEFAULTS.get(k)
        out[k] = int(v) if isinstance(cur, int) and not isinstance(cur, bool) else v
    if "probe_mode" in out and out["probe_mode"] not in ("auto", "det", "raw"):
        raise HTTPException(400, f'probe_mode 只能是 auto / det / raw（收到 {out["probe_mode"]!r}）')
    return out


def _params_of(extra) -> dict:
    """把请求里的 params（JSON 字符串或字典）解析并校验成参数字典。

    单独抽出来是为了能在**开跑之前**就调用它：读目录评测原先要先把整个目录
    扫一遍（probes/2 有 16454 张，要几秒），才发现 {"det_size": NaN} 少个引号，
    用户等完这几秒只收到一个失败的任务。现在 open / run_dir 先调一次，
    拼错立刻 400；_probe_setup 里再调一次是幂等的，兜住绕过入口的调用。
    """
    if isinstance(extra, str):
        try:
            extra = json.loads(extra or "{}")
        except ValueError as e:
            raise HTTPException(400, f"params 不是合法的 JSON：{e}") from e
    if extra is None:
        extra = {}
    if not isinstance(extra, dict):
        raise HTTPException(400, 'params 得是对象（字典），例如 {"det_size": 320}')
    return _clean_probe_params(extra)


def _probe_setup(opts: dict) -> dict:
    """备好一次评测要用的东西：底库、推理后端、参数、截图目录。

    目录评测要分块喂（一万张一次读进内存是几个 GB），所以把「准备」和
    「跑一块」拆开；整批 multipart 上传、逐张上传、读目录三条路都收在这里，
    指标口径不会走偏。params 允许是字典或 JSON 字符串。
    """
    gallery = get_gallery()
    if gallery.matrix.size == 0:
        raise HTTPException(400, "底库为空，请先注册底库")
    try:
        bk = get_backend()
    except BackendError as e:
        raise HTTPException(400, str(e))
    p = {**DEFAULTS, **_params_of(opts.get("params"))}
    run_id = uuid.uuid4().hex[:8]
    out_dir = PROBE_DIR / run_id
    # bare = 只要数字、不要复核图（扫描表用）。一套扫描表九格，每格都存一遍
    # 整图缩略和对齐小脸就是几千次 JPEG 编码，纯属给扫描本身加戏。
    bare = bool(opts.get("bare"))
    if not bare:
        out_dir.mkdir(parents=True, exist_ok=True)
        _probe_hold(run_id, out_dir)
    # 并发路数兜个底：前端传 0 / 负数 / 1e9 都会让线程池要么退化成串行、要么瞬间起上千线程
    p["workers"] = max(1, min(int(p.get("workers") or 1), 16))
    # set(gallery.persons) 是裸遍历：后台正在注册时 add() 会改这个字典，
    # 撞上就是「dictionary changed size during iteration」，几千张白跑
    with gallery.lock:
        persons = set(gallery.persons)
    return {"gallery": gallery, "bk": bk, "p": p, "run_id": run_id, "out_dir": out_dir,
            "bare": bare,
            "person_mode": str(opts.get("person_mode") or "name"),
            "delimiter": str(opts.get("delimiter") or "__"),
            "max_side": int(opts.get("max_side") or 1280),
            "persons": persons, "raw_of": {}, "wall_s": 0.0}


def _probe_prepare(ctx: dict, batch, idx_base: int, progress=None) -> list:
    """解码 + 定真值 + 存整图缩略。batch 每项是 (相对路径, 文件名, 原始字节)。

    缩略图名用全局序号（idx_base 起），分块之间不会撞名；meta.jsonl 每块
    只开一次文件，一万张就是一万行而不是两万次打开。
    """
    items: list = []
    meta: list[str] = []
    for k, (rel, name, raw) in enumerate(batch):
        i = idx_base + k
        if progress:
            progress(i)
        nm = rel or name
        try:
            img = decode(raw, max_side=ctx["max_side"])
        except Exception:
            img = None
        if img is None:
            # 解码失败：原来这里塞一张 112×112 纯黑小图继续往下跑，等于骗下游
            # 「这是一张正常照片，只是没检出脸」。后果有三个，全都落在选型要看的眼前：
            #   1) 白白多出一次检测推理，还挤进 detect_rate 的分母（坏图被冤枉成漏检）；
            #   2) bench._probe_one 里专为 img is None 写的 error 分支在生产路径永远走不到，
            #      n_error 恒等于 0，坏图全被记成 no_face / fallback，让人去改检测阈值；
            #   3) 真值没算，坏图一律掉进「非底库」那一类，被放行时还倒过来污染 FAR。
            # 真值照样按文件名算（坏图也可能是「张三__0001.jpg」），判定就如实是 error。
            truth, truth_raw = _probe_truth(rel, name, ctx["person_mode"],
                                            ctx["delimiter"], ctx["persons"])
            if truth_raw:
                ctx["raw_of"][nm] = truth_raw
            # 没有像素可以缩略：meta 里 thumb 留空，界面对这张只出判定不出复核图，
            # 而不是去加载一张根本不存在的 jpg 变成裂图图标。
            meta.append(json.dumps({"i": i, "name": nm, "thumb": ""}, ensure_ascii=False))
            items.append((nm, None, truth))
            continue
        thumb = f"{i}.jpg"
        if not ctx.get("bare"):
            # 一张写不进去不能让整批评测散场：一万六千张里只要有一张撞上磁盘满、
            # 或者目录被人中途清掉，异常会一路冒出 _probe_chunk，前面几千张的判定全丢。
            # 丢一张复核图就如实留空，判定和汇总照样是完整的。
            try:
                (ctx["out_dir"] / thumb).write_bytes(encode_jpg(limit_side(img, 320), 80))
            except Exception:
                log.warning("整图缩略图 %s 写盘失败，这张只出判定不出复核图", thumb, exc_info=True)
                thumb = ""
        truth, truth_raw = _probe_truth(rel, name, ctx["person_mode"],
                                        ctx["delimiter"], ctx["persons"])
        if truth_raw:
            ctx["raw_of"][nm] = truth_raw
        items.append((nm, img, truth))
        # 交给 json.dumps 转义，别手写 f-string：以前只剥双引号，文件名里带反斜杠
        # （从 Windows 拷过来的照片常有）就写出 "name": "a\\新建"，
        # 这一行 json.loads 直接 Invalid \\escape，清单从这条往后全废。
        meta.append(json.dumps({"i": i, "name": nm, "thumb": thumb}, ensure_ascii=False))
    if meta and not ctx.get("bare"):
        with (ctx["out_dir"] / "meta.jsonl").open("a", encoding="utf-8") as fh:
            fh.write("".join(m + "\n" for m in meta))
    return items


def _probe_chunk(ctx: dict, batch, idx_base: int, progress=None) -> list:
    """跑一小块探针，返回记录列表。"""
    items = _probe_prepare(ctx, batch, idx_base, progress)
    idx_of = {nm: idx_base + k for k, (nm, _, _) in enumerate(items)}

    def crop_saver(name: str, fidx: int, crop) -> str:
        """存对齐后的 112×112 人脸，命名 <图序号>-f<脸序号>.jpg，供逐脸复核。"""
        fn = f"{idx_of.get(name, idx_base)}-f{fidx}.jpg"
        try:
            (ctx["out_dir"] / fn).write_bytes(encode_jpg(crop, 85))
        except Exception:
            return ""
        return fn

    res = bench.run_probe(ctx["bk"], ctx["gallery"], items, ctx["p"],
                          crop_saver=None if ctx.get("bare") else crop_saver)
    # 分块之间把推理墙钟累加起来。一万张分 20 块只记最后一块的话，
    # 并发吞吐会被虚报 20 倍，而选型要看的恰恰就是这个数
    ctx["wall_s"] = ctx.get("wall_s", 0.0) + float(res.get("wall_s") or 0.0)
    recs = res["records"]
    for r in recs:
        if ctx["raw_of"].get(r["name"]):
            r["truth_raw"] = ctx["raw_of"][r["name"]]
    return recs


def _discard_probe_dir(ctx: dict) -> None:
    """评测中途失败时，把这一批已经写出去的截图目录一并清掉。

    `_probe_setup` 一进来就 mkdir，之后每解码一张就写一张整图缩略；一万六千张的
    目录跑到第 9000 张推理炸了，run_id 从来没进 RUNS，历史批次下拉里既看不见也
    删不掉，几百 MB 就这么烂在 data/probes/ 里，只能靠人肉 ls 发现。
    只在「这一批最终没进历史」时删：万一 `_probe_finish` 已经登记完批次、只是落盘
    报了错，界面还点得开它，那时候把截图端走就只剩一片裂图了。
    """
    try:
        if ctx.get("bare") or ctx.get("run_id") in RUNS:
            return
        shutil.rmtree(ctx["out_dir"], ignore_errors=True)
    except Exception:
        log.warning("失败批次的截图目录 %s 没清掉", ctx.get("out_dir"), exc_info=True)
    finally:
        # 三条分支（bare / 已进历史 / 真删了）都要撤登记，否则「清空历史」
        # 会永远绕着一个早就不写的目录，越攒越多。
        _probe_release(ctx)


def _probe_finish(ctx: dict, records: list, t0: float) -> dict:
    """汇总整批，附上底库/阈值/规模曲线，落盘并进历史批次。"""
    p = ctx["p"]
    wall = ctx.get("wall_s") or 0.0
    res = {
        "records": records,
        "summary": bench.summarize(records, float(p["threshold"]), int(p["topk"]),
                                   wall_s=wall or None, workers=int(p.get("workers") or 1)),
        "params": p,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        # 推理墙钟单独存一份：elapsed_s 还含了解码、存图、汇总。改真值 / 拖阈值之后
        # 要按同一口径重算吞吐，不然一动滑杆吞吐就跟着变
        "wall_s": round(wall, 3),
        "run_id": ctx["run_id"],
        "created_at": time.time(),
        "backend": ctx["bk"].info(),
        "gallery": ctx["gallery"].stats(),
        "scale_curve": bench.search_scale_curve(ctx["gallery"]),
        # 把刚跑完的这批一起喂进去：底库每人只有 1 张模板时（最常见的部署形态）
        # 纯底库凑不出同人互配对，阈值建议整栏是空的。带真值的探针天然带这两类样本，
        # 不重新推理就能补上，跑完一批这一栏立刻有内容。
        "threshold_report": bench.threshold_report(ctx["bk"], ctx["gallery"], p,
                                                  records=records),
    }
    # 登记进内存表要持锁：worker 线程在这儿塞新批次，同一时刻前端轮询历史批次的
    # 那个请求正拿着 LOCK 遍历 RUNS，裸插入偶发把对面顶成 500。
    # 落盘和压壳都不在锁里干 —— 报告 JSON 有几十 MB，别让轮询请求陪着等磁盘。
    with LOCK:
        RUNS[ctx["run_id"]] = res
    config.write_json(REPORT_DIR / f"probe_{ctx['run_id']}.json", res)
    _probe_release(ctx)      # 目录改由历史批次管，删除交给界面/接口
    _trim_runs()
    return res


def _run_probe(batch: list[tuple[str, str, bytes]], opts: dict, progress=None) -> dict:
    """跑一批探针图，batch 每项是 (相对路径, 文件名, 原始字节)。"""
    ctx = _probe_setup(opts)
    t0 = time.perf_counter()
    try:
        return _probe_finish(ctx, _probe_chunk(ctx, batch, 0, progress), t0)
    except Exception:
        # 截图目录已经建好、也可能写了几百张，批次却没进历史，界面永远删不到它
        _discard_probe_dir(ctx)
        raise


@app.post("/api/probe/run")
def api_probe_run(
    files: list[UploadFile] = File(...),
    relpaths: str = Form(""),
    params: str = Form("{}"),
    person_mode: str = Form("name"),
    delimiter: str = Form("__"),
    max_side: int = Form(1280),
):
    """整批 multipart 上传跑一次评测（脚本/命令行方便）；浏览器走下面逐张三个接口。"""
    rels = relpaths.split("||") if relpaths else []
    batch = [(rels[i] if i < len(rels) else "", up.filename or f"probe{i}", up.file.read())
             for i, up in enumerate(files)]
    return _run_probe(batch, {"params": params, "person_mode": person_mode,
                              "delimiter": delimiter, "max_side": max_side})


PROBE_MAX_IMAGES = 2000        # 浏览器逐张上传，一批的上限
PROBE_DIR_MAX = 50000          # 服务端直接读目录，一批的上限（分块跑，内存恒定）
PROBE_CHUNK = 64               # 读目录时每读 64 张就先跑去推理，不让一万张原图同时压在内存里
PROBE_DIR_MAX_BYTES = 16 * 1024 * 1024 * 1024   # 只是防手滑选错成整个硬盘的兜底


def _scan_image_dir(raw: str) -> tuple[Path, list[Path], int]:
    """校验一个本机图片目录并列出里面的图，返回 (根目录, 图片列表, 总字节)。

    /api/probe/run_dir 和 /api/sweep 都要读本机目录，规则必须一模一样：
    相对路径按 cwd 解析、递归进子目录、张数和总字节两道兜底。
    复制两份的下场是以后改了一边忘了另一边，扫描表会比评测多吃一套限制。
    """
    root = Path(raw).expanduser()
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    if not root.exists():
        raise HTTPException(400, f"路径不存在：{root}")
    if not root.is_dir():
        raise HTTPException(400, f"这是个文件不是文件夹：{root}")
    found = sorted(p for p in root.rglob("*")
                   if p.suffix.lower() in IMG_EXT and p.is_file())
    if not found:
        raise HTTPException(400, f"{root} 里没找到图片（支持 jpg / png / webp / bmp，含子目录）")
    if len(found) > PROBE_DIR_MAX:
        raise HTTPException(400, f"这个目录有 {len(found)} 张图，单次最多 {PROBE_DIR_MAX} 张，"
                                "请拆成几个子目录分批跑")
    total_bytes = 0
    for p in found:
        try:
            total_bytes += p.stat().st_size
        except OSError:
            continue
    if total_bytes > PROBE_DIR_MAX_BYTES:
        raise HTTPException(400, f"这 {len(found)} 张图共 {round(total_bytes / 1073741824, 1)} GB，"
                                "一次实在吃不下，请拆成几个子目录分批跑")
    return root, found, total_bytes
PROBE_UPLOADS: dict[str, dict] = {}
# 上面那张会话表的写锁。浏览器一次开三个页签就会同时在三个请求线程里开评测，
# 而 _api_probe_open 里「插入新会话 + 顺手清掉过期会话」要遍历这张表：
# 没有锁就是偶发的「dictionary changed size during iteration」，
# 症状是随机某一次点「开始评测」报 500，重点一次又好。
_UPLOADS_LOCK = threading.Lock()


@app.post("/api/probe/open")
def api_probe_open(request: Request, payload: dict = Body(...)):
    """开一个探针上传会话：图片逐张推上来，close 时一次性评测。

    探针往往只有几十张，但同样会撞上「这个窗口发不出 multipart 请求体」。
    和底库用同一套原始字节上传，浏览器这边就再没有表单要拼了。
    """
    total = int(payload.get("total") or 0)
    if total <= 0:
        raise HTTPException(400, "没有探针图片，请先拖入摄像头照片")
    if total > PROBE_MAX_IMAGES:
        raise HTTPException(400, f"一批最多 {PROBE_MAX_IMAGES} 张探针图，当前 {total} 张，请分批评测")
    # 参数在这里就验一次：会话开起来之后浏览器要逐张推几百张图，
    # 全部传完、在 close 里才发现阈值写错，等于让人白等一趟上传
    _params_of(payload.get("params"))
    uid = uuid.uuid4().hex[:8]
    with _UPLOADS_LOCK:
        PROBE_UPLOADS[uid] = {"items": [], "at": {}, "opts": payload, "last": time.time()}
        for stale in [k for k, v in PROBE_UPLOADS.items()
                      if k != uid and time.time() - v["last"] > 1800]:
            PROBE_UPLOADS.pop(stale, None)
    log.warning("探针会话 %s：total=%s UA=%s", uid, total, (request.headers.get("user-agent") or "-")[:160])
    return {"upload_id": uid, "total": total}


def probe_total(up: dict) -> int:
    return int((up.get("opts") or {}).get("total") or 0)


@app.post("/api/probe/push")
async def api_probe_push(request: Request, id: str = "", rel: str = "", name: str = ""):
    """收一张探针图的原始字节；评测在 close 时统一做，这里只攒图。"""
    up = PROBE_UPLOADS.get(id)
    if not up:
        raise HTTPException(404, f"探针上传会话 {id or '(空)'} 已过期，请重新点「开始评测」")
    raw = await _read_upload(request, name or (rel.rsplit("/", 1)[-1] if rel else "探针图"),
                             "评测本机目录")
    if not raw:
        nm = name or (rel.rsplit("/", 1)[-1] if rel else "探针图")
        raise HTTPException(400, f"「{nm}」这张浏览器没发出内容（0 字节）。"
                                 "请用 Safari 直接打开 http://127.0.0.1:8617 再试一次")
    if len(up["items"]) >= PROBE_MAX_IMAGES:
        raise HTTPException(400, f"一批最多 {PROBE_MAX_IMAGES} 张探针图，剩下的请分批评测")
    up["last"] = time.time()
    nm = name or (rel.rsplit("/", 1)[-1] if rel else f"probe{len(up['items'])}")
    # 按文件名去重：浏览器补送同一张时（读盘失败重发、请求超时后重试）覆盖上一次，
    # 而不是攒出两条记录——否则一张图被算两次，识别率和耗时的分母都脏了。
    at = up.setdefault("at", {})
    key = rel or nm
    if key in at:
        up["items"][at[key]] = (rel, nm, raw)
    else:
        at[key] = len(up["items"])
        up["items"].append((rel, nm, raw))
    return {"recved": len(up["items"]), "total": probe_total(up)}


@app.post("/api/probe/close")
def api_probe_close(id: str = ""):
    """探针图传完了：用攒下的图跑一次评测，返回完整结果。

    评测放在 close 而不是每张 push 里——合影要整批一起看才有意义，
    而且 multipart 整批上传和逐张上传两条路都收在 _run_probe，
    指标口径不会走偏。
    """
    up = PROBE_UPLOADS.pop(id, None)
    if not up:
        raise HTTPException(404, f"探针上传会话 {id or '(空)'} 已过期"
                                 "（页面刷新过或太久没传完），请重新点「开始评测」")
    items = up["items"]
    if not items:
        raise HTTPException(400, "一张探针图都没收到，请重新拖入摄像头照片")
    want = probe_total(up)
    if want and len(items) < want:
        log.warning("探针只收到 %s/%s 张，按已收到的出结果", len(items), want)
    return _run_probe(items, up["opts"])


@app.post("/api/probe/run_dir")
def api_probe_run_dir(payload: dict = Body(...)):
    """直接评测服务端本机目录里的探针图，不经过浏览器上传。

    摄像头一天的抓拍动辄上千张：走浏览器要逐张推上千次请求，中间任何一次
    「这个窗口读不到磁盘文件」都会让整批变成「一张都没传上去」。评测机和
    照片在同一台机器上时（部署到 RK3588 就是这样），让服务端自己读盘最快也最稳。
    真值规则、参数口径和拖拽上传完全一致，结果同样进历史批次。
    """
    raw_path = str(payload.get("path") or "").strip().rstrip(chr(92) + "/")
    if not raw_path:
        raise HTTPException(400, "请先填探针图片目录的完整路径")
    # 先把参数验了再扫盘：一万六千张的目录扫一遍要几秒，
    # 顺序反过来就是「转圈几秒 → 报一个少引号的错」
    _params_of(payload.get("params"))
    root, found, total_bytes = _scan_image_dir(raw_path)

    log.warning("本机目录评测：%s（%d 张 / %d MB）", root, len(found), max(1, total_bytes // 1048576))
    opts = {k: payload.get(k) for k in ("params", "person_mode", "delimiter", "max_side")}
    job_id, j = _new_job(len(found), "probe")
    j["prepared"] = 0

    def worker():
        """分块读盘 + 推理：读 64 张就跑这一块，再读下一块。

        摄像头一天的抓拍动辄一两万张。整批先读进内存要几个 GB，而且
        「读完才开始识别」时进度条要干等好几分钟。分块之后内存占用恒定，
        进度按真正评完的张数往上涨。
        """
        try:
            ctx = _probe_setup(opts)
        except HTTPException as e:
            j.update(status="error", error=str(e.detail))
            return
        except Exception as e:
            j.update(status="error", error=f"{type(e).__name__}: {str(e)[:300]}")
            return
        # 批次号早早登记进任务：本来它只在评测结束时才出现，那之前「清空历史」
        # 看不出这个截图目录有人在写。（截图目录的占用登记在 _probe_hold。）
        j["run_id"] = ctx["run_id"]
        t0 = time.perf_counter()
        records: list[dict] = []
        for s in range(0, len(found), PROBE_CHUNK):
            part = found[s:s + PROBE_CHUNK]
            batch = []
            for path in part:
                try:
                    batch.append((str(path.relative_to(root)), path.name, path.read_bytes()))
                except Exception as e:
                    _note_bad(j, path.name, f"读不出来：{type(e).__name__} {str(e)[:120]}")
            if not batch:
                j.update(done=min(s + len(part), len(found)))
                continue
            try:
                chunk = _probe_chunk(ctx, batch, s,
                                     progress=lambda i: j.update(prepared=i + 1))
            except HTTPException as e:
                _discard_probe_dir(ctx)
                j.update(status="error", error=str(e.detail))
                return
            except Exception as e:
                _discard_probe_dir(ctx)
                j.update(status="error", error=f"{type(e).__name__}: {str(e)[:300]}")
                return
            records += chunk
            # ok = 真的评出结果的张数。注册那边一直有这个名字（入库的模板数），
            # 评测这边以前永远是 0：脚本轮询 /api/jobs/<id> 看 done=202 / ok=0，
            # 会以为整批全失败（本机的端到端脚本就是这么被唬住的）。
            # 坏图不算 ok，但它们另有 bad_total，两边加起来正好是 total。
            with _JOB_LOCK:
                j["ok"] += len([r for r in chunk if r.get("verdict") != "error"])
            j.update(done=min(s + len(part), len(found)))
        if not records:
            _discard_probe_dir(ctx)
            j.update(status="error",
                     error=f"{root} 里的图片一张都没评成功，检查权限或磁盘是否已卸载")
            return
        try:
            run = _probe_finish(ctx, records, t0)
        except HTTPException as e:
            _discard_probe_dir(ctx)
            j.update(status="error", error=str(e.detail))
            return
        except Exception as e:
            _discard_probe_dir(ctx)
            j.update(status="error", error=f"{type(e).__name__}: {str(e)[:300]}")
            return
        # 任务里只放批次号和汇总。一万张明细塞进任务对象，前端每 600ms 拉一次
        # 就是每次几十 MB，白等；明细由前端按页向 /api/runs/<批次> 去要。
        # 别改写 total：total 全程是「扫到多少张」，收尾时换成「评出多少条记录」
        # 会让 done 直接超过 total（进度条跑到 130%），而且毁掉
        # ok + bad_total == total 这个不变式 —— 目录里有读不出来的文件时准撞。
        # 记录条数另起一个名字给脚本用。
        j.update(status="done", run_id=run["run_id"], n_records=len(records),
                 summary=run["summary"], elapsed_s=run["elapsed_s"])

    threading.Thread(target=_guarded, args=(worker, j), daemon=True).start()
    return {"job_id": job_id, "found": len(found), "path": str(root),
            "total_bytes": total_bytes}


# ---------------- 并发 / 线程扫描（选型主输出） ----------------
SWEEP_GRID_WORKERS = (1, 2, 4)      # 并发路数默认档
SWEEP_GRID_THREADS = (0, 4)         # 推理线程数默认档，0 = 交给 onnxruntime 自己定
SWEEP_DEFAULT_IMAGES = 200
SWEEP_MIN_IMAGES = 24               # 再少的话 P95 就是噪声，不如不报
SWEEP_MAX_IMAGES = 800              # 原始字节要一次驻留内存，这是给板子上限留的余量
SWEEP_MAX_CELLS = 24
# 并发路数的上限。之所以要封顶而不是随便填：每路都在占一份全图字节和一个推理线程，
# 本机实测 5 路就把吞吐顶到 ~40 张/秒的平台上，再往上只剩延迟变差。留到 12 是给
# 「看拐点之后怎么掉」留出空间，同时挡住手滑填 500 把内存打爆。
SWEEP_WORKERS_MAX = 12
# 建议档的延迟可用线，单位毫秒。取 120 是照着「摄像头抽帧进 1:N 闸门」的场景定的：
# 本机实测单路 P50 约 32ms，并发拉到 5 路会涨到约 91ms，120ms 既能放过多路并行、
# 又能挡住延迟最糟那一档。用 payload.latency_target_ms 可覆盖（闸机要求更严就调低）。
SWEEP_LATENCY_MS = 120.0


def _sweep_grid(payload: dict) -> tuple[list[int], list[int]]:
    """把要扫的「并发路数 × 推理线程数」归一化成两个去重列表。

    threads=0 表示不指定、用 onnxruntime 自己的默认线程数。这一档必须留在表里：
    它是装完机什么都不调时的真实状态，拿一个用户根本不会去设的参数当基线，
    整张表的结论就没有意义了。
    """
    def clean(key: str, default: tuple[int, ...], lo: int, hi: int) -> list[int]:
        raw = payload.get(key)
        if raw is None or raw == "" or raw == []:
            raw = list(default)
        elif isinstance(raw, (int, float)) and not isinstance(raw, bool):
            raw = [int(raw)]
        out: list[int] = []
        for v in raw:
            try:
                iv = int(v)
            except (TypeError, ValueError):
                continue
            if lo <= iv <= hi and iv not in out:
                out.append(iv)
        return out or list(default)

    w = clean("workers", SWEEP_GRID_WORKERS, 1, SWEEP_WORKERS_MAX)
    t = clean("threads", SWEEP_GRID_THREADS, 0, 64)
    if len(w) * len(t) > SWEEP_MAX_CELLS:
        raise HTTPException(
            400, f"并发路数 × 线程数一共 {len(w)}×{len(t)}={len(w) * len(t)} 格，"
                 f"单次最多 {SWEEP_MAX_CELLS} 格 —— 每格都要把探针图整批重跑一遍")
    return w, t


@app.post("/api/sweep")
def api_sweep(payload: dict = Body(...)):
    """扫「并发路数 × 推理线程数」，输出选型表。后台任务，进度走 /api/jobs/<id>。

    选型真正要回答的是「这块板子该开几路、每个模型给几个核」，而这两个数互相牵制：
    路数换吞吐、线程数换单帧延迟。单跑一次评测只给得出一个点，换块板子就得重猜；
    把两个维度摊成一张表，就能直接挑「延迟够用的最低配置」。

    每格报的吞吐是墙钟口径（总张数 ÷ 真实墙钟），含排队和并行，
    和评测批次里的 throughput_basis="wall" 同一个算法 —— 用 1000/P50 折算的话，
    并发越高这个数越虚高，恰好把最需要看清的权衡说反。
    """
    raw_path = str(payload.get("path") or "").strip().rstrip(chr(92) + "/")
    if not raw_path:
        raise HTTPException(400, "请先填探针图片目录的完整路径（扫描必须是真图，不能凭空造）")
    root, found, _bytes = _scan_image_dir(raw_path)
    workers, threads = _sweep_grid(payload)

    # 先校验再夹上限：原来先把 limit 抬到 SWEEP_MIN_IMAGES 再判「小于下限」，
    # 那条分支永远走不到，用户填 5 张会被悄悄当成 24 张用（多跑四倍时间还不吭声）。
    limit = _clean_num(payload.get("limit"), "扫描张数", SWEEP_DEFAULT_IMAGES,
                       1, SWEEP_MAX_IMAGES)
    if limit < SWEEP_MIN_IMAGES:
        raise HTTPException(400, f"扫描至少要用 {SWEEP_MIN_IMAGES} 张图，你填了 {limit} 张，"
                                "不然 P95 就是噪声，换个顺序跑结果就变")
    # 上限不再静默夹掉：填 5000 的人是真想跑 5000 张，悄悄按 800 跑完，
    # 他只会以为这块板子变慢了。（#s-limit 的 max 本来就是 SWEEP_MAX_IMAGES，
    # 走到这里说明是手打或脚本直传，报错比夹掉有用。）
    # 等间隔抽样而不是取前 N 张：目录基本是按时间/机位排的，
    # 前 200 张往往是同一台相机同一段光照，扫出来的吞吐只代表那一刻。
    step = max(1, len(found) // limit)
    picked = found[::step][:limit]

    # 九格共用同一份参数，所以在这里就整份验掉（含 NaN 和范围）：
    # 少个引号要等到开跑以后才发现，就是九格各记一条一样的失败 + 白扫一遍盘
    base = _params_of(payload.get("params"))

    # 原始字节一次读进内存复用：九格就是九遍同样的图，
    # 反复读盘会把「换板子换出来的差异」淹在 IO 抖动里。
    batch: list[tuple[str, str, bytes]] = []
    for p in picked:
        try:
            batch.append((str(p.relative_to(root)), p.name, p.read_bytes()))
        except Exception:
            continue
    if len(batch) < SWEEP_MIN_IMAGES:
        raise HTTPException(400, f"只读出 {len(batch)} 张图（要至少 {SWEEP_MIN_IMAGES} 张），"
                                "检查目录权限或是否有 iCloud 没下下来的占位文件")

    n_cells = len(workers) * len(threads)
    job_id, j = _new_job(n_cells, "sweep")
    j["cells"] = []
    j["grid"] = {"workers": workers, "threads": threads}
    j["n_images"] = len(batch)
    j["path"] = str(root)
    j["cpu_count"] = os.cpu_count()
    opts = {k: payload.get(k) for k in ("person_mode", "delimiter", "max_side")}
    opts["bare"] = True
    orig_threads = STATE.get("threads")
    latency_target = _clean_num(payload.get("latency_target_ms"), "延迟可用线(ms)",
                                SWEEP_LATENCY_MS, 1.0, 60000.0, cast=float)

    def worker():
        """换一次线程数就重载一次模型，然后在这一档下把所有并发路数跑完。

        线程数是在建 session 时定死的，所以换线程数必须重建后端（要重新载入
        两百多 MB 权重）；但重载之后所有并发档共用同一个 session，
        九格的扫描只重载两次而不是九次。

        换线程数会临时改掉全局 STATE["threads"]：跑扫描期间别人点「换后端」
        或发起评测，用的就是这一档的线程数。这是把两条路都收在同一份 STATE 上的
        代价，好处是扫描用的配置和真实评测完全一致，不会自成一套。
        """
        try:
            # 记下「现在内存里这个 session 是按哪一档线程数建的」。用 loop 变量 th
            # 在 finally 里判断不可靠：某一档载入失败会 continue 掉，th 已经是失败那档，
            # 而内存里跑的还是上一档的 session，会把该重载误判成不用重载。
            loaded_th = orig_threads
            for th in threads:
                with LOCK:
                    STATE["threads"] = th
                    try:
                        get_backend(force=True)
                        loaded_th = th
                    except BackendError as e:
                        for w in workers:
                            j["cells"].append({"threads": th, "workers": w,
                                               "error": f"载入模型失败：{e}"})
                            with _JOB_LOCK:
                                j["done"] += 1
                        continue
                for w in workers:
                    cell: dict = {"threads": th, "workers": w}
                    try:
                        ctx = _probe_setup({**opts, "params": {**base, "workers": w}})
                        t0 = time.perf_counter()
                        recs = _probe_chunk(ctx, batch, 0)
                        # 优先用推理墙钟（ctx 里逐块累加的那份）；它为空说明一块都没跑成，
                        # 退回整格墙钟，别让一格失败把吞吐报成 0 而不是缺失
                        wall = float(ctx.get("wall_s") or 0.0) or (time.perf_counter() - t0)
                        s = bench.summarize(recs, float(ctx["p"]["threshold"]),
                                            int(ctx["p"]["topk"]), wall_s=wall, workers=w)
                        lat = s.get("latency") or {}
                        cell.update({
                            "n": len(batch), "faces": s["n_faces"],
                            "detect_rate": s["detect_rate"],
                            "throughput_fps": s["throughput_fps"],
                            "throughput_faces_per_s": s["throughput_faces_per_s"],
                            "throughput_basis": s["throughput_basis"],
                            "total_ms": lat.get("total_ms"),
                            "detect_ms": lat.get("detect_ms"),
                            "embed_ms": lat.get("embed_ms"),
                            "search_ms": lat.get("search_ms"),
                            "gallery": ctx["gallery"].stats().get("templates"),
                            "wall_s": round(wall, 2),
                        })
                    except HTTPException as e:
                        cell["error"] = str(e.detail)
                    except Exception as e:
                        cell["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                    j["cells"].append(cell)
                    with _JOB_LOCK:
                        j["done"] += 1
        except Exception as e:      # 兜底：线程里炸了要给前端看到，别留一个永远 running 的任务
            j["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        finally:
            with LOCK:
                # 恢复调用方原来的线程设置，扫描不该改变服务状态 —— 但只覆盖
                # 「还是我们留下那一档」的情况：扫描要跑几分钟，期间别人在
                # 「换后端」里把线程数改成 8 的话，无条件写回 orig_threads
                # 会把他的选择悄悄吞掉，他还以为自己设上了。
                cur = STATE.get("threads")
                if cur == loaded_th:
                    STATE["threads"] = orig_threads
                    need_reload = orig_threads != loaded_th
                else:
                    need_reload = cur != loaded_th
                    log.warning("扫描期间线程设置被改成 %s，保留对方的选择，不还原成 %s",
                                cur, orig_threads)
                # 线程数从头到尾没变就别重载：重载要重读两百多 MB 权重（本机 2~4 秒），
                # threads=[0] 这种单档扫描根本没动过它，无条件重载等于给扫描结束后的
                # 第一个请求白加几秒延迟。变了才必须重建 session，线程数建 session 时才定死。
                if need_reload:
                    try:
                        get_backend(force=True)
                    except BackendError:
                        log.warning("扫描结束后恢复线程档失败，下次请求会自行重载")
        ok_cells = [c for c in j["cells"] if not c.get("error")]
        fast = max(ok_cells, key=lambda c: c.get("throughput_faces_per_s") or 0) if ok_cells else None
        # ok 回填成功的格数：扫描和注册共用同一份 job 结构，前端 pollSweep 是自己
        # 数 cells 的所以看不出来，但直接读 /api/jobs 的脚本和命令行会拿到恒 0 的
        # ok，以为 24 格全失败。bad_total 同步记失败格数，口径和 bad 一致。
        with _JOB_LOCK:
            j["ok"] = len(ok_cells)
            j["bad_total"] = len(j["cells"]) - len(ok_cells)
        j.update(status="done", finished=time.time(), best_throughput=fast,
                 recommend=_sweep_recommend(ok_cells, latency_target))

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id, "grid": j["grid"], "n_images": len(batch),
            "cells": n_cells, "path": str(root), "cpu_count": os.cpu_count()}


def _sweep_recommend(cells: list[dict], latency_target: float) -> dict:
    """从扫描结果里挑一个建议配置，并把理由一并说清楚。

    不是无脑取吞吐最高那格：并发最高的一档通常延迟也最高，
    而摄像头接入要的是「单帧别拖太久」。这里先按 P50 卡一条可用线
    （payload.latency_target_ms，默认 SWEEP_LATENCY_MS），在达标的格里取吞吐最高的，
    一格都不达标时才退回吞吐最高，并如实说明延迟超标。
    """
    if not cells:
        return {"note": "没有任何一格跑成功，无法给建议"}
    target = float(latency_target or 0) or None
    # 用 is not None 而不是真值判断：p50 真的可能是 0.0（det_size 很小、
    # 或者按 3 位小数被抹平），那恰恰是最达标的一格，真值判断会把它当「没数据」剔掉。
    fits = [c for c in cells if target
            and (c.get("total_ms") or {}).get("p50") is not None
            and c["total_ms"]["p50"] <= target]
    if fits:
        best = max(fits, key=lambda c: c.get("throughput_faces_per_s") or 0)
        return {"cell": best, "basis": f"P50 ≤ {target:.0f}ms 的格里取吞吐最高",
                "note": "该格延迟在可用线内，且是这些格里吞吐最高的"}
    best = max(cells, key=lambda c: c.get("throughput_faces_per_s") or 0)
    p50 = (best.get("total_ms") or {}).get("p50")
    note = "取吞吐最高（没有一格达到延迟可用线）"
    if p50 is not None and target:
        note += f"：最快的一格 P50 仍要 {p50:.0f}ms，考虑降 det_size、换更小的检测模型或减少并发"
    return {"cell": best, "basis": note, "note": note}


@app.post("/api/probe/relabel")
def api_probe_relabel(payload: dict = Body(...)):
    """人工纠正真值后重算指标，无需重新推理。

    不带 face_idx → 改整图主体：该图所有人脸先回到“都算这个主体”的初始态；
    带 face_idx  → 只改那一张脸的真值（用于标注合影里的同框他人 / 非底库人员）。
    """
    run = _ensure_run(str(payload.get("run_id") or ""))
    rec = next((r for r in run["records"] if r["name"] == payload.get("name")), None)
    if not rec:
        raise HTTPException(404, "探针记录不存在")
    thr = float(run["params"]["threshold"])
    truth = payload.get("truth") or None
    fidx = payload.get("face_idx")
    faces = rec.get("faces") or []
    if fidx is None:
        prev = rec.get("truth")
        rec["truth"] = truth
        if rec.get("verdict") != "error":
            rec.pop("error", None)
        # 评失败的那张留着原因：recompute 只做比较、不会重新推理，
        # 标个真值不可能让「这张根本没评出来」变成过去式。把原因一起抹掉，
        # 判定期望还是 error、理由却空了，翻报告的人只能对着一个孤零零的失败标签猜。
        rec.pop("truth_raw", None)
        bench.auto_label(rec, thr, prev)   # 重新分配：命中主体的脸认领，人工标过的旁人不动
    else:
        # 序号也走 _clean_num：改脸级真值时以前传个 "abc" 会冒出 500，
        # 报错还说不出是哪个字段错了；越界的那句「人脸序号不合法」同理没说清
        # 到底有几张脸可选，界面上一次点七张脸的合影时很容易踩到。
        if not faces:
            # 无脸 / 解码失败的记录压根没有「第几张脸」，得先挡住再校验序号：
            # 否则上限算出 len(faces)-1 = -1，faces[0] 直接 IndexError 变 500。
            raise HTTPException(400, "这张记录里一张脸都没有，改不了单张脸的真值")
        fi = _clean_num(fidx, "人脸序号", 0, 0, max(0, len(faces) - 1))
        target = faces[fi]
        target["truth"] = truth
        if target.get("is_subject") and rec.get("truth") != truth:
            # 主体脸和整图真值说的是同一件事，不能各说各话。
            # 以前容许「整图真值=张三、主体脸标成非底库」并存，于是同一次放行
            # 既进 Rank-1 分子（认对了张三）又当一次误放行（非底库被放行），
            # far_estimated 能算出 100% 这种荒谬值。以刚改的这张脸为准，倒推整图真值。
            rec["truth"] = truth
            rec["_truth_auto"] = truth
    bench.recompute(rec, thr)
    run["summary"] = bench.summarize(run["records"], thr, int(run["params"]["topk"]),
                                     wall_s=run.get("wall_s") or None,
                                     workers=int(run["params"].get("workers") or 1))
    _mark_dirty(run["run_id"])
    return {"summary": run["summary"], "facets": _facets(run["records"]),
            "record": rec, "threshold": thr,
            # 真值就是阈值分布的原料：改一条真值，genuine/冒认两侧各多（少）一个样本。
            # 不带回来的话，用户把探针真值一条条标全了，阈值建议那栏还是「样本不足」。
            "threshold_report": _fresh_threshold_report(run["records"])}


@app.post("/api/rethreshold")
def api_rethreshold(payload: dict = Body(...)):
    """只做阈值重判定（纯 numpy，毫秒级），用于 UI 上拖阈值滑杆。"""
    run = _ensure_run(str(payload.get("run_id") or ""))
    thr = _clean_threshold(payload.get("threshold"))
    run["params"]["threshold"] = thr
    for r in run["records"]:
        bench.recompute(r, thr)
    run["summary"] = bench.summarize(run["records"], thr, int(run["params"]["topk"]),
                                     wall_s=run.get("wall_s") or None,
                                     workers=int(run["params"].get("workers") or 1))
    _mark_dirty(run["run_id"])
    return {"summary": run["summary"], "facets": _facets(run["records"]), "threshold": thr,
            "threshold_report": _fresh_threshold_report(run["records"])}


def _fresh_threshold_report(records: list | None = None) -> dict | None:
    """按当前底库（和这批带真值的探针）重出一份阈值分布，供刷新阈值曲线。

    曲线跟阈值无关，但跟两件事有关：底库在两次评测之间被注册过，历史报告里那份
    就旧了；这批探针的真值被人工改过，genuine/冒认样本也跟着变。
    底库那一份 bench.threshold_report 内部按 gallery.version 缓存，
    探针那一份是纯 numpy 拼接，拖一次滑杆几毫秒。

    取不到就返回 None 而不是报错：重判定用的是已经缓存的 Top-K，
    完全不碰推理，模型文件被移走也应当能继续翻旧批次和拖滑杆。
    """
    try:
        return bench.threshold_report(get_backend(), get_gallery(), records=records)
    except Exception:
        log.warning("阈值分布重算失败（不影响重判定），已跳过", exc_info=True)
        return None


@app.get("/api/probe/thumb/{run_id}/{name}")
def api_probe_thumb(run_id: str, name: str):
    p = (PROBE_DIR / run_id / name).resolve()
    if not _inside(PROBE_DIR / run_id, p) or not p.exists():
        raise HTTPException(404, "缩略图不存在")
    return FileResponse(p, media_type="image/jpeg")


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "任务不存在")
    return j


@app.get("/api/runs")
def api_runs():
    # 先把 RUNS 拷出来再排序：跑本机目录评测 / 并发扫描时 worker 线程正在
    # _probe_finish 里写 RUNS，这里裸遍历偶发 500（dictionary changed size during
    # iteration），而且偏偏在浏览器每几秒就轮询一次的接口上。和 api_runs_clear 同口径。
    with LOCK:
        items = list(RUNS.items())
    return sorted(
        ({"run_id": k, "summary": v["summary"], "params": v["params"],
          "gallery": v.get("gallery"), "created_at": v.get("created_at")}
         for k, v in items),
        key=lambda x: x.get("created_at") or 0, reverse=True,
    )


@app.get("/api/runs/{run_id}")
def api_run(run_id: str, offset: int = Query(0, ge=0),
            limit: int = Query(150, ge=1, le=2000), only: str = Query("all")):
    """按页取批次明细：一万张的批次整包吐给浏览器是几十 MB，页面直接卡死。

    只回当前筛选页的记录（每条带全局序号 `_i`，界面按它取缩略图），
    外加各筛选项的计数，界面据此显示「共 N 条 · 命中 M 条」。
    """
    run = _ensure_run(run_id)
    recs = run.get("records") or []
    only = only if only in _FACETS else "all"
    hit = [i for i, r in enumerate(recs) if _facet_hit(r, only)]
    # offset 是客户端传来的，而筛选结果随时会缩：停在第 5 页时把「判错」改成
    # 「命中」，或者改完真值这一类变少了，切片就是空的，可 matched 还是 100 ——
    # 页面上写着「第 100–100 条 · 共 100 条」外加零行，下一页按钮永远按不动。
    # 夹到最后一页，并且把夹后的值回给前端，位置立刻自洽。
    if hit:
        offset = min(offset, (len(hit) - 1) // limit * limit)
    else:
        offset = 0
    page = [{**recs[i], "_i": i} for i in hit[offset:offset + limit]]
    # 复核图在不在，这里以磁盘为准现算，不等重启时才标：扫描表用的 bare 批次本来就不存图，
    # 探针目录被手工清掉的批次也一样，两种都得在界面上说明白，否则屏幕上只剩一片裂图图标，
    # 看着像评测跑坏了。现算比信 RUNS 里那份老标记可靠——目录可能刚被人删掉。
    gone = bool(run.get("bare")) or not (PROBE_DIR / run_id).is_dir()
    return {**{k: v for k, v in run.items() if k != "records"},
            "images_gone": gone,
            "records": page, "total": len(recs), "matched": len(hit),
            "offset": offset, "limit": limit, "only": only,
            "facets": _facets(recs)}


def _tree_bytes(path: Path) -> int:
    """单个文件或整棵目录树的字节数，只用来给用户报「释放了多少空间」。"""
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _rm_tree(path: Path) -> tuple[bool, int]:
    """删掉一个文件或整棵目录树，返回 (是否删净, 释放字节)。

    「本来就不存在」算删净：清历史批次时要连磁盘上的残留一起扫，
    只剩内存记录的批次删文件拿到 ENOENT，那正是我们想要的结果。
    """
    try:
        freed = _tree_bytes(path)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        return True, freed
    except FileNotFoundError:
        return True, 0
    except OSError as e:
        log.warning("清理失败 %s：%s", path, e)
        return False, 0


@app.delete("/api/runs")
def api_runs_clear():
    """清空历史批次：内存里的批次、reports/probe_*.json、data/probes/<批次>/ 一起删。

    报告和探针原图必须成对删——报告引用的是 `data/probes/<批次>/` 里的裁剪图，
    只剩报告的孤儿在重启时会被 load_runs_from_disk 跳过，界面上再也没机会删掉。
    命令行写的 cli_*.json 不是历史批次，底库也不在范围内，两者都不动。
    """
    with LOCK:
        ids = set(RUNS)     # 别的线程可能正在收尾新批次，裸遍历会撞上并发修改
    for f in REPORT_DIR.glob("probe_*.json"):
        rid = f.stem[len("probe_"):]
        # 批次号来自磁盘文件名，挡一下 `probe_..json` 之类拼出来的意外目录名
        if re.fullmatch(r"[0-9A-Za-z_-]{4,64}", rid):
            ids.add(rid)
    freed, cleared, failed = 0, [], []
    # 用集合而不是列表：一个批次在两个落盘位置各有一份，跳过时会各撞一次，
    # 列表就变成「跳过了 2 个批次」，把 1 个批次说成 2 个。
    skipped: set[str] = set()
    # 正在评测的批次跳过：那一刻 out_dir 里是 worker 正在往里写的截图，
    # rmtree 掉它，下一块开 meta.jsonl 就是 FileNotFoundError，几分钟推理全丢。
    busy = _active_probe_ids()

    # 两个落盘位置整体扫一遍，顺带清掉「只剩报告的孤儿」和「只剩原图的死目录」
    for p in sorted(REPORT_DIR.glob("probe_*.json")) + sorted(PROBE_DIR.iterdir()):
        rid = p.stem[len("probe_"):] if p.suffix == ".json" else p.name
        if rid in busy:
            skipped.add(rid)
            continue
        ok, n = _rm_tree(p)
        freed += n
        if not ok:
            failed.append(p.name)
    # 文件真没了才撤内存记录：删失败的批次下次重启还会读回来，记录得跟着留在下拉里
    for rid in sorted(ids):
        if not (REPORT_DIR / f"probe_{rid}.json").exists() and not (PROBE_DIR / rid).exists():
            with LOCK:
                RUNS.pop(rid, None)
            with _DIRTY_LOCK:
                DIRTY.pop(rid, None)    # 文件都删了，别让后台线程又把旧版本写回去
            cleared.append(rid)
    return {"cleared": len(cleared), "failed": failed, "skipped": sorted(skipped),
            "freed_bytes": freed, "runs": len(RUNS)}


@app.delete("/api/runs/{run_id}")
def api_run_delete(run_id: str):
    """删掉单个历史批次：报告、探针复核图、内存记录一起走。

    以前只有「清空全部」：想撤掉一次跑歪的评测（阈值设错、探针目录选错），
    就得把旁边几个好批次一起删掉重跑，而一次一万六千张的评测要跑十几分钟。
    删除口径和清空全部一致 —— 报告和图必须成对消失，否则要么留下界面上下不掉的
    孤儿报告，要么留下没人认领的几百 MB 目录。删失败如实报 failed，不假装删掉了。
    """
    if not re.fullmatch(r"[0-9A-Za-z_-]{4,64}", run_id):
        raise HTTPException(400, f"批次号格式不对：{run_id!r}")
    if run_id in _active_probe_ids():
        # 这一批评测还在往里写截图。删一半的效果最难解释：报告没了、目录还在写，
        # 收尾时 _probe_finish 又把报告落回来，界面上它又冒出来了。
        raise HTTPException(409, "这批评测还在跑，等它结束再删")
    rep = REPORT_DIR / f"probe_{run_id}.json"
    imgs = PROBE_DIR / run_id
    with LOCK:
        known = run_id in RUNS
    if not known and not rep.is_file() and not imgs.exists():
        raise HTTPException(404, "评测批次不存在")
    failed, freed = [], 0
    for p in (imgs, rep):
        ok, n = _rm_tree(p)
        freed += n
        if not ok:
            failed.append(p.name)
    cleared = 0
    # 两处文件真都没了才撤内存记录：只删掉一半就撤，下拉里它会消失，
    # 下次重启又被 load_runs_from_disk 读回来，看着像删除没生效
    if not rep.exists() and not imgs.exists():
        with LOCK:
            RUNS.pop(run_id, None)
        with _DIRTY_LOCK:
            DIRTY.pop(run_id, None)      # 别让后台落盘线程把报告又写回来
        cleared = 1
    return {"cleared": cleared, "run_id": run_id, "failed": failed,
            "freed_bytes": freed, "runs": len(RUNS)}


def load_runs_from_disk() -> int:
    """重启后把 reports/probe_*.json 读回内存，历史批次仍可查看与改标。

    只读最近几个批次的完整明细，其余收成「只剩汇总」的壳：一万张的批次
    一份明细几十 MB，三个全读进来就该担心内存了，壳照样能让历史批次列出来，
    点开时 _ensure_run 再从磁盘读回。
    """
    n = 0
    files = sorted(REPORT_DIR.glob("probe_*.json"),
                   key=lambda f: f.stat().st_mtime, reverse=True)
    for f in files:
        run_id = f.stem[len("probe_"):]
        if run_id in RUNS:
            continue
        try:
            run = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if run.get("records"):
            # 复核图目录没了也照样读回来。以前直接跳过，于是扫描表（bare）
            # 跑出来的批次重启后从下拉里凭空消失 —— 那些批次本来就不存复核图，
            # 报告和判定却一样在，删不掉也看不了。缺图如实标一下就够了。
            run["images_gone"] = not (PROBE_DIR / run_id).is_dir()
            if n >= RUNS_KEEP:
                # 超出配额就地收成壳，别先整份读进内存再等 _trim_runs 压：
                # json.loads 出来的那一刻全部批次的所有 records 同时挂在内存里，
                # 峰值是几个批次之和（万张一批几十 MB × 批次个数），
                # 而文档说的「只读最近几个批次的完整明细」本来就不需要那份峰值。
                run = {**{k: v for k, v in run.items() if k != "records"},
                       "records": [], "_stub": True}
            else:
                n += 1
            RUNS[run_id] = run
    _trim_runs()      # 兜底：内存里可能已经有别的整份批次（热重载/重复调用）
    return n


@app.get("/api/export/{run_id}")
def api_export(run_id: str):
    _ensure_run(run_id)
    _flush_run(run_id)     # 导出必须拿到最新判定，不能是 5 秒前的旧版本
    return FileResponse(REPORT_DIR / f"probe_{run_id}.json", filename=f"face_bench_{run_id}.json")


class NoCacheStatic(StaticFiles):
    """HTML/JS 必须每次校验，否则浏览器会把「新 app.js + 旧 index.html」混在一起用，
    新代码去取旧页面里不存在的控件，报一句看不懂的 null 错误。"""

    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


@app.get("/")
def index_page():
    """首页把版本号现场写进去，省掉「改了代码要手动改版本号」这一步。"""
    html = (STATIC / "index.html").read_text(encoding="utf-8").replace("__BUILD__", build_id())
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


app.mount("/static", NoCacheStatic(directory=str(STATIC)), name="static")


def main() -> None:
    import uvicorn
    load_runs_from_disk()

    port = int(os.environ.get("PORT", "8617"))
    uvicorn.run(app, host="0.0.0.0", port=port,
                log_level=os.environ.get("FACE_LOG_LEVEL", "info"))


if __name__ == "__main__":
    main()
