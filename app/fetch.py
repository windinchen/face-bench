#!/usr/bin/env python3
"""模型文件按需下载：缺失或校验不过就补，验过 SHA256 才原子换上去。

清单和校验值以 app/config.MODELS 为准。两种来源：
  url        单个 .onnx 直链（hf-mirror），curl 断点续传直接下。
  zip+member 官方只以 zip 发行的成员，用 HTTP range 只抽那一段字节 ——
             det_2.5g 只有 3 MB，没必要拉 276 MB 的 buffalo_m.zip 整包。
             解压时按 zip 中央目录里的 CRC 和原始尺寸逐块复核。

github 直连在国内常抽风（连接超时或 404），失败自动换镜像站重试；板子走内网时
用 FACE_MODEL_BASES 整表换成自己的镜像。只用标准库 + curl，装依赖之前、
系统 python3 上也能跑（bash models/fetch.sh 调的就是这里）。

谁怎么用：
  cli/run.py / models/fetch.sh —— 阻塞下载并打进度（批量工具等得起，180 MB 也是明账）；
  app/server.py —— 后台下载 + /api/env 出进度，绝不把一个 HTTP 请求卡住三分钟；
  app/backend.py —— OnnxBackend(auto_fetch=True/False) 决定走哪条路。
"""
from __future__ import annotations

import os
import shutil
import struct
import subprocess
import threading
import time
import zlib
from pathlib import Path

from .config import MODELS, MODEL_DIR, sha256_of

# github 直连失败时依次尝试的镜像前缀（前者快，后者兜底）
MIRRORS = ("", "https://ghfast.top/", "https://gh-proxy.com/")
CHUNK = 16 << 20
RANGE_TIMEOUT_S = 900     # 单段 range 下载的超时（秒），够抽 16MB 的块
STALL_TIME_S = 30         # 速度低于下限持续这么久就判定这条连接已经僵了
STALL_LIMIT_BPS = 4096    # 4KB/s：真慢的链路也远不止这个数，够不到就是没在传字节
SPACE_MARGIN = 64 << 20   # 落盘前至少多留这么多，别把板子的 eMMC 写到满
RETRY_COOLDOWN_S = 120.0  # 自动下载失败后隔这么久才再试：/api/env 20 秒轮一次，
                          # 不然每 20 秒就去敲一次已经抽风的镜像，日志刷屏还占连接
POLL_S = 0.4              # wait() 打进度的间隔


class ModelError(RuntimeError):
    """模型补不上（网络、校验、磁盘空间）。"""


class NoSpace(ModelError):
    """磁盘装不下。镜像换多少个都一样，不重试。"""


_LOCK = threading.Lock()          # 保护 _STATES / _KEYS
_STATES: dict[str, dict] = {}     # 模型 key -> 最近一次下载的状态
_KEYS: dict[str, threading.Lock] = {}   # 同一个模型同时只允许一个下载者


# ---------------------------------------------------------------- 登记表读法

def auto_fetch() -> bool:
    """FACE_AUTO_FETCH=0 时永不自动下载。

    板子上没外网、或不想让一次评测顺手吃掉 180 MB eMMC 时关掉；
    关掉以后缺模型直接报错，提示用 bash models/fetch.sh 手动补。
    """
    return os.environ.get("FACE_AUTO_FETCH", "1").strip().lower() not in (
        "0", "false", "no", "off")


def needed(key: str) -> bool:
    """这个模型还需不需要补：没文件，或者文件的 SHA256 跟登记表不符。

    校验不过也要算「要补」：下到一半断网留下的半截文件、被人换成另一个导出的
    权重，都还能被 InferenceSession 正常加载，推理不报错却算出错乱的分数，
    而界面早就把「校验失败」标红了 —— 现在能自己修，不用人再去删。
    """
    cfg = MODELS.get(key)
    if not cfg:
        return False
    out = MODEL_DIR / cfg["file"]
    if not out.exists():
        return True
    want = cfg.get("sha256")
    return bool(want) and sha256_of(out) != want


def _source(cfg: dict):
    """(方式, 参数…)；direct=直链，member=从远程 zip 里抽成员，None=地址没配。"""
    if cfg.get("url"):
        return ("direct", cfg["url"])
    if cfg.get("zip"):
        return ("member", cfg["zip"], cfg.get("member") or cfg["file"])
    return None


# -------------------------------------------------------------------- 状态表

def _set(key: str, **kw) -> None:
    with _LOCK:
        st = _STATES.setdefault(key, {"key": key})
        st.update(kw)


def _note(msg: str, quiet: bool = False) -> None:
    """往终端写一行进度。后台线程和自检都传 quiet=True，别把别人的输出搅乱。"""
    if not quiet:
        print(msg, flush=True)


def _snapshot(key: str) -> dict | None:
    with _LOCK:
        st = _STATES.get(key)
        return dict(st) if st else None


def status(key: str) -> dict | None:
    """最近一次下载的状态，附实时进度。没下过返回 None。

    进度不用下载线程回写：curl 直接往 <模型>.part 写，这里 stat 一下就是
    真实字节数，省一个线程也省一把进度锁，而且断线重连后接着算不会倒退。
    """
    st = _snapshot(key)
    if not st:
        return None
    live = st.get("state") in ("queued", "download", "verify")
    done = int(st.get("bytes_done") or 0)
    total = int(st.get("bytes_total") or 0)
    if live:
        chunk = st.get("chunk")
        if chunk:
            # 从 zip 抽成员时正在下的那一段先落在 .chunk 里，还没并进 .part。
            # 不算进来，慢网下一段 16MB 要几分钟，顶栏就贴着上个数一动不动，
            # 用户以为服务断了 —— 而这段字节其实已经在盘上了，stat 得到。
            try:
                done = min(done + Path(chunk).stat().st_size, total or done)
            except OSError:
                pass
        elif st.get("part"):
            # 直链：curl 直接写 .part（带断点续传），文件多大就是多大
            try:
                done = max(done, Path(st["part"]).stat().st_size)
            except OSError:
                pass
    st["bytes_done"] = done
    spent = max(time.time() - float(st.get("started") or time.time()), 0.001)
    st["mbps"] = round(done / spent / 1e6, 1)
    # 下载中最多报 99%：curl 写完了还要过 SHA256，卡在 100% 半天才变「就位」更像卡死了
    st["pct"] = min(99, done * 100 // total) if total else None
    st["retry_in"] = max(0, int(RETRY_COOLDOWN_S - (time.time() - float(st.get("failed_at") or 0)))) \
        if st.get("state") == "failed" else 0
    return st


def statuses() -> dict[str, dict]:
    out = {}
    for key in list(MODELS):
        st = status(key)
        if st:
            out[key] = st
    return out


def fetching() -> list[str]:
    return [k for k, s in statuses().items() if s.get("state") in ("queued", "download", "verify")]


def describe(key: str) -> str:
    """给人看的一句话，用在报错信息和顶栏 pill 上。"""
    st = status(key) or {}
    state = st.get("state")
    if state in ("queued", "download", "verify"):
        got, total = int(st.get("bytes_done") or 0), int(st.get("bytes_total") or 0)
        pct = st.get("pct")
        how = f"{pct}%" if pct is not None else f"{got >> 20}MB"
        tail = f"/{total >> 20}MB · {st.get('mbps')}MB/s" if total else ""
        return f"模型 {key} 正在后台下载（{how}{tail}），下载完成前用不了这个模型"
    if state == "failed":
        left = int(st.get("retry_in") or 0)
        back = f"，{left} 秒后自动重试" if left > 0 else ""
        return f"模型 {key} 自动下载失败：{st.get('error') or '原因不明'}{back}"
    if state == "wait":
        return f"模型 {key} 没配下载地址（app/config.MODELS 里 url / zip 都是空的）"
    if state == "done":
        return f"模型 {key} 刚下载完"
    if not auto_fetch():
        return (f"模型 {key} 未下载，而自动下载已被 FACE_AUTO_FETCH=0 关掉："
                f"bash models/fetch.sh {key} 补上，或把开关打开")
    return f"模型 {key} 未下载"


# ------------------------------------------------------------------ 下载内核

def _run(args: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, timeout=timeout)


def _stall_flags() -> list[str]:
    """连接僵住（连上了不吐字节）要在几十秒内断开，而不是抱着 -m 干等。

    实测踩过：det_2.5g 从 github 直连抽 zip 成员，curl 挂在那儿三分钟零字节，
    而 -m 900 是总超时，这期间它既不断开也不换镜像 —— 顶栏进度贴在一个数上
    一动不动，底下那条几百 KB/s 的 ghfast 一直没轮到上场。少这一副卡死判定，
    「模型自动下载」在国内直连上就是最长 15 分钟的假死。

    交给 curl 自己判：低于 STALL_LIMIT_BPS 持续 STALL_TIME_S 秒就断开，
    断开之后各走各的既有退路 —— _curl_range 重试、fetch_direct 的 --retry -C -
    续传、fetch_member 换下一个镜像。
    """
    return ["--speed-time", str(STALL_TIME_S), "--speed-limit", str(STALL_LIMIT_BPS)]


def _curl_range(url: str, start: int, end: int, retries: int = 4,
                sink: Path | None = None) -> bytes:
    """取 [start, end] 闭区间字节；长度不符就重试（镜像偶尔截断）。

    sink 是给大段下载用的：不落盘的话这一段字节只存在于 curl 的 stdout 里，
    整段下完之前 /api/env 问到的进度一动不动，几百 KB/s 的链路上顶栏能连着
    几分钟贴在 0%，看着和服务断了一个样。让 curl 直接写文件，进度就是实时的。
    """
    want = end - start + 1
    last = ""
    for attempt in range(retries):
        args = ["curl", "-sL", "--fail", "--connect-timeout", "20", "-m", str(RANGE_TIMEOUT_S),
                *_stall_flags(), "-H", "User-Agent: curl/8.7.1", "--range", f"{start}-{end}"]
        if sink:
            args += ["-o", str(sink)]
        p = _run(args + [url], RANGE_TIMEOUT_S + 60)
        if sink:
            try:
                got = Path(sink).stat().st_size
            except OSError:
                got = 0
        else:
            got = len(p.stdout)
        if p.returncode == 0 and got == want:
            return Path(sink).read_bytes() if sink else p.stdout
        last = f"rc={p.returncode} got={got} want={want}"
        time.sleep(2 + attempt)
    raise ModelError(f"分段下载失败 {start}-{end}：{last}")


def _bases(url: str) -> list[str]:
    """github 的包按镜像顺序试，其他地址原样返回。

    FACE_MODEL_BASES 用逗号分隔整表覆盖，板子内网就填自己那台
    （`,https://lan.invalid/` 这种留一个空前缀表示「先直连再走内网」）。
    """
    env = os.environ.get("FACE_MODEL_BASES", "")
    if env.strip():
        picked = [b.strip() + url for b in env.split(",") if b.strip()]
        return picked or [url]
    if "github.com" in url:
        return [m + url for m in MIRRORS]
    return [url]


def remote_size(url: str) -> int:
    """问远端「这个文件多大」，用来算百分比和落盘前查磁盘。

    重定向链上**每一跳都带 content-length**，第一跳那个是跳转页的大小
    （hf-mirror 的 302 只有 1034 字节，真文件 16 MB）。所以一路覆盖到
    最后一次才算数；带 content-range 的回包最权威，直接采信。
    取错了的后果是百分比永远贴在 0%，而磁盘检查形同虚设。
    """
    for u in _bases(url):
        p = _run(["curl", "-sIL", "--fail", "--connect-timeout", "20", "-m", "120",
                  "-H", "User-Agent: curl/8.7.1", u], 130)
        if p.returncode != 0:
            continue
        size = None
        for line in p.stdout.decode("latin1").splitlines():
            line = line.strip().lower()
            if line.startswith("content-range:"):
                return int(line.rsplit("/", 1)[-1])
            elif line.startswith("content-length:"):
                try:
                    v = int(line.split(":", 1)[1])
                except ValueError:
                    continue
                if v:                   # 0 多半是 HEAD 的空回包，别拿它顶掉真尺寸
                    size = v
        if size:
            return size
    raise ModelError(f"拿不到远端尺寸：{url}")


def zip_members(url: str, size: int) -> dict[str, dict]:
    """只读 zip 尾部的中央目录，返回 {成员名: 位置与校验信息}。"""
    tail = _curl_range(url, max(0, size - (1 << 18)), size - 1)
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise ModelError("zip 尾部找不到 EOCD")
    count = struct.unpack("<H", tail[i + 10:i + 12])[0]
    cdsize = struct.unpack("<I", tail[i + 12:i + 16])[0]
    cdoff = struct.unpack("<I", tail[i + 16:i + 20])[0]
    cd = _curl_range(url, cdoff, cdoff + cdsize - 1) if cdsize > len(tail) - i - 22 else tail[i + 22:]
    out, p = {}, 0
    while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        method = struct.unpack("<H", cd[p + 10:p + 12])[0]
        crc = struct.unpack("<I", cd[p + 16:p + 20])[0]
        csize = struct.unpack("<I", cd[p + 20:p + 24])[0]
        usize = struct.unpack("<I", cd[p + 24:p + 28])[0]
        nlen, elen, clen = struct.unpack("<HHH", cd[p + 28:p + 34])
        lho = struct.unpack("<I", cd[p + 42:p + 46])[0]
        out[cd[p + 46:p + 46 + nlen].decode("utf-8")] = dict(
            method=method, crc=crc, csize=csize, usize=usize, lho=lho)
        p += 46 + nlen + elen + clen
    if len(out) < count:
        raise ModelError(f"中央目录只解析出 {len(out)}/{count} 条")
    return out


def _check_space(need_bytes: int) -> None:
    """落盘前先看磁盘够不够，别下到一半把 eMMC 写满。

    写满是「下到 99% 突然失败、盘上多个 180 MB 的 .part、下次接着写还是满」，
    提前看一眼剩下的空间就能换成一句人话。尺寸拿不到（远端不报 content-length）
    就不拦，照样能下。
    """
    if need_bytes <= 0:
        return
    try:
        free = shutil.disk_usage(MODEL_DIR).free
    except OSError:
        return
    if free < need_bytes + SPACE_MARGIN:
        raise NoSpace(f"磁盘只剩 {free >> 20}MB，装不下 {need_bytes >> 20}MB 的模型："
                      f"腾地方，或用 FACE_MODEL_DIR 指到别的盘")


def fetch_member(zip_url: str, member: str, part: Path, key: str = "") -> Path:
    """从远程 zip 里按 range 抽出成员，边下边解压，CRC 不过就不算完成。"""
    chunk = Path(str(part) + ".chunk")     # 正在下的那一段，只为让进度实时可见
    for base in _bases(zip_url):
        try:
            size = remote_size(base)
            members = zip_members(base, size)
            if member not in members:
                raise ModelError(f"包里没这个成员，有：{list(members)[:8]}")
            e = members[member]
            lh = _curl_range(base, e["lho"], e["lho"] + 29)
            if lh[:4] != b"PK\x03\x04":
                raise ModelError("局部头损坏")
            nlen, elen = struct.unpack("<HH", lh[26:30])
            start = e["lho"] + 30 + nlen + elen
            _check_space(e["usize"])
            _set(key, part=str(part), chunk=str(chunk), bytes_total=e["csize"], bytes_done=0)
            get = zlib.decompressobj(-15) if e["method"] == 8 else None
            done = 0
            part.parent.mkdir(parents=True, exist_ok=True)
            with open(part, "wb") as f:
                while done < e["csize"]:
                    n = min(CHUNK, e["csize"] - done)
                    data = _curl_range(base, start + done, start + done + n - 1, sink=chunk)
                    f.write(get.decompress(data) if get else data)
                    done += n
                    _set(key, bytes_done=done)       # 压缩字节数：和 total 同一口径，进度单调
                f.write(get.flush() if get else b"")
            _set(key, chunk=None)           # 往后的进度回到 .part，别让这段字节重复计数
            chunk.unlink(missing_ok=True)
            if part.stat().st_size != e["usize"]:
                raise ModelError(f"解出 {part.stat().st_size} 字节，期望 {e['usize']}")
            crc = 0
            with open(part, "rb") as f:
                while block := f.read(1 << 20):
                    crc = zlib.crc32(block, crc)
            if crc & 0xFFFFFFFF != e["crc"]:
                part.unlink(missing_ok=True)
                _set(key, bytes_done=0)
                raise ModelError("CRC 校验不通过")
            return part
        except NoSpace:
            raise
        except Exception as exc:                     # 换下一个镜像重来
            _set(key, chunk=None)
            chunk.unlink(missing_ok=True)
            print(f"    {base[:40]}… 失败：{exc}")
    raise ModelError(f"{member} 抽取失败")


def fetch_direct(url: str, part: Path, key: str = "") -> Path:
    """单个 .onnx 直链，断点续传 + 重试。"""
    last = ""
    for base in _bases(url):
        try:
            total = 0
            try:
                total = remote_size(base)
            except ModelError:
                pass                                  # 报不出尺寸也要能下，只是没百分比
            _check_space(total)
            part.parent.mkdir(parents=True, exist_ok=True)
            _set(key, part=str(part), bytes_total=total,
                 bytes_done=part.stat().st_size if part.exists() else 0)
            p = _run(["curl", "-L", "--fail", "-C", "-", "--retry", "8", "--retry-delay", "3",
                      "--retry-all-errors", "--connect-timeout", "20", *_stall_flags(),
                      "-H", "User-Agent: curl/8.7.1", "-o", str(part), base], 7200)
            if p.returncode != 0:
                raise ModelError((p.stderr.decode("utf-8", "ignore").strip()
                                  or f"curl rc={p.returncode}")[-300:])
            return part
        except NoSpace:
            raise
        except Exception as exc:
            last = str(exc)
            print(f"    {base[:40]}… 失败：{exc}")
    raise ModelError(f"下载失败：{last or url}")


def _verify_and_install(key: str, cfg: dict, part: Path) -> Path:
    """校验临时文件，通过才原子换到正式名上。

    顺序很要紧：先下新的、验完再 replace。原先是「校验不过就删掉原件让用户重来」，
    网络一断就是原来的文件也没了、新的也没下来，整个模型直接消失；正式名上
    始终留着旧文件，替换是同一文件系统上的 rename，读者只会看到旧的那份或新的那份。
    """
    out = MODEL_DIR / cfg["file"]
    want = cfg.get("sha256")
    got = sha256_of(part)
    if want and got != want:
        part.unlink(missing_ok=True)      # 半截文件留着下次续传也还是坏的，直接扔
        _set(key, bytes_done=0)
        raise ModelError(f"SHA256 不符：{got[:16]}… != {want[:16]}…（下到的不是登记的那个模型）")
    with open(part, "rb") as f:           # curl 只 close 不 fsync，板子断电会留下
        try:                              # 目录项在、数据没落盘的「假就位」
            os.fsync(f.fileno())
        except OSError:
            pass
    part.replace(out)
    return out


# ------------------------------------------------------------- 对外：补一个模型

def _ensure_locked(key: str, force: bool, quiet: bool = False) -> str:
    """调用方必须已持有该模型的锁。返回 skip / ok / wait。"""
    cfg = MODELS[key]
    if not force and not needed(key):
        _set(key, state="ready", error=None, part=None, chunk=None)
        return "skip"
    src = _source(cfg)
    if src is None:
        _set(key, state="wait", error="地址未配置", part=None)
        return "wait"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        tries = int(_STATES.get(key, {}).get("tries") or 0) + 1
    _set(key, state="download", started=time.time(), ended=None, error=None,
         bytes_done=0, bytes_total=0, chunk=None, tries=tries)
    _note(f"  下载 {key}（{cfg['file']}）…", quiet)
    try:
        part = (fetch_direct(src[1], _part(cfg), key) if src[0] == "direct"
                else fetch_member(src[1], src[2], _part(cfg), key))
        _set(key, state="verify")
        out = _verify_and_install(key, cfg, part)
    except Exception as exc:
        _set(key, state="failed", error=str(exc)[:300], ended=time.time(),
             part=None, chunk=None, failed_at=time.time())
        raise ModelError(f"{key} 下载失败：{exc}") from exc
    size = out.stat().st_size
    _set(key, state="done", bytes_done=size, bytes_total=size, part=None,
         chunk=None, error=None, ended=time.time())
    _note(f"  完成 {key}：{size >> 20}MB sha256={sha256_of(out)[:12]}…", quiet)
    return "ok"


def _part(cfg: dict) -> Path:
    return MODEL_DIR / (cfg["file"] + ".part")


def _key_lock(key: str) -> threading.Lock:
    """一个模型一把锁。dict.setdefault 在 CPython 里是原子的，够撑住这个用法。"""
    return _KEYS.setdefault(key, threading.Lock())


def ensure(key: str, wait: bool = True, force: bool = False, quiet: bool = False) -> str:
    """把模型补到「就位」。wait=False 只保证有人在下，不等它下完。

    force=True 是「不管在没就位都重下一遍」（models/fetch.sh --force），
    跟 start(force=True) 那个 force 不是一回事 —— 那边只是「别管开关和冷却」。
    """
    if key not in MODELS:
        raise ModelError(f"模型 {key} 没登记（app/config.MODELS）")
    lk = _key_lock(key)
    if wait:
        with lk:
            return _ensure_locked(key, force, quiet)
    if not lk.acquire(blocking=False):
        return "download"           # 已经有线程在下了，别开第二路
    try:
        return _ensure_locked(key, force, quiet)
    finally:
        lk.release()


def start(key: str, force: bool = False, quiet: bool = True) -> dict | None:
    """后台补一个模型，立刻返回状态。Web 服务和并发场景用这个。

    force=True 表示「用户明说要下这一个」：越过 FACE_AUTO_FETCH 开关和失败冷却，
    但**不**越过「文件已就位且校验通过」—— 手动来一次也不该把 180 MB 重下，
    那是 ensure(force=True) 的活。

    锁在这里是交棒而不是互斥一下：主线程抢到锁就把锁交给发起线程，发起线程
    跑完再放。中间不能有缝，否则两轮轮询会各起一个下载线程，第二个虽然最后
    被 needed() 挡掉，可状态已被第一个改成 download，界面上就成了同一个模型
    两个进度条互相打架。
    """
    if key not in MODELS:
        return None
    if not force and not auto_fetch():
        return None
    lk = _key_lock(key)
    if not needed(key):
        _set(key, state="ready", error=None, part=None, chunk=None)
        return status(key)
    st = status(key)
    if st and st.get("state") in ("queued", "download", "verify"):
        return st
    if not force and st and st.get("state") == "failed" and st.get("retry_in"):
        return st                                   # 冷却期内不反复敲同一个镜像
    if not lk.acquire(blocking=False):
        return status(key)
    _set(key, state="queued", started=time.time(), error=None, chunk=None)
    threading.Thread(target=_worker, args=(key, lk, quiet, False), daemon=True,
                     name=f"model-fetch-{key}").start()
    return status(key)


def _worker(key: str, lk: threading.Lock, quiet: bool, force: bool) -> None:
    try:
        _ensure_locked(key, force=force, quiet=quiet)
    except Exception:
        pass          # 原因已写进状态表，调用方按 describe() 说话，这里不再往外抛
    finally:
        lk.release()


def force_refetch(key: str, quiet: bool = False) -> dict | None:
    """不管在没就位都重下一遍（models/fetch.sh --force）。立刻返回。

    先抢锁再发起线程：状态必须在发起之前就写成 queued，否则调用方紧接着
    读一轮状态会看到上一轮的 ready / done，进度条一秒就「完成」了，
    而 curl 其实还没起步。
    """
    if key not in MODELS:
        raise ModelError(f"模型 {key} 没登记（app/config.MODELS）")
    lk = _key_lock(key)
    lk.acquire()                      # 撞上下一个键的重下就排队，别两个 curl 抢同一份带宽
    _set(key, state="queued", started=time.time(), error=None, chunk=None)
    threading.Thread(target=_worker, args=(key, lk, quiet, True), daemon=True,
                     name=f"model-fetch-{key}").start()
    return status(key)


def prefetch(keys: list[str]) -> list[str]:
    """把缺的几个模型发起后台下载，返回真的发起了哪几个（已就位的不在里面）。"""
    out = []
    for key in keys:
        st = start(key)
        if st and st.get("state") in ("queued", "download", "verify"):
            out.append(key)
    return out


def watch(keys: list[str], show: bool = True, timeout: float | None = None) -> dict[str, str]:
    """盯状态表打进度，一直等到这批模型都到终态。自己不发起下载。

    返回 {模型: ready|failed|wait}。不抛异常 —— 命令行一次要等识别 + 检测两个，
    第一个失败也该让第二个下完再一起报，否则重试一次前面下好的又白跑了。
    """
    keys = [k for k in keys if k in MODELS]
    done: dict[str, str] = {}
    t0 = time.time()
    last = ""
    while len(done) < len(keys):
        for key in keys:
            if key in done:
                continue
            state = (status(key) or {}).get("state")
            if state in ("ready", "done"):
                done[key] = "ready"
            elif state in ("failed", "wait"):
                done[key] = state
            elif state is None:
                # 状态表里压根没这个键 = 没人认领它（发起线程没起来），别永久等下去
                done[key] = "failed"
            # queued / download / verify 是进行中，继续等，不能当终态结算
        if show:
            bits = []
            for k in keys:
                if k in done:
                    bits.append(f"{k} {done[k]}")
                else:
                    d = describe(k)
                    bits.append(f"{k} {d[len(f'模型 {k} '):] if d.startswith(f'模型 {k} ') else d}")
            line = "  " + " · ".join(bits)
            if line != last:
                print("\r" + line[:200], end="", flush=True)
                last = line
        if len(done) >= len(keys):
            break
        if timeout and time.time() - t0 > timeout:
            for key in keys:
                done.setdefault(key, "wait")      # 超时时仍在下载，报「没等到」而不是「下坏了」
            break
        time.sleep(POLL_S)
    if show and last:
        print()
    return done


def wait(keys: list[str], show: bool = True, timeout: float | None = None) -> dict[str, str]:
    """发起下载并等这批模型就位（命令行用）。

    start(force=True)：命令行是用户明着要下的，FACE_AUTO_FETCH=0
    和上一轮的失败冷却都不该拦路。
    """
    keys = [k for k in keys if k in MODELS]
    for key in keys:
        start(key, force=True, quiet=True)
    return watch(keys, show=show, timeout=timeout)


def report(keys: list[str] | None = None) -> tuple[int, int]:
    """命令行汇总：返回 (就位数, 总数)。"""
    keys = keys or list(MODELS)
    return sum(1 for k in keys if not needed(k)), len(keys)
