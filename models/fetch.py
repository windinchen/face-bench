#!/usr/bin/env python3
"""按 app/config.py 的模型登记表补齐缺失模型，已就位且 SHA256 通过就跳过。

两种来源：
  url        单个 .onnx 直链（hf-mirror），断点续传直接下。
  zip+member 官方只以 zip 发行的成员，用 HTTP range 只抽那一段字节 ——
             det_2.5g 只有 3 MB，没必要拉 276 MB 的 buffalo_m.zip 整包。
             解压时按 zip 中央目录里的 CRC 和原始尺寸逐块复核。

github 直连在国内常抽风（连接超时或 404），失败自动换镜像站重试。
只用标准库 + curl，不需要装任何东西，系统 python3 就能跑。
"""
from __future__ import annotations

import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import MODELS, MODEL_DIR, sha256_of  # noqa: E402

# github 直连失败时依次尝试的镜像前缀（前者快，后者兜底）
MIRRORS = ("", "https://ghfast.top/", "https://gh-proxy.com/")
CHUNK = 16 << 20


def _run(args: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, timeout=timeout)


def _curl_range(url: str, start: int, end: int, retries: int = 4) -> bytes:
    """取 [start, end] 闭区间字节；长度不符就重试（镜像偶尔截断）。"""
    want = end - start + 1
    last = ""
    for attempt in range(retries):
        p = _run(["curl", "-sL", "--fail", "--connect-timeout", "20", "-m", "900",
                  "-H", "User-Agent: curl/8.7.1", "--range", f"{start}-{end}", url], 960)
        if p.returncode == 0 and len(p.stdout) == want:
            return p.stdout
        last = f"rc={p.returncode} got={len(p.stdout)} want={want}"
        time.sleep(2 + attempt)
    raise RuntimeError(f"分段下载失败 {start}-{end}：{last}")


def _try_bases(url: str):
    """github 的包按镜像顺序试，其他地址原样返回。"""
    if "github.com" in url:
        return [m + url for m in MIRRORS]
    return [url]


def remote_size(url: str) -> int:
    for u in _try_bases(url):
        p = _run(["curl", "-sIL", "--fail", "--connect-timeout", "20", "-m", "120",
                  "-H", "User-Agent: curl/8.7.1", u], 130)
        if p.returncode != 0:
            continue
        size = None
        for line in p.stdout.decode("latin1").splitlines():
            line = line.strip().lower()
            if line.startswith("content-range:"):
                size = int(line.rsplit("/", 1)[-1])
            elif line.startswith("content-length:") and size is None:
                size = int(line.split(":", 1)[1])
        if size:
            return size
    raise RuntimeError(f"拿不到远端尺寸：{url}")


def zip_members(url: str, size: int) -> dict[str, dict]:
    """只读 zip 尾部的中央目录，返回 {成员名: 位置与校验信息}。"""
    tail = _curl_range(url, max(0, size - (1 << 18)), size - 1)
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise RuntimeError("zip 尾部找不到 EOCD")
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
        raise RuntimeError(f"中央目录只解析出 {len(out)}/{count} 条")
    return out


def fetch_member(zip_url: str, member: str, out: Path) -> int:
    """从远程 zip 里按 range 抽出成员，边下边解压，CRC 不过就不落盘。"""
    bases = _try_bases(zip_url)
    for base in bases:
        try:
            size = remote_size(base)
            members = zip_members(base, size)
            if member not in members:
                raise RuntimeError(f"包里没这个成员，有：{list(members)[:8]}")
            e = members[member]
            lh = _curl_range(base, e["lho"], e["lho"] + 29)
            if lh[:4] != b"PK\x03\x04":
                raise RuntimeError("局部头损坏")
            nlen, elen = struct.unpack("<HH", lh[26:30])
            start = e["lho"] + 30 + nlen + elen
            get = zlib.decompressobj(-15) if e["method"] == 8 else None
            tmp = out.with_suffix(out.suffix + ".part")
            done = t0 = 0
            t0 = time.time()
            with open(tmp, "wb") as f:
                while done < e["csize"]:
                    n = min(CHUNK, e["csize"] - done)
                    data = _curl_range(base, start + done, start + done + n - 1)
                    f.write(get.decompress(data) if get else data)
                    done += n
                    print(f"\r    {out.name} {done * 100 // max(e['csize'], 1)}% "
                          f"({done >> 20}MiB/{e['csize'] >> 20}MiB, "
                          f"{done / max(time.time() - t0, .001) / 1e6:.1f}MB/s)   ",
                          end="", flush=True)
                f.write(get.flush() if get else b"")
            print()
            if tmp.stat().st_size != e["usize"]:
                raise RuntimeError(f"解出 {tmp.stat().st_size} 字节，期望 {e['usize']}")
            crc = 0
            with open(tmp, "rb") as f:
                while block := f.read(1 << 20):
                    crc = zlib.crc32(block, crc)
            if crc & 0xFFFFFFFF != e["crc"]:
                tmp.unlink(missing_ok=True)
                raise RuntimeError("CRC 校验不通过")
            tmp.replace(out)
            return e["usize"]
        except Exception as exc:  # 换下一个镜像重来
            print(f"    {base[:40]}… 失败：{exc}")
    raise RuntimeError(f"{member} 抽取失败")


def fetch_direct(url: str, out: Path) -> int:
    """单个 .onnx 直链，断点续传 + 重试。"""
    tmp = out.with_suffix(out.suffix + ".part")
    p = _run(["curl", "-L", "--fail", "-C", "-", "--retry", "8", "--retry-delay", "3",
              "--retry-all-errors", "--connect-timeout", "20", "-H", "User-Agent: curl/8.7.1",
              "-#", "-o", str(tmp), url], 7200)
    if p.returncode != 0:
        raise RuntimeError((p.stderr.decode("utf-8", "ignore").strip() or f"curl rc={p.returncode}")[-300:])
    tmp.replace(out)
    return out.stat().st_size


def ensure(key: str, cfg: dict) -> tuple[str, str]:
    out = MODEL_DIR / cfg["file"]
    want = cfg.get("sha256")
    if out.exists():
        got = sha256_of(out)
        if want is None or got == want:
            return "skip", f"已就位 {out.stat().st_size >> 20}MB"
        raise RuntimeError(f"SHA256 不符（可能是半截文件），删掉 {out} 重来")
    for d in ("url", "zip"):
        if cfg.get(d):
            size = (fetch_member(cfg["zip"], cfg["member"], out) if d == "zip"
                    else fetch_direct(cfg[d], out))
            got = sha256_of(out)
            if want and got != want:
                out.unlink(missing_ok=True)
                raise RuntimeError(f"SHA256 不符：{got} != {want}")
            return "ok", f"{size >> 20}MB sha256={got[:12]}…"
    return "wait", "地址未配置（占位，不参与评测）"


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bad = 0
    for key, cfg in MODELS.items():
        print(f"== {key} ({cfg['kind']}) {cfg['file']}")
        try:
            state, msg = ensure(key, cfg)
        except Exception as exc:
            state, msg = "fail", str(exc)
        print(f"  {state}: {msg}")
        bad += state == "fail"
    print(f"\n{len(MODELS) - bad}/{len(MODELS)} 就位" if bad else "\n全部就位")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
