#!/usr/bin/env python3
"""命令行补齐 models/ 里的模型：bash models/fetch.sh [模型名…] [--force] [--list]

真正的逻辑在 app/fetch.py —— Web 服务首启自动下载走的是同一个模块，
校验、镜像回退、磁盘余量检查、进度都在那边，这里只管参数和终端输出。
只用标准库 + curl，装依赖之前的系统 python3 就能跑。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import fetch                              # noqa: E402
from app.config import MODELS, MODEL_DIR, sha256_of  # noqa: E402


def _list() -> None:
    print(f"{'模型':<12} {'用途':<12} {'文件':<20} 状态")
    for key, cfg in MODELS.items():
        p = MODEL_DIR / cfg["file"]
        if not p.exists():
            state = "未下载" if fetch._source(cfg) else "未配地址"
        elif cfg.get("sha256") and sha256_of(p) != cfg["sha256"]:
            state = "校验不符（重下会修好）"
        else:
            state = f"已就位 {p.stat().st_size >> 20}MB"
        print(f"{key:<12} {cfg['kind']:<12} {cfg['file']:<20} {state}")


def main(argv: list[str]) -> int:
    args = [a for a in argv if a.strip()]
    if "--help" in args or "-h" in args:
        print(__doc__)
        return 0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if "--list" in args:
        _list()
        return 0
    force = "--force" in args
    keys = [a for a in args if not a.startswith("-")]
    unknown = [k for k in keys if k not in MODELS]
    if unknown:
        print(f"没登记的模型：{'、'.join(unknown)}（可选：{'、'.join(MODELS)}）")
        return 2
    keys = keys or list(MODELS)

    print(f"目标：{' '.join(keys)}   目录：{MODEL_DIR}"
          + ("   （--force：全部重下）" if force else ""))
    if force:
        for key in keys:
            fetch.force_refetch(key)
        states = fetch.watch(keys)
    else:
        states = fetch.wait(keys)

    bad = 0
    for key in keys:
        state = states.get(key, "failed")
        if state == "ready":
            p = MODEL_DIR / MODELS[key]["file"]
            print(f"  ✓ {key}：{p.name} {p.stat().st_size >> 20}MB "
                  f"sha256={sha256_of(p)[:12]}…")
        elif state == "wait":
            print(f"  – {key}：{fetch.describe(key)}")
        else:
            bad += 1
            print(f"  ✗ {key}：{fetch.describe(key)}")
    ok, tot = fetch.report(keys)
    print(f"\n{ok}/{tot} 就位" if ok < tot else "\n全部就位")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
