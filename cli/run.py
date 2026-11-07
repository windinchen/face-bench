#!/usr/bin/env python3
"""命令行跑一遍评测（不走 Web），适合接 CI 或在板子上批量回归。

用法：
  python -m cli.run --gallery /Users/windinchen/work/项目任务/人脸识别/_rp/face-bench/_tmp/demo/gallery --probe /Users/windinchen/work/项目任务/人脸识别/_rp/face-bench/_tmp/demo/probe \
                    --det-size 320 --threshold 0.4078 --mode auto

人员 ID 默认自动判别：底库里还有子目录就按「每人一个文件夹」认人，
平铺（一张图一个人）就按文件名认人。要强制哪种用 --gallery-mode / --probe-mode。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from app import bench
from app import fetch as model_fetch
from app.backend import build
from app.config import DEFAULTS, REPORT_DIR, infer_person, write_json
from app.gallery import Gallery
from app.imaging import decode, limit_side


IMG = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def person_of(p: Path, root: Path, mode: str = "auto") -> str | None:
    """认人。auto 看的是相对 --gallery/--probe 那一层的深度：

    <root>/<人员ID>/<图片> → 父目录名；<root>/<图片>（一张图一个人）→ 文件名。
    必须相对 root 判断，绝对路径的层数是没有意义的。
    """
    if mode == "auto":
        try:
            rel = p.relative_to(root)
        except ValueError:
            rel = Path(p.name)
        mode = "path" if len(rel.parts) >= 2 else "name"
    return infer_person(str(p), p.name, mode, "_")


def collect(root: Path):
    out = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMG and p.is_file():
            out.append(p)
    return out


def keep_template(person: str | None, per: dict[str, int], cap: int) -> bool:
    """这张底库图还该不该再给他注册一个模板。

    cap = 0 是「不限」，不是「一张都不许」—— DEFAULTS["max_templates"] 和
    `bench.enroll_image` 都是这个口径（那边写的是 `if cap and ...`）。
    写成 `per.get(person,0) >= cap` 的话，`--templates 0` 恒真，
    一个模板都不注册，探针整批「没匹配上」，屏幕上只写着「0 人 / 0 模板」。
    负数按不限处理：那是手滑，不是「一张都别存」。
    """
    cap = max(0, int(cap))     # 负数是手滑，按「不限」比按「一张都别存」讲得通
    return bool(person) and not (cap and per.get(person, 0) >= cap)


def cli_gallery_key(model_key: str) -> str:
    """命令行回归专用的底库键：和界面上登记的那份错开目录，互不覆盖。"""
    return f"cli_{model_key}"


def collect_required(root: Path, what: str) -> list[Path]:
    """取图片，一张都没有就当场退出。

    --gallery 路径写错时 rglob 只会安静地返回空（不抛异常），而 clear() 已经把库删了：
    结果是底库被清空、注册 0 人，然后若无其事地拿空库跑探针，整批记成「没匹配上」，
    屏幕上只留下一行「0 人 / 0 模板，0.0s」。删库之前就拦住，报错带上填错的路径。
    """
    out = collect(root)
    if not out:
        raise SystemExit(f"{what}里没有图片：{root}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--rec", default="w600k_r50")
    ap.add_argument("--det", default="det_10g")
    ap.add_argument("--mode", default="auto", choices=["auto", "det", "raw"])
    ap.add_argument("--det-size", type=int, default=320)
    ap.add_argument("--threshold", type=float, default=0.4078)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--templates", type=int, default=4,
                    help="每人最多模板数，0 = 不限（与界面/DEFAULTS 同口径）")
    modes = ["auto", "name", "path", "id_num", "prefix", "none"]
    ap.add_argument("--gallery-mode", default="auto", choices=modes,
                    help="底库认人方式：auto=自动(默认) name=文件名即人名 path=父目录名")
    ap.add_argument("--probe-mode-id", default="auto", choices=modes,
                    help="探针真值来源，含义同 --gallery-mode")
    a = ap.parse_args()

    # 缺模型就在开跑前补齐，并把进度打在终端上。build() 自己也会等（auto_fetch 默认开），
    # 但那副样子是「屏幕一片安静，三分钟后突然开始跑」，板子上看着就是卡死了。
    # 命令行是明着要跑批的，等下载合理，不说明才不合理。
    ready = model_fetch.wait([a.rec, a.det], show=True)
    if ready.get(a.rec) != "ready":
        raise SystemExit(f"识别模型没就位：{model_fetch.describe(a.rec)}")
    if a.mode != "raw" and ready.get(a.det) != "ready":
        # raw 根本不检测，缺检测模型照样能跑；auto/det 缺了它每张图都算 error，
        # 跑完十几分钟只换来一张检出率 0 的表，比当场报错坑人得多。
        raise SystemExit(f"检测模型没就位（--mode {a.mode} 要用它）：{model_fetch.describe(a.det)}")

    bk = build("onnx", rec_key=a.rec, det_key=a.det)
    # 用命令行专属的底库目录（data/gallery_cli_<key>），不碰界面上那份。
    # 以前 Gallery(a.rec) 拿到的就是 Web 在用的 data/gallery_w600k_r50，而 clear()
    # 现在是真的删文件（gallery.db/-wal/-shm、向量、全部缩略图）：跑一次命令行回归
    # 等于把用户在界面里注册的底库整个清掉、换成 --gallery 里的人，纯速度回归也要陪葬。
    g = Gallery(cli_gallery_key(a.rec))
    src = collect_required(Path(a.gallery), "底库目录")
    # 探针目录同样在跑之前拦一次。底库那侧已经拦了，探针写错路径还是照跑：
    # 空列表一路走到 summarize，只回一个 {"n": 0}，屏幕上是一片看着像
    # 「检出率 0%、识别全错」的汇总，而真正的原因只是路径少写了一层。
    probe_src = collect_required(Path(a.probe), "探针目录")
    cap = int(a.templates)
    g.clear()
    print(f"注册底库 {a.gallery} …")
    per: dict[str, int] = {}
    t0 = __import__("time").perf_counter()
    for p in src:
        person = person_of(p, Path(a.gallery), a.gallery_mode)
        if not keep_template(person, per, cap):
            continue
        try:
            bench.enroll_image(bk, g, limit_side(decode(p.read_bytes()), 1280), person, str(p))
        except Exception as e:
            # 底库里一张坏 JPEG（下载截断、0 字节、后缀是 .jpg 其实是 HTML）
            # 不该让整条命令行跑到一半崩掉：已经注册的向量全在内存里，
            # 崩了连下面那句 g.save() 都执行不到，等于前面几百张白跑。
            # 下面探针那个循环本来就是这么容错的，注册这边对齐口径。
            # 计数放在成功之后，坏掉的那张不占「每人最多几张」的名额。
            print(f"  跳过底库 {p}: {type(e).__name__}: {str(e)[:120]}")
            continue
        per[person] = per.get(person, 0) + 1
    g.save()
    el = __import__("time").perf_counter() - t0
    print(f"  {g.stats()['persons']} 人 / {g.stats()['templates']} 模板，{el:.1f}s "
          f"（{g.stats()['templates']/max(el,.001):.1f} 张/秒）")

    items = []
    for p in probe_src:
        try:
            items.append((str(p), limit_side(decode(p.read_bytes()), 1280),
                          person_of(p, Path(a.probe), a.probe_mode_id)))
        except Exception as e:
            print(f"  跳过 {p}: {e}")
    res = bench.run_probe(bk, g, items, {**DEFAULTS, "probe_mode": a.mode,
                                        "det_size": a.det_size, "threshold": a.threshold,
                                        "topk": a.topk})
    s = res["summary"]
    print("\n=== 汇总 ===")
    print(json.dumps({k: v for k, v in s.items() if k != "latency"}, ensure_ascii=False, indent=2))
    # 一张都没评出来时 metrics.summarize 只回 {"n": 0}，压根没有 latency 这一项
    # （探针目录写错、或者整目录都是坏图就是这个样子）。原来这里直接 s["latency"]
    # 抛 KeyError，把「路径填错了」这种真实原因埋在一串 traceback 底下。
    if "latency" in s:
        print("耗时 ms:", json.dumps(s["latency"], ensure_ascii=False))
    else:
        print("耗时 ms: —（这一批没有可统计的样本）")
    print("规模曲线:", json.dumps(bench.search_scale_curve(g), ensure_ascii=False))
    name = f"cli_{__import__('time').strftime('%Y%m%d_%H%M%S')}.json"
    write_json(REPORT_DIR / name, res)
    print("报告:", REPORT_DIR / name)


if __name__ == "__main__":
    main()
