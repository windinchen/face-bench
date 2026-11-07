"""评测编排：底库注册、探针 1:N 识别、耗时与规模基准。

底库约定是单人大头照（每人 1~N 张），探针帧里可能出现多个人，
因此探针侧对**每一张检测到的人脸**独立检索，并区分：
  · 主体人脸（真值对应的脸）→ 算 Rank-1 / 漏识 / 误识
  · 同框他人人脸 → 单独统计“被接受成了谁”，用于看多人场景下的误报
"""
from __future__ import annotations

import time

import numpy as np

from .config import DEFAULTS, UNKNOWN, UNKNOWN_LABEL
from .gallery import Gallery, Template, tid_for
from .imaging import encode_jpg, limit_side
from .metrics import probe_roc, sample_impostors, summarize, sweep_thresholds
from concurrent.futures import ThreadPoolExecutor


def _area(f) -> float:
    return float((f.box[2] - f.box[0]) * (f.box[3] - f.box[1]))


def enroll_image(backend, gallery: Gallery, img, person: str, source: str,
                 params: dict | None = None) -> list[dict]:
    """注册一张底库图：检测 -> 对齐 -> 提特征 -> 入库。返回逐条记录。

    底库图默认只取最大的一张人脸（证件照/头像里的主角），其余按“同图另有人脸”
    忽略并上报，避免把旁人存成这个人的模板。max_faces=0 表示全部保留。
    检测不到脸就直接上报失败（不静默丢弃），enroll_fallback_whole=1 时才按整图入库
    并标记为未对齐。每人模板数受 max_templates 限制。
    """
    p = {**DEFAULTS, **(params or {})}
    max_faces = int(p.get("max_faces", 1))
    cap = int(p.get("max_templates", 0))
    have = gallery.count(person)      # 走锁读：注册并行后裸读 persons 会读到半个数
    if cap and have >= cap:
        return [{"source": source, "person": person, "skipped": [f"已达每人模板上限 {cap} 张"]}]

    thresh = float(p["det_thresh"])
    # 底库默认必须检到人脸；enroll_fallback_whole=1 时允许整图兜底（未对齐，会被标注）
    mode = "auto" if int(p.get("enroll_fallback_whole", 0)) else "det"
    faces, t = backend.faces_from_image(img, mode=mode, det_size=int(p["det_size"]),
                                       thresh=thresh, nms=float(p["det_nms"]))
    if not faces:
        return [{"source": source, "person": person,
                 "skipped": [f"未检测到人脸(阈值{thresh:.2f})"]}]

    faces.sort(key=_area, reverse=True)
    out: list[dict] = []
    if max_faces > 0 and len(faces) > max_faces:
        for _ in faces[max_faces:]:
            out.append({"source": source, "person": person,
                        "skipped": [f"同图检测到 {len(faces)} 张脸，已忽略较小的人脸"]})
        faces = faces[:max_faces]
    room = (cap - have) if cap else len(faces)
    inserted: list[str] = []
    for i, f in enumerate(faces):
        if cap and i >= room:
            out.append({"source": source, "person": person,
                        "skipped": [f"超出每人模板上限 {cap} 张"]})
            continue
        # 是否做过关键点对齐 / 是否靠重试才检出，都从 quality 里读，不靠外部状态
        aligned = bool(f.quality.get("aligned", True))
        note = list(f.quality.get("reasons") or [])
        if not aligned:
            note.append("按整图入库，未做关键点对齐，建议重拍")
        q = {**f.quality, "reasons": note, "aligned": aligned,
             "ok": bool(f.quality.get("ok", True)) and aligned}
        vec, ems = backend.embed(f.crop)
        tid = tid_for(person, source) + (f"#{i}" if i else "")
        thumb = f"{tid}.jpg"
        gallery.thumbs_dir.mkdir(parents=True, exist_ok=True)
        (gallery.thumbs_dir / thumb).write_bytes(encode_jpg(f.crop, 80))
        try:
            stored = gallery.add(Template(tid=tid, person=person, source=source, vec=vec,
                                          box=[round(float(x), 1) for x in f.box],
                                          score=round(float(f.score), 4), quality=q, thumb=thumb),
                                 max_per_person=cap)
        except Exception:
            # 落库失败（gallery.db 只读、磁盘写满）时 add 现在是抛出来的。
            # 这里把刚写下去的缩略图一起带走再原样抛：不然库里没这个人、
            # 盘上却多一张谁都不引用的 JPG，几千张底库重试几轮就是几百 MB 孤儿。
            (gallery.thumbs_dir / thumb).unlink(missing_ok=True)
            raise
        if not stored:
            # 上限改在 gallery 锁内判定后，这里可能比上面那次预检更紧：
            # 并行注册时另一个 worker 刚好把最后一点名额占了。这条必须如实上报，
            # 并把刚写下去的缩略图一起带走 —— 否则底库没涨，磁盘上却多了一张
            # 谁都不引用的图，几千张底库反复重注册就是几百 MB 的静默泄漏。
            (gallery.thumbs_dir / thumb).unlink(missing_ok=True)
            out.append({"source": source, "person": person,
                        "skipped": [f"超出每人模板上限 {cap} 张"]})
            continue
        inserted.append(tid)
        out.append({"source": source, "person": person, "tid": tid,
                    "det_ms": round(t["detect"], 2), "embed_ms": round(ems, 2),
                    "box": [round(float(x), 1) for x in f.box],
                    "score": round(float(f.score), 4), "quality": q, "thumb": thumb})
    # 同一张底库图重新注册时，上一次多出来的模板必须一起带走。
    # 例：第一次「全部人脸」收了 3 张脸（base / base#1 / base#2），第二次改成只取主角，
    # 只会覆盖 base，另外两条旁人的脸仍然挂在这个人名下 —— 检索时会把他认成路人，
    # 底库人数也看不出异常。
    keep = set(inserted)
    stale = gallery.stale_tids(person, source, keep)
    if stale:
        try:
            gallery.remove_templates(stale)
        except Exception as e:
            # 清旧模板失败不该把这次注册整张判失败：新模板已经好好入库了，
            # 只是上一次多收的旁人还挂在名下。记一条待复查、写清该删谁，
            # 比让整张图变成「没注册上」准确 —— 重跑一遍也还是删不掉。
            out.append({"source": source, "person": person,
                        "skipped": [f"重新注册时旧模板没清掉（{type(e).__name__}: {e}），"
                                    "建议手动删除 " + "、".join(sorted(stale))]})
            return out
        for tid_ in stale:
            out.append({"source": source, "person": person,
                        "skipped": [f"重新注册时清掉了上一次的旧模板 {tid_}"]})
    return out


def _identify(backend, gallery: Gallery, crop, topk: int):
    vec, ems = backend.embed(crop)
    s0 = time.perf_counter()
    scores, persons = gallery.search(vec, topk=max(topk, 1))
    sms = (time.perf_counter() - s0) * 1000
    return vec, scores, persons, ems, sms


def auto_label(rec: dict, thr: float, prev_truth: str | None = None) -> None:
    """把图级真值落到具体某一张脸上。

    选哪张脸当主体只看「谁的 Top1 命中真值、并且分数最高」，**不看阈值**：
    阈值是放行判定线，一旦参与选主体，把滑杆拖高一点就没人「够格」，
    主体会从主角跳到旁边的路人脸上，同一份数据在「拖阈值」和「改真值」
    两条路上得出完全相反的结论（一个算漏识、一个算误识）。
    thr 参数保留是为了兼容老调用点，函数体不再依赖它。

    其余人脸默认留空，由用户在界面上逐张补标 —— 合影里的旁人本来就不该
    被默认当成主体。但留空只针对「系统自己认领的那张」：人工标过的旁人
    必须原样留着，否则改一次整图真值就悄悄抹掉几条脸级样本，分母变小还没提示。
    """
    truth = rec.get("truth")
    faces = rec.get("faces") or []
    cands = [f for f in faces if (f.get("top") or []) and f["top"][0]["person"] == truth] if truth else []
    # 多张脸都命中真值时取分数最高的那张（README 承诺的口径），不是面积最大的那张
    target = max(cands, key=lambda f: f["top"][0]["score"]) if cands else None
    if target is None and faces:
        target = faces[0]          # 一张都没命中：按「最大脸是主角」的约定兜给第一张
    for f in faces:
        if f is target and truth:
            f["truth"] = truth
            f["_truth_auto"] = truth
            continue
        cur = f.get("truth")
        if cur is None:
            continue
        # 只有「系统自动认领的」和「等于上一版整图真值的（老数据没打标记）」才回收，
        # 剩下的是人工标注的同框他人，保持不动
        if f.get("_truth_auto") or (prev_truth and cur == prev_truth):
            f["truth"] = None
            f.pop("_truth_auto", None)


def recompute(rec: dict, thr: float) -> dict:
    """按当前阈值与真值标签重算一张探针记录，纯比较、不重新推理。

    评测、人工改真值、拖阈值滑杆三条路径共用这里，保证图级/人脸级口径永远一致。
    输入只需要 rec["faces"][*]["top"]（已缓存的 Top-K 命中）和各自的 truth。
    """
    faces = rec.get("faces") or []
    truth = rec.get("truth")
    if not faces:
        if rec.get("verdict") != "error":
            rec["verdict"] = "no_face"
        return rec
    for f in faces:
        persons = [t["person"] for t in (f.get("top") or [])]
        scores = np.asarray([t["score"] for t in (f.get("top") or [])], np.float32)
        best = round(float(scores[0]), 4) if scores.size else None
        f["top1_person"] = persons[0] if persons else None
        f["top1_score"] = best
        f["cos"] = best
        f["accepted"] = bool(best is not None and best >= thr)
        ft = f.get("truth")
        f["is_impostor"] = ft == UNKNOWN
        if ft == UNKNOWN:
            # 非底库人员：库里没有他，期望是「不放行」。被认成库里某人就是误识
            f["truth_rank"] = None
            f["rank1_correct"] = bool(not f["accepted"])
        else:
            f["truth_rank"] = _truth_rank(persons, scores, ft) if ft else None
            f["rank1_correct"] = bool(f["accepted"] and ft and persons and persons[0] == ft)
    # 主体 = 标为该图真值的脸里分数最高的那张；一张都没标上就退到最大脸
    cands = [f for f in faces if truth and f.get("truth") == truth]
    subject = max(cands, key=lambda x: x.get("top1_score") or -1.0) if cands else faces[0]
    for f in faces:
        f["is_subject"] = f is subject
    # 图级判定只看「整图真值 + 主体脸的 Top1」，不能直接抄主体脸的 rank1_correct：
    # 用户可能在人脸下拉里把主体改成了「未标注」，此时脸级标签为空，
    # 照抄会把明明认对的人判成误识（误识表里出现「张三 → 张三」这种自相矛盾）。
    s_persons = [t["person"] for t in (subject.get("top") or [])]
    s_scores = np.asarray([t["score"] for t in (subject.get("top") or [])], np.float32)
    rec["is_impostor"] = truth == UNKNOWN
    rec["accepted"] = subject["accepted"]
    rec["top1_person"] = subject["top1_person"]
    rec["top1_score"] = subject["top1_score"]
    rec["cos"] = subject["cos"]
    if truth == UNKNOWN:
        rec["rank1_correct"] = bool(not subject["accepted"])
        rec["truth_rank"] = None
    elif truth:
        rec["rank1_correct"] = bool(subject["accepted"] and s_persons and s_persons[0] == truth)
        rec["truth_rank"] = _truth_rank(s_persons, s_scores, truth)
    else:
        rec["rank1_correct"] = False
        rec["truth_rank"] = None
    # 推理中途炸掉的那张不能被拖阈值 / 改真值洗白：recompute 只做比较、不重新推理，
    # 这种记录的脸是残缺的（比如合影五张脸只比完两张就抛了异常）。
    # 无条件按比较结果改判会让 n_error 莫名变小，而「到底有几张根本没评出来」
    # 恰恰是查坏图时要看的那个数。上面 `if not faces` 的分支早就留了这道闸，
    # 有脸的那条路径漏了。
    if rec.get("verdict") != "error":
        rec["verdict"] = _verdict(rec, bool(truth))
    rec["extra_accepted"] = [
        {"idx": x["idx"], "person": x["top1_person"], "score": x["top1_score"]}
        for x in faces if not x["is_subject"] and x["accepted"]
    ]
    return rec


def _probe_one(backend, gallery: Gallery, p: dict, name, img, truth,
               crop_saver=None) -> dict:
    """跑一张探针图，返回一条记录。不碰任何共享状态，所以可以并行调用。"""
    thr = float(p["threshold"])
    topk = int(p["topk"])
    mode = p["probe_mode"] if p["probe_mode"] in ("auto", "det", "raw") else "auto"
    all_faces = bool(int(p.get("all_faces", 1)))
    t0 = time.perf_counter()
    rec = {"name": name, "truth": truth or None, "faces_found": 0, "detected": False,
           "faces": [], "verdict": None}
    try:
        if img is None:
            # 上游解码就失败了（坏图 / 不是图片）：别拿一张纯色图去喂检测器，
            # 那会白白产生一条「检不到脸」，把检出率和耗时分布一起带偏
            rec["error"] = "图片解码失败（文件损坏、格式不支持或下载不完整）"
            rec["verdict"] = "error"
            rec["timings"] = _timings({}, 0.0, 0.0, t0, 0)
            return rec
        faces, t = backend.faces_from_image(
            img, mode=mode, det_size=int(p["det_size"]), thresh=float(p["det_thresh"]),
            nms=float(p["det_nms"]), min_face=int(p.get("min_face", 0)),
        )
        if not faces:
            rec["verdict"] = "no_face"
            rec["error"] = "未检测到人脸"
            rec["timings"] = _timings(t, 0.0, 0.0, t0, 0)
            return rec
        rec["faces_found"] = len(faces)
        # 「整图兜底」那张脸不是检测器认出来的：探针是摄像头抓拍时，上游常裁出
        # 草地、衣领，兜底会让检出率恒等于 100%，把一批坏输入全藏起来。
        rec["detected"] = mode == "raw" or any(
            f.quality.get("aligned", True) for f in faces)
        rec["fallback"] = bool(faces) and not rec["detected"]
        faces.sort(key=_area, reverse=True)
        if not all_faces:
            faces = faces[:1]
        ems_all = sms_all = 0.0
        for i, f in enumerate(faces):
            _, scores, persons, ems, sms = _identify(backend, gallery, f.crop, topk)
            ems_all += ems
            sms_all += sms
            sub = {"idx": i,
                   "box": [round(float(x), 1) for x in f.box],
                   "det_score": round(float(f.score), 4),
                   "truth": None,
                   "top": [{"person": persons[j], "score": round(float(scores[j]), 4)}
                           for j in range(len(persons))],
                   "quality": f.quality}
            if crop_saver is not None:
                sub["thumb"] = crop_saver(name, i, f.crop)
            rec["faces"].append(sub)
        auto_label(rec, thr)
        recompute(rec, thr)
        rec["timings"] = _timings(t, ems_all, sms_all, t0, len(rec["faces"]))
    except Exception as e:  # 单张失败不影响整批
        if not rec.get("error"):
            rec["error"] = f"{type(e).__name__}: {e}"
        rec["verdict"] = "error"
        rec.setdefault("timings", _timings({}, 0.0, 0.0, t0, 0))
    return rec


def run_probe(backend, gallery: Gallery, items: list[tuple[str, object, str | None]],
              params: dict | None = None, crop_saver=None) -> dict:
    """跑一批探针。items = [(名称, BGR图, 真值 or None)]，每张图可含多张人脸。

    params["workers"] > 1 时用线程池并行推理：检测/提特征都在 onnxruntime 里
    跑，run() 会放开 GIL，所以多线程在多核机器上是真加速（本机 10 核实测
    串行 23.9 → 5 路 41.6 张/秒）。记录顺序仍按输入顺序返回，
    界面的序号、缩略图命名、分页都依赖这个顺序。

    并行时单张图的 total_ms 会有所上升（几个 worker 抢同一批核），
    所以吞吐一律按整批墙钟算，别再用 1000/P50 折算。
    """
    p = {**DEFAULTS, **(params or {})}
    thr = float(p["threshold"])
    topk = int(p["topk"])
    workers = max(1, int(p.get("workers", 1) or 1))
    t0 = time.perf_counter()
    job = lambda it: _probe_one(backend, gallery, p, it[0], it[1], it[2], crop_saver)  # noqa: E731
    if workers > 1 and len(items) > 1:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="probe") as ex:
            records = list(ex.map(job, items))
    else:
        records = [job(it) for it in items]
    wall = time.perf_counter() - t0
    return {"records": records, "wall_s": round(wall, 3),
            "summary": summarize(records, thr, topk, wall_s=wall, workers=workers),
            "params": p}


def _verdict(rec, has_truth) -> str:
    """图级结论。读的是 rec 上刚算好的图级字段，不是脸级标签。"""
    if not has_truth:
        return "no_truth"
    if rec.get("truth") == UNKNOWN:
        return "correct_unknown" if rec["rank1_correct"] else "wrong"
    if rec["rank1_correct"]:
        return "correct"
    return "wrong" if rec["accepted"] else "reject"


def _truth_rank(persons, scores, truth):
    """真值在 Top-k 候选里的名次。

    刻意不卡阈值：Recall@k 衡量的是检索质量（真值有没有被捞进前 k），
    和放行阈值无关。带上阈值会让界面上拖滑杆时 Recall@k 跟着乱跳。
    """
    for i, p in enumerate(persons):
        if p == truth:
            return i + 1
    return None


def _timings(t, embed_ms, search_ms, t0, n_faces):
    total = (time.perf_counter() - t0) * 1000
    return {
        "total_ms": round(total, 3),
        "detect_ms": round(float(t.get("detect", 0.0) or 0.0), 3),
        "align_ms": round(float(t.get("align", 0.0) or 0.0), 3),
        "embed_ms": round(float(embed_ms), 3),
        "search_ms": round(float(search_ms), 3),
        "n_faces": int(n_faces),
    }


def search_scale_curve(gallery: Gallery, n_probes: int = 40,
                       sizes=(256, 512, 1024, 2048, 4096, 8192, 16384)) -> list[dict]:
    """底库规模 -> 1:N 检索耗时曲线。库不足时用有放回复制放大。"""
    if gallery.matrix.size == 0:
        return []
    base = gallery.matrix
    n = base.shape[0]
    rng = np.random.default_rng(7)
    q = rng.standard_normal((n_probes, base.shape[1]), dtype=np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    out = []
    for s in sizes:
        reps = int(np.ceil(s / n))
        big = base[rng.integers(0, n, size=(reps * n,))][:s] if reps > 1 else base[:min(s, n)]
        pid = rng.integers(0, max(2, big.shape[0] // 2), big.shape[0]).astype(np.int64)
        nu = int(pid.max()) + 1
        ms = []
        for v in q:
            t1 = time.perf_counter()
            sc = big @ v
            best = np.full(nu, np.float32(-1.0))
            # 与 search() 用同一个 np.maximum.at 做按人聚合。以前写的是
            # best[pid[np.argsort(sc)]] = sc[np.argsort(sc)]，两次全量 argsort，
            # 16k 库实测 2.02 ms，而 maximum.at 只要 0.35 ms（差约 5 倍）——
            # 曲线比真实检索慢这么多，README 的选型头条数字就是虚的。
            np.maximum.at(best, pid, sc)
            k = min(5, nu)
            part = np.argpartition(-best, k - 1)[:k]
            _ = best[part]
            ms.append((time.perf_counter() - t1) * 1000)
        out.append({"size": int(big.shape[0]),
                    "p50_ms": round(float(np.percentile(ms, 50)), 4),
                    "p95_ms": round(float(np.percentile(ms, 95)), 4)})
    return out


_NO_PROBE_INFO = {"probe_faces": 0, "probe_genuine_n": 0, "probe_impostor_n": 0,
                  "probe_censored": 0, "probe_unlabeled_faces": 0, "probe_unknown_faces": 0}
_NO_IMPOSTOR_INFO = {"impostor_n": 0, "impostor_pairs": 0, "impostor_persons": 0,
                     "pair_basis": "per-person"}


def _gallery_imp_info(imp_info: dict) -> dict:
    """底库那侧的抽样交代，但 impostor_n 让位给合并后的总数。

    sample_impostors 的 impostor_n 说的是「底库抽了多少对」，直接 update 进来会把
    sweep 算出的合并总数顶掉：曲线明明吃了 12 个底库对 + 3 个探针冒认 = 15 个样本，
    报告却写 12，前端按 far × impostor_n 复核命中数就对不上。
    底库那一份另有 impostor_from_gallery 说。
    """
    return {k: v for k, v in imp_info.items() if k != "impostor_n"}


def _threshold_note(gen_g: int, imp_info: dict, pinfo: dict) -> str:
    """交代曲线是根据哪些样本算的、缺口在哪，别只丢一句「跑探针即可生成」。"""
    pg, pi, pc = pinfo["probe_genuine_n"], pinfo["probe_impostor_n"], pinfo["probe_censored"]
    if not (gen_g or pg):
        return ("估不出「同一人互配」分布：底库里没有人有多张模板，这批探针里也没有可用的"
                "带真值人脸（真值需命中 Top-K）；给几个人再注册一张不同场合的照片，"
                "或把探针的真值标全即可生成")
    g_src = ([f"底库同人互配 {gen_g} 对"] if gen_g else []) + \
            ([f"探针真值命中 {pg} 张"] if pg else [])
    i_src = ([f"底库跨人抽样 {imp_info.get('impostor_n', 0)} 对"
              f"（{imp_info.get('impostor_pairs', 0)} 个异人对）"]
             if imp_info.get("impostor_n") else []) + \
            ([f"探针冒认 {pi} 张"] if pi else [])
    note = "依据 genuine=" + " + ".join(g_src) + "；impostor=" + (" + ".join(i_src) or "无") + "。"
    if gen_g and pg:
        note += " 底库互配多是同场合照片、偏乐观，以探针那部分为准。"
    if pc:
        note += f" 另有 {pc} 张脸的真值没进 Top-K，分数只知道低于第 K 名，按删失处理未计入分布。"
    return note


def threshold_report(backend, gallery: Gallery, params: dict | None = None,
                     use_cache: bool = True, records: list | None = None) -> dict:
    """阈值建议：底库自身分布 + 这批带真值探针，合并成一份经验分布。

    genuine（同一人互配）与 impostor（异人冒认）各有两个来源：
      · 底库侧 —— 同一人多张模板互配 / 跨人模板对（按人聚合，口径同 search）；
      · 探针侧 —— 标了真值且真值命中 Top-K 的脸给出同人分，它对非真值人员的最高分给出冒认分
        （见 metrics.probe_roc，两边都和 per_person=True 对齐，不是另起一套口径）。

    底库每人只有 1 张模板时（README 的推荐用法，也是本机那份 3475 人底库的实际形态）
    底库侧凑不出任何同人互配对，以前整栏「阈值建议」是空的，而提示语让用户
    「跑一批带真值的探针即可生成」—— 探针其实压根没参与计算，照着跑完还是空的。
    现在探针参与，跑完带真值的批次这一栏就有数了。

    分布只跟底库和这批探针有关、跟阈值无关，所以底库那一份按版本号缓存（含纯底库的报告，
    拖滑杆和翻旧批次直接命中，省掉二十万点的一次扫描）；探针样本每次现并，几毫秒。

    遍历 persons 必须在 gallery.lock 里做快照：后台注册线程正在 add()，
    裸遍历会撞「dictionary changed size during iteration」，
    而且是几千张推理全跑完之后再炸，整批结果白丢。
    """
    # 版本号和快照必须在同一次持锁里取。以前先在锁外读 version 当缓存键、
    # 再进锁拷 matrix，中间插进一次 add() 的话，缓存键记的是 N、内容却是 N+1，
    # 上层拿到的「推荐阈值」就永远比界面上的底库少一张模板，且不会自己纠正。
    base = None
    with gallery.lock:
        key = (gallery.model_key, gallery.version, gallery.matrix.shape[0])
        cached = getattr(gallery, "_thr_cache", None)
        if use_cache and cached and cached[0] == key:
            base = cached[1]
        else:
            matrix = gallery.matrix.copy()
            persons = [t.person for t in gallery.index]
            groups = {p: [t.vec for t in lst] for p, lst in gallery.persons.items()
                      if len(lst) > 1}
    if base is None:
        # 底下两步是二十万个模板对的量级，必须留在锁外算，
        # 否则后台注册会被卡住几百毫秒，前端看起来就是「注册卡住了」。
        # 不足 4 张模板时不抽底库样本，但探针样本照样能出曲线，所以不提前 return。
        if matrix.shape[0] < 4:
            imp0, imp_info = np.zeros((0,), np.float32), dict(_NO_IMPOSTOR_INFO)
            gen0: list[float] = []
        else:
            imp0, imp_info = sample_impostors(matrix, persons)
            gen0 = []
            for lst in groups.values():
                v = np.stack(lst)
                s = v @ v.T
                iu = np.triu_indices(len(lst), 1)
                gen0.extend(s[iu].tolist())
        rep0 = sweep_thresholds(gen0, imp0)
        rep0.update(_gallery_imp_info(imp_info))
        rep0.update(_NO_PROBE_INFO)
        rep0["basis"] = "gallery"
        rep0["genuine_from_gallery"] = len(gen0)
        rep0["impostor_from_gallery"] = int(imp_info.get("impostor_n", 0))
        rep0["note"] = _threshold_note(len(gen0), imp_info, _NO_PROBE_INFO)
        base = (gen0, imp0, imp_info, rep0)
        gallery._thr_cache = (key, base)

    gen0, imp0, imp_info, rep0 = base
    if not records:
        return rep0

    pgen, pimp, pinfo = probe_roc(records)
    if pgen.size == 0 and pimp.size == 0:
        return rep0
    rep = sweep_thresholds(np.concatenate([np.asarray(gen0, np.float32), pgen]),
                           np.concatenate([imp0, pimp]))
    rep.update(_gallery_imp_info(imp_info))
    rep["impostor_from_gallery"] = int(imp_info.get("impostor_n", 0))
    rep.update(pinfo)
    rep["basis"] = "gallery+probe"
    rep["genuine_from_gallery"] = len(gen0)
    rep["note"] = _threshold_note(len(gen0), imp_info, pinfo)
    return rep
