"""评测指标：Rank-1 / Recall@k / TAR@FAR 阈值扫描 / 分阶段耗时分位数。"""
from __future__ import annotations

import numpy as np

from .config import UNKNOWN, UNKNOWN_LABEL


def pct(arr, qs=(50, 95, 99)):
    a = np.asarray([x for x in arr if x is not None and np.isfinite(x)], np.float64)
    if a.size == 0:
        # 空样本也要把 max 补齐：非空分支有这个键，少一个就让同一份报表的
        # 两组指标结构不一致，按固定键遍历的消费方（导出脚本）会 KeyError
        return {f"p{q}": None for q in qs} | {"mean": None, "max": None, "n": 0}
    return {
        f"p{q}": round(float(np.percentile(a, q)), 3) for q in qs
    } | {"mean": round(float(a.mean()), 3), "max": round(float(a.max()), 3), "n": int(a.size)}


def sample_impostors(matrix: np.ndarray, persons: list[str], max_pairs: int = 200_000,
                     seed: int = 20261106, per_person: bool = True):
    """抽「一个探针人脸 vs 另一个人」的分数对，估计冒认者分布（决定安全阈值）。

    口径刻意和 search(per_person=True) 对齐：先抽一个**人**当探针（从他名下随机一张
    模板出发），再抽另一个**人**当被比对者，取他与该人所有模板的最高分。
    以前抽的是「模板对」，一个人有 4 张模板时 far 是**模板对**误配率，
    而 gallery_far_1e4 的指数 9999 说的是**人**，量纲对不上，
    每人模板越多就越低估「某个人被放行」的真实概率（约 1-(1-p)^4）。

    抽样改成不放回：有放回时 39800 个不同对会被重复抽满 20 万条，
    far 的分辨率其实卡在 1/不同对数，报出来的 at_far_1e6 只是「零命中的最低阈值」。
    per_person=False 保留旧的模板对口径，便于对照两种算法差多少。

    返回 (scores, info)，info 里如实交代抽了多少个不同对、覆盖多少人。
    """
    info = {"impostor_n": 0, "impostor_pairs": 0, "impostor_persons": 0,
            "pair_basis": "per-person" if per_person else "per-template"}
    if matrix.shape[0] < 2 or not persons:
        return np.zeros((0,), np.float32), info
    rng = np.random.default_rng(seed)
    p = np.asarray(persons)
    if not per_person:
        out, got = [], 0
        for _ in range(300):        # 底库只有一个人时永远抽不到异人对，必须有循环上限
            if got >= max_pairs:
                break
            n = min(4096, max_pairs - got)
            i = rng.integers(0, matrix.shape[0], n)
            j = rng.integers(0, matrix.shape[0], n)
            keep = p[i] != p[j]
            if not keep.any():
                continue
            out.append(np.einsum("ij,ij->i", matrix[i[keep]], matrix[j[keep]]))
            got += int(keep.sum())
        s = np.concatenate(out) if out else np.zeros((0,), np.float32)
        info.update(impostor_n=int(s.size), impostor_pairs=int(s.size),
                    impostor_persons=int(len(set(persons))))
        return s, info

    # 每人一组模板，补齐成 (人数, 最多模板数, 维度)。空位填 -1.0 只是占位，
    # 它不是单位向量（模长是 sqrt(512)），所以取 max 之前必须按 counts 掩掉，见下面。
    uniq, inv = np.unique(p, return_inverse=True)
    n = int(uniq.size)
    info["impostor_persons"] = n
    if n < 2:
        return np.zeros((0,), np.float32), info
    mmax = int(np.bincount(inv).max())
    grp = np.full((n, mmax, matrix.shape[1]), -1.0, np.float32)
    counts = np.bincount(inv, minlength=n)
    start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    order = np.argsort(inv, kind="stable")          # 按人排好，槽位就是组内偏移
    grp[inv[order], np.arange(inv.size) - start[inv[order]]] = matrix[order]
    cap = n * (n - 1)               # 有序异人对总数（谁当探针、谁被比对是有方向的）
    want = min(int(max_pairs), cap)
    ids = np.zeros((0,), np.int64)
    # 不放回抽样：在 n×n 的编号空间里抽不重复的编号，扔掉对角线（同一人自己）
    if want >= cap:                 # 人能全枚举就全枚举，抽样反而覆盖不全
        ids = np.arange(n * n, dtype=np.int64)
        ids = ids[(ids // n) != (ids % n)]
    for _ in range(20):
        need = want - ids.size
        if need <= 0:
            break
        cand = np.unique(rng.integers(0, n * n, min(cap, need * 3 + 128)))
        cand = cand[(cand // n) != (cand % n)]
        ids = np.unique(np.concatenate([ids, cand]))[:want]
    if ids.size == 0:
        return np.zeros((0,), np.float32), info
    a, b = ids // n, ids % n
    # 探针取 a 名下随机一张模板；被比对侧对他名下所有模板取最高分（= search 的按人聚合）
    ta = grp[a, rng.integers(0, np.maximum(1, counts[a]))]
    d = np.einsum("ik,ijk->ij", ta, grp[b])
    # 补齐出来的空槽一定要掩掉再取 max。原来注释以为「-1 占位在 max 时自然落选」，
    # 但空槽是整条 -1.0：它的模长是 sqrt(512) 而不是 1，和探针的点积等于 -sum(ta)，
    # 根本不是余弦。本机用 w600k_r50 真实向量实测（800 人、每人 1~4 张不等）：
    # 冒认分最高 2.58、12.5% 的样本 > 1.0，而余弦的上界就是 1（掩掉之后是 1.0）。
    # 且模板数少于 mmax 的人人一进被比对侧就常常让空槽赢下 max。后果不是偏一点点：
    # far 虚高到阈值拉到 1.0 都压不住，sweep 里满足 far<=1e-4 的档位整段消失，
    # best 退化成 {t:1.0, tar:0.0}、at_far_1e6 变成 null —— 整栏「阈值建议」等于没有。
    # 底库里只要有人模板数不一致就会触发（README 恰好推荐「补一张不同场合的照片」）。
    live = np.arange(mmax)[None, :] < counts[b][:, None]
    s = np.where(live, d, np.float32(-1.0)).max(axis=1).astype(np.float32)
    info.update(impostor_n=int(s.size), impostor_pairs=int(ids.size))
    return s, info


def probe_roc(records, max_impostor: int = 200_000):
    """从带真值的探针批次里取两组相似度：同一人互配（genuine）/ 异人冒认（impostor）。

    底库每人只有一张模板时（README 推荐、也是本机那份 3475 人底库的实际形态）
    底库自己凑不出同人互配对，整栏「阈值建议」是空的。而一批标了真值的探针
    天然带这两类样本，不用重新推理：Top-K 已经是 search(per_person=True) 按人聚合过的，
    所以「真值那一行」就是这张脸对真值人员的最高分（同人互配分），
    「除真值外的最高分」就是这张脸对某个陌生人员的最高分（冒认分），
    两个口径都和 sample_impostors 刻意对齐，不是另起一套。

    真值没进 Top-K 的同人样本是**删失**观测：真值分数只知道低于第 K 名，
    随手补一个数（补第 K 名会系统性偏高）会让 TAR 虚高，所以只计数不进分布。
    没标真值的脸（合影里没标注的旁人）两侧都不进 —— 压根不知道他在不在库里。
    真值为「非底库人员」的脸对全库的最高分是标准冒认样本，进 impostor。

    返回 (genuine, impostor, info)，前两个是一维 float32 数组。
    """
    gen: list[float] = []
    imp: list[float] = []
    n_face = n_censored = n_unlabeled = n_unknown = 0
    for r in records or []:
        if r.get("verdict") == "error":       # 根本没评出来的那张，脸是残缺的
            continue
        for f in (r.get("faces") or []):
            top = [t for t in (f.get("top") or []) if t.get("person")]
            if not top:
                continue
            n_face += 1
            truth = f.get("truth")
            if truth == UNKNOWN:
                imp.append(float(top[0].get("score") or 0.0))
                n_unknown += 1
                continue
            if not truth:
                n_unlabeled += 1
                continue
            same = [float(t.get("score") or 0.0) for t in top if t["person"] == truth]
            diff = [float(t.get("score") or 0.0) for t in top if t["person"] != truth]
            if same:
                gen.append(max(same))
            else:
                n_censored += 1
            if diff and len(imp) < max_impostor:
                imp.append(max(diff))
    info = {"probe_faces": n_face, "probe_genuine_n": len(gen),
            "probe_impostor_n": len(imp), "probe_censored": n_censored,
            "probe_unlabeled_faces": n_unlabeled, "probe_unknown_faces": n_unknown}
    return (np.asarray(gen, np.float32), np.asarray(imp, np.float32), info)


def sweep_thresholds(genuine, impostor, steps: int = 120):
    """扫描阈值给出 TAR/FAR 曲线、万库累计误识率与推荐阈值。"""
    g = np.asarray(genuine, np.float32)
    m = np.asarray(impostor, np.float32)
    if g.size == 0:
        # 底库每人只有 1 张模板时（README 推荐的最常见形态）永远凑不出 genuine 对。
        # 原来这里直接返回两个键就结束，前端整块「阈值建议」凭空消失，
        # 连「为什么没有」都看不出来 —— 键必须补齐，并说明缺的是哪一侧的样本。
        # 说法要经得起照着做：能补上这一侧的只有「同人互配」样本，来源就两条路 ——
        # 底库里有人有第二张照片，或者这批探针里有标了真值的人脸（见 probe_roc）。
        # 原来写的「跑一批带真值的探针即可生成」在探针没参与计算时是句空话：
        # 照着跑完 202 张全带真值的批次，这一栏还是原样一句话。
        return {"curve": [], "best": None, "at_far_1e6": None,
                "genuine_n": 0, "impostor_n": int(m.size),
                "note": "估不出「同一人互配」分布：底库里没有人有多张模板，"
                        "这批探针里也没有可用的带真值人脸（真值需命中 Top-K）；"
                        "给几个人再注册一张不同场合的照片，或把探针的真值标全即可生成"}
    lo = float(min(g.min(), m.min())) if m.size else float(g.min())
    hi = float(max(g.max(), m.max())) if m.size else float(g.max())
    ts = np.linspace(max(-1.0, lo - 0.02), min(1.0, hi + 0.02), steps)
    curve = []
    for t in ts:
        tar = float((g >= t).mean())
        far = float((m >= t).mean()) if m.size else 0.0
        curve.append(
            {
                "t": round(float(t), 4),
                "tar": round(tar, 4),
                "far": round(far, 6),
                # 万库累计误报：至少一次误识的概率。far 是「单个库内人员被误放行」的概率，
                # 指数 9999 也是人数，两边量纲一致（见 sample_impostors 的 per-person 口径）
                "gallery_far_1e4": round(1.0 - (1.0 - far) ** 9999, 4) if m.size else None,
            }
        )
    # 推荐：满足 FAR<=1e-4 的最高 TAR；否则取 TAR-FAR 差值最大点。
    # TAR 并列时必须取**更高**的那个阈值：TAR 相同意味着放行率一样，阈值越高
    # 冒认者越少，选低端等于白白多放一批人。genuine 样本一少 TAR 就是粗台阶，
    # 并列几乎是必然（400 点台阶只 4 个人时实测并列 4 个点：
    # t=0.5251/far=8.5e-5 与 t=0.5545/far=3.5e-5，万库累计误报 0.57 → 0.30，
    # 放行率一模一样，误报却差一倍），而 max() 取的是第一个最大值、正好挑走低端。
    ok = [c for c in curve if c["far"] <= 1e-4]
    best = max(ok, key=lambda c: (c["tar"], c["t"])) if ok else max(
        curve, key=lambda c: c["tar"] - c["far"] * 20
    )
    far1e6 = [c for c in curve if c["far"] <= 1e-6]
    return {
        "curve": curve,
        "best": best,
        # 这里反过来：1e-6 档本来就贴着「几乎全拒」，TAR 并列时要挑**最低**的阈值，
        # 否则会把还能放行的人一并关在门外（台阶顶端往往连着好几个 TAR=0 的点）
        "at_far_1e6": max(far1e6, key=lambda c: (c["tar"], -c["t"])) if far1e6 else None,
        "genuine_n": int(g.size),
        "impostor_n": int(m.size),
    }


def summarize(records: list[dict], threshold: float, topk: int,
              wall_s: float | None = None, workers: int = 1) -> dict:
    """汇总一批探针结果，分两套口径。

    图级（主体人脸）：每张图按真值对应的那张脸判定 Rank-1 / 漏识 / 误识。
    人脸级（所有已标真值的脸）：把用户手工标注的同框他人也算进来，
    多人合影里每张脸都是独立的 1:N 判定样本。

    wall_s 是这一批的真实墙钟耗时：给了它，吞吐就按「总张数 ÷ 总时间」算，
    这才是开多路并发时该有的数字。以前吞吐一律按 1000/P50 折算，开 4 路并发
    时单图 P50 会涨到三倍，折算出来的吞吐反而往下掉，看着像「加了线程更慢」——
    那是口径错了，不是机器慢了。没给 wall_s 时退回按 P50 折算，
    并用 throughput_basis 讲清是哪种口径。
    """
    n = len(records)
    if n == 0:
        return {"n": 0}
    faces = [f for r in records for f in (r.get("faces") or [])]
    truth_imgs = [r for r in records if r.get("truth")]
    labeled = [f for f in faces if f.get("truth")]
    # 坏图（解码失败 / 推理抛异常）根本没有可用输入，算进检出率分母会把
    # 好好的检测器冤枉成漏检。注意「检不到脸」不算坏图：那是真实的漏检，
    # 它自带 detected=False，必须留在分母里，否则检出率永远 100%。
    errored = [r for r in records if r.get("verdict") == "error"]
    usable = [r for r in records if r.get("verdict") != "error"]
    # 图级分母 = 有真值且不是坏图的**全部**图，一张脸都没检到的也算在内。
    # 原来多加了「得有 faces」这一条，漏检的图整个退出 Rank-1 分母：
    # 检测器越差、能算 Rank-1 的样本越少，分数反而越好看，和 README 承诺的
    # 「每张有真值的图」也对不上。漏检本来就是没认对，必须计入分母。
    subjects = [r for r in truth_imgs if r.get("verdict") != "error"]
    subj_nodetect = [r for r in subjects if not r.get("faces")]
    detected = [r for r in usable if r.get("detected")]
    fallback = [r for r in records if r.get("fallback")]
    accepted = [r for r in usable if r.get("accepted")]
    multi = [r for r in records if (r.get("faces_found") or 0) > 1]
    extras = [f for r in records for f in (r.get("faces") or []) if not f.get("is_subject")]
    extras_acc = [f for f in extras if f.get("accepted")]
    # 同框他人被放行，得分清是「认对了旁人」还是「认错了人」，两者风险完全不同。
    # 以前全塞进 extra_accepted 一个数，合影全对时这个「核心风险项」反而 100%，
    # 越准看着越危险。原始计数保留（口径不变），另列三个可解释的分项。
    extras_right = [f for f in extras_acc
                    if f.get("truth") and f.get("truth") == f.get("top1_person")]
    extras_wrong = [f for f in extras_acc if f.get("truth") and f.get("truth") != UNKNOWN
                    and f.get("truth") != f.get("top1_person")]
    extras_nonmember = [f for f in extras_acc if f.get("truth") == UNKNOWN]
    extras_unlabeled = [f for f in extras_acc if not f.get("truth")]

    # 「非底库人员」可以判「拒绝对不对」，但库里根本没有他，
    # 算进 Rank-1 / Recall@k 的分母只会稀释指标，所以单独统计
    named_subj = [r for r in subjects if r.get("truth") != UNKNOWN]
    named_faces = [f for f in labeled if f.get("truth") != UNKNOWN]
    unknown_subj = [r for r in subjects if r.get("truth") == UNKNOWN]
    # 脸级口径覆盖所有标了非底库的脸，合影里的旁人同样算一个样本。
    # 唯一例外是「主体脸标了非底库、整图真值却是个有名的人」：同一张脸的同一件事，
    # 图级算「认对了张三」、脸级又算一次「非底库被误放行」，一次通过同时进
    # Rank-1 分子和 FAR 分子（far_estimated 能离谱到 100%）。这是标注冲突，
    # 以整图真值为准，同时单独计数，让界面能提示用户去修而不是静默吞掉。
    conflict_ids = {id(f) for r in records for f in (r.get("faces") or [])
                    if f.get("truth") == UNKNOWN and f.get("is_subject")
                    and r.get("truth") and r["truth"] != UNKNOWN}
    conflict_faces = [f for f in labeled if id(f) in conflict_ids]
    unknown_faces = [f for f in labeled if f.get("truth") == UNKNOWN and id(f) not in conflict_ids]

    img_ok = [r for r in named_subj if r.get("rank1_correct")]
    img_rej = [r for r in named_subj if not r.get("accepted")]
    img_wrong = [r for r in subjects if r.get("accepted") and not r.get("rank1_correct")]
    # 命中名次用 None 表示「真值不在候选里」。以前写成 `or 99`，一旦 topk 大于 99
    # 就把压根没命中的样本算进 Recall 分子，虚高还没人发现
    img_topk = [r for r in named_subj if r.get("truth_rank") is not None
                and r["truth_rank"] <= topk]

    face_ok = [f for f in named_faces if f.get("rank1_correct")]
    face_rej = [f for f in named_faces if not f.get("accepted")]
    # 标注冲突的脸（脸级标了非底库、整图真值却是有名的人）整条退出脸级统计：
    # 上面已经定了「以整图真值为准」，而 named_faces 本来就不收 UNKNOWN 真值，
    # 它只出现在分子里就是不一致 —— 认对时同一次通过既进 Rank-1 分子又进
    # 脸级误放行，混淆表还会冒出「非底库人员 → 张三」这种自相矛盾的行；
    # 认错时（认成了李四）图级已经按真名记了一次「张三 → 李四」，脸级再记一次就是重复计价。
    # 冲突条数由 n_truth_conflict_faces 如实交代，不静默吞掉。
    face_wrong = [f for f in labeled if f.get("accepted") and not f.get("rank1_correct")
                  and id(f) not in conflict_ids]
    face_topk = [f for f in named_faces if f.get("truth_rank") is not None
                 and f["truth_rank"] <= topk]

    def rate(x, d):
        return round(len(x) / d, 4) if d else None

    def disp(p: str | None) -> str:
        return UNKNOWN_LABEL if p == UNKNOWN else (p or "(拒绝)")

    conf: dict[str, int] = {}
    for r in img_wrong:
        key = f"{disp(r.get('truth'))} → {disp(r.get('top1_person'))}"
        conf[key] = conf.get(key, 0) + 1
    for f in face_wrong:
        if f.get("is_subject"):
            continue
        # 具名的同框他人要报真名：「(同框) → 张三」答不出「王五被认成了谁」
        who = UNKNOWN_LABEL if f.get("truth") == UNKNOWN else (f.get("truth") or "(同框)")
        conf[f"{who} → {disp(f.get('top1_person'))}"] = conf.get(
            f"{who} → {disp(f.get('top1_person'))}", 0) + 1

    # 耗时只看正常跑完的记录：坏图那几条各阶段都是 0，混进来会把 P50 往下拽，
    # 让一次「输入其实有问题」的批次看起来比实际更快
    ok_t = [r["timings"] for r in usable if r.get("timings")]
    tm = [t["total_ms"] for t in ok_t] or [t["total_ms"] for t in
                                           (r["timings"] for r in records if r.get("timings"))]
    # 每张脸的平均耗时按「脸」展开成同样条数，而不是先除再取分位数：
    # 混进一堆 n_faces=0 的记录（值是 0）会把 P50 直接拽到 0，
    # 而合影里 3 张脸共用一个 total 时，按图取分位数等于把三张脸折成一张计价
    per_face = [v for t in ok_t if t.get("n_faces", 0) > 0
                for v in ((t["embed_ms"] / t["n_faces"], t["search_ms"] / t["n_faces"])
                          for _ in range(t["n_faces"]))]
    pf_embed = [a for a, _ in per_face]
    pf_search = [b for _, b in per_face]
    # 平均人脸数按实际检出数（faces_found）算，不用 timings.n_faces：
    # all_faces=0 只分析最大一张时 timings 恒记 1，合影占比会被压成 1.00
    nf = [int(r.get("faces_found") or 0) for r in usable] or [0]

    return {
        "n": n,
        "n_faces": len(faces),
        "n_multi_face_images": len(multi),
        "n_fallback_faces": len(fallback),
        "n_error": len(errored),
        "n_truth_known": len(truth_imgs),
        "n_rank1_nodetect": len(subj_nodetect),      # 有真值却一张脸都没检到，已计入 Rank-1 分母
        "n_truth_conflict_faces": len(conflict_faces),  # 图级真值与主体脸级真值打架
        "n_faces_labeled": len(labeled),
        "n_truth_named": len(named_subj),
        "n_truth_unknown": len(unknown_subj),
        "unknown_correct_reject": len([r for r in unknown_subj if not r.get("accepted")]),
        "unknown_false_accept": len([r for r in unknown_subj if r.get("accepted")]),
        "n_unknown_faces": len(unknown_faces),
        "unknown_faces_correct_reject": len([f for f in unknown_faces if not f.get("accepted")]),
        "unknown_faces_false_accept": len([f for f in unknown_faces if f.get("accepted")]),
        "detect_rate": rate(detected, len(usable)),
        # 图级：真值对应的主体人脸
        "rank1": rate(img_ok, len(named_subj)),
        f"recall@{topk}": rate(img_topk, len(named_subj)),
        "false_accept": len(img_wrong),
        "false_reject": len(img_rej),
        # 误识率的分母只能是「不在底库的人」。之前用有名真值当分母，
        # 标注一多就冒出 1250% 这种荒谬值（250 个误放行 ÷ 20 个有真值的人）。
        "far_estimated": rate([f for f in unknown_faces if f.get("accepted")], len(unknown_faces)),
        # 人脸级：含手工标注的同框他人
        "face_rank1": rate(face_ok, len(named_faces)),
        f"face_recall@{topk}": rate(face_topk, len(named_faces)),
        "face_false_accept": len(face_wrong),
        "face_false_reject": len(face_rej),
        # 多人合影里的“闯入”统计：非主体人脸被放行成库内某人
        "extra_faces": len(extras),
        "extra_accepted": len(extras_acc),
        # 放行之后拆三种：认对旁人 / 认错人 / 放进来的是非底库人员 / 还没标注
        "extra_identified_correct": len(extras_right),
        "extra_identified_wrong": len(extras_wrong),
        "extra_nonmember_accepted": len(extras_nonmember),
        "extra_unlabeled_accepted": len(extras_unlabeled),
        "extra_accept_rate": rate(extras_acc, len(extras)),
        "accept_rate": rate(accepted, len(usable)),
        "avg_faces_per_image": round(float(np.mean(nf)), 2) if nf else 0,
        "confusions": sorted(conf.items(), key=lambda kv: -kv[1])[:12],
        "threshold": threshold,
        "latency": {
            "total_ms": pct(tm),
            "detect_ms": pct([t["detect_ms"] for t in ok_t]),
            "align_ms": pct([t["align_ms"] for t in ok_t]),
            "embed_ms": pct(pf_embed) if pf_embed else pct(
                [t["embed_ms"] / max(1, t.get("n_faces", 1)) for t in ok_t]),
            "search_ms": pct(pf_search) if pf_search else pct(
                [t["search_ms"] / max(1, t.get("n_faces", 1)) for t in ok_t]),
        },
        # 有墙钟耗时就走真吞吐（含排队、含并行），没有才退回按 P50 折算。
        # 分子是「真评完成的图/脸」（usable），不是 n：坏图不占分子，可它们解码失败前
        # 花掉的时间仍留在整批墙钟里，所以这个吞吐只会偏保守、不会虚高。别拿 n ÷ 墙钟
        # 去对账 —— 那个数把只解了半下码就抛异常的图也算成了产出，会高于这里报的值。
        "throughput_fps": (round(len(usable) / wall_s, 1) if wall_s and wall_s > 0
                           else (round(1000.0 / np.percentile(tm, 50), 1) if tm else None)),
        "throughput_faces_per_s": (
            round(sum(nf) / wall_s, 1) if wall_s and wall_s > 0
            else (round(sum(nf) * 1000.0 / max(1e-9, sum(tm)), 1) if tm and sum(tm) else None)),
        "throughput_basis": "wall" if wall_s and wall_s > 0 else "p50",
        "workers": int(workers or 1),
    }
