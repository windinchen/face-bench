#!/usr/bin/env python3
"""逻辑自检：不加载 ONNX，用桩后端在秒级验证评测口径。

盯的都是不会报错、只会静默算错的地方：
  1) 单人底库会不会把阈值建议卡死
  2) 「非底库人员」是不是按「期望拒绝」判定，且不污染 Rank-1 / Recall@k 分母
  3) Recall@k 会不会跟着判定阈值一起乱跳
  4) 同一个底库文件夹拖两次，底库会不会膨胀
  5) 老底库自动迁进 SQLite 时会不会拿错位的行去检索
 10) 老格式底库（features.npy + meta 内嵌向量）升级后会不会丢人
 11) 一边注册一边评测会不会读到半个库
 12) 清空历史批次会不会留下删不掉的孤儿报告和没人认领的探针目录
 13) 清空底库会不会留下几千张孤儿缩略图和没删掉的数据库
 14) 底库列表分页会不会重漏、搜索会不会把通配符当字面、删人会不会留幽灵行
  6) 探针真值推断会不会从文件名造出假真值
  7) 底库平铺（一张图一个人、文件名就是人名）能不能正确认人
15) 通道式 2-anchor 的检测头会不会整层被静默丢掉（检出一律为 0 且不报错）
16) 「每人模板上限」并行注册时会不会被超发
17) 坏图会不会混进检出率分母；并发下吞吐是不是还按 P50 折算
18) 检索用的「模板→人员」增量登记会不会和全量重算对不上
19) 重新注册同一个人时，多出来的旧模板和缩略图会不会留在盘上
20) 底库写不进去（只读目录、磁盘写满）时会不会默默「注册成功」
21) 全员名单会不会被分页上限截断，和 stats.persons 各说各话
22) TAR 并列时推荐阈值会不会挑走最宽松那一档（同样的放行率多放一倍人）
23) 主体脸被手工改成「非底库」时，会不会同一次通过既进 Rank-1 又进脸级误放行
24) 模型摘要缓存会不会互相挤掉，导致每次轮询都把 455MB 重算一遍
25) 删单个批次会不会连累别的批次，或只删掉报告留下探针目录
26) 删单个人会不会连累别人，或人不在库里还报「删除成功」
27) 解码失败的坏图会不会被一张假像素冒充成「漏检」，害得 n_error 恒为 0

用法：python -m cli.selftest
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from app.backend import Backend, Face
from app.bench import recompute, run_probe, threshold_report
from app.config import DATA_DIR, UNKNOWN, infer_person
from app import config as cfg
from app.metrics import (pct, probe_roc, sample_impostors, summarize as bench_summarize,
                         sweep_thresholds)
from app.gallery import Gallery, Template, gallery_dir_for, tid_for
from app.scrfd import Scrfd
from app import server as srv
from app.server import _probe_truth

DIM = 8


def basis(i: int) -> np.ndarray:
    v = np.zeros(DIM, np.float32)
    v[i] = 1.0
    return v


def mix(**kw: float) -> np.ndarray:
    """按 {基向量下标: 分量} 造一个单位向量。"""
    v = np.zeros(DIM, np.float32)
    for k, x in kw.items():
        v[int(k[1:])] = x
    rest = 1.0 - float((v ** 2).sum())
    free = [i for i in range(DIM) if v[i] == 0.0]
    if rest > 1e-9 and free:
        v[free[0]] = np.sqrt(rest)
    return (v / np.linalg.norm(v)).astype(np.float32)


class Stub(Backend):
    """桩后端：把「图片」当成一串人脸编号，向量查表返回，绕开推理。"""

    key = "stub"

    def __init__(self, vecs: dict[int, np.ndarray]):
        self.vecs = vecs
        self.delay = 0.0        # 让桩后端慢一点，好验证「并发确实换到吞吐」而不是口径错觉

    def detect(self, img, det_size, thresh, nms):
        return []

    def embed(self, crop):
        if self.delay:
            time.sleep(self.delay)
        return np.asarray(self.vecs[int(crop[0, 0, 0])], np.float32), 0.1

    def info(self) -> dict:
        return {"key": "stub", "det": "stub", "rec": "stub"}

    def faces_from_image(self, img, **_kw):
        faces = [
            Face(box=np.array([i * 40, 0, i * 40 + 30, 30], np.float32), score=0.99,
                 crop=np.full((112, 112, 3), fid, np.uint8),
                 quality={"ok": True, "aligned": True, "reasons": []})
            for i, fid in enumerate(int(x) for x in np.asarray(img).reshape(-1))
        ]
        return faces, {"detect": 1.0, "align": 0.4}


def img(*ids: int) -> np.ndarray:
    return np.array(list(ids), np.uint8)


def new_gallery(key: str) -> Gallery:
    shutil.rmtree(gallery_dir_for(key), ignore_errors=True)
    return Gallery(key)


def tpl(tid: str, person: str, vec) -> Template:
    return Template(tid=tid, person=person, source=f"{tid}.jpg", vec=np.asarray(vec, np.float32),
                    box=[0, 0, 30, 30], score=0.9, quality={"ok": True, "aligned": True},
                    thumb=f"{tid}.jpg")


# 人脸编号 -> 特征向量：1/2/3 是底库里的 A/B/C，4/5 是库外的人（一个会被放行一个不会），
# 6 是「真值 B 只排第 2 位」的难题样本
VECS = {
    1: basis(0), 2: basis(1), 3: basis(2),
    4: mix(b0=0.6, b3=0.8),      # 库外但和 A 有 0.6 相似度 → 会被误放行
    5: mix(b0=0.2, b3=0.98),     # 库外且谁都不像 → 应该被拒绝
    6: mix(b0=0.7, b1=0.55),     # 更像 A，真值 B 在第 2 位
}


def t1_single_person_gallery():
    g = new_gallery("selftest_t1")
    for i in range(5):
        g.add(tpl(f"a{i}", "A", basis(0)))
    t0 = time.perf_counter()
    rep = threshold_report(None, g)
    cost = time.perf_counter() - t0
    assert cost < 5.0, f"单人底库阈值建议耗时 {cost:.1f}s，疑似死循环"
    assert rep["impostor_n"] == 0, rep
    res = run_probe(Stub(VECS), g, [("p/A/1.jpg", img(4), UNKNOWN)], {"threshold": 0.4})
    assert res["summary"]["n"] == 1
    print(f"  1 单人底库：阈值建议 {cost*1000:.0f}ms 返回，整批评测不卡死  ✓")


def t2_unknown_semantics():
    g = new_gallery("selftest_t2")
    for tid, person, i in (("a", "A", 0), ("b", "B", 1), ("c", "C", 2)):
        g.add(tpl(tid, person, basis(i)))
    bk = Stub(VECS)
    res = run_probe(bk, g, [
        ("cam/A/1.jpg", img(1), "A"),
        ("cam/outsider/1.jpg", img(4), UNKNOWN),
        ("cam/ghost/1.jpg", img(5), UNKNOWN),
    ], {"threshold": 0.4, "topk": 5})
    by = {r["name"]: r for r in res["records"]}
    assert by["cam/A/1.jpg"]["verdict"] == "correct", by["cam/A/1.jpg"]
    acc, rej = by["cam/outsider/1.jpg"], by["cam/ghost/1.jpg"]
    assert acc["verdict"] == "wrong" and acc["accepted"], acc["verdict"]
    assert rej["verdict"] == "correct_unknown" and not rej["accepted"], rej["verdict"]
    assert rej["faces"][0]["is_impostor"] is True
    s = res["summary"]
    assert (s["n_truth_known"], s["n_truth_named"], s["n_truth_unknown"]) == (3, 1, 2), s
    assert (s["unknown_correct_reject"], s["unknown_false_accept"]) == (1, 1), s
    assert s["false_accept"] == 1 and s["false_reject"] == 0, s
    assert s["rank1"] == 1.0, f"库外样本混进了 Rank-1 分母：{s['rank1']}"
    assert any(k.startswith("非底库人员") for k, _ in s["confusions"]), s["confusions"]
    print("  2 非底库人员：拒绝=正确拒绝、放行=误识，且不进 Rank-1 分母  ✓")


def t3_recall_independent_of_threshold():
    g = new_gallery("selftest_t3")
    for tid, person, i in (("a", "A", 0), ("b", "B", 1), ("c", "C", 2)):
        g.add(tpl(tid, person, basis(i)))
    rec = run_probe(Stub(VECS), g, [("cam/B/1.jpg", img(6), "B")],
                    {"threshold": 0.4, "topk": 5})["records"][0]
    assert rec["faces"][0]["top1_person"] == "A" and rec["truth_rank"] == 2, rec["faces"][0]
    seen = {}
    for thr in (0.4, 0.65, 0.75, 0.95):
        recompute(rec, thr)
        s = res_summary([rec], thr, 5)
        seen[thr] = (s["recall@5"], s["rank1"], s["false_reject"])
    recalls = {v[0] for v in seen.values()}
    assert recalls == {1.0}, f"Recall@5 随阈值漂移：{seen}"
    assert seen[0.4][1] == 0.0 and seen[0.95][2] == 1, seen
    print("  3 Recall@5 恒为 100%，只有 Rank-1/漏识随阈值变  ✓")


def res_summary(records, thr, topk):
    from app.metrics import summarize
    return summarize(records, thr, topk)


def t4_dedupe_on_reenroll():
    g = new_gallery("selftest_t4")
    g.add(tpl("a", "A", basis(0)))
    g.add(tpl("b", "B", basis(1)))
    for _ in range(3):                      # 同一个文件夹连着拖三次
        g.add(tpl("a", "A", mix(b0=0.5, b1=0.866)))
    assert len(g.index) == 2 and len(g.persons["A"]) == 1, (len(g.index), g.persons)
    assert g.matrix.shape[0] == 2
    assert np.allclose(g.matrix[0], g.index[0].vec, atol=1e-6), "matrix 行没跟着替换"
    assert np.allclose(g.persons["A"][0].vec, g.index[0].vec, atol=1e-6), "分组里还是旧向量"
    q = mix(b0=0.5, b1=0.866)               # 换过的新向量应当真正生效，而不只是改了元数据
    s, p = g.search(q, topk=2)
    assert p == ["A", "B"] and abs(float(s[0]) - 1.0) < 1e-5 and abs(float(s[1]) - 0.866) < 1e-4, (p, s)
    print("  4 重复注册同一张底库图：原地替换，底库不膨胀、阈值建议不被污染  ✓")


def _rec(t: Template) -> dict:
    """老格式 meta.json 里的一条模板记录（不含向量）。"""
    d = asdict(t)
    d.pop("vec", None)
    return d


def drop_db(g: Gallery) -> None:
    """删掉数据库，让下一次 Gallery() 只能从老格式文件读。"""
    g._close_db()
    for f in (g.db_file, Path(str(g.db_file) + "-wal"), Path(str(g.db_file) + "-shm")):
        f.unlink(missing_ok=True)
    shutil.rmtree(g.legacy_dir, ignore_errors=True)


def t5_recover_row_mismatch():
    """老底库（features.npz + meta.json）自动迁进数据库时，向量行必须还和自己的
    tid 绑在一起：「向量没写完」只会少人，绝不会把 C 的向量配成 B 的名字——
    错配比缺失更难被发现。"""
    key = "selftest_t5"
    g = new_gallery(key)
    for tid, person, i in (("a", "A", 0), ("b", "B", 1), ("c", "C", 2)):
        g.add(tpl(tid, person, basis(i)))
    g.save()
    mat = g.matrix.copy()
    tids = np.asarray([t.tid for t in g.index])
    assert len(tids) == 3 and mat.shape[0] == 3

    def as_legacy(m, ts, rs):
        """把内存里的库伪装成一个老格式底库目录：只剩 npz + meta.json。"""
        drop_db(g)
        np.savez(g.vec_file, matrix=m, tids=ts)
        g.meta_file.write_text(json.dumps(
            {"model_key": key, "built_at": 1.0, "templates": rs}, ensure_ascii=False),
            encoding="utf-8")

    as_legacy(mat, tids, [_rec(t) for t in g.index])       # 完整的老底库
    g2 = Gallery(key)
    assert [t.person for t in g2.index] == ["A", "B", "C"], g2.stats()
    assert g2.db_file.exists(), "老底库没自动迁进数据库，几千张得重传"
    assert (g2.legacy_dir / "meta.json").exists(), "迁移后老文件要留档，不能直接删"
    assert g2.search(basis(2))[1][0] == "C"
    g2.save()
    assert Gallery(key).search(basis(2))[1][0] == "C"

    as_legacy(mat[:2], tids[:2], [_rec(t) for t in g2.index])   # 向量少一条（当年写断了）
    g3 = Gallery(key)
    assert [t.person for t in g3.index] == ["A", "B"], g3.stats()
    s, p = g3.search(basis(1), topk=3)
    assert p[0] == "B", (p, s)                            # 行没跟错人名
    g3.add(tpl("d", "D", basis(3)))          # 少一个人的库继续往上涨，扩容不能错位
    assert g3.matrix.shape[0] == len(g3.index) == 3, (g3.matrix.shape, len(g3.index))
    assert g3.search(basis(3))[1][0] == "D"
    g3.save()

    # 元数据落盘前被打断：向量里有 d 的行，元数据里没有 d 这个人
    as_legacy(mat[:3], np.asarray(["a", "b", "d"]), [_rec(t) for t in g3.index[:2]])
    g4 = Gallery(key)
    assert len(g4.index) == 2, g4.stats()
    assert g4.stats()["integrity"], "缺元数据的向量要报出来，不能悄悄吞掉"
    assert g4.search(basis(0))[1][0] == "A"
    print("  5 老底库自动迁库 + 向量/元数据不一致：只少人不串位，并如实告知  ✓")


def t10_legacy_storage():
    """用户手上的存量底库是老格式，升级到 npz 必须无损，否则几千张得重传。"""
    key = "selftest_t10"
    g = new_gallery(key)
    for i, person in enumerate(("A", "B", "C")):
        g.add(tpl(tid_for(person, f"{person}.jpg"), person, basis(i)))
    g.save()
    recs = []
    for t in g.index:
        d = asdict(t)
        d["vec"] = t.vec.tolist()
        recs.append(d)
    np.save(g.dir / "features.npy", g.matrix)      # 新存储是 gallery.db，这里手动退回最老格式
    g.meta_file.write_text(json.dumps(
        {"model_key": key, "built_at": 1.0, "templates": recs}, ensure_ascii=False), encoding="utf-8")
    drop_db(g)                                            # 只留最老那一份文件

    g2 = Gallery(key)
    assert [t.person for t in g2.index] == ["A", "B", "C"], g2.stats()
    assert g2.search(basis(2))[1][0] == "C"
    assert not (g2.dir / "features.npy").exists() and (g2.legacy_dir / "features.npy").exists(), \
        "老文件要归档进 legacy/，既不能被重复读取也不能被直接删掉"
    g2.save()                                             # 存进数据库
    assert g2.db_file.exists()
    g3 = Gallery(key)
    assert len(g3.index) == 3 and g3.search(basis(2))[1][0] == "C", g3.stats()
    assert not g3.stats()["integrity"]
    print(" 10 老格式底库自动升级：人名、向量、检索结果都不变  ✓")


def t11_concurrent_enroll_and_search():
    """注册在后台线程跑，评测同时在请求线程读库（几千张底库的真实用法）。"""
    key = "selftest_t11"
    g = new_gallery(key)
    for i in range(8):
        g.add(tpl(f"seed{i}", f"P{i}", basis(i % DIM)))
    g.save()
    errs: list[str] = []

    def adder():
        try:
            for i in range(200, 260):
                g.add(tpl(f"n{i}", f"P{i}", basis(i % DIM)))
            g.save()
        except Exception as e:                            # noqa: BLE001
            errs.append(f"注册线程：{type(e).__name__}: {e}")

    def reader():
        for i in range(400):
            try:
                names = g.search(basis(i % DIM), topk=3)[1]
                if not names:
                    errs.append("读到空库")
                    break
            except Exception as e:                        # noqa: BLE001
                errs.append(f"检索线程：{type(e).__name__}: {e}")
                break

    ts = [threading.Thread(target=adder), threading.Thread(target=reader)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errs, errs[:3]
    assert len(g.index) == 68, g.stats()
    g2 = Gallery(key)
    sc, names = g2.search(basis(3), topk=3)   # 同分的人有多个，只断言「满分命中」不断言具体名字
    assert len(g2.index) == 68 and names and float(sc[0]) > 0.999, (len(g2.index), names, sc)
    print(" 11 边注册边评测：不崩、不读到半个库、落盘后一致  ✓")


def t6_probe_truth():
    persons = {"张三", "李四"}
    got = _probe_truth("cam1/张三/IMG_0001.jpg", "IMG_0001.jpg", "delim", "_", persons)
    assert got == ("张三", None), got
    got = _probe_truth("IMG_0001.jpg", "IMG_0001.jpg", "delim", "_", persons)
    assert got == (None, None), f"相机编号 IMG 被当成了人名：{got}"
    got = _probe_truth("cam1/王五/1.jpg", "1.jpg", "path", "_", persons)
    assert got == (UNKNOWN, "王五"), got
    assert _probe_truth("a/b/1.jpg", "1.jpg", "none", "_", persons) == (None, None)
    # 批次目录不是人名：cam1 命中不了底库，必须回落到文件名规则
    got = _probe_truth("cam1/李四_0007.jpg", "李四_0007.jpg", "id_num", "_", persons)
    assert got == ("李四", None), f"批次目录 cam1 抢在了人名前面：{got}"
    got = _probe_truth("cam1/李四.jpg", "李四.jpg", "name", "_", persons)
    assert got == ("李四", None), f"选了「文件名即人名」却被父目录覆盖：{got}"
    assert _probe_truth("cam1/路人.jpg", "路人.jpg", "name", "_", persons) == (UNKNOWN, "路人")
    # 探针默认也是「每张图一个人」：文件名先上，命中不了才轮到父目录，
    # 相机自动编号既不命中底库就不标真值，留给界面手工标注。
    assert _probe_truth("张三.jpg", "张三.jpg", "name", "_", persons) == ("张三", None)
    got = _probe_truth("cam1/李四_0007.jpg", "李四_0007.jpg", "name", "_", persons)
    assert got == ("李四", None), f"默认口径没把带序号的文件名认出来：{got}"
    assert _probe_truth("cam1/9999.jpg", "9999.jpg", "name", "_", persons) == (UNKNOWN, "cam1")
    assert _probe_truth("IMG_0001.jpg", "IMG_0001.jpg", "name", "_", persons) == (None, None)
    assert _probe_truth("1792052059096.jpg", "1792052059096.jpg", "name", "_", persons) == (None, None)
    # 显式选「文件夹名优先」时也一样：文件名只有命中底库才被采用，
    # 底库平铺（文件名即人名）的探针不用手点几千次，批次目录也不会压过真名。
    ids = {"1792052059096"}
    assert _probe_truth("1792052059096.jpg", "1792052059096.jpg", "path", "_", ids) == ("1792052059096", None)
    assert _probe_truth("cam1/1792052059096.jpg", "1792052059096.jpg", "path", "_", ids) == ("1792052059096", None)
    assert _probe_truth("cam1/9999.jpg", "9999.jpg", "path", "_", ids) == (UNKNOWN, "cam1")
    print("  6 探针真值：默认按文件名（每张图一个人），命中底库才算数；"
          "批次目录不当人名，相机编号不造假  ✓")


def t7_multi_face_same_frame():
    g = new_gallery("selftest_t7")
    for tid, person, i in (("a", "A", 0), ("b", "B", 1), ("c", "C", 2)):
        g.add(tpl(tid, person, basis(i)))
    res = run_probe(Stub(VECS), g, [("cam/A/合影.jpg", img(1, 5, 4), "A")],
                    {"threshold": 0.4, "topk": 5})
    rec = res["records"][0]
    assert rec["faces_found"] == 3 and rec["faces"][0]["is_subject"], rec
    assert all(f["truth"] is None for f in rec["faces"][1:]), "同框旁人不该被默认认领成主体"
    assert res["summary"]["extra_accepted"] == 1, res["summary"]
    rec["faces"][1]["truth"] = UNKNOWN      # 库外且谁都不像 → 期望被拒绝
    rec["faces"][2]["truth"] = UNKNOWN      # 库外但与 A 有 0.6 → 被误放行
    recompute(rec, 0.4)
    s = res_summary([rec], 0.4, 5)
    assert s["n_faces_labeled"] == 3 and s["face_rank1"] == 1.0, s
    assert (s["n_unknown_faces"], s["unknown_faces_correct_reject"],
            s["unknown_faces_false_accept"]) == (2, 1, 1), s
    print("  7 合影：旁人默认未标注；补标后分别计入人脸级与「非底库」统计  ✓")


def t8_image_truth_without_face_label():
    """整图真值在、人脸标签被改成「未标注」：图级判定不能跟着塌成误识。"""
    g = new_gallery("selftest_t8")
    for tid, person, i in (("a", "A", 0), ("b", "B", 1), ("c", "C", 2)):
        g.add(tpl(tid, person, basis(i)))
    rec = run_probe(Stub(VECS), g, [("cam/A/2人.jpg", img(1, 3), "A")],
                    {"threshold": 0.4, "topk": 5})["records"][0]
    assert rec["verdict"] == "correct", rec["verdict"]
    for f in rec["faces"]:
        f["truth"] = None            # 用户在人脸下拉里选了「— 未标注 —」
    recompute(rec, 0.4)
    assert rec["rank1_correct"] is True and rec["verdict"] == "correct", rec["verdict"]
    assert rec["truth_rank"] == 1, rec["truth_rank"]
    s = res_summary([rec], 0.4, 5)
    assert s["false_accept"] == 0 and s["rank1"] == 1.0, s
    assert s["n_faces_labeled"] == 0 and s["n_truth_named"] == 1, s
    assert not s["confusions"], s["confusions"]
    print("  8 主体标签被清空：图级仍按整图真值判定，不会把认对的人记成误识  ✓")


def t9_flat_gallery_one_image_per_person():
    """底库最常见的形态：没有「每人一个文件夹」，一张图就是一个人，文件名即人名。"""
    assert infer_person("底库/张三.jpg", "张三.jpg", "name") == "张三"
    assert infer_person("张三.jpg", "张三.jpg", "name") == "张三"
    assert infer_person("底库/张三 .jpg", "张三 .jpg", "name") == "张三"
    assert infer_person("底库/李四/01.jpg", "01.jpg", "path") == "李四"   # 旧结构不受影响
    assert infer_person("底库/张三.jpg", "张三.jpg", "none") is None

    g = new_gallery("selftest_t9")
    fid = {"张三": 1, "李四": 2, "王五": 3}
    for name, face_id in fid.items():
        rel = f"底库/{name}.jpg"
        person = infer_person(rel, f"{name}.jpg", "name")
        g.add(Template(tid=tid_for(person, rel), person=person, source=rel,
                      vec=VECS[face_id], box=[0, 0, 30, 30], score=0.9,
                      quality={"ok": True}, thumb=""))
    assert sorted(g.persons) == ["张三", "李四", "王五"], sorted(g.persons)
    scores, names = g.search(VECS[2], topk=3)
    assert names[0] == "李四" and float(scores[0]) > 0.99, (names, scores)

    persons = set(g.persons)
    assert _probe_truth("cam/李四.jpg", "李四.jpg", "name", "_", persons) == ("李四", None)
    assert _probe_truth("cam/路人.jpg", "路人.jpg", "name", "_", persons) == (UNKNOWN, "路人")
    res = run_probe(Stub(VECS), g, [("cam/李四.jpg", img(2), "李四")],
                    {"threshold": 0.4, "topk": 5})
    assert res["records"][0]["verdict"] == "correct", res["records"][0]["verdict"]
    assert res["summary"]["rank1"] == 1.0, res["summary"]
    print("  9 底库平铺（一张图一个人）：文件名即人名，认人/检索/探针真值都对  ✓")


def t12_clear_history_batches():
    """「清空历史批次」要内存批次、报告 JSON、探针原图一起删。

    少删一样都留垃圾：只剩报告的孤儿在重启时会被 load_runs_from_disk 跳过，
    界面上再也没机会删；只剩原图就是几百 MB 没人认领的死数据。
    底库和命令行的 cli_*.json 不在删除范围内。
    """
    tmp = Path(tempfile.mkdtemp(prefix="selftest_t12_"))
    reports, probes = tmp / "reports", tmp / "probes"
    reports.mkdir()
    # 一个批次真正落盘的只有这三样：整图缩略 <i>.jpg、对齐人脸 <i>-f<j>.jpg、
    # 探针清单 meta.jsonl。探针向量不落盘，features.npz 是底库专属，删不得。
    for rid, n in (("aaa11111", 2048), ("deadbeef", 1024)):
        d = probes / rid
        d.mkdir(parents=True)
        (d / "0.jpg").write_bytes(b"x" * n)
        (d / "0-f0.jpg").write_bytes(b"x" * n)
        (d / "1-f0.jpg").write_bytes(b"x" * n)
        (d / "meta.jsonl").write_text(
            '{"i": 0, "name": "a.jpg", "thumb": "0.jpg"}\n', encoding="utf-8")
    (probes / "bbb22222").mkdir()             # 只有内存记录，报告还没落盘
    for rid in ("aaa11111", "bbb22222"):
        (reports / f"probe_{rid}.json").write_text('{"records": [1]}', encoding="utf-8")
    # 孤儿报告：探针图早被手工删了，只剩报告，重启读不回来、界面上删不掉
    (reports / "probe_orphan99.json").write_text('{"records": [1]}', encoding="utf-8")
    cli_report = reports / "cli_20261107_000000.json"
    cli_report.write_text('{"summary": {}}', encoding="utf-8")
    gallery = tmp / "gallery_w600k_r50"
    (gallery / "thumbs").mkdir(parents=True)
    gallery_meta, gallery_feat = gallery / "meta.json", gallery / "features.npz"
    gallery_thumb = gallery / "thumbs" / "t1.jpg"
    gallery_meta.write_text("{}", encoding="utf-8")
    gallery_feat.write_bytes(b"PK" * 64)
    gallery_thumb.write_bytes(b"y" * 128)

    keep = (srv.RUNS, srv.REPORT_DIR, srv.PROBE_DIR)
    srv.RUNS = {"aaa11111": {"records": [1]}, "bbb22222": {"records": [1]}}
    srv.REPORT_DIR, srv.PROBE_DIR = reports, probes
    try:
        out = srv.api_runs_clear()
        assert out["cleared"] == 3 and out["failed"] == [] and out["runs"] == 0, out
        assert out["freed_bytes"] >= 9216, out        # 六张图 + 两份 meta.jsonl + 三份报告
        assert list(probes.iterdir()) == [], f"探针原图没清干净：{list(probes.iterdir())}"
        assert [p.name for p in reports.iterdir()] == [cli_report.name], \
            f"报告目录清乱了：{[p.name for p in reports.iterdir()]}"
        assert gallery_meta.exists() and gallery_feat.exists() and gallery_thumb.exists(), \
            "底库特征文件或缩略图被误删"
        again = srv.api_runs_clear()                   # 空库再点一次不该报错
        assert again["cleared"] == 0 and again["freed_bytes"] == 0, again
    finally:
        srv.RUNS, srv.REPORT_DIR, srv.PROBE_DIR = keep
        shutil.rmtree(tmp, ignore_errors=True)
    print(" 12 清空历史批次：内存/报告/探针图与 meta.jsonl 一起删，cli 报告和底库特征不动  ✓")


def t13_clear_gallery_removes_files():
    """「清空底库」要连缩略图和特征文件一起删。

    只清内存再 save() 等于在磁盘上留一个空库加几千张 JPG：界面不再引用它们，
    /api/thumb 却还认得这些文件名，几百 MB 就一直躺着。删完必须重启还是空库，
    而且不能顺手把别的模型的底库删了。删单个人同理，得带走他自己的缩略图。
    """
    key, other = "selftest_t13", "selftest_t13_other"
    g, other_g = new_gallery(key), new_gallery(other)
    for tid, person, i in (("a", "A", 0), ("b", "B", 1), ("c", "C", 2)):
        g.add(tpl(tid, person, basis(i)))
        (g.thumbs_dir / f"{tid}.jpg").write_bytes(b"jpg" * 512)
    other_g.add(tpl("z", "Z", basis(3)))
    (other_g.thumbs_dir / "z.jpg").write_bytes(b"jpg" * 512)
    g.save()
    other_g.save()
    assert g.db_file.exists(), "底库没落盘，测了个寂寞"

    def clear_via_api(model_key: str, gal: Gallery):
        keep = (srv.GALLERY, srv.STATE["rec"])
        srv.GALLERY, srv.STATE["rec"] = gal, model_key
        try:
            return srv.api_gallery_clear()
        finally:
            srv.GALLERY, srv.STATE["rec"] = keep

    out = clear_via_api(key, g)
    assert (out["persons"], out["templates"]) == (0, 0), out
    assert out["thumbs_removed"] == 3 and out["freed_bytes"] >= 3 * 512, out
    assert not g.db_file.exists(), "数据库还在，重启就复活成一个满库"
    assert list(g.thumbs_dir.iterdir()) == [], f"缩略图没删干净：{list(g.thumbs_dir.iterdir())}"
    assert g.thumbs_dir.is_dir(), "thumbs/ 目录得留着，注册还要往里写"
    assert g.stats()["integrity"] == [], f"清空后不该还挂着告警：{g.stats()}"
    assert Gallery(key).stats()["templates"] == 0, "重启后必须还是空库"
    assert Gallery(other).stats()["templates"] == 1, "别的模型的底库被误删"
    (g.thumbs_dir / "a.jpg").write_bytes(b"jpg" * 512)   # 清空之后能继续注册
    g.add(tpl("a", "A", basis(0)))
    g.save()
    assert g.search(basis(0))[1][0] == "A"

    keep = (srv.GALLERY, srv.STATE["rec"])
    srv.GALLERY, srv.STATE["rec"] = g, key
    try:
        srv.api_gallery_del("A")
    finally:
        srv.GALLERY, srv.STATE["rec"] = keep
    assert not (g.thumbs_dir / "a.jpg").exists(), "删掉的人留下了孤儿缩略图"
    assert Gallery(key).stats()["templates"] == 0, "删人后重启又活了"
    shutil.rmtree(other_g.dir, ignore_errors=True)
    print(" 13 清空底库：模板/特征文件/缩略图一起删，删人带走他的图，重启不复活  ✓")



def t14_sqlite_gallery_pagination():
    """底库落在 SQLite 里，列表按页取。

    几千张底库以前是一次把所有人渲染到界面上，浏览器要卡好几秒；改成服务端分页后
    必须保证：翻页不重不漏、越界页给空列表而不是报错、搜索按字面匹配（人名里带
    % 或 _ 不能被当成通配符）、删掉的人不会留下 0 张的幽灵人员。
    """
    key = "selftest_t14"
    g = new_gallery(key)
    names = [f"1792052059{i:03d}" for i in range(35)]      # 相机自动编号式的纯数字人名
    for i, name in enumerate(names):
        for k in range(2):
            g.add(tpl(f"{name}#{k}", name, basis((i + k) % DIM)))
        (g.thumbs_dir / f"{name}#0.jpg").write_bytes(b"jpg" * 128)
    g.add(Template(tid="zhangsan#0", person="张三", source="张三.jpg", vec=basis(0),
                   box=[0, 0, 9, 9], score=0.5, quality={"ok": False, "aligned": False},
                   thumb=""))                              # 整图入库未对齐 → 要能被数成「待查」
    g.save()
    all_names = sorted(names + ["张三"])

    seen, offset, size = [], 0, 8
    while True:
        pg = g.page(offset=offset, size=size)
        seen += [x["person"] for x in pg["items"]]
        if offset + size >= pg["total"]:
            break
        offset += size
    assert pg["total"] == len(all_names), pg["total"]
    assert sorted(seen) == all_names, f"翻页有重复或漏人：{len(set(seen))}/{len(all_names)}"
    assert g.page(offset=10000, size=8)["items"] == [], "越界页要给空列表"
    assert g.page(size=0)["size"] == 80 and g.page(size=99999)["size"] == 500

    one = g.page(offset=0, size=8)
    assert len(one["items"]) == 8 and one["items"][0]["templates"] == 2, one["items"][:2]
    zs = [x for x in g.page(q="张三")["items"] if x["person"] == "张三"]
    assert zs and zs[0]["bad"] == 1 and zs[0]["thumb"] == "", zs
    assert [x["person"] for x in g.page(q="003")["items"]] == ["1792052059003"], "搜索命中不对"
    assert g.page(q="1%9")["total"] == 0, "LIKE 通配符没转义，搜 % 会把整库都匹配上"

    g2 = Gallery(key)                                       # 重启后必须原样读回
    st = g2.stats()
    assert (st["persons"], st["templates"]) == (len(all_names), len(names) * 2 + 1), st
    assert st["storage"] == "sqlite" and st["db_bytes"] > 0 and st["integrity"] == [], st
    assert g2.names() == all_names
    assert g2.search(basis(0), topk=3)[1], "重启后检索不能变空"

    g2.remove_person("1792052059003")
    g3 = Gallery(key)
    assert "1792052059003" not in g3.persons, "数据库里的人没删掉，重启又活了"
    assert g3.page()["total"] == len(all_names) - 1, g3.page()["total"]
    assert g3.page(q="003")["total"] == 0, "删空的人留下 0 张的幽灵行，页码会对不上"
    print(" 14 底库落 SQLite：重启无损读回，列表分页/搜索/删人都对  ✓")


class _FakeNode:
    """假 onnxruntime 输出描述，只提供 .name。"""

    def __init__(self, name: str):
        self.name = name


class _FakeSession:
    """假 session：按固定名字返回预先造好的三个 stride 的 score/bbox/kps。"""

    def __init__(self, outs: list[np.ndarray]):
        self._names = ([f"score{i}" for i in range(3)] + [f"bbox{i}" for i in range(3)]
                       + [f"kps{i}" for i in range(3)])
        self.outs = outs

    def get_inputs(self):
        return [_FakeNode("input")]

    def get_outputs(self):
        return [_FakeNode(n) for n in self._names]

    def run(self, _out_names, _feed):
        return self.outs


def _chw_outs(size: int, na: int, stride: int, hh: int, ww: int) -> list[np.ndarray]:
    """造一套通道式 [1, C, fh, fw] 输出，只在 (stride, hh, ww) 的第 1 号 anchor 上埋一个高分。"""
    outs = []
    for st in (8, 16, 32):
        fh = size // st
        a = np.zeros((1, na, fh, fh), np.float32)
        if st == stride:
            a[0, 1 % na, hh, ww] = 0.92
        outs.append(a)
    for st in (8, 16, 32):
        fh = size // st
        b = np.zeros((1, 4 * na, fh, fh), np.float32)
        if st == stride:
            # 四条边各留 0.4 个 stride，凑出一个正方形框，方便核对中心
            b[0, (1 % na) * 4:(1 % na) * 4 + 4, hh, ww] = st * 0.4
        outs.append(b)
    for st in (8, 16, 32):
        fh = size // st
        outs.append(np.zeros((1, 10 * na, fh, fh), np.float32))
    return outs


def t15_scrfd_chw_two_anchor_layout():
    """通道式 2-anchor 的检出层不能被整层丢掉。

    回归的那条 bug：后处理先把 score 的 anchor 维 max 掉、网格却还按 na 份展开，
    行数必然差 na 倍，撞上自检就 continue，于是**三个 stride 全丢**，
    检出框恒为 0 而且一声不响。det_2.5g / scrfd-500m 这类小模型正是这个导出，
    在板子上换小模型选型的人会以为「一张脸都检不出来」。
    """
    size, na, hh, ww = 640, 2, 12, 7
    s = Scrfd(_FakeSession(_chw_outs(size, na, 32, hh, ww)), input_size=size)
    levels = s.collect(s.session.run(None, {}), size)
    assert len(levels) == 3, f"三个 stride 只解析出 {sorted(levels)}，疑似整层被丢"
    for st, slot in levels.items():
        assert slot["mode"] == "chw" and slot["na"] == na, slot
        n_grid = s._grid(slot["fh"], slot["fw"], st, na).shape[0]
        assert slot["score"].reshape(-1).shape[0] == n_grid, (
            f"stride={st} 分数行数 {slot['score'].size} 对不上网格 {n_grid}，检出必被丢弃")
        assert slot["bbox"].reshape(-1, 4).shape[0] == n_grid, f"stride={st} 边框行数对不上"

    # 原图 400(高)×300(宽)，letterbox 只右下贴零，反算就是除以 r=min(640/400,640/300)=1.6
    img = np.full((400, 300, 3), 128, np.uint8)
    det, kps = s.detect(img, thresh=0.5)
    assert det.shape[0] == 1, f"2-anchor 通道式检出 {det.shape[0]} 个框，应为 1 个"
    cx, cy = (det[0][0] + det[0][2]) / 2, (det[0][1] + det[0][3]) / 2
    assert abs(cx - ww * 32 / 1.6) < 1.0 and abs(cy - hh * 32 / 1.6) < 1.0, \
        f"框中心 ({cx:.1f},{cy:.1f}) 与 letterbox 反算的 ({ww * 20:.0f},{hh * 20:.0f}) 不符"
    assert kps is not None and kps.shape == (1, 5, 2), "框和关键点必须一一对应"

    # 官方 det_10g 是扁平 1-anchor，改动不能把它带崩
    flat = []
    for st, n in ((8, 6400), (16, 1600), (32, 400)):
        a = np.full((n, 1), 0.01, np.float32)
        if st == 32:
            a[hh * (size // st) + ww, 0] = 0.92
        flat.append(a)
    for st, n in ((8, 6400), (16, 1600), (32, 400)):
        b = np.full((n, 4), 0.3, np.float32)
        flat.append(b)
    flat += [np.zeros((n, 10), np.float32) for n in (6400, 1600, 400)]
    s2 = Scrfd(_FakeSession(flat), input_size=size)
    det2, _ = s2.detect(img, thresh=0.5)
    assert det2.shape[0] == 1, f"扁平 1-anchor（官方 det_10g）回归成 {det2.shape[0]} 个框"
    print(" 15 通道式 2-anchor 检测层不再被静默丢弃，扁平导出无回归  ✓")


def t16_max_templates_per_person_is_atomic():
    """「每人模板上限」在并行注册下不能被超发。

    上限原先在上层「先数再写」，两个 worker 会同时数到还差一张然后双双写入，
    上限就成了「至少这么多」。判定挪到写库那把锁里之后，超发的口子才堵上。
    """
    g = new_gallery("selftest_t16")
    cap, threads, per_thread = 5, 8, 30
    lock = threading.Lock()
    accepted: list[str] = []

    def worker(tid_base: int) -> None:
        for k in range(per_thread):
            i = tid_base * per_thread + k
            t = tpl(f"a{i}", "A", mix(**{"b0": 0.9, f"b{i % (DIM - 1) + 1}": 0.44}))
            if g.add(t, max_per_person=cap):
                with lock:
                    accepted.append(t.tid)

    # 每个线程各造 30 张不同的脸，八线程一起涌向同一个人，名额只有 5 个
    g.add(tpl("warm", "B", basis(3)))          # 先放个人占位，顺便验证不影响别人的名额
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(accepted) == cap, f"8×30 次并发注册，上限 {cap} 张却成功了 {len(accepted)} 次"
    assert len(set(accepted)) == cap, "同一个 tid 被认领了两次"
    assert g.count("A") == cap, f"库里实际存了 {g.count('A')} 条，和上报的对不上"
    assert g.count("B") == 1, "上限把别人也挡住了"
    # 重复注册（同 tid）不算新增名额，满了也得能原地替换
    assert g.add(tpl(accepted[0], "A", basis(0)), max_per_person=cap) is True, \
        "重复注册同一个人的一张旧图被上限挡掉了，重新注册会整批失败"
    assert g.count("A") == cap, "原地替换把底库撑大了"
    assert g.add(tpl("a999", "A", basis(0)), max_per_person=cap) is False, "还能继续超发"
    print(" 16 每人模板上限在并发注册下不超发，重复注册不受影响  ✓")


def t17_bad_images_and_wall_clock_throughput():
    """坏图不进检出率分母，吞吐按墙钟而不是 1000/P50 折算。

    两个都是「不报错但数字是错的」：坏图算进分母会把好检测器冤枉成漏检；
    吞吐按 P50 折算的话，开多路并发单图 P50 涨三倍，数字反而往下掉，
    看着像「加线程更慢」，恰好把选型最需要的结论说反。
    """
    g = new_gallery("selftest_t17")
    for i, person in enumerate(("A", "B", "C")):
        g.add(tpl(person.lower(), person, basis(i)))
    items = [("cam/A/1.jpg", img(1), "A"), ("cam/B/1.jpg", img(2), "B"),
             ("cam/C/1.jpg", img(3), "C"),
             ("cam/A/2.jpg", img(), "A"),                    # 检不到脸：真实漏检，得留在分母
             ("cam/A/3.jpg", None, "A")]                     # 坏图：根本没有输入
    res = run_probe(Stub(VECS), g, items, {"threshold": 0.4, "topk": 5})
    s = res["summary"]
    assert s["n_error"] == 1, f"坏图没被单独数出来：{s['n_error']}"
    assert s["detect_rate"] == round(3 / 4, 4), \
        f"检出率 {s['detect_rate']} 像是把坏图也算进了分母（应为 3/4=0.75，不是 3/5=0.6）"
    assert s["n_rank1_nodetect"] == 1, "有真值却检不到脸的图必须进 Rank-1 分母并被单独计数"
    # 漏检那张图必须在 Rank-1 分母里：4 张有名真值里 3 张认对，Rank-1 就该是 0.75。
    # 以前多了一条「得有 faces」，漏检的图整个退出分母，检测器越差分反而越好看。
    assert s["rank1"] == 0.75, f"Rank-1={s['rank1']}，漏检的图没被算进分母（应为 3/4）"
    assert s["throughput_basis"] == "wall", f"吞吐口径是 {s['throughput_basis']}，并发下会虚高"
    # 墙钟直接喂给 summarize 做精确校验：桩后端跑完不到 1ms，
    # run_probe 里 round(wall, 3) 会把它压成 0.0，没法从返回值反推，只能自己给一个已知的 W。
    # 分母是「可用图」4 张（坏图没有输入，不该算进吞吐），W=0.08 → 50.0 张/秒
    probe = bench_summarize(res["records"], 0.4, 5, wall_s=0.08, workers=4)
    assert probe["throughput_fps"] == 50.0, \
        f"给了 0.08s 墙钟却报 {probe['throughput_fps']} 张/秒，没按「可用图 ÷ 墙钟」算"
    assert probe["throughput_basis"] == "wall" and probe["workers"] == 4
    no_wall = bench_summarize(res["records"], 0.4, 5)
    assert no_wall["throughput_basis"] == "p50", \
        "没给墙钟时必须声明是 P50 折算口径，不能让人误读成真吞吐"

    # 并行只能改速度，不能改判定：每条记录逐字段对齐；
    # 顺便确认按墙钟算出来的并发吞吐真的更高 —— 按 1000/P50 折算的话并发会「更慢」，
    # 那正是当初把这个数报反的地方。桩后端每张脸睡 8ms，四路并行肉眼可见地省时间。
    slow = Stub(VECS)
    slow.delay = 0.008
    serial = run_probe(slow, g, items * 4, {"threshold": 0.4, "topk": 5})
    par = run_probe(slow, g, items * 4, {"threshold": 0.4, "topk": 5, "workers": 6})
    assert [ {k: v for k, v in r.items() if k != "timings"} for r in par["records"] ] == \
        [ {k: v for k, v in r.items() if k != "timings"} for r in serial["records"] ], \
        "并行推理改变了判定结果"
    assert par["summary"]["workers"] == 6 and par["summary"]["throughput_basis"] == "wall"
    assert par["summary"]["detect_rate"] == s["detect_rate"], "并行前后检出率不一致"
    gain = (par["summary"]["throughput_fps"] or 0) / max(1e-9, serial["summary"]["throughput_fps"] or 0)
    assert gain > 1.5, (f"四路并行的墙钟吞吐只有串行的 {gain:.2f} 倍，"
                        "要么并行没生效，要么吞吐又退回按 P50 折算")
    print(f" 17 坏图不进检出率分母、吞吐按墙钟算，并行不改变判定（{gain:.1f}×）  ✓")


def t18_incremental_grouping_matches_full_recount():
    """检索用的「模板→人员」增量登记必须和全量重算逐条一致。

    为了不让每次 add 都在锁里重算一遍 np.unique（3475 人实测 0.64ms，
    而整次检索才 1.18ms），分组改成增量追加。追加错了不会报错，
    只会让「按人取最高分」悄悄把张三的分记到李四头上，所以每加一条就核一次。
    """
    g = new_gallery("selftest_t18")
    people = ["张三", "李四", "王五", "赵六"]
    seq = [(people[(i * 7) % 4], i) for i in range(23)]        # 交错加，逼出下标错位
    for i, (person, k) in enumerate(seq):
        g.add(tpl(f"t{i}", person, mix(**{"b0": 0.8, f"b{k % (DIM - 1) + 1}": 0.6})))
        decoded = [g._uniq[i] for i in g._pid]
        assert decoded == [t.person for t in g.index], \
            f"加到第 {i} 条时分组就错了：{decoded[-3:]} != {[t.person for t in g.index][-3:]}"
    before = ([g._uniq[i] for i in g._pid], sorted(g._uniq))
    g._group()                                                  # 全量重算
    assert [g._uniq[i] for i in g._pid] == before[0], "增量登记和全量重算分的不是一批人"
    assert sorted(g._uniq) == before[1], "人员集合对不上"

    # 分组错了不会报错，但按人聚合的检索会当场露馅：同一个人名下只该出一条
    scores, names = g.search(mix(**{"b0": 0.8, "b1": 0.6}), topk=3)
    assert len(names) == len(set(names)) == 3, f"按人聚合失效，top3 里出现了重名：{names}"
    assert set(names) <= set(people), names
    assert len(scores) == 3 and float(scores[0]) >= float(scores[-1]), scores

    # 分组被外部搞脏（长度对不上）时，下一次 add 必须自己退回全量重算
    g._pid = g._pid[: max(1, g._pid.size - 1)]
    g.add(tpl("late", "新来的", basis(5)))
    assert [g._uniq[i] for i in g._pid] == [t.person for t in g.index], "脏分组没被修回来"
    assert "新来的" in g._uniq
    print(" 18 检索分组增量登记与全量重算逐条一致，脏分组能自愈  ✓")


def t19_reenroll_prunes_stale_templates_and_thumbs():
    """同一个人重新注册，上一次多收的旁人模板和缩略图要一起带走。

    第一次用「全部人脸」收了 3 张脸，第二次改成只取主角，只覆盖主角那条 tid，
    另外两条旁人脸还挂在他名下 —— 检索时把他认成路人，人数统计也看不出异常。
    并行注册撞上限时刚写的缩略图也得删掉，否则底库没涨、盘上却多张孤儿 JPG。
    """
    from app.bench import enroll_image

    g = new_gallery("selftest_t19")
    p_all = {"max_faces": 0, "max_templates": 0}
    out = enroll_image(Stub(VECS), g, img(1, 4, 5), "张三", "张三/全家福.jpg", p_all)
    assert sum(1 for o in out if o.get("tid")) == 3, out
    assert g.count("张三") == 3, g.count("张三")

    # 改成只取最大的一张脸重新注册：旁人的两条模板必须消失，缩略图不能留在盘上
    out2 = enroll_image(Stub(VECS), g, img(1, 4, 5), "张三", "张三/全家福.jpg",
                        {"max_faces": 1, "max_templates": 0})
    assert g.count("张三") == 1, f"重新注册后还挂着 {g.count('张三')} 条模板，旁人没被清掉"
    assert any("清掉了上一次的旧模板" in " ".join(o.get("skipped") or []) for o in out2), \
        "清了旧模板却没上报，用户会以为底库还是 3 张"
    left = sorted(p.name for p in g.thumbs_dir.glob("*.jpg"))
    assert len(left) == 1, f"thumbs/ 里还剩 {left}，缩略图泄漏了"
    assert g.search(basis(1), topk=3)[1][:1] == ["张三"], "清完模板反而认不出本人了"

    # 并发撞上限：不管谁抢到名额，最后「磁盘上的 jpg 数」必须等于「模板数」
    g2 = new_gallery("selftest_t19_other")
    cap, threads = 6, 6
    lock = threading.Lock()
    report: list[list] = []

    def worker(i: int) -> None:
        r = enroll_image(Stub(VECS), g2, img(1), "李四", f"李四/证件照{i}.jpg",
                         {"max_faces": 1, "max_templates": cap})
        with lock:
            report.append(r)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(threads * 3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert g2.count("李四") == cap, f"并发注册把上限 {cap} 撑到 {g2.count('李四')}"
    thumbs = sorted(p.name for p in g2.thumbs_dir.glob("*.jpg"))
    assert len(thumbs) == g2.count("李四"), \
        f"盘上 {len(thumbs)} 张缩略图但库里只有 {g2.count('李四')} 条模板，撞上限的没回收"
    print(" 19 重新注册清掉旧模板，撞上限的缩略图不泄漏  ✓")


def t20_readonly_gallery_fails_loudly():
    """底库落不进去时必须当场响，不能报「成功」等重启以后少人。

    底库目录被设成只读、磁盘写满、gallery.db 被另一个进程占着，
    以前 _write / _delete_db_person 只往 integrity 里塞一句话就 return：
    注册那边一路绿灯，人数在涨、界面全对，重启这批人凭空消失。
    顺带验内存和磁盘的改动顺序 —— 半途失败不能留下「库里有人、图上没头像」
    或者「界面上没这个人、库里行还在」这种两边都对不上的中间状态。
    """
    g = new_gallery("selftest_t20")
    g.add(tpl("a", "A", basis(0)))
    g.add(tpl("c1", "C", basis(2)))
    g.add(tpl("c2", "C", basis(3)))
    (g.thumbs_dir / "a.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    g.save()
    g._close_db()                      # WAL 落盘并断开，才轮得到目录权限说话
    mode = g.db_file.parent.stat().st_mode
    for suffix in ("-wal", "-shm"):    # 留着的话写入还是能进 WAL，测不出只读
        Path(str(g.db_file) + suffix).unlink(missing_ok=True)
    os.chmod(g.db_file.parent, 0o555)
    try:
        try:
            g.add(tpl("d", "D", basis(4)))
            raise AssertionError("只读底库上注册居然静默成功了，重启后这个人会凭空消失")
        except RuntimeError as e:
            assert "不可用" in str(e) or "落库失败" in str(e), e
        assert "D" not in g.persons, \
            "落库失败但人已经进了内存：检索会命中一个数据库里根本不存在的幽灵"
        try:
            g.remove_person("A")
            raise AssertionError("只读底库上删人居然静默成功了，数据库里的行一条没少")
        except RuntimeError as e:
            assert "不可用" in str(e), e
        assert "A" in g.persons, "数据库没删成人却先从界面上消失了，重启以后他会复活"
        assert (g.thumbs_dir / "a.jpg").is_file(), "库没删成就先删了头像，复活以后头像全是 404"
    finally:
        os.chmod(g.db_file.parent, mode)
    g.remove_person("A")               # 恢复可写：同一个人就该删得掉，头像跟着走
    assert "A" not in g.persons and not (g.thumbs_dir / "a.jpg").exists(), \
        "恢复可写以后删人不干净，说明前面的失败把状态弄坏了"
    print(" 20 底库只读时注册/删人当场报错，不留半套状态  ✓")


def t21_person_list_not_truncated_by_page_size():
    """全员名单必须真的全：分页上限是给一屏渲染用的，不是给「取全部」用的。

    以前 person_list() 写的是 page(size=10**6)，被 page() 的 500 上限夹住，
    三千多人的底库只回 500 条，还和同一份响应里的 stats.persons 直接打架。
    """
    g = new_gallery("selftest_t21")
    total = 520
    for i in range(total):
        g.add(tpl(f"p{i}", f"路人{i:03d}", basis(i % DIM)))
    assert g.stats()["persons"] == total, g.stats()
    listed = g.person_list()
    assert len(listed) == total, \
        f"名单只有 {len(listed)} 人，底库实际 {total} 人：分页上限把「取全部」夹住了"
    assert {p["person"] for p in listed} == {f"路人{i:03d}" for i in range(total)}, "翻页拼起来少了人"
    print(" 21 全员名单不再被分页上限截在 500 条  ✓")


def t22_threshold_tie_prefers_higher_cut():
    """TAR 并列时必须推荐**更高**的阈值：放行率一样，冒认者必须更少。

    genuine 样本一少，TAR 就是几个百分点一格的粗台阶，台阶顶上必然并列一片点，
    而 max() 取的是第一个最大值 —— 正好挑走最左边（最宽松）那一档。
    真机底库上实测过：并列 4 个点，t=0.5251/far=8.5e-5 与 t=0.5545/far=3.5e-5，
    放行率一模一样，万库累计误报却从 0.57 涨到 0.30 的差距白扔了。
    """
    # 9999 个冒认者 0.30 分、1 个 0.55 分：阈值只要高于 0.30，FAR 就落到 1e-4 以内；
    # genuine 16 张里 11 张 0.9 分，TAR 台阶顶是 11/16，顶上并排一百多个点
    imp = np.concatenate([np.full(9999, 0.30, np.float32), np.full(1, 0.55, np.float32)])
    gen = np.array([0.9] * 11 + [0.2] * 5, np.float32)
    rep = sweep_thresholds(gen, imp, steps=200)
    top = [c for c in rep["curve"] if c["far"] <= 1e-4 and c["tar"] == rep["best"]["tar"]]
    assert len(top) >= 2, f"这个用例没造出并列点（只有 {len(top)} 个），换一组分数再试"
    assert rep["best"]["t"] == max(c["t"] for c in top), \
        f"并列 {len(top)} 档却推荐了 t={rep['best']['t']}，取了最宽松那一档"
    assert rep["best"]["far"] == min(c["far"] for c in top), \
        f"同一放行率下 FAR 不是最低：{rep['best']['far']}，应取 {min(c['far'] for c in top)}"
    # 1e-6 档反过来：那里 TAR 常常是平的 0，要挑并列里**最低**的阈值，
    # 否则贴着「几乎全拒」还会顺手把还能放行的人关在门外
    floor = [c for c in rep["curve"] if c["far"] <= 1e-6 and c["tar"] == rep["at_far_1e6"]["tar"]]
    assert rep["at_far_1e6"]["t"] == min(c["t"] for c in floor), \
        f"FAR<=1e-6 档并列 {len(floor)} 个，取了 t={rep['at_far_1e6']['t']}，应取最低那档"
    assert pct([]) == pct([]) and set(pct([])) == set(pct([1.0, 2.0])), \
        "空样本与有样本的分位数结构不一致，按固定键遍历会 KeyError"
    print(" 22 阈值推荐在 TAR 并列时取更高一档（FAR 更低），空分位数结构一致  ✓")


def t23_subject_face_labeled_unknown_not_double_counted():
    """主体脸被手工改成「非底库」、整图真值却是有名的人：一次判定只能算一次。

    图级按整图真值算「认对张三」，脸级按脸级标签算「非底库被误放行」，
    同一张脸的同一件事进了两个口径，far_estimated 能离谱到 100%，
    混淆表还会印出「非底库人员 → 张三」这种自相矛盾的行。
    """
    g = new_gallery("selftest_t23")
    for tid, person, i in (("a", "A", 0), ("b", "B", 1)):
        g.add(tpl(tid, person, basis(i)))
    # 认对的情形：探针里是 A，图级真值也是 A
    ok_res = run_probe(Stub(VECS), g, [("cam/A/1.jpg", img(1), "A")],
                       {"threshold": 0.4, "topk": 5})
    rec = ok_res["records"][0]
    assert rec["rank1_correct"] and rec["faces"][0]["is_subject"], rec
    rec["faces"][0]["truth"] = UNKNOWN
    recompute(rec, 0.4)
    s = res_summary([rec], 0.4, 5)
    assert s["n_truth_conflict_faces"] == 1 and s["n_unknown_faces"] == 0, s
    assert s["rank1"] == 1.0 and s["false_accept"] == 0, s
    assert s["face_false_accept"] == 0, \
        f"冲突的脸又被算了一次脸级误放行（face_false_accept={s['face_false_accept']}）"
    assert not any(k.startswith(UNKNOWN_LABEL) for k, _ in s["confusions"]), \
        f"混淆表里还有自相矛盾的行：{s['confusions']}"
    # 认错的情形：探针里是 B，图级真值写 A —— 图级记一次「A → B」就够了
    bad = run_probe(Stub(VECS), g, [("cam/A/2.jpg", img(2), "A")],
                    {"threshold": 0.4, "topk": 5})["records"][0]
    assert not bad["rank1_correct"] and bad["accepted"], bad
    bad["faces"][0]["truth"] = UNKNOWN
    recompute(bad, 0.4)
    s2 = res_summary([bad], 0.4, 5)
    assert s2["n_truth_conflict_faces"] == 1 and s2["face_false_accept"] == 0, s2
    assert s2["false_accept"] == 1, f"图级必须仍然记下这次误放行：{s2['false_accept']}"
    assert ["A → B"] == [k for k, _ in s2["confusions"]], \
        f"误放行只该按真名记一行，实际 {s2['confusions']}"
    print(" 23 主体脸与整图真值打架时只按整图真值记一次  ✓")


def t24_model_hash_cache_actually_hits():
    """模型摘要缓存必须真命中：按路径分槽，别让后一个模型把前一个挤掉。

    /api/env 前端每 20 秒轮一次，一次连问四个模型（本机 455 MB）。
    原先算完一个就 _SHA_CACHE.clear()，四个模型互相挤掉，实测轮询三次
    十二次查询零命中，每 20 秒白烧 0.3 秒 CPU（RK3588 的 A55 上 1~3 秒，
    还正好和被测的推理抢核，把吞吐数字一起拖低）。
    """
    tmp = Path(tempfile.mkdtemp(prefix="fb_sha_"))
    calls: list[str] = []
    orig = cfg._hash_file
    try:
        a, b = tmp / "a.onnx", tmp / "b.onnx"
        a.write_bytes(b"x" * 4096)
        b.write_bytes(b"y" * 4096)
        cfg._hash_file = lambda p, c: (calls.append(p.name), orig(p, c))[1]
        try:
            first = (cfg.sha256_of(a), cfg.sha256_of(b))
            assert calls == ["a.onnx", "b.onnx"], f"首轮就该各算一次：{calls}"
            for _ in range(3):                                   # 模拟连续轮询
                again = (cfg.sha256_of(a), cfg.sha256_of(b))
            assert again == first, "命中缓存却算出了另一个摘要"
            assert calls == ["a.onnx", "b.onnx"], f"轮询又重算了：{calls}"
        finally:
            cfg._hash_file = orig
        b.write_bytes(b"z" * 8192)          # 重新下载模型：大小变了必须重算
        assert calls == ["a.onnx", "b.onnx"], "文件大小变了却没重算，缓存会把旧摘要当成新的"
        assert cfg.sha256_of(b) != first[1], "换文件后摘要没变"
    finally:
        cfg._hash_file = orig
        shutil.rmtree(tmp, ignore_errors=True)
    print(" 24 模型摘要按路径分槽缓存，轮询不再重算，换文件自动失效  ✓")


def t25_delete_single_run_removes_report_and_images():
    """删单个批次只能删掉它自己，而且报告与探针图必须成对消失。

    只有「清空全部」时，一次跑歪的评测（阈值设错、探针目录选错）要把旁边
    几个好批次一起删掉重跑，而一万六千张探针要跑十几分钟。反过来删干净也重要：
    报告和图只删掉一半，剩下的报告会变成重启后又读回来的删不掉的孤儿，
    或者留下几百 MB 没人认领的探针目录。批次号还要挡在拼路径之前，
    否则 ../../etc 这种直接变成任意目录删除。
    """
    tmp = Path(tempfile.mkdtemp(prefix="selftest_t25_"))
    reports, probes = tmp / "reports", tmp / "probes"
    reports.mkdir()
    for rid in ("keep0001", "gone0002"):
        d = probes / rid
        d.mkdir(parents=True)
        (d / "0.jpg").write_bytes(b"x" * 512)
        (d / "0-f0.jpg").write_bytes(b"x" * 256)
        (d / "meta.jsonl").write_text('{"i": 0}\n', encoding="utf-8")
        (reports / f"probe_{rid}.json").write_text('{"records": [1]}', encoding="utf-8")
    # 只剩报告的孤儿（探针图被手工删过）：仍然要能从界面上删掉
    (reports / "probe_orphan3.json").write_text('{"records": [1]}', encoding="utf-8")
    cli = reports / "cli_20261107_000000.json"
    cli.write_text('{"summary": {}}', encoding="utf-8")

    keep = (srv.RUNS, srv.REPORT_DIR, srv.PROBE_DIR)
    srv.RUNS = {r: {"records": [1]} for r in ("keep0001", "gone0002", "orphan3")}
    srv.REPORT_DIR, srv.PROBE_DIR = reports, probes
    try:
        out = srv.api_run_delete("gone0002")
        assert out["cleared"] == 1 and out["failed"] == [], out
        assert not (probes / "gone0002").exists(), "探针目录还在：删了一半"
        assert not (reports / "probe_gone0002.json").exists(), "报告还在：删了一半"
        assert out["runs"] == 2 and set(srv.RUNS) == {"keep0001", "orphan3"}, srv.RUNS
        assert out["freed_bytes"] >= 700, out
        assert (probes / "keep0001" / "0.jpg").exists(), "误删了别的批次的探针图"
        assert (reports / "probe_keep0001.json").exists(), "误删了别的批次的报告"
        assert cli.exists(), "命令行的 cli_*.json 被误删"

        orphan = srv.api_run_delete("orphan3")     # 只剩报告也要删得掉
        assert orphan["cleared"] == 1 and "orphan3" not in srv.RUNS, orphan

        # 越界与不存在的批次号必须被拒，而且拒了就不能动过磁盘
        for bad, code in (("nosuch99", 404), ("../etc", 400), ("abc", 400)):
            try:
                srv.api_run_delete(bad)
            except srv.HTTPException as e:
                assert e.status_code == code, f"{bad} 应返回 {code}，实际 {e.status_code}"
            else:
                raise AssertionError(f"{bad} 该被拒，却返回了成功")
        assert (probes / "keep0001" / "meta.jsonl").exists(), "被拒的请求动到了磁盘"
    finally:
        srv.RUNS, srv.REPORT_DIR, srv.PROBE_DIR = keep
        shutil.rmtree(tmp, ignore_errors=True)
    print(" 25 删单个批次：只删它自己，报告与探针图成对消失，越界批次号被拒  ✓")


def t26_delete_single_person_via_api():
    """界面上删一个人只能删掉他自己，人不在库里还得报错。

    以前只有整库清空：底库里混进一个错标的人（把路人注册成了张三），
    只能把几千张全删了重注册。改成按人删之后要盯两件事：
    一是别顺手带走别人的模板和头像（头像删一半，重启就是一堆 404）；
    二是库里根本没这个人时必须报错 —— remove_person 对不存在的人
    只返回 persons_removed=0 不抛异常，接口照这返回 200 的话，
    界面会弹「已删除张三」，可张三一直在库里，用户再也找不到他去哪了。
    """
    key = "selftest_t26"
    g = new_gallery(key)
    for i, name in enumerate(("张三", "李四")):
        for k in range(2):
            g.add(tpl(f"{name}#{k}", name, basis(i * 2 + k)))
        (g.thumbs_dir / f"{name}#0.jpg").write_bytes(b"jpg" * 128)
        (g.thumbs_dir / f"{name}#1.jpg").write_bytes(b"jpg" * 128)
    g.save()

    orig = srv.get_gallery
    srv.get_gallery = lambda: g
    try:
        out = srv.api_gallery_del("张三")
        assert out["persons_removed"] == 1 and out["templates_removed"] == 2, out
        assert out["persons"] == 1 and out["templates"] == 2, f"统计还是两人的：{out}"
        # 界面提示「删了几张头像、释放多少」全靠这两个字段，缺了就是整句 0：
        # 几百 MB 的底库删一个人只回一句「已删除」，没人相信盘真的空了
        assert out["thumbs_removed"] == 2, out
        assert out["freed_bytes"] >= 2 * 384, out
        assert not g.has_person("张三") and g.has_person("李四"), sorted(g.persons)
        assert not (g.thumbs_dir / "张三#0.jpg").exists(), "他的头像没跟着删：thumbs 里变孤儿"
        assert not (g.thumbs_dir / "张三#1.jpg").exists(), "他的头像没跟着删：thumbs 里变孤儿"
        assert (g.thumbs_dir / "李四#0.jpg").exists(), "误删了别人的头像"
        assert (g.thumbs_dir / "李四#1.jpg").exists(), "误删了别人的头像"

        for bad, code in (("王五", 404), ("", 400)):
            try:
                srv.api_gallery_del(bad)
            except srv.HTTPException as e:
                assert e.status_code == code, f"{bad!r} 应返回 {code}，实际 {e.status_code}"
            else:
                raise AssertionError(f"{bad!r} 该被拒，却返回了成功")
    finally:
        srv.get_gallery = orig

    # 删除必须真的落盘：重启后又把人读回来等于白删
    g2 = Gallery(key)
    assert not g2.has_person("张三"), "删了没落盘，重启张三又回来了"
    assert g2.has_person("李四"), "李四被一起删掉了"
    assert g2.stats()["templates"] == 2, g2.stats()
    shutil.rmtree(gallery_dir_for(key), ignore_errors=True)
    print(" 26 删单个人：只删他自己、头像成对消失、没这个人时报错  ✓")


def t27_undecodable_image_reported_as_error():
    """坏图如实记成 error，不能被一张假像素冒充成「漏检」。

    `_probe_prepare` 以前在 decode() 抛异常时塞进一张 112×112 纯黑小图接着往下跑，
    等于骗下游「这是张正常照片，只是没检出脸」。三个后果全是静默的，
    而且恰好都落在选型要看的眼前：白跑一次检测推理还挤进 detect_rate 的分母
    （坏图被冤枉成漏检，检测器越跑越难看）；bench._probe_one 里专为
    img is None 写的 error 分支在生产路径永远走不到，n_error 恒等于 0，
    人跑去调检测阈值，调半天没用；真值也没算，坏图一律掉进「非底库」那一类，
    被放行时还倒过来污染 FAR。顺带守住另一半：拖阈值 / 改真值时
    `recompute` 不能把这种根本没评出来的记录洗白成「认对了」。
    """
    key = "selftest_t27"
    g = new_gallery(key)
    g.add(tpl("z1", "张三", basis(0)))
    g.add(tpl("l1", "李四", basis(1)))
    g.save()
    out = DATA_DIR / "selftest_t27_out"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    # ctx 手工拼，不走 _probe_setup（那要真后端才能建）。补丁必须打在 srv.decode 上：
    # server.py 是 `from .imaging import decode`，打 app.imaging 的话
    # _probe_prepare 里的 decode 这个名字仍绑着真函数，补丁等于没打。
    ctx = {"out_dir": out, "bare": False, "person_mode": "name", "delimiter": "__",
           "max_side": 1280, "persons": set(g.persons), "raw_of": {}}
    orig_decode = srv.decode

    def fake_decode(raw, max_side=1600):
        if raw == b"bad":
            raise ValueError("无法解码图片，可能已损坏或不是图片")
        # 必须是二维图：桩后端把像素值当人脸编号，(1,1,3) 摊平是 3 张脸，
        # 编号 2 → vecs[2] → 李四。给一维数组 limit_side 会拿 shape[:2] 取错维度。
        return np.full((1, 1, 3), 2, np.uint8)

    srv.decode = fake_decode
    try:
        batch = [("李四.jpg", "李四.jpg", b"good"), ("张三.jpg", "张三.jpg", b"bad")]
        items = srv._probe_prepare(ctx, batch, 0)
        assert len(items) == 2, items
        assert items[1][1] is None, "坏图没被留成 None，下游又拿假像素去推理了"
        assert items[1][2] == "张三", f"坏图的真值没按文件名算出来：{items[1][2]!r}"
        lines = [json.loads(x) for x in (out / "meta.jsonl").read_text("utf-8").splitlines()]
        assert [x["name"] for x in lines] == ["李四.jpg", "张三.jpg"], lines
        assert lines[0]["thumb"] == "0.jpg" and (out / "0.jpg").exists(), lines
        assert lines[1]["thumb"] == "", \
            "坏图没有像素可以缩略，meta 里得留空，不然逐图复核就是一枚裂图图标"
        assert not (out / "1.jpg").exists(), "坏图凭空多出一张不存在的整图缩略"

        res = run_probe(Stub(VECS), g, items, {"threshold": 0.4, "topk": 5})
        s = res["summary"]
        assert s["n"] == 2 and s["n_faces"] == 3, s
        assert s["n_error"] == 1, f"n_error={s['n_error']}，坏图还是没被单独数出来"
        assert s["detect_rate"] == 1.0, \
            f"检出率 {s['detect_rate']}：坏图还赖在分母里（能评的 1 张全检出，该是 1.0）"
        # 坏图有真值、没脸：它进「有真值」这个数（说明真值还在，人看得见），
        # 但不进 Rank-1 也不进任何误识分母，否则要么冤枉检测器要么凭空给 FAR 加样本
        assert s["n_truth_known"] == 2, f"坏图的真值丢了：n_truth_known={s['n_truth_known']}"
        assert s["n_truth_named"] == 1, f"坏图混进了 Rank-1 分母：n_truth_named={s['n_truth_named']}"
        assert s["n_truth_unknown"] == 0 and s["n_unknown_faces"] == 0, \
            "坏图被当成了「非底库人员」，放行时反过来污染 FAR"
        bad = [r for r in res["records"] if r["name"] == "张三.jpg"][0]
        assert bad["verdict"] == "error" and bad["faces"] == [], bad
        assert bad["truth"] == "张三", f"坏图判定没有真值：{bad['truth']!r}"
        assert "解码" in bad["error"], f"失败原因没说是输入问题：{bad['error']!r}"
        assert bad["timings"]["total_ms"] >= 0 and not bad["faces_found"], \
            "坏图根本没输入，检出脸数必须是 0"

        # 另一半：这张根本没评出来的记录，不能被拖阈值 / 改真值洗白成「认对了」
        good = [r for r in res["records"] if r["name"] == "李四.jpg"][0]
        assert good["verdict"] == "correct", good["verdict"]
        good["verdict"] = "error"
        good["error"] = "RuntimeError: 推理炸了"
        recompute(good, 0.4)
        assert good["verdict"] == "error", "带脸的失败记录被阈值/真值重算洗白，n_error 会莫名变小"
        assert good["error"] == "RuntimeError: 推理炸了", "重算顺手抹掉了失败原因"
        # 但正常记录该照常改判，闸门锁的只有 error。桩向量和底库那条完全相同，
        # cos=1.0，所以得抬到 1.01 才翻得动，抬到 0.99 反而会误判成「重算失效」
        good["verdict"] = "correct"
        recompute(good, 1.01)
        assert good["verdict"] == "reject", f"阈值重算失效：{good['verdict']}"
    finally:
        srv.decode = orig_decode
        shutil.rmtree(out, ignore_errors=True)
    print(" 27 坏图如实记 error、不进任何分母，且不会被阈值/真值重算洗白  ✓")


def t28_remote_gallery_key_stays_inside_data_dir():
    """remote 模式的模型键带斜杠时，底库目录仍要是 data/ 下面扁平的一层。

    remote 模式下键是 f"remote_{remote_url}"，而 URL 天生带斜杠：以前把键直接拼进
    目录名，http://10.0.0.9:8080/v1/embed 会在 data/ 下面一路建出
    gallery_remote_http:/10.0.0.9:8080/v1/embed/thumbs 四五层以冒号结尾的目录，
    缩略图散在里面、清空底库时容易只删掉最外层；URL 路径里塞一段 .. 更是能把
    整个底库建到 data/ 外面去（跟当初 _apply_backend_state 要校验 rec 是同一类问题，
    只是这条从 remote_url 绕进来）。
    """
    key = "remote_http://10.0.0.9:8080/v1/embed"
    d = gallery_dir_for(key)
    assert d.parent == DATA_DIR, f"底库不在 DATA_DIR 下：{d}"
    assert ":" not in d.name and "/" not in d.name, f"目录名没扁平化：{d.name}"
    assert gallery_dir_for(key) == d, "同一个 URL 每次算出不同目录，重启就读不到库了"
    # 已经跑了很久的纯键必须原样映射，否则老底库直接凭空消失
    assert gallery_dir_for("w600k_r50").name == "gallery_w600k_r50", \
        "改了纯键的目录名，data/gallery_w600k_r50 会读不到"
    # 只在一个字后缀上不同的两个服务，不能因为截断撞进同一个底库
    pre = "http://h/" + "a" * 120 + "/"
    assert gallery_dir_for("remote_" + pre + "x") != gallery_dir_for("remote_" + pre + "y"), \
        "长 URL 截断后撞名，两个底库会共用一个目录"
    # .. 不能把库带出 DATA_DIR
    esc = gallery_dir_for("remote_http://h/../../../../Users/windinchen/work/项目任务/人脸识别/_rp/face-bench/_tmp/fb_escape_t28")
    assert esc.parent == DATA_DIR, f".. 让底库逃出了 DATA_DIR：{esc}"
    assert not Path("/Users/windinchen/work/项目任务/人脸识别/_rp/face-bench/_tmp/fb_escape_t28").exists(), "目录真的建到 /tmp 下面去了"

    g = new_gallery(key)
    try:
        assert g.dir == d and g.thumbs_dir == d / "thumbs", (g.dir, g.thumbs_dir)
        g.thumbs_dir.mkdir(parents=True, exist_ok=True)
        stray = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir() and ":" in p.name)
        assert not stray, f"键里的冒号又开出了嵌套目录：{stray}"
        assert g.stats()["model_key"] == key, "换目录名不能改统计里报的模型键"
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print(" 28 remote 键带斜杠时底库扁平不越界，纯键目录名照旧  ✓")


def t29_frontend_kpi_keys_match_summary():
    """前端 KPI 引用的每个 summary 键，服务端必须真的产出。

    界面上的数字全是 `${s.xxx}` 插值：服务端改了键名前端不会报错，只会取到 undefined
    走进 `? :` 的假分支，那半句就静默消失。`n_truth_nodetect` 就是这么哑掉的 ——
    「含 N 张没检到脸已算失败」在任何批次上都不显示，检测器越差越没人知道难看在哪儿，
    而自检一直只盯服务端侧的键名（t17 断言 n_rank1_nodetect 存在），两边对不上照样全绿。
    这里改成拿真实 summary 逐个核对前端引用的键，拼错、改名、少产出当场红。
    """
    appjs = Path(__file__).resolve().parent.parent / "static" / "app.js"
    src = appjs.read_text(encoding="utf-8")
    refs: list[str] = []
    for fn in ("kpiHeaders", "kpiExtra", "kpiThroughput"):
        # 抓不到函数就直接抛：KPI 渲染被改名等于这块自检失去靶子，不能默默放过
        i = src.index("function " + fn + "(")
        refs += re.findall(r"\bs\.([A-Za-z_][A-Za-z0-9_]*)", src[i:src.index("\n}\n", i)])
    assert refs, "app.js 里没抓到任何 s.<键> 引用，抓取方式失效了，这条自检等于没跑"

    g = new_gallery("selftest_t29")
    for i, person in enumerate(("A", "B", "C")):
        g.add(tpl(person.lower(), person, basis(i)))
    items = [("cam/A/1.jpg", img(1), "A"), ("cam/B/1.jpg", img(2), "B"),
             ("cam/C/1.jpg", img(3), "C"),
             ("cam/A/2.jpg", img(), "A"),        # 有真值却检不到脸 → n_rank1_nodetect
             ("cam/A/3.jpg", None, "A")]         # 坏图 → n_error
    s = run_probe(Stub(VECS), g, items, {"threshold": 0.4, "topk": 5})["summary"]
    missing = sorted({k for k in refs if k not in s})
    assert not missing, (f"前端 KPI 引用了 summary 里不存在的键 {missing}："
                         "界面对应的那半句会静默不显示，不报错也没人发现")
    # `recall@${k}` 这种模板键正则抓不到，按界面实际取的那两档单独核对
    for k in ("recall@5", "face_recall@5"):
        assert k in s, f"summary 里没有 {k}，Recall 那一格会显示 undefined"
    assert s["n_rank1_nodetect"] == 1 and s["n_error"] == 1, \
        f"这批本该同时构造出漏检和坏图：{s['n_rank1_nodetect']} / {s['n_error']}"
    print(f" 29 前端 KPI 引用的 {len(set(refs))} 个 summary 键服务端全都产出  ✓")


def t30_impostor_scores_stay_inside_cosine_range():
    """底库模板数不齐时，冒认分数不能被补齐用的空槽抬过余弦上界。

    抽样把每人模板补齐成 (人数, 最多模板数, 维度) 的方阵，空槽填整条 -1.0。
    这个占位向量的模长是 sqrt(维度) 而不是 1，它和探针的点积等于 -sum(探针)，
    根本不是余弦；原来对整行直接取 max，只要被比对的人模板数少于 mmax，
    空槽就常常赢下 max，冒认分布右尾被造假抬高（本机 w600k_r50 真实向量实测
    最高 2.58、12.5% 的样本 > 1.0）。far 于是虚高到阈值拉到 1.0 都压不住：
    满足 far<=1e-4 的档位整段消失，推荐阈值退化成 t=1.0/tar=0、at_far_1e6 变 null，
    整栏「阈值建议」等于没有。而 README 恰好推荐「给几个人再补一张不同场合的照片」，
    正好制造模板数不一致 —— 越按建议认真建库，这一栏坏得越彻底。
    """
    rng = np.random.default_rng(7)
    dim = 128                       # 用高一点的空间模拟 512 维真实特征：8 维随机向量本身
    mat, persons = [], []           # 就能撞出 0.9 的余弦，看不出空槽造假的那一分
    for i in range(30):
        for _ in range(1 + i % 5):          # 每人 1~5 张，故意不齐，保证绝大多数人有空槽
            v = rng.normal(size=dim).astype(np.float32)
            mat.append(v / np.linalg.norm(v))
            persons.append(f"p{i}")
    mat = np.stack(mat)
    s, info = sample_impostors(mat, persons, max_pairs=4000)
    assert s.size > 500, f"冒认对没抽够，这条自检没跑到：{info}"
    assert float(s.max()) <= 1.0 + 1e-5, \
        f"冒认分 {float(s.max()):.4f} 越过余弦上界 1.0：补齐的空槽又把 max 抢走了"

    g = np.full(200, 0.72, np.float32)      # 同一人互配的典型分数
    sw = sweep_thresholds(g, s)
    assert sw["best"] and sw["best"]["tar"] > 0 and sw["best"]["t"] < 1.0, \
        f"阈值建议被空槽污染成 {sw['best']}"
    assert sw["at_far_1e6"] is not None, \
        "far<=1e-4 的档位整段消失，界面「阈值建议」那一栏会空掉"
    # 人人一张模板时（README 推荐的最常见形态）本来就没有空槽，口径必须一点没变
    only = list(dict.fromkeys(persons))
    one = np.stack([mat[persons.index(nm)] for nm in only])
    s1, _ = sample_impostors(one, only, max_pairs=4000)
    assert float(s1.max()) <= 1.0 + 1e-5, "人人一张的口径被改坏了"
    assert float(s1.max()) < float(s.max()), \
        "模板数不齐的冒认分布反而比人人一张更低，这个用例没构造出空槽，断言失去鉴别力"
    print(" 30 模板数不齐时空槽不再冒充高分，冒认分仍在 [-1,1]、阈值建议不消失  ✓")


def t31_cli_regression_does_not_wipe_web_gallery():
    """命令行回归要用自己的底库，不能把界面上登记的那份删掉。

    cli/run.py 以前直接 Gallery(a.rec)：拿到的就是 Web 在用的 data/gallery_<key>，
    而 clear() 现在是真的删文件（gallery.db/-wal/-shm、向量、全部缩略图），于是
    「在板子上跑一次命令行回归」等于把用户注册好的底库整个清掉换成 --gallery 的内容。
    另一个坑是 --gallery 路径写错：rglob 安静返回空、不抛异常，删库却已经发生了，
    屏幕只剩「0 人 / 0 模板」，随后还拿空库跑完整批探针。
    """
    from cli.run import cli_gallery_key, collect_required

    g = new_gallery("selftest_t31")
    g.add(tpl("a", "A", basis(0)))
    g.save()
    cg = Gallery(cli_gallery_key("selftest_t31"))
    try:
        assert cg.dir != g.dir, "命令行还在用界面那份底库目录"
        assert cg.dir.parent == DATA_DIR, f"命令行底库跑到了 DATA_DIR 外面：{cg.dir}"
        cg.clear()
        assert g.stats()["templates"] == 1, f"命令行清库把界面底库也清了：{g.stats()}"
        assert g.db_file.exists(), "界面底库的 gallery.db 被命令行的 clear() 删了"
    finally:
        shutil.rmtree(cg.dir, ignore_errors=True)
        shutil.rmtree(g.dir, ignore_errors=True)

    empty = DATA_DIR / "selftest_t31_empty"
    empty.mkdir(parents=True, exist_ok=True)
    try:
        for bad in (empty, DATA_DIR / "selftest_t31_missing"):
            try:
                collect_required(bad, "底库目录")
            except SystemExit as e:
                assert str(bad) in str(e), f"报错没带上填错的路径，看不出哪儿写错了：{e}"
            else:
                raise AssertionError(f"{bad} 里没有图片却静默返回，底库会被白白清掉")
        (empty / "x.jpg").write_bytes(b"x")
        assert [p.name for p in collect_required(empty, "底库目录")] == ["x.jpg"]
    finally:
        shutil.rmtree(empty, ignore_errors=True)
    print(" 31 命令行回归用独立底库不删界面的库，底库目录为空当场报错  ✓")


def t32_id_num_keeps_pure_numeric_names_in_truth():
    """纯数字文件名的探针不能被踢出真值口径。

    id_num 是给「张三_0001.jpg」这类摄像头命名用的。12345.jpg 正则同样匹配，
    只是剥完尾部序号什么都不剩，原来落到 `or None` 判成「推不出真值」：这批图静默
    退出 Rank-1/Recall 的分母，而 detect_rate 的分母仍是全部图，两个指标分母悄悄打架，
    n_truth_known 比图片数少且界面毫无提示，换个命名规则 Rank-1 就变还没法解释。
    现在跟 prefix 模式同一个口径：交回整个文件名，由底库去命中，命中才算数、不造假真值。
    """
    assert infer_person("cam/12345.jpg", "12345.jpg", "id_num") == "12345", \
        "纯数字文件名推不出真值，整批会静默退出 Rank-1 分母"
    assert infer_person("cam/0007.jpg", "0007.jpg", "id_num") == "0007"
    # 原本剥得出来的命名不能受影响，人名里的数字也不许被剥掉
    assert infer_person("cam/张三_0002.jpg", "张三_0002.jpg", "id_num") == "张三"
    assert infer_person("cam/li4-0012.jpg", "li4-0012.jpg", "id_num") == "li4"
    assert infer_person("cam/x.jpg", "x.jpg", "id_num") == "x"
    assert infer_person("cam/12345.jpg", "12345.jpg", "none") is None, "none 模式还得是不推"

    g = new_gallery("selftest_t32")
    g.add(tpl("v1", "12345", basis(0)))
    g.add(tpl("v2", "0007", basis(1)))
    items = [("cam/12345.jpg", img(1), infer_person("cam/12345.jpg", "12345.jpg", "id_num")),
             ("cam/0007.jpg", img(2), infer_person("cam/0007.jpg", "0007.jpg", "id_num"))]
    s = run_probe(Stub(VECS), g, items, {"threshold": 0.4, "topk": 5})["summary"]
    assert s["n_truth_known"] == 2 and s["n_truth_named"] == 2, \
        f"纯数字命名的探针没全进真值口径：{s['n_truth_known']}/{s['n_truth_named']}"
    assert s["rank1"] == 1.0, f"Rank-1 不该因为文件名是数字就掉：{s['rank1']}"
    print(" 32 纯数字文件名照样有真值，Rank-1 与检出率的分母不再打架  ✓")


def t33_probe_truth_fills_the_threshold_curve():
    """底库每人 1 张模板时，带真值的探针必须真的能把「阈值建议」算出来。

    README 推荐每人先存一张大头照，本机那份 3475 人的底库也正是这个形态：
    底库侧永远凑不出「同一人互配」对，原来 threshold_report 只吃底库，
    于是整栏「阈值建议」是空的，而提示语却让用户「跑一批带真值的探针即可生成」——
    探针根本没参与计算，202 张全标了真值跑完，那一栏照样一句空话。
    """
    g = new_gallery("selftest_t33")
    for i, who in enumerate("ABCD"):
        g.add(tpl(f"t{i}", who, basis(i)))
    items = [
        ("cam/A/1.jpg", img(1), "A"),        # 命中真值 → genuine 1.0
        ("cam/B/1.jpg", img(2), "B"),        # 命中真值 → genuine 1.0
        ("cam/B/2.jpg", img(6), "B"),        # top1 是 A(0.7)，真值 B 没进 topk=1 → 删失
        ("cam/x/1.jpg", img(4), UNKNOWN),    # 库外，与 D 有 0.80 → 冒认样本
        ("cam/x/2.jpg", img(5), UNKNOWN),    # 库外，与 D 有 0.98 → 最强的那个冒认者
        ("cam/raw/1.jpg", img(3), None),     # 没标真值：两侧都不进分布
    ]
    res = run_probe(Stub(VECS), g, items, {"threshold": 0.4, "topk": 1})
    recs = res["records"]

    bare = threshold_report(None, g)
    assert bare["basis"] == "gallery", bare.get("basis")
    assert bare["curve"] == [] and bare["best"] is None, "纯底库该是「样本不足」，别硬造曲线"
    assert bare["probe_genuine_n"] == 0 and bare["genuine_from_gallery"] == 0, bare
    assert "同一人互配" in bare["note"] and bare["note"], bare["note"]

    rep = threshold_report(None, g, records=recs)
    assert rep["basis"] == "gallery+probe", rep["basis"]
    assert rep["curve"] and rep["best"] is not None, "带真值的批次跑完，阈值曲线还是空的"
    assert (rep["probe_genuine_n"], rep["probe_impostor_n"]) == (2, 3), rep
    assert rep["probe_censored"] == 1, f"真值没进 Top-K 的那张该记删失，不是瞎补一个数：{rep}"
    assert (rep["probe_unknown_faces"], rep["probe_unlabeled_faces"]) == (2, 1), rep
    assert rep["genuine_from_gallery"] == 0 and rep["genuine_n"] == 2, rep
    assert rep["impostor_n"] == 15, f"底库那 12 个有序异人对不能丢：{rep['impostor_n']}"

    gen, imp, info = probe_roc(recs)
    assert gen.size == 2 and np.allclose(gen, 1.0), gen
    # 删失那张的 A 分 0.7 只能进冒认侧：它是「真值没上榜的脸 vs 陌生人」，
    # 顺手补进 genuine 等于把冒认者当前锋，TAR 直接虚高
    assert np.isclose(imp.max(), 0.9798, atol=1e-3), (imp.max(), info)
    assert not np.isclose(gen, 0.7).any(), "删失样本漏进了 genuine 分布"
    assert rep["at_far_1e6"] and rep["at_far_1e6"]["t"] > 0.9798, \
        f"推荐阈值得把最强的那个冒认者挡在外面：{rep['at_far_1e6']}"
    assert rep["at_far_1e6"]["tar"] == 1.0, "卡住冒认者不该把两个真人一起关在门外"

    # 喂过探针之后不能污染底库那份缓存：下一个不带探针的调用还得拿到纯底库口径
    again = threshold_report(None, g)
    assert again["basis"] == "gallery" and again["curve"] == [], \
        f"探针样本漏进了底库缓存：{again['basis']}/{len(again['curve'])}"
    # 报错的记录没有可用的 Top-K，不能当成样本
    assert probe_roc(recs[:1])[2]["probe_faces"] == 1
    assert probe_roc([dict(recs[0], verdict="error")])[2]["probe_faces"] == 0, \
        "整图没评成却仍被当成带真值样本"

    g.add(tpl("t0b", "A", basis(0)))          # 给 A 补第二张：底库侧终于有互配对
    after = threshold_report(None, g)
    assert after["basis"] == "gallery" and after["genuine_from_gallery"] >= 1, \
        "底库变了却没让缓存失效，阈值建议还停在少一张模板的时候"
    assert after["curve"], "底库自己凑出互配对之后就该有曲线"
    print(" 33 带真值的探针把空的「阈值建议」补上，删失不瞎补、缓存不串味  ✓")


def t34_truncated_first_vector_does_not_void_the_gallery():
    """一条截断的向量不能带走整库，尤其不能因为它排在最前面就重新定义全库维度。

    以前 _load_db 拿「第一条读到的向量长度」当全库维度：写盘时断过一次电，
    最前面那条 512 维落盘成了 511 维，后面几千条完好的全被判「长度对不上」跳过，
    几千人的底库加载完只剩那半条残向量 —— 界面显示「底库是空的」，人还在库里躺着。
    """
    import sqlite3
    key = "selftest_t34"
    g = new_gallery(key)
    for i, who in enumerate("ABCDE"):
        g.add(tpl(f"v{i}", who, basis(i)))
    g.save()
    con = sqlite3.connect(gallery_dir_for(key) / "gallery.db")
    con.execute("UPDATE templates SET vec=substr(vec,1,length(vec)-4)"
                " WHERE rowid=(SELECT MIN(rowid) FROM templates)")
    con.commit()
    con.close()

    g2 = Gallery(key)
    assert len(g2.index) == 4, f"一条截断的向量带走了 {5 - len(g2.index)} 条好数据：{g2.stats()}"
    assert [t.person for t in g2.index] == ["B", "C", "D", "E"], [t.person for t in g2.index]
    assert g2.matrix.shape[1] == DIM, f"维度被残向量带偏了：{g2.matrix.shape}"
    warns = g2.stats()["integrity"]
    assert len(warns) == 1 and warns[0].startswith("1 条模板向量损坏"), warns
    assert g2.search(basis(4))[1][0] == "E", "剩下的库要照常认得人"
    g2.add(tpl("v9", "F", basis(5)))          # 扩容也得按 512 维走，不能顺着 511 长
    assert g2.matrix.shape[1] == DIM and g2.search(basis(5))[1][0] == "F", g2.stats()
    print(" 34 最前面那条向量被截断，只损失他自己，其余几千条照旧能用  ✓")


def t35_sweep_recommend_keeps_zero_p50_cells():
    """P50 = 0.0ms 的那一格是最达标的一格，不能因为「它是 0」被当成没数据剔掉。"""
    cells = [
        {"workers": 1, "throughput_faces_per_s": 30.0, "total_ms": {"p50": 33.0}},
        {"workers": 4, "throughput_faces_per_s": 90.0, "total_ms": {"p50": 0.0}},
        {"workers": 8, "throughput_faces_per_s": 120.0, "total_ms": {"p50": 400.0}},
        {"workers": 2, "throughput_faces_per_s": 50.0},          # 压根没测到耗时
    ]
    rec = srv._sweep_recommend(cells, 120.0)
    assert rec["cell"]["workers"] == 4, \
        f"P50=0 的那格被真值判断剔掉了：{rec['cell']} / {rec['basis']}"
    assert "可用线内" in rec["note"], rec
    # 全部超线时如实说超线，并报出最快那格的 P50（P50=0 那格任何可用线都达标，得剔掉才能测这条分支）
    rec2 = srv._sweep_recommend([c for c in cells if (c.get("total_ms") or {}).get("p50")], 5.0)
    assert rec2["cell"]["workers"] == 8 and "没有一格达到" in rec2["note"], rec2
    assert "400ms" in rec2["note"], rec2
    # 没有耗时数据的格子不能被当成「达标」推荐
    rec3 = srv._sweep_recommend([cells[3]], 120.0)
    assert "没有一格达到" in rec3["note"], rec3
    assert srv._sweep_recommend([], 120.0)["note"], "空格也要有话说"
    print(" 35 扫描建议不再把 P50=0 的最优格当空值剔掉，缺耗时也不算达标  ✓")


def t36_parallel_probe_matches_serial_record_by_record():
    """并行评测只许改变快慢，不许改变结果。

    README 把「记录顺序仍按输入顺序返回」当成界面序号 / 缩略图命名 / 分页的前提，
    可这个前提以前没有任何断言守着：哪天 map 换成 as_completed，界面会静默把
    复核图和照片对错位，指标一张不少、名字全串了。
    """
    g = new_gallery("selftest_t36")
    for i, who in enumerate("ABCDEFGH"):
        g.add(tpl(f"t{i}", who, basis(i)))
    items = [(f"cam/{who}/{n}.jpg", img(*fids), truth) for who, n, fids, truth in (
        ("A", 1, (1,), "A"), ("B", 1, (2,), "B"), ("C", 1, (3,), "C"),
        ("D", 1, (4,), "D"), ("E", 1, (5,), "E"), ("F", 1, (6,), "F"),      # 真值排第 2
        ("G", 1, (7,), "G"), ("H", 1, (8,), "H"),
        ("A", 2, (1, 3, 5), "A"),      # 合影：主角 + 两个旁人
        ("C", 2, (3, 1), "C"),
        ("E", 2, (4, 4, 6), "E"),
    ) * 2]
    p = {"threshold": 0.4, "topk": 5}
    bk = Stub(VECS)
    bk.delay = 0.004      # 让推理「慢一点」，四路才真的在抢，而不是瞬间跑完假装并行
    serial = run_probe(bk, g, items, {**p, "workers": 1})
    par = run_probe(bk, g, items, {**p, "workers": 4})
    assert par["wall_s"] < serial["wall_s"], \
        f"并行比串行还慢（{par['wall_s']} vs {serial['wall_s']}），多半根本没并行起来"

    def strip(res):
        recs = [json.loads(json.dumps(r, ensure_ascii=False, default=str))
                for r in res["records"]]
        for r in recs:
            r.pop("timings", None)        # 墙钟本来就不可能一样
        s = dict(res["summary"])
        for k in ("latency", "throughput_fps", "throughput_faces_per_s", "workers"):
            s.pop(k, None)
        return recs, s

    a_recs, a_sum = strip(serial)
    b_recs, b_sum = strip(par)
    names = [it[0] for it in items]
    assert [r["name"] for r in a_recs] == names, "串行批次顺序就错了"
    assert [r["name"] for r in b_recs] == names, \
        "并行之后记录顺序变了，界面缩略图会整批错位"
    assert a_recs == b_recs, next(
        f"{x['name']} 并行与串行不一致：{x} vs {y}"
        for x, y in zip(a_recs, b_recs) if x != y)
    assert a_sum == b_sum, (a_sum, b_sum)
    assert a_sum["n"] == len(items) and a_sum["rank1"] is not None, a_sum
    print(" 36 并行评测与串行逐条一致、顺序不变，只有快慢不同  ✓")


def t37_clear_history_spares_the_running_batch():
    """「清空历史批次」不能把正在跑的评测删废。

    目录评测一跑就是十几分钟，data/probes/<批次>/ 里是 worker 正在往里写的截图。
    清空把 PROBE_DIR 整个 rmtree，下一个分块开 meta.jsonl 就是 FileNotFoundError，
    整批 status=error、几分钟推理全丢 —— 而用户只是按了一个「清空历史」。
    收尾之后登记必须释放，否则那个目录以后永远删不掉。
    """
    tmp = Path(tempfile.mkdtemp(prefix="selftest_t37_"))
    reports, probes = tmp / "reports", tmp / "probes"
    reports.mkdir()
    for rid in ("busy11111", "idle22222"):
        d = probes / rid
        d.mkdir(parents=True)
        (d / "0.jpg").write_bytes(b"x" * 256)
        (d / "meta.jsonl").write_text('{"i": 0}\n', encoding="utf-8")
        (reports / f"probe_{rid}.json").write_text('{"records": [1]}', encoding="utf-8")
    keep = (srv.RUNS, srv.REPORT_DIR, srv.PROBE_DIR, dict(srv._ACTIVE_PROBE))
    srv.RUNS = {"idle22222": {"records": [1]}}
    srv.REPORT_DIR, srv.PROBE_DIR = reports, probes
    try:
        srv._probe_hold("busy11111", probes / "busy11111")
        assert srv._active_probe_ids() == {"busy11111"}
        out = srv.api_runs_clear()
        assert out["skipped"] == ["busy11111"], f"没认出正在评测的批次：{out}"
        assert out["cleared"] == 1, out
        assert (probes / "busy11111" / "meta.jsonl").exists(), "正在写的截图目录被删了"
        assert (reports / "probe_busy11111.json").exists(), "正在评测的报告被删了"
        assert not (probes / "idle22222").exists(), "空闲批次居然没删掉"

        # 单独删这一批也要被拒，而且拒了就不能碰到磁盘
        try:
            srv.api_run_delete("busy11111")
        except srv.HTTPException as e:
            assert e.status_code == 409, f"应返回 409，实际 {e.status_code}"
        else:
            raise AssertionError("正在评测的批次居然删掉了，这一批推理全废")
        assert (probes / "busy11111" / "0.jpg").exists(), "被拒的请求动到了磁盘"

        # 评测收尾之后登记要撤掉，否则这个目录永远删不掉
        srv._discard_probe_dir({"run_id": "busy11111",
                               "out_dir": probes / "busy11111", "bare": False})
        assert srv._active_probe_ids() == set(), "释放登记漏了，这目录以后永远删不掉"
        assert srv.api_run_delete("busy11111")["cleared"] == 1, "收尾之后还是删不掉"
    finally:
        srv.RUNS, srv.REPORT_DIR, srv.PROBE_DIR = keep[0], keep[1], keep[2]
        srv._ACTIVE_PROBE.clear()
        srv._ACTIVE_PROBE.update(keep[3])
        shutil.rmtree(tmp, ignore_errors=True)
    print(" 37 清空历史批次绕开正在评测那一批，收尾释放登记后才真删得掉  ✓")


def t38_flush_never_resurrects_a_deleted_batch():
    """落盘不能把刚删掉的批次写回磁盘，也不能吞掉写盘期间的那次改标。

    _flush_run 在锁外写几十 MB，这几毫秒里两件事都可能发生：
      1) 用户点了「删除本批」：报告删了、内存撤了、脏标记也撤了，然后这一次写
         又把文件落回磁盘 —— 重启后 load_runs_from_disk 把它整个读回来，
         一个已经被删掉的批次在界面里复活了；
      2) 手还压在阈值滑杆上又标了一次脏：写完无脑清脏标记，
         最新那一版判定就永久没了（写出去的字节是改标之前序列化好的）。
    """
    tmp = Path(tempfile.mkdtemp(prefix="selftest_t38_"))
    reports = tmp / "reports"
    reports.mkdir()
    keep = (srv.RUNS, srv.REPORT_DIR, srv.DIRTY, srv._DIRTY_REV, srv.FLUSH_DELAY_S)
    srv.RUNS, srv.DIRTY, srv._DIRTY_REV = {}, {}, {}
    srv.REPORT_DIR = reports
    srv.FLUSH_DELAY_S = 3600.0      # 别让后台线程插进来，这一轮时序要自己控制
    real_write = cfg.write_json
    try:
        srv.RUNS["run0000a"] = {"run_id": "run0000a", "records": [{"name": "a.jpg"}]}
        srv._mark_dirty("run0000a")
        assert srv._flush_run("run0000a") is True
        assert (reports / "probe_run0000a.json").is_file(), "第一次落盘就没写出去"
        assert "run0000a" not in srv.DIRTY

        def delete_during_write(path, obj):
            # api_run_delete 正好插在「取出 run」和「写完文件」之间
            srv.RUNS.pop("run0000a", None)
            srv.DIRTY.pop("run0000a", None)
            srv._DIRTY_REV.pop("run0000a", None)
            (reports / "probe_run0000a.json").unlink(missing_ok=True)
            real_write(path, obj)   # 这次写就是一个「已经删掉的批次」留下的最后一份

        cfg.write_json = delete_during_write
        try:
            srv._mark_dirty("run0000a")
            assert srv._flush_run("run0000a") is True, "删掉的批次不该被当成没写完"
        finally:
            cfg.write_json = real_write
        assert not (reports / "probe_run0000a.json").exists(), \
            "删掉的批次被落盘线程写回磁盘了，重启后它会复活"
        assert "run0000a" not in srv.DIRTY

        srv.RUNS["run0000b"] = {"run_id": "run0000b", "records": [{"name": "b.jpg"}]}
        srv._mark_dirty("run0000b")

        def edit_during_write(path, obj):
            real_write(path, obj)
            srv._mark_dirty("run0000b")      # 写的时候手又拖了一下滑杆

        cfg.write_json = edit_during_write
        try:
            assert srv._flush_run("run0000b") is False, \
                "写盘期间的新改动被当成写完了，脏标记会被清掉"
            assert "run0000b" in srv.DIRTY, "脏标记被抹掉：最后一次改标永久丢失"
        finally:
            cfg.write_json = real_write
        assert srv._flush_run("run0000b") is True, "补写那一轮没认定已经落干净"
        assert "run0000b" not in srv.DIRTY, "补写完还不撒手，后台线程会一直写同一批"

        # 只剩汇总的壳不该被写回磁盘：那会把真报告抹成没有明细的空壳
        srv.RUNS["run0000c"] = {"run_id": "run0000c", "records": [], "_stub": True}
        srv._mark_dirty("run0000c")
        assert srv._flush_run("run0000c") is True
        assert not (reports / "probe_run0000c.json").exists(), "空壳被写回磁盘会抹掉真报告"
        assert "run0000c" not in srv.DIRTY
    finally:
        srv.RUNS, srv.REPORT_DIR = keep[0], keep[1]
        srv.DIRTY, srv._DIRTY_REV = keep[2], keep[3]
        srv.FLUSH_DELAY_S = keep[4]
        cfg.write_json = real_write
        shutil.rmtree(tmp, ignore_errors=True)
    print(" 38 落盘不会让删掉的批次复活，写盘期间的改标也不会被吞  ✓")


def t39_job_pruning_keeps_the_newest():
    """淘汰旧任务要留最新的，运行中的永远留着。

    JOBS 按插入顺序遍历，筛出来的列表是最旧在前。写 [50:] 等于留最旧 50 个、
    把刚跑完的那批删掉：脚本正要去轮询 /api/jobs/<刚提交的 id> 拿到 404，
    而界面上结果明明白白摆着。反过来一个都不淘汰也不对，一次会话能积累
    几千个任务对象，内存和 /api/jobs 的序列化一起涨。
    """
    keep = (srv.JOBS, srv.STREAMS)
    srv.JOBS, srv.STREAMS = {}, {}
    try:
        running_id = srv._new_job(1000, "probe")[0]      # 一直在跑，不能被淘汰
        done_ids = []
        for _ in range(srv.MAX_JOBS + 25):
            jid, j = srv._new_job(4)
            j["status"] = "done"
            done_ids.append(jid)
        assert running_id in srv.JOBS, "运行中的任务被淘汰了，进度条会突然 404"
        finished = [k for k in srv.JOBS if srv.JOBS[k].get("status") == "done"]
        # 淘汰发生在插新任务那一刻，而新任务当时还是 running、算不进候选，
        # 所以稳态是 MAX_JOBS + 1 个已完成任务，不是正好 MAX_JOBS 个。
        assert len(finished) <= srv.MAX_JOBS + 1, f"已完成任务没淘汰，攒了 {len(finished)} 个"
        assert set(finished) == set(done_ids[-(srv.MAX_JOBS + 1):]), (
            f"淘汰留下的不是最新一批：缺 {set(done_ids[-(srv.MAX_JOBS + 1):]) - set(finished)}，"
            f"多留 {set(finished) - set(done_ids[-(srv.MAX_JOBS + 1):])}")
        for k in done_ids[-(srv.MAX_JOBS + 1):]:
            assert k in srv.JOBS, f"刚跑完的任务 {k} 被淘汰了，脚本轮询会拿到 404"
        for k in done_ids[:-(srv.MAX_JOBS + 1)]:
            assert k not in srv.JOBS, f"最旧的 {k} 还留着，内存会一直涨"
        assert srv._new_job(1)[0] in srv.JOBS
    finally:
        srv.JOBS, srv.STREAMS = keep
    print(" 39 任务淘汰留最新、运行中的不动，不会把刚跑完的批次淘汰掉  ✓")


def t40_startup_load_keeps_only_a_few_full_runs():
    """重启读盘要「只把最近几批的明细读进内存」，不是全读进来再压壳。

    json.loads 出来的那一刻，所有批次的 records 同时挂在内存里：万张一批
    几十 MB × 磁盘上批次的个数，峰值是它们之和 —— 而文档承诺的
    「只读最近几个批次的完整明细」根本不需要这份峰值。壳照样能让历史批次
    列出来，点开时 _ensure_run 再从磁盘读回。
    """
    tmp = Path(tempfile.mkdtemp(prefix="selftest_t40_"))
    reports = tmp / "reports"
    reports.mkdir()
    ids = [f"load000{i}" for i in range(6)]
    for i, rid in enumerate(ids):
        recs = [{"name": f"{i}-{k}.jpg", "faces_found": 1} for k in range(50)]
        (reports / f"probe_{rid}.json").write_text(
            json.dumps({"run_id": rid, "records": recs, "summary": {"n": 50},
                        "params": {"threshold": 0.4}, "created_at": 1000.0 + i}),
            encoding="utf-8")
        # 谁算「最近几批」看 mtime，逐个拉开，别让时间戳精度打平
        os.utime(reports / f"probe_{rid}.json", (2000 + i, 2000 + i))
    keep = (srv.RUNS, srv.REPORT_DIR, srv.PROBE_DIR, srv.DIRTY, srv._DIRTY_REV)
    srv.RUNS, srv.DIRTY, srv._DIRTY_REV = {}, {}, {}
    srv.REPORT_DIR, srv.PROBE_DIR = reports, tmp / "probes"
    try:
        n = srv.load_runs_from_disk()
        assert n == srv.RUNS_KEEP, f"整份明细留了 {n} 份，配额只有 {srv.RUNS_KEEP}"
        full = {k for k, v in srv.RUNS.items() if not v.get("_stub")}
        assert full == set(ids[-srv.RUNS_KEEP:]), f"整份留存的不是最近几批：{sorted(full)}"
        stubs = sorted(k for k, v in srv.RUNS.items() if v.get("_stub"))
        assert stubs == ids[:len(ids) - srv.RUNS_KEEP], stubs
        for k in stubs:
            assert srv.RUNS[k]["records"] == [], "壳里还留着明细，白占内存"
            assert srv.RUNS[k]["summary"]["n"] == 50, "壳丢了汇总，下拉列不出指标"
        assert len(srv.api_runs()) == len(ids), "历史批次列表少了几批"
        assert len(srv._ensure_run(ids[0])["records"]) == 50, "点开旧批次读不回完整明细"
    finally:
        (srv.RUNS, srv.REPORT_DIR, srv.PROBE_DIR,
         srv.DIRTY, srv._DIRTY_REV) = keep
        shutil.rmtree(tmp, ignore_errors=True)
    print(" 40 重启读盘只把最近几批的明细留在内存，旧批次点开才读回  ✓")


def t41_record_page_offset_clamped_to_filtered_set():
    """筛选结果缩了要把 offset 夹回最后一页，别交出「有范围、没行」的一页。

    停在第 5 页时拖一下阈值滑杆，命中数从 900 掉到 40：服务端还是按 offset=600
    切片，切出来是空的，可 matched 照抄 40 —— 页面上写着「第 40–40 条 ·
    命中 40 条」外加零行，下一页按钮永远按不动，只能靠换筛选项才逃得出去。
    """
    recs = []
    for i in range(200):
        hitme = i % 5 == 0                       # 40 条命中、160 条未命中
        recs.append({"name": f"p{i}.jpg", "faces_found": 1, "truth": f"P{i}",
                     "faces": [{"accepted": hitme, "top1_person": "A" if hitme else ""}],
                     "rank1_correct": hitme})
    rid = "page0001"
    keep = (srv.RUNS, srv.REPORT_DIR, srv.PROBE_DIR)
    srv.RUNS = {rid: {"run_id": rid, "records": recs, "summary": {"n": 200},
                      "params": {"threshold": 0.4, "topk": 5}, "created_at": 1.0}}
    tmp = Path(tempfile.mkdtemp(prefix="selftest_t41_"))
    srv.REPORT_DIR = tmp
    srv.PROBE_DIR = srv.REPORT_DIR / "probes"
    try:
        base = srv.api_run(rid, offset=0, limit=150, only="all")
        assert base["total"] == 200 and len(base["records"]) == 150, base["total"]
        assert base["facets"]["hit"] == 40 and base["facets"]["miss"] == 160, base["facets"]

        far = srv.api_run(rid, offset=600, limit=150, only="hit")
        assert far["matched"] == 40, far["matched"]
        assert far["offset"] == 0, f"越界的 offset 原样回给前端了：{far['offset']}"
        assert len(far["records"]) == 40, f"命中 40 条却只回了 {len(far['records'])} 行"

        # 多页的筛选集要夹到最后一页，而不是退回第 0 页
        multi = srv.api_run(rid, offset=900, limit=150, only="miss")
        assert multi["matched"] == 160 and multi["offset"] == 150, multi["offset"]
        assert len(multi["records"]) == 10, len(multi["records"])

        # 空筛选集不能算出负的 offset
        empty = srv.api_run(rid, offset=400, limit=150, only="error")
        assert empty["offset"] == 0 and empty["records"] == [], empty["offset"]
    finally:
        srv.RUNS, srv.REPORT_DIR, srv.PROBE_DIR = keep
        shutil.rmtree(tmp, ignore_errors=True)
    print(" 41 筛选变少后 offset 自动夹回最后一页，不再交出空的明细页  ✓")


def t42_number_box_rejects_infinity_with_400():
    """数字框里的 Infinity 要报 400，不能变成 500。

    Python 的 json 认 Infinity / -Infinity，而 cast 是 int 时
    int(float('inf')) 抛的是 OverflowError，不在原来那两个 except 里，
    于是「扫描张数填了个 Infinity」又是一句看不懂的「服务器内部错误」，
    下面那句 math.isfinite 的贴心提示根本轮不到执行。
    """
    for raw in (float("inf"), float("-inf"), "Infinity", "-Infinity", "1e999"):
        try:
            srv._clean_num(raw, "扫描张数", 200, 1, 800)
        except srv.HTTPException as e:
            assert e.status_code == 400, f"{raw!r} 应返回 400，实际 {e.status_code}"
        else:
            raise AssertionError(f"{raw!r} 被静默接受了，扫描会当场把内存打爆")
    assert srv._clean_num("200", "扫描张数", 200, 1, 800) == 200
    assert srv._clean_num("", "扫描张数", 200, 1, 800) == 200
    assert srv._clean_num(12, "延迟可用线(ms)", 120.0, 1.0, 60000.0, cast=float) == 12.0
    print(" 42 数字框里的 Infinity 报 400 讲清楚哪个框，不再抛 500  ✓")


def t43_ensure_run_hands_back_the_full_records():
    """从磁盘读回旧批次时，要交自己刚读的那份，不要回表里再查一次。

    keep=run_id 只挡住「_ensure_run 自己这次触发的压壳」；评测收尾时 worker
    也在 _probe_finish 里调 _trim_runs，它可能先一步把这一批换成了 _stub
    （records 是空的）。那时候 RUNS.get(run_id) or run 取出的是那个空壳 ——
    用户点开一个几千张的旧批次，屏幕上写着「本批共 0 张」，刷新还是 0。
    """
    tmp = Path(tempfile.mkdtemp(prefix="selftest_t43_"))
    rid = "shell001"
    recs = [{"name": f"{i}.jpg", "faces_found": 1} for i in range(120)]
    (tmp / f"probe_{rid}.json").write_text(
        json.dumps({"run_id": rid, "records": recs, "summary": {"n": 120},
                    "params": {"threshold": 0.4, "topk": 5}, "created_at": 5.0}),
        encoding="utf-8")
    keep = (srv.RUNS, srv.REPORT_DIR, srv.PROBE_DIR, srv._trim_runs, srv.DIRTY, srv._DIRTY_REV)
    srv.RUNS, srv.DIRTY, srv._DIRTY_REV = {}, {}, {}
    srv.REPORT_DIR, srv.PROBE_DIR = tmp, tmp / "probes"

    def trim_steals_it(keep=None):
        # 模拟并发的评测收尾：正好在这一批刚塞回表之后把它压成空壳
        srv.RUNS[rid] = {**srv.RUNS[rid], "records": [], "_stub": True}

    try:
        srv._trim_runs = trim_steals_it
        got = srv._ensure_run(rid)
        assert len(got["records"]) == 120, \
            f"并发压壳之后交出的是空壳，只剩 {len(got['records'])} 条明细"
        page = srv.api_run(rid, offset=0, limit=150, only="all")
        assert page["total"] == 120 and len(page["records"]) == 120, page["total"]
    finally:
        (srv.RUNS, srv.REPORT_DIR, srv.PROBE_DIR, srv._trim_runs,
         srv.DIRTY, srv._DIRTY_REV) = keep
        shutil.rmtree(tmp, ignore_errors=True)
    print(" 43 读回旧批次时并发压壳也不交空明细，界面不会写「本批共 0 张」  ✓")


def t44_cli_templates_zero_means_unlimited():
    """命令行的 `--templates 0` 是「每人不限张数」，不是「一张都不注册」。

    DEFAULTS 和 bench.enroll_image 都把 0 当不限（那边写的是 `if cap and ...`），
    命令行那侧却写成 `per.get(person,0) >= a.templates`，0 时恒真：
    传 `--templates 0` 想要「每人全收」，拿到的是空库 + 整批「没匹配上」，
    屏幕上只剩「0 人 / 0 模板」，完全看不出是参数被理解反了。
    """
    from cli.run import keep_template

    assert keep_template("A", {}, 0) is True
    assert keep_template("A", {"A": 999}, 0) is True, "0 被当成「每人 0 张」，一个都不注册"
    assert keep_template("A", {"A": 3}, 4) is True
    assert keep_template("A", {"A": 4}, 4) is False, "到了上限还在注册"
    assert keep_template("A", {"A": 2}, -1) is True, "负数该按不限处理，而不是全拒"
    assert keep_template(None, {}, 0) is False, "认不出人名也去注册"
    print(" 44 命令行 --templates 0 是不限张数，不会注册出空底库  ✓")


def main() -> int:
    print("自检（桩后端，不加载模型）：")
    for fn in (t1_single_person_gallery, t2_unknown_semantics, t3_recall_independent_of_threshold,
               t4_dedupe_on_reenroll, t5_recover_row_mismatch, t6_probe_truth,
               t7_multi_face_same_frame, t8_image_truth_without_face_label,
               t9_flat_gallery_one_image_per_person, t10_legacy_storage,
               t11_concurrent_enroll_and_search, t12_clear_history_batches,
               t13_clear_gallery_removes_files, t14_sqlite_gallery_pagination,
               t15_scrfd_chw_two_anchor_layout, t16_max_templates_per_person_is_atomic,
               t17_bad_images_and_wall_clock_throughput,
               t18_incremental_grouping_matches_full_recount,
               t19_reenroll_prunes_stale_templates_and_thumbs,
               t20_readonly_gallery_fails_loudly,
               t21_person_list_not_truncated_by_page_size,
               t22_threshold_tie_prefers_higher_cut,
               t23_subject_face_labeled_unknown_not_double_counted,
               t24_model_hash_cache_actually_hits,
               t25_delete_single_run_removes_report_and_images,
               t26_delete_single_person_via_api,
               t27_undecodable_image_reported_as_error,
               t28_remote_gallery_key_stays_inside_data_dir,
               t29_frontend_kpi_keys_match_summary,
               t30_impostor_scores_stay_inside_cosine_range,
               t31_cli_regression_does_not_wipe_web_gallery,
               t32_id_num_keeps_pure_numeric_names_in_truth,
               t33_probe_truth_fills_the_threshold_curve,
               t34_truncated_first_vector_does_not_void_the_gallery,
               t35_sweep_recommend_keeps_zero_p50_cells,
               t36_parallel_probe_matches_serial_record_by_record,
               t37_clear_history_spares_the_running_batch,
               t38_flush_never_resurrects_a_deleted_batch,
               t39_job_pruning_keeps_the_newest,
               t40_startup_load_keeps_only_a_few_full_runs,
               t41_record_page_offset_clamped_to_filtered_set,
               t42_number_box_rejects_infinity_with_400,
               t43_ensure_run_hands_back_the_full_records,
               t44_cli_templates_zero_means_unlimited):
        fn()
    for key in ("selftest_t1", "selftest_t2", "selftest_t3", "selftest_t4",
                "selftest_t5", "selftest_t7", "selftest_t8", "selftest_t9",
                "selftest_t10", "selftest_t11", "selftest_t13", "selftest_t13_other",
                "selftest_t14", "selftest_t16", "selftest_t17", "selftest_t18",
                "selftest_t19", "selftest_t19_other", "selftest_t20", "selftest_t21",
                "selftest_t23", "selftest_t26", "selftest_t27", "selftest_t29",
                "selftest_t31", "cli_selftest_t31", "selftest_t32", "selftest_t33",
                "selftest_t34", "selftest_t36"):
        shutil.rmtree(gallery_dir_for(key), ignore_errors=True)
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
