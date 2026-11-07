"""SCRFD 人脸检测后处理（自实现，不依赖 insightface 包）。

与 insightface/model_zoo/scrfd.py 等价：anchor-free 距离回归 + NMS。
自动兼容两种导出布局：
  1) 扁平 [n, C]（官方 det_10g.onnx 即此型，anchor 维已折进行里，
     n = (size/stride)^2 * num_anchors，C 恒为 1/4/10）
  2) 通道式 [1, C, fh, fw]（C = na / 4na / 10na，一行含同格全部 anchor）
stride 由行数或特征图边长反推，因此同一份代码也能跑 det_2.5g。
"""
from __future__ import annotations

import numpy as np

from .imaging import letterbox, to_blob

STRIDES = (8, 16, 32)


def _kind(cols: int) -> str | None:
    """按列数判断张量语义。"""
    if cols in (1, 2):
        return "score"
    if cols in (4, 8):
        return "bbox"
    if cols in (10, 20):
        return "kps"
    return None


def distance2bbox(points: np.ndarray, dist: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            points[:, 0] - dist[:, 0],
            points[:, 1] - dist[:, 1],
            points[:, 0] + dist[:, 2],
            points[:, 1] + dist[:, 3],
        ],
        axis=-1,
    )


def distance2kps(points: np.ndarray, dist: np.ndarray) -> np.ndarray:
    preds = []
    for i in range(0, dist.shape[1], 2):
        preds.append(points[:, 0] + dist[:, i])
        preds.append(points[:, 1] + dist[:, i + 1])
    return np.stack(preds, axis=-1)


def nms(dets: np.ndarray, thresh: float) -> list[int]:
    if dets.shape[0] == 0:
        return []
    x1, y1, x2, y2, s = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    order = s.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(xx2 - xx1, 0) * np.maximum(yy2 - yy1, 0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= thresh]
    return keep


class Scrfd:
    def __init__(self, session, input_size: int = 640):
        self.session = session
        self.input_size = input_size
        self.nms_thresh = 0.4
        self.input_name = session.get_inputs()[0].name
        self.out_names = [o.name for o in session.get_outputs()]
        self._grids: dict = {}

    # ---- 布局解析 -------------------------------------------------------
    def _resolve_flat(self, n_rows: int, size: int):
        """扁平布局：由行数反推 (stride, fh, fw, na)。"""
        for na in (1, 2):
            for st in STRIDES:
                fh = size / st
                if fh == int(fh) and int(fh) ** 2 * na == n_rows:
                    return st, int(fh), int(fh), na
        return None

    def collect(self, outs, size: int) -> dict:
        """把原始输出整理成 {stride: {score, bbox, kps, na, mode}}。"""
        levels: dict[int, dict] = {}
        for name, raw in zip(self.out_names, outs):
            a = np.asarray(raw, dtype=np.float32)
            slot = None
            if a.ndim == 2:  # [n, C]
                cols = int(a.shape[1])
                kind = _kind(cols)
                got = self._resolve_flat(int(a.shape[0]), size)
                if kind and got:
                    st, fh, fw, na = got
                    if kind == "score" and cols != 1:
                        continue
                    slot = (st, {"mode": "flat", "na": na, "fh": fh, "fw": fw, kind: a})
            elif a.ndim == 3 and a.shape[0] == 1:  # [1, n, C]
                b = a[0]
                cols = int(b.shape[1])
                kind = _kind(cols)
                got = self._resolve_flat(int(b.shape[0]), size)
                if kind and got:
                    st, fh, fw, na = got
                    slot = (st, {"mode": "flat", "na": na, "fh": fh, "fw": fw, kind: b})
            elif a.ndim == 4 and a.shape[0] == 1:  # [1, C, fh, fw]
                cols, fh, fw = int(a.shape[1]), int(a.shape[2]), int(a.shape[3])
                kind = _kind(cols)
                if kind and fh == fw and size % fh == 0:
                    st = size // fh
                    na = cols if kind == "score" else cols // (4 if kind == "bbox" else 10)
                    # 通道式布局的 anchor 数只能从通道数反推，score/bbox/kps 各推各的。
                    # 万一某个导出里 score 是 2 个 anchor 通道而 bbox 只有 1 个，
                    # 原来会把两者混进同一层：网格按 2 份展开、bb 只有 1 份行，
                    # keep 里的下标直接越界，整次检测抛 IndexError（界面就是一个 500）。
                    # 这里要求同一层三类张量推出来的 anchor 数一致，不一致就丢这个张量。
                    prev = levels.get(st)
                    if prev is not None and int(prev.get("na") or na) != na:
                        continue
                    b = np.transpose(a[0], (1, 2, 0)).reshape(fh * fw, cols)
                    slot = (st, {"mode": "chw", "na": na, "fh": fh, "fw": fw, kind: b})
            if slot is None:
                continue
            st, d = slot
            tgt = levels.setdefault(st, d)
            tgt.update({k: v for k, v in d.items() if k in ("score", "bbox", "kps")})
            tgt.setdefault("na", d["na"])
        return levels

    def _grid(self, fh: int, fw: int, stride: int, na: int) -> np.ndarray:
        """anchor 中心网格。

        官方导出把 [1, C*k, fh, fw] 做 permute(0,2,3,1) 再展平，行序是
        (h, w, anchor)，同一格的全部 anchor 连续排列，因此这里用
        repeat_interleave 而不是 tile；SCRFD 各 anchor 共用格中心。
        """
        key = (fh, fw, stride, na)
        g = self._grids.get(key)
        if g is None:
            hw = np.stack(np.mgrid[:fh, :fw][::-1], axis=-1).astype(np.float32).reshape(-1, 2)
            g = np.repeat(hw, na, axis=0) * stride
            if len(self._grids) < 64:
                self._grids[key] = g
        return g

    # ---- 检测 -----------------------------------------------------------
    def detect(
        self, img: np.ndarray, input_size: int | None = None, thresh: float = 0.5,
        nms_thresh: float | None = None, max_num: int = 0,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """返回 (det[N,5]=x1,y1,x2,y2,score, kps[N,5,2] 或 None)，坐标已还原到原图。"""
        size = int(input_size or self.input_size)
        size = max(32, (size // 32) * 32)
        nms_thresh = self.nms_thresh if nms_thresh is None else nms_thresh
        canvas, r, _, _ = letterbox(img, size)
        blob = to_blob(canvas, size, 127.5, 128.0)
        outs = self.session.run(self.out_names, {self.input_name: blob})
        levels = self.collect(outs, size)

        scores_l, boxes_l, kps_l = [], [], []
        for stride in sorted(levels):
            slot = levels[stride]
            if "score" not in slot or "bbox" not in slot:
                continue
            na = max(1, int(slot.get("na") or 1))
            sc, bb = slot["score"], slot["bbox"]
            # 两类布局展平后是同一个行序：(h, w, anchor)，同一格的全部 anchor 连续。
            # 原来通道式在这里先 max(axis=1) 把 anchor 维折叠掉，网格却仍按 na 份展开，
            # 行数必然差 na 倍，撞上下面 grid.shape[0] != sc.shape[0] 的自检，
            # 于是**三个 stride 的层全被 continue 丢掉**。合成 2-anchor 输出实测
            # 检出框恒为 0：det_2.5g / scrfd-500m 这类小模型正是 2 anchor 通道式导出，
            # 在 RK3588 上换小模型选型的人会看到「一张脸都检不出来」，还不报错。
            sc = sc.reshape(-1)
            bb = bb.reshape(-1, 4)
            grid = self._grid(slot["fh"], slot["fw"], stride, na)
            if grid.shape[0] != sc.shape[0] or bb.shape[0] != sc.shape[0]:
                continue
            bb = bb * stride
            keep = np.where(sc >= thresh)[0]
            if keep.size == 0:
                continue
            boxes_l.append(distance2bbox(grid[keep], bb[keep]))
            scores_l.append(sc[keep][:, None])
            if "kps" in slot:
                kp = slot["kps"].reshape(-1, 10)
                if kp.shape[0] != sc.shape[0]:
                    continue
                kps_l.append(distance2kps(grid[keep], kp[keep] * stride).reshape(-1, 5, 2))

        if not boxes_l:
            return np.zeros((0, 5), np.float32), None
        pre = np.hstack((np.vstack(boxes_l) / r, np.vstack(scores_l))).astype(np.float32)
        order = pre[:, 4].argsort()[::-1]
        pre = pre[order]
        # 关键点必须和框一一对应：只要有一层的 kps 行数对不上（没进 kps_l），
        # 拼出来的行数就和 pre 不等，按 order 重排会整体错位 —— 张三的眉眼装到李四脸上，
        # 五个点看着还像张脸，但对齐是错的，比干脆没有关键点更糟。
        kps_all = np.vstack(kps_l) / r if kps_l and len(kps_l) == len(boxes_l) else None
        if kps_all is not None:
            kps_all = kps_all[order]
        keep = nms(pre, nms_thresh)
        det = pre[keep]
        if kps_all is not None:
            kps_all = kps_all[keep]
        if max_num and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            idx = np.sort(area.argsort()[::-1][:max_num])
            det, kps_all = det[idx], (kps_all[idx] if kps_all is not None else None)
        return det, kps_all
