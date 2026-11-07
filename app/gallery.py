"""底库：一份 SQLite 数据库（人员 / 模板 / 512 维向量），按模型版本命名空间隔离。

换识别模型必须重建特征库（不同模型向量空间不通用），因此目录名带模型 key：
  data/gallery_<model>/gallery.db   persons(人员) + templates(模板+向量BLOB) + meta
  data/gallery_<model>/thumbs/      每条模板一张裁剪缩略图，文件名就是 <tid>.jpg

检索仍用 numpy 精确矩阵乘，1 万库 ≈1ms，无需 faiss 近似索引：数据库只当持久化，
启动时把 BLOB 一次性读进一块连续矩阵，几千人的检索速度和以前逐条比一样。

为什么用数据库而不是 features.npz + meta.json：
  · 向量在 add() 时就落库（攒几十条提交一次事务），注册到一半进程被杀最多只少
    最后没提交的那几十条，不会再出现「向量文件写完了、元数据没写完」的半套库；
  · 底库列表能按人名分页查，几千人不靠把整份 JSON 塞给浏览器；
  · RK3588/安卓上只有一个文件要备份，也不用担心写一半掉电。
老的 features.npz / features.npy + meta.json 底库第一次打开时自动迁进数据库，
原文件移到 legacy/ 留档，几千张底库不用重传。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import numpy as np

from .config import DATA_DIR

DIM = 512
COMMIT_EVERY = 200          # 攒够这么多条提交一次事务；逐条 commit 会让几千张底库慢一个数量级

# 底库目录名以前是 f"gallery_{model_key}" 直接拼出来的。识别模型键来自 config.MODELS，
# 只有字母数字下划线，拼出来没事；但 remote 模式下键是 f"remote_{remote_url}"，
# URL 天生带斜杠，于是 http://10.0.0.9:8080/v1/embed 会在 data/ 下面一路建出
#   data/gallery_remote_http:/10.0.0.9:8080/v1/embed/thumbs/
# 四五层以冒号结尾的目录，缩略图全散在里面（清空底库时也更容易只删掉最外层漏掉子树）。
# 更糟的是 URL 路径里只要有一段 .. ，底库就直接建到 data/ 外面去了 —— 跟当初
# _apply_backend_state 要校验 rec 是同一个道理，只是这条是从 remote_url 绕进来的。
# 所以键里出现不安全字符就整体转义成一个扁平名字；纯 [A-Za-z0-9._-] 的键原样保留，
# 已经跑了很久的 data/gallery_w600k_r50 一个字节都不用动。
_SAFE_KEY = re.compile(r"^[A-Za-z0-9._-]+$")


def gallery_dir_for(model_key: str) -> Path:
    """把模型键变成 DATA_DIR 下一个扁平、不会越界的目录名。"""
    key = str(model_key or "")
    if _SAFE_KEY.match(key):
        return DATA_DIR / f"gallery_{key}"
    # 转义后再挂 8 位摘要：URL 可能被下面的截断切掉尾巴，两个只差后缀的服务
    # 截断后会撞进同一个目录，共用了底库比撞名还难查。摘要取自已转义的 key 之外
    # 的原始串，保证同一 URL 在任何机器上都算出同一个目录。
    slug = quote(key, safe="")[:96]
    return DATA_DIR / f"gallery_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:8]}_{slug}"

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS persons(
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  name       TEXT NOT NULL UNIQUE,
  updated_at REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS templates(
  tid        TEXT PRIMARY KEY,
  person_id  INTEGER NOT NULL,
  source     TEXT NOT NULL DEFAULT '',
  box        TEXT NOT NULL DEFAULT '[]',
  score      REAL NOT NULL DEFAULT 0,
  quality    TEXT NOT NULL DEFAULT '{}',
  bad        INTEGER NOT NULL DEFAULT 0,
  thumb      TEXT NOT NULL DEFAULT '',
  dim        INTEGER NOT NULL DEFAULT 0,
  vec        BLOB NOT NULL,
  created_at REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_tpl_person ON templates(person_id);
"""


def _modal_dim(rows) -> int:
    """这一库向量本该是多少维：按声明维度投票，没声明的按落盘字节数投票。

    templates.dim 是写盘时从内存里的向量量出来的，所以一条被截断的记录
    声明的是 512、实际只有 511 —— 投出来的正好是完好那批的维度。
    票数一样的时候取较大的那个：截断只会把维度变小，不会变大。
    """
    votes: dict[int, int] = {}
    for row in rows:
        n = int(row[7] or 0) or (len(row[8] or b"") // 4)
        if n > 0:
            votes[n] = votes.get(n, 0) + 1
    if not votes:
        return DIM
    top = max(votes.values())
    return max(k for k, v in votes.items() if v == top)


@dataclass
class Template:
    tid: str
    person: str
    source: str
    vec: np.ndarray
    box: list[float] = field(default_factory=list)
    score: float = 0.0
    quality: dict = field(default_factory=dict)
    thumb: str = ""


class Gallery:
    def __init__(self, model_key: str):
        self.model_key = model_key
        self.dir = gallery_dir_for(model_key)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.thumbs_dir.mkdir(exist_ok=True)
        self.persons: dict[str, list[Template]] = {}
        self.matrix: np.ndarray = np.zeros((0, DIM), np.float32)
        self.index: list[Template] = []
        self.loaded_at = 0.0
        self.integrity: list[str] = []           # 载入时发现的存储异常，界面要能看见
        self._pid: np.ndarray | None = None      # 每条模板 -> 人员序号
        self._uniq: list[str] = []               # 人员序号 -> 人员名
        self._row: dict[str, int] = {}           # tid -> matrix 行号，供原地替换
        self._buf: np.ndarray | None = None      # matrix 的底层存储，几何扩容
        self._db: sqlite3.Connection | None = None
        self._pid_of: dict[str, int] = {}        # 人名 -> persons.id，省掉每条模板一次查询
        self._pending = 0                        # 已写入但未提交事务的模板条数
        self.version = 0                         # 底库内容版本号，任何增删改都 +1，给上层做缓存键
        self._uniq_idx: dict[str, int] = {}      # 人员名 -> _uniq 下标，供增量登记分组
        self._pid_buf: np.ndarray | None = None  # _pid 的底层存储，和 matrix 一样几何扩容
        # 注册在后台线程跑、评测在请求线程跑，两边动的是同一个库；
        # 分组要遍历 index 列表，边插边遍历会抛「list changed size during iteration」。
        # 数据库连接是 check_same_thread=False 开的，所有读写都靠这把锁排队。
        self.lock = threading.RLock()
        self.load()

    # ---------- 路径 ----------
    @property
    def db_file(self) -> Path:
        return self.dir / "gallery.db"

    @property
    def meta_file(self) -> Path:
        """旧格式元数据，只用于自动迁移，本版本不再写它。"""
        return self.dir / "meta.json"

    @property
    def vec_file(self) -> Path:
        """旧格式向量（矩阵 + tid 同文件），只用于自动迁移。"""
        return self.dir / "features.npz"

    @property
    def legacy_vec_file(self) -> Path:
        return self.dir / "features.npy"

    @property
    def legacy_dir(self) -> Path:
        """迁移成功后的老文件归档处，留着但不再被读取。"""
        return self.dir / "legacy"

    @property
    def thumbs_dir(self) -> Path:
        """底库唯一存图片的地方：每张模板一张裁剪缩略图，文件名就是 tid.jpg。"""
        return self.dir / "thumbs"

    # ---------- 数据库连接 ----------
    def _connect(self, create: bool) -> sqlite3.Connection | None:
        """取连接。create=False 时库文件不存在就返回 None，别凭空造一个空库。

        调用方必须已持有 self.lock（连接是跨线程共用的）。
        """
        if self._db is not None:
            return self._db
        if not create and not self.db_file.exists():
            return None
        try:
            db = sqlite3.connect(self.db_file, check_same_thread=False, timeout=20.0)
            try:
                db.execute("PRAGMA journal_mode=WAL")   # 后台注册在写、评测在读，别互相堵死
            except sqlite3.Error:
                pass                                     # 个别文件系统/U 盘不支持 WAL，用默认日志
            db.execute("PRAGMA synchronous=NORMAL")
            db.executescript(SCHEMA)
            db.commit()
        except sqlite3.Error as e:
            self.integrity.append(f"gallery.db 打不开：{type(e).__name__}: {e}")
            return None
        self._db = db
        return db

    def _close_db(self) -> None:
        """删库文件前必须先关连接，否则 Windows 上删不掉、WAL 里还压着没落盘的数据。"""
        if self._db is None:
            return
        try:
            self._db.commit()
        except sqlite3.Error:
            pass
        try:
            self._db.close()
        except sqlite3.Error:
            pass
        self._db = None
        self._pid_of = {}
        self._pending = 0

    def _flush(self, db: sqlite3.Connection) -> None:
        db.commit()
        self._pending = 0

    def _person_id(self, db: sqlite3.Connection, name: str) -> int:
        pid = self._pid_of.get(name)
        if pid is None:
            db.execute("INSERT OR IGNORE INTO persons(name,updated_at) VALUES(?,?)",
                       (name, time.time()))
            row = db.execute("SELECT id FROM persons WHERE name=?", (name,)).fetchone()
            pid = int(row[0])
            self._pid_of[name] = pid
        return pid

    def _insert(self, db: sqlite3.Connection, t: Template) -> None:
        q = t.quality or {}
        bad = 0 if q.get("ok", True) else 1        # 单独存一列：分页统计「待查」不用解析 JSON
        db.execute(
            "INSERT OR REPLACE INTO templates"
            "(tid,person_id,source,box,score,quality,bad,thumb,dim,vec,created_at)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (t.tid, self._person_id(db, t.person), t.source or "",
             json.dumps(t.box or [], ensure_ascii=False), float(t.score or 0.0),
             json.dumps(q, ensure_ascii=False, default=str), bad, t.thumb or "",
             int(np.asarray(t.vec).reshape(-1).shape[0]),
             np.asarray(t.vec, np.float32).reshape(-1).tobytes(), time.time()))

    # ---------- 载入 ----------
    def load(self) -> None:
        if self.db_file.exists():
            self._load_db()
            return
        self._migrate_legacy()

    def _load_db(self) -> None:
        db = self._connect(create=False)
        if db is None:
            return
        try:
            mk = db.execute("SELECT value FROM meta WHERE key='model_key'").fetchone()
            built = db.execute("SELECT value FROM meta WHERE key='built_at'").fetchone()
            rows = db.execute(
                "SELECT t.tid, p.name, t.source, t.box, t.score, t.quality, t.thumb,"
                " t.dim, t.vec FROM templates t JOIN persons p ON p.id=t.person_id"
                " ORDER BY t.rowid").fetchall()
        except sqlite3.Error as e:
            self.integrity.append(f"gallery.db 读取失败：{type(e).__name__}")
            return
        if mk and str(mk[0]) != self.model_key:
            self.integrity.append(f"gallery.db 属于模型 {mk[0]}，与当前 {self.model_key} "
                                  "向量空间不通用，按空库处理（换模型请重建底库）")
            return
        self.persons, self.index = {}, []
        vecs: list[np.ndarray] = []
        # 全库维度按「多数派」定，不是按第一条读到的长度。
        # 以前 dim 由第一条向量定死：只要有一条被截断过（写盘时断电、磁盘满，
        # 512 个 float 落盘成了 511 个）恰好排在最前面，后面几千条完好的 512 维
        # 就全被判成「长度对不上」跳过 —— 几千人的底库加载完只剩半条残向量，
        # 界面上是「底库是空的」，人却还整整齐齐躺在库里。
        want = _modal_dim(rows)
        broken: list[str] = []
        for tid, name, source, box, score, quality, thumb, vdim, blob in rows:
            try:
                v = np.frombuffer(blob or b"", dtype=np.float32)
                if v.size == 0:
                    raise ValueError("向量是空的")
                if v.size != want:
                    raise ValueError(f"只有 {v.size} 维，全库是 {want} 维")
                rec = json.loads(quality or "{}")
                bx = json.loads(box or "[]")
            except Exception:
                # 单条坏了就跳过这一条：整库打不开比少一个人更糟
                broken.append(f"{tid}（{name}）")
                continue
            t = Template(tid=tid, person=name, source=source or "", vec=v,
                         box=bx if isinstance(bx, list) else [],
                         score=float(score or 0.0),
                         quality=rec if isinstance(rec, dict) else {},
                         thumb=thumb or "")
            self.persons.setdefault(name, []).append(t)
            self.index.append(t)
            vecs.append(t.vec)
        if broken:
            # 一条一人：3475 人的库坏 357 条时以前会灌出 357 条告警，
            # 界面那一片红根本读不动，真正的信号（坏了几条）反而被埋了。
            self.integrity.append(
                f"{len(broken)} 条模板向量损坏或维度不符，已跳过："
                + "、".join(broken[:5]) + ("…" if len(broken) > 5 else ""))
        self._set_matrix(vecs)
        try:
            self.loaded_at = float(built[0]) if built else 0.0
        except (TypeError, ValueError):
            self.loaded_at = 0.0
        self._group()
        self._row = {t.tid: i for i, t in enumerate(self.index)}

    def _set_matrix(self, vecs: list[np.ndarray]) -> None:
        """把向量摊成一块连续矩阵，并预留 25% 余量给后续注册，避免每次追加都整块拷贝。"""
        n = len(vecs)
        dim = int(vecs[0].reshape(-1).shape[0]) if n else DIM
        buf = np.zeros((max(64, int(n * 1.25) + 32), dim), np.float32)
        if n:
            buf[:n] = np.stack([np.asarray(v, np.float32).reshape(-1) for v in vecs])
        self._buf = buf
        self.matrix = buf[:n]
        for i, t in enumerate(self.index):
            t.vec = self.matrix[i]      # 模板向量和矩阵行共用同一块内存，检索不会读到两份数据

    def _migrate_legacy(self) -> None:
        """老底库（features.npz / features.npy + meta.json）自动搬进数据库。

        用户手上可能有几千张已注册好的底库，读一次就升级完，不用重传。
        只有确认数据库读回的条数和内存一致，才把原文件挪进 legacy/。
        """
        if not self.meta_file.exists():
            return
        try:
            meta = json.loads(self.meta_file.read_text(encoding="utf-8"))
        except Exception as e:
            self.integrity.append(f"meta.json 读取失败：{type(e).__name__}")
            return
        if meta.get("model_key") != self.model_key:
            return
        recs = meta.get("templates", [])
        by_tid = {r.get("tid"): r for r in recs if r.get("tid")}

        rows, tids = self._read_rows(by_tid, recs)
        if rows is None:
            return

        self.persons, self.index = {}, []
        for i, tid in enumerate(tids):
            rec = dict(by_tid[tid])
            rec.pop("tid", None)
            rec.pop("vec", None)
            try:
                t = Template(tid=tid, vec=rows[i], **rec)
            except TypeError:
                # 老版本存的字段对不上时跳过这一条，别让整个库打不开
                self.integrity.append(f"模板 {tid} 元数据字段异常，已跳过")
                continue
            self.persons.setdefault(t.person, []).append(t)
            self.index.append(t)

        self._set_matrix([t.vec for t in self.index])
        self.loaded_at = meta.get("built_at", 0.0) or time.time()
        self._group()
        self._row = {t.tid: i for i, t in enumerate(self.index)}
        if self.index and self._dump_db():
            self._stash_legacy()

    def _read_rows(self, by_tid: dict, recs: list):
        """返回 (矩阵行, 与之一一对应的 tid)。向量永远带着自己的 tid 一起读出来。"""
        if self.vec_file.exists():
            try:
                with np.load(self.vec_file, allow_pickle=False) as f:
                    mat = np.asarray(f["matrix"], np.float32)
                    tids = [str(x) for x in np.asarray(f["tids"]).tolist()]
            except Exception as e:
                self.integrity.append(f"features.npz 读取失败：{type(e).__name__}")
                return None, None
            if mat.shape[0] != len(tids):
                self.integrity.append(
                    f"features.npz 行数 {mat.shape[0]} 与 tid 数 {len(tids)} 不一致，已按较小值取用")
                n = min(mat.shape[0], len(tids))
                mat, tids = mat[:n], tids[:n]
            keep = [i for i, tid in enumerate(tids) if tid in by_tid]
            lost = len(tids) - len(keep)
            if lost:
                self.integrity.append(f"{lost} 条向量缺少元数据（最后一次注册被打断），需重传这些图片")
            return mat[keep], [tids[i] for i in keep]
        if self.legacy_vec_file.exists():
            # 更老的格式：向量在 features.npy，同时又冗长地抄在 meta.json 里。
            try:
                m = np.load(self.legacy_vec_file)
            except Exception as e:
                self.integrity.append(f"features.npy 读取失败：{type(e).__name__}")
                return None, None
            tids = [r.get("tid") for r in recs]
            tids = [t for t in tids if t]
            if m.ndim == 2 and m.shape[0] == len(recs):
                return m, tids
            self.integrity.append("features.npy 与 meta.json 行数不一致，已按 meta 内冗余向量重建")
            try:
                return np.stack([np.asarray(r["vec"], np.float32) for r in recs]), tids
            except Exception:
                return None, None
        return np.zeros((0, DIM), np.float32), []

    def _dump_db(self) -> bool:
        """全量写入数据库，读回条数对得上才算成功（迁移用）。"""
        with self.lock:
            db = self._connect(create=True)
            if db is None:
                return False
            try:
                db.execute("DELETE FROM templates")
                db.execute("DELETE FROM persons")
                db.execute("DELETE FROM meta")
                self._pid_of = {}
                for t in self.index:
                    self._insert(db, t)
                db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('model_key',?)",
                           (self.model_key,))
                db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('built_at',?)",
                           (str(self.loaded_at),))
                db.commit()
                self._pending = 0
                n = int(db.execute("SELECT COUNT(*) FROM templates").fetchone()[0])
            except sqlite3.Error as e:
                self.integrity.append(f"迁移到 gallery.db 失败：{type(e).__name__}: {e}")
                try:
                    db.rollback()
                except sqlite3.Error:
                    pass
                return False
            if n != len(self.index):
                self.integrity.append(
                    f"迁移后数据库只有 {n} 条，应为 {len(self.index)} 条，原文件已保留")
                return False
            return True

    def _stash_legacy(self) -> None:
        """迁移成功后把老文件挪进 legacy/，不直接删：万一数据库文件被误改还能回退。"""
        try:
            self.legacy_dir.mkdir(exist_ok=True)
            for p in (self.vec_file, self.meta_file, self.legacy_vec_file):
                if p.exists():
                    shutil.move(str(p), str(self.legacy_dir / p.name))
        except OSError as e:
            self.integrity.append(f"老底库文件归档失败：{type(e).__name__}: {e}")

    def save(self) -> None:
        """提交事务并把统计口径写回 meta。

        向量早在 add() 里就进数据库了，这里只是收尾：注册到一半进程被 Killed，
        最多只少最后没提交的那几十条，不会出现界面上有人、向量缺行的半套库。
        """
        with self.lock:
            db = self._connect(create=True)
            if db is None:
                raise RuntimeError("; ".join(self.integrity[-3:]) or "gallery.db 不可用")
            self.loaded_at = time.time()
            try:
                db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('model_key',?)",
                           (self.model_key,))
                db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('built_at',?)",
                           (str(self.loaded_at),))
                self._flush(db)
                try:
                    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")   # 别让 -wal 一直涨到几百 MB
                except sqlite3.Error:
                    pass
            except sqlite3.Error as e:
                raise RuntimeError(f"gallery.db 提交失败 {type(e).__name__}: {e}") from e

    # ---------- 写入 ----------
    def _ensure(self, n: int, dim: int | None = None) -> None:
        """matrix 是 _buf 的前 n 行视图。按几何倍数扩容，否则每加一条都要整块拷贝，
        万级底库注册会退化成 O(n²)。dim 取第一条向量的维度：空库时 matrix 只是占位。"""
        dim = int(dim or (self.matrix.shape[1] if self.matrix.size else DIM))
        if self._buf is None or self._buf.shape[0] < n or self._buf.shape[1] != dim:
            cap = max(64, int(n * 1.5) + 8)
            nb = np.zeros((cap, dim), np.float32)
            if self._buf is not None and self.matrix.size:
                nb[: self.matrix.shape[0]] = self.matrix
            self._buf = nb
            self.matrix = self._buf[:len(self.index)]   # 先指向新块，再重绑，否则老 index 越界
            # 换过底层 buffer 后，老 Template.vec 还指着旧块。此刻内容一致（刚拷过来），
            # 但之后 matrix 行内一改它们就看不到，np.stack([t.vec ...]) 会拿到改之前的快照。
            # 扩容次数是 O(log n)，这里整体重绑一次，代价可以忽略，心智负担少一层。
            for i, t in enumerate(self.index):
                t.vec = self.matrix[i]
        self.matrix = self._buf[:n]

    def add(self, t: Template, max_per_person: int = 0) -> bool:
        """写入模板。同一 tid（同一人 + 同一源图）原地替换而不是追加。

        底库文件夹拖两次是常见操作，膨胀出的重复模板会让「同人互配」分数虚高，
        把阈值建议带偏，也会让底库人数统计失真。数据库侧靠主键 REPLACE 天然去重。

        max_per_person>0 时在锁内判定这个人是否已经存满，满了就不写并返回 False。
        「每人模板上限」原来是在 bench.enroll_image 里先数一遍再写的：注册并行起来以后
        两个 worker 会同时数到「还差一张」然后双双写入，上限就成了「至少这么多」。
        判定挪到真正写入的这一刻，这个竞态就不存在了；重复注册（同 tid）不受上限影响。
        """
        with self.lock:
            if max_per_person:
                same = self.persons.get(t.person) or []
                if (len(same) >= max_per_person
                        and not any(x.tid == t.tid for x in same)):
                    return False
            # 先落库，再动内存。顺序反过来的话，gallery.db 只读或磁盘写满时
            # _write 抛出来，人已经进了 persons/index、_row 也登记了，
            # 内存里人多了一个、matrix 里没有他、数据库里没这行 ——
            # 检索能命中一个「幽灵」，重启后人消失，两边都查不出为什么。
            self._write(t)
            row = self._row.get(t.tid)
            if row is not None and row < len(self.index):
                self.index[row] = t
                for j, old_t in enumerate(self.persons.get(t.person) or []):
                    if old_t.tid == t.tid:
                        self.persons[t.person][j] = t
                        break
                if row < self.matrix.shape[0] and self.matrix.shape[1] == t.vec.shape[0]:
                    self.matrix[row] = t.vec
                else:
                    self.rebuild(sync_db=True)     # 维度对不上，整块重建才能对齐行号
            else:
                self.persons.setdefault(t.person, []).append(t)
                self.index.append(t)
                # 先抓住原向量再扩容：_ensure 换底层块时会把 index 里所有模板的 vec
                # 重绑到新块，而这条的数据此刻还没写进去，重绑等于把它清成一整行 0，
                # 于是新入库的人检索出来永远 0 分（认不出来，也不报错）
                vec = t.vec
                self._ensure(len(self.index), vec.shape[0])
                self.matrix[len(self.index) - 1] = vec
                t.vec = self.matrix[len(self.index) - 1]
                self._row[t.tid] = len(self.index) - 1
                self._append_group(t.person)       # 增量登记分组，别让每次检索都全量重算
            self.version += 1
            return True

    def _write(self, t: Template) -> None:
        """一条模板落库，攒够 COMMIT_EVERY 条提交一次事务。失败直接抛。

        以前出错只往 integrity 里塞一句话然后 return，注册那边照样报「成功」：
        底库目录被设成只读、磁盘写满、gallery.db 被别的进程占着的时候，
        内存里人数在涨、界面看着全对，重启以后这批人静悄悄全没了。
        integrity 只在顶栏亮个小标记，没有人会一直盯着它看。
        抛出去之后 _enroll_one 会记一条「需复查」，_finish_enroll 又会把
        gallery.save() 的异常变成 status=error，两条路都会响。
        """
        with self.lock:
            db = self._connect(create=True)
            if db is None:
                raise RuntimeError("gallery.db 不可用："
                                   + ("; ".join(self.integrity[-3:]) or "无法创建数据库连接"))
            try:
                self._insert(db, t)
            except sqlite3.Error as e:
                raise RuntimeError(f"模板 {t.tid} 落库失败：{type(e).__name__}: {e}") from e
            self._pending += 1
            if self._pending >= COMMIT_EVERY:
                try:
                    self._flush(db)
                except sqlite3.Error as e:
                    raise RuntimeError(f"底库提交失败：{type(e).__name__}: {e}") from e

    def clear(self) -> dict:
        """清空底库：内存模板、数据库、缩略图一起删，删完目录里不留任何底库数据。

        以前只清内存再 save()，等于在磁盘上留了一个空库加几千张缩略图：
        界面不再引用它们，/api/thumb 却还认得这些文件名，几百 MB 就这么一直躺着。
        库里没模板了，磁盘上就不该留图和向量。
        """
        with self.lock:
            self.persons, self.index = {}, []
            self.matrix = np.zeros((0, DIM), np.float32)
            self._buf = None
            self._pid, self._uniq, self._row = None, [], {}
            self.loaded_at = 0.0
            self.integrity = []       # 文件都删了，上次载入时的告警已经没有对应的库
            return self.purge_files()

    def purge_files(self) -> dict:
        """删掉落在磁盘上的底库文件，返回删了什么、释放多少字节。

        删失败要写进 integrity，让顶栏亮起来——只删掉一半却报「已清空」，
        比不清更误导人。
        """
        with self.lock:
            self._close_db()            # 先关连接再删，WAL 里的数据才会真正落盘后一起消失
            removed, freed, thumbs = 0, 0, 0
            for p in (self.db_file, Path(str(self.db_file) + "-wal"),
                      Path(str(self.db_file) + "-shm"), self.vec_file, self.meta_file,
                      self.legacy_vec_file):
                try:
                    freed += p.stat().st_size
                    p.unlink()
                    removed += 1
                except FileNotFoundError:
                    pass
                except OSError as e:
                    self.integrity.append(f"{p.name} 删除失败：{type(e).__name__}: {e}")
            for d, count in ((self.thumbs_dir, True), (self.legacy_dir, False)):
                if d.is_dir():
                    try:
                        for f in d.rglob("*"):
                            if f.is_file():
                                freed += f.stat().st_size
                                if count:
                                    thumbs += 1
                        shutil.rmtree(d)
                    except OSError as e:
                        self.integrity.append(f"{d.name}/ 删除失败：{type(e).__name__}: {e}")
            self.thumbs_dir.mkdir(parents=True, exist_ok=True)   # 注册还要往里写缩略图
            # 磁盘上的东西没了，内存里那套也已经是空的，version 必须一起涨：
            # 阈值分布报告按 (gallery.version) 缓存，不涨的话清空底库以后
            # 界面拿到的还是三千万个 genuine 对算出来的推荐阈值，看着像没清干净。
            self.version += 1
            return {"files_removed": removed, "thumbs_removed": thumbs, "freed_bytes": freed}

    def _drop_one(self, name: str) -> tuple[bool, int]:
        """删一张缩略图，回报「删掉了没有」和「释放了多少字节」。

        要字节数是因为删除的提示得说人话：只说「已删除张三」，而底库缩略图动辄几百 MB，
        没人相信盘真的空出来了。两个数分开返回而不是只回字节：0 字节的图删成功了
        和一张删不动的图都表现为 0，混成一个数就会把没删干净的删除说成删干净了。
        文件名来自数据库、可能被手改过，先确认没跑出 thumbs/ 再动手。
        """
        if not name:
            return False, 0
        root = self.thumbs_dir.resolve()
        p = (root / name).resolve()
        try:
            if p.is_file() and str(p).startswith(str(root) + os.sep):
                size = p.stat().st_size
                p.unlink()
                return True, size
        except OSError:
            return False, 0
        return False, 0

    def _drop_thumb(self, name: str) -> bool:
        """删一张缩略图，只关心成没成。"""
        return self._drop_one(name)[0]

    def has_person(self, person: str) -> bool:
        """库里有没有这个人（O(1)，给删除接口在动手之前验一次）。

        remove_person 对不存在的人只返回 persons_removed=0、不报错，接口照这返回 200
        的话界面会弹「已删除」而人其实还在库里 —— 多半是列表比这份新，或人名差个空格。
        """
        with self.lock:
            return person in self.persons

    def remove_person(self, person: str) -> dict:
        """删一个人：内存模板、数据库行、向量、他的缩略图一起走。

        缩略图不跟着删的话，底库来回删改几十次，thumbs/ 就全是没人认领的孤儿 JPG。
        """
        with self.lock:
            gone = self.persons.get(person) or []
            # 数据库排第一位：gallery.db 只读时它抛出来，内存和缩略图都还是原样。
            # 以前先删缩略图再删库，库删失败只记一条 integrity，结果人从界面上没了、
            # JPG 也没了，数据库里他的行还在 —— 重启整个人带着一堆 404 头像复活。
            self._delete_db_person(person)
            self.persons.pop(person, None)
            hits = [self._drop_one(t.thumb) for t in gone]
            dropped = sum(1 for ok, _ in hits if ok)
            freed = sum(n for _, n in hits)
            self.rebuild()
            return {"persons_removed": 1 if gone else 0, "templates_removed": len(gone),
                    "thumbs_removed": dropped, "freed_bytes": freed}

    def remove_templates(self, tids) -> int:
        """按 tid 精确删掉若干模板：内存、数据库行、缩略图一起走，返回删掉的条数。

        remove_person 只能整人删除，但「同一张底库图重新注册」只会覆盖主角那条
        （tid 相同原地替换）。上一次用「全部人脸」多收的旁人模板 base#1 / base#2
        会一直挂在这个人名下，检索时把他认成路人，底库人数也看不出异常。
        删空的人要连 persons 行的空壳一起清掉，否则分页列表出现「0 张」的幽灵人员。
        """
        want = {str(t) for t in (tids or ()) if t}
        if not want:
            return 0
        with self.lock:
            # 先看清要删谁，再删库，最后才动内存和图：和 remove_person 同一个理由，
            # 库删不动就整笔不做，不能留下「图上没了、库里还在」的半套删除。
            hit = {p: [t for t in lst if t.tid in want]
                   for p, lst in self.persons.items() if any(t.tid in want for t in lst)}
            if not hit:
                return 0
            self._delete_db_templates(want)
            removed = 0
            for person, gone in hit.items():
                removed += len(gone)
                for t in gone:
                    self._drop_thumb(t.thumb)     # 缩略图不删就是 thumbs/ 里的孤儿 JPG
                keep = [t for t in self.persons[person] if t.tid not in want]
                if keep:
                    self.persons[person] = keep
                else:
                    self.persons.pop(person, None)
                    self._pid_of.pop(person, None)
            self.rebuild()        # 行号、_row 映射、分组都得跟着重排
            return removed

    def _require_db(self) -> sqlite3.Connection | None:
        """删除路径专用的取连接：库文件在、却打不开，必须报出来。

        _connect(create=False) 返回 None 有两种含义：库文件压根不存在
        （内存里也没东西，静默返回是对的），和 gallery.db 存在但只读 /
        被别的进程占着 / 目录不可写建不出 -wal。后者静默返回 None 的话，
        调用方照样把人从内存和 thumbs/ 里删掉，数据库里的行一条没少 ——
        界面立刻空了，重启整个人带着一堆 404 头像复活。
        """
        db = self._connect(create=False)
        if db is None and self.db_file.exists():
            raise RuntimeError("gallery.db 不可用："
                               + ("; ".join(self.integrity[-3:]) or "无法打开数据库"))
        return db

    def _delete_db_templates(self, tids) -> None:
        """按 tid 删数据库行。tid 是主键，直接删即可；人走空了就顺手清壳。"""
        with self.lock:
            db = self._require_db()
            if db is None:
                return          # 库文件不存在，本来就没有行可删
            try:
                for tid in tids:
                    db.execute("DELETE FROM templates WHERE tid=?", (tid,))
                db.execute("DELETE FROM persons WHERE id NOT IN"
                           " (SELECT DISTINCT person_id FROM templates)")
                self._flush(db)
            except sqlite3.Error as e:
                raise RuntimeError(f"删除模板失败：{type(e).__name__}: {e}") from e

    def _delete_db_person(self, person: str) -> None:
        with self.lock:
            db = self._require_db()
            if db is None:
                return          # 库文件不存在，本来就没有行可删
            try:
                pid = self._pid_of.pop(person, None)
                if pid is None:
                    row = db.execute("SELECT id FROM persons WHERE name=?", (person,)).fetchone()
                    pid = int(row[0]) if row else None
                if pid is not None:
                    db.execute("DELETE FROM templates WHERE person_id=?", (pid,))
                    db.execute("DELETE FROM persons WHERE id=?", (pid,))
                # 模板被删空的人不留空行，否则分页列表会出现「0 张」的幽灵人员
                db.execute("DELETE FROM persons WHERE id NOT IN"
                           " (SELECT DISTINCT person_id FROM templates)")
                self._flush(db)
            except sqlite3.Error as e:
                raise RuntimeError(f"删除人员 {person} 失败：{type(e).__name__}: {e}") from e

    def rebuild(self, sync_db: bool = False) -> None:
        with self.lock:
            self.version += 1
            self.index = [t for lst in self.persons.values() for t in lst]
            self._buf = (
                np.ascontiguousarray(np.stack([t.vec for t in self.index]).astype(np.float32))
                if self.index else np.zeros((0, DIM), np.float32)
            )
            self.matrix = self._buf[: len(self.index)]
            self._pid = None
            self._row = {t.tid: i for i, t in enumerate(self.index)}
            for i, t in enumerate(self.index):
                t.vec = self.matrix[i]
            if sync_db and self.index:
                self._dump_db()

    def count(self, person: str) -> int:
        """某人当前有几张模板（在锁里读，避免边注册边读到半个数）。"""
        with self.lock:
            return len(self.persons.get(person, []))

    def stale_tids(self, person: str, source: str, keep) -> list[str]:
        """列出某人在某张源图上留下的、本次没有再写入的旧模板 tid。

        同样必须走锁：注册并行以后另一些 worker 正在改 persons，
        裸遍历照样会撞「dictionary changed size during iteration」。
        """
        with self.lock:
            return [t.tid for t in self.persons.get(person, ())
                    if t.source == source and t.tid not in keep]

    def snapshot_groups(self, min_templates: int = 2) -> dict[str, np.ndarray]:
        """取「有多张模板的人」的向量组快照，供同人互配之类的统计用。

        必须在锁里拷：后台注册线程随时在 add()，裸遍历 persons 会撞
        「dictionary changed size during iteration」，而且往往发生在几千张
        推理全跑完之后，整批结果一起丢。
        """
        with self.lock:
            return {p: np.stack([t.vec for t in lst])
                    for p, lst in self.persons.items() if len(lst) >= min_templates}

    # ---------- 分组 ----------
    def _append_group(self, person: str) -> None:
        """增量登记一条新模板的「模板 -> 人员」分组。

        以前每次 add 都把 _pid 置 None，于是注册期间**每一次** 1:N 检索都要在锁里
        重算一遍 np.unique(object array)：3475 人实测 0.64 ms，而热态整次 search
        才 1.18 ms —— 边注册边评测时约七成的检索耗时花在重复分组上。
        分组落后、形状对不上或还没建过，一律退回全量 _group()，宁可慢不可错。
        """
        if (self._pid is None or self._pid_buf is None
                or self._pid.size != len(self.index) - 1
                or self._pid_buf.shape[0] < len(self.index)):
            self._group()
            return
        idx = self._uniq_idx.get(person)
        if idx is None:
            idx = len(self._uniq)
            self._uniq.append(person)
            self._uniq_idx[person] = idx
        self._pid_buf[self._pid.size] = idx
        self._pid = self._pid_buf[: self._pid.size + 1]

    def _group(self) -> None:
        """建立 模板->人员 映射，供按人聚合使用。"""
        with self.lock:
            if not self.index:
                self._pid, self._uniq, self._pid_buf, self._uniq_idx = None, [], None, {}
                return
            names = [t.person for t in list(self.index)]
            uniq, pid = np.unique(
                np.asarray(names, dtype=object), return_inverse=True
            )
            self._uniq = [str(x) for x in uniq]
            self._pid = np.asarray(pid, dtype=np.int64)
            # 留出自定义容量，之后 add() 才能 O(1) 追加而不是每次整块拷贝
            cap = max(64, int(self._pid.size * 1.5) + 8)
            self._pid_buf = np.zeros(cap, np.int64)
            self._pid_buf[: self._pid.size] = self._pid
            self._pid = self._pid_buf[: self._pid.size]
            self._uniq_idx = {n: i for i, n in enumerate(self._uniq)}

    # ---------- 检索 ----------
    def search(self, vec: np.ndarray, topk: int = 5, per_person: bool = True):
        """精确 1:N 余弦检索。

        per_person=True 时同一人的多张模板只保留最高分，返回的 top-k 是 k 个
        不同的人——这是 1:N 身份识别的正确口径（否则同一人的 5 个模板会占满 topk）。
        返回 (scores, person_names)。
        """
        with self.lock:
            mat, pid, uniq, idx = self.matrix, self._pid, list(self._uniq), list(self.index)
            if mat.size == 0:
                return np.zeros((0,), np.float32), []
            if per_person and pid is None:
                self._group()
                pid, uniq = self._pid, list(self._uniq)
            s = mat @ vec
            k = min(topk, mat.shape[0]) if not per_person else min(topk, len(uniq) or 1)
            if k <= 0:
                return np.zeros((0,), np.float32), []
            if per_person:
                best = np.full(len(uniq), np.float32(-1.0))
                np.maximum.at(best, pid, s)
                part = np.argpartition(-best, k - 1)[:k]
                part = part[np.argsort(-best[part])]
                return best[part], [uniq[int(i)] for i in part]
            part = np.argpartition(-s, k - 1)[:k]
            part = part[np.argsort(-s[part])]
            return s[part], [idx[int(i)].person for i in part]

    # ---------- 列表 / 分页 ----------
    def page(self, q: str = "", offset: int = 0, size: int = 80) -> dict:
        """按人名分页取底库：几千个人一次全渲染会把浏览器卡住，界面只需要当前这一屏。

        返回 {total, offset, size, q, items:[{person,templates,bad,thumb,avg_score}]}。
        total 只算「还有模板的人」，删空的人不算，免得页码对不上。
        """
        size = max(1, min(int(size or 80), 500))
        offset = max(0, int(offset or 0))
        kw = (q or "").strip()
        cond, args = "", []
        if kw:
            esc = kw.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            cond, args = " AND p.name LIKE ? ESCAPE '\\' ", [f"%{esc}%"]
        empty = {"total": 0, "offset": offset, "size": size, "q": kw, "items": []}
        with self.lock:
            db = self._connect(create=False)
            if db is None:
                return empty
            has = "EXISTS(SELECT 1 FROM templates t WHERE t.person_id=p.id)"
            try:
                total = int(db.execute(
                    f"SELECT COUNT(*) FROM persons p WHERE {has}{cond}", args).fetchone()[0])
                rows = db.execute(
                    "SELECT p.name, COUNT(t.tid), COALESCE(SUM(t.bad),0), AVG(t.score),"
                    " (SELECT x.thumb FROM templates x WHERE x.person_id=p.id"
                    "  ORDER BY x.rowid LIMIT 1)"
                    " FROM persons p JOIN templates t ON t.person_id=p.id"
                    f" WHERE 1=1{cond}"
                    " GROUP BY p.id ORDER BY p.name LIMIT ? OFFSET ?",
                    args + [size, offset]).fetchall()
            except sqlite3.Error as e:
                self.integrity.append(f"底库分页失败：{type(e).__name__}: {e}")
                return empty
        items = [{"person": r[0], "templates": int(r[1]), "bad": int(r[2] or 0),
                  "avg_score": round(float(r[3]), 4) if r[3] is not None else 0,
                  "thumb": r[4] or ""} for r in rows]
        return {"total": total, "offset": offset, "size": size, "q": kw, "items": items}

    def person_list(self) -> list[dict]:
        """全量人员名单（带模板数/缩略图）。

        原来写的是 page(size=10**6)，但 page() 把 size 夹在 500 以内
        —— 那个上限是给前端一屏渲染用的，不是给「取全部」用的。
        结果是 3475 人的底库这里只回 500 条，和同一份响应里的
        stats.persons=3475 直接打架。改成按上限翻页累加，
        比把 page() 的上限改掉安全：那个上限还在替前端挡误传的巨大 size。
        """
        step, size = 500, 0
        out = []
        while True:
            part = self.page(offset=size, size=step)
            out.extend(part["items"])
            size += step
            if size >= part["total"] or not part["items"]:
                return out

    def names(self) -> list[str]:
        """全部人名（不带缩略图/质量），给探针真值下拉用，几千人也只有几十 KB。"""
        with self.lock:
            db = self._connect(create=False)
            if db is None:
                return [t.person for t in self.index]
            try:
                return [str(r[0]) for r in db.execute(
                    "SELECT name FROM persons ORDER BY name").fetchall()]
            except sqlite3.Error:
                return sorted({t.person for t in self.index})

    def stats(self) -> dict:
        with self.lock:
            try:
                db_bytes = self.db_file.stat().st_size if self.db_file.exists() else 0
            except OSError:
                db_bytes = 0
            return {
                "model_key": self.model_key,
                "persons": len(self.persons),
                "templates": len(self.index),
                "dim": int(self.matrix.shape[1]) if self.matrix.size else 0,
                "built_at": self.loaded_at,
                "storage": "sqlite",
                "db_bytes": db_bytes,
                "integrity": list(self.integrity),
            }


def tid_for(person: str, source: str) -> str:
    return hashlib.sha1(f"{person}::{source}".encode()).hexdigest()[:16]
