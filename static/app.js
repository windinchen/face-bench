const $ = (s) => document.querySelector(s);
const S = { env: null, gallerySel: [], probeSel: [], run: null, persons: [], unreadable: [],
        gp: 1, gs: 80, gq: '', gloading: false,
        page: { offset: 0, only: 'hit', total: 0, matched: 0 } };
/* 模型还在下载时点「应用」的那次切换：整份请求体存下来，下载完成自动补发一次 */
S.pendingApply = null;

function toast(msg, ms = 2600) {
  const t = $('#toast'); t.textContent = msg; t.classList.add('on');
  clearTimeout(t._t); t._t = setTimeout(() => t.classList.remove('on'), ms);
}
const pct = (v) => (v === null || v === undefined ? '—' : (v * 100).toFixed(1) + '%');
const ms = (v) => (v === null || v === undefined ? '—' : v.toFixed(1) + ' ms');
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
/* 浏览器把「新版 app.js + 旧版 index.html」拼在一起用时，控件会取不到，
   直接 .value 就抛 Cannot read properties of null，用户只看到「失败」两个字。 */
const val = (sel, d) => { const el = $(sel); return el && el.value !== '' ? el.value : d; };
function requireUi(sels) {
  const miss = sels.filter((x) => !$(x));
  if (!miss.length) return true;
  toast(`页面控件缺失 ${miss.join(', ')}——浏览器在用旧版页面，请按 Cmd+Shift+R 强制刷新后重试`, 7000);
  return false;
}

/* 服务端 422 的 detail 是个数组（FastAPI 校验错误），直接 String() 会变成
   [object Object]，用户只能回一句「提示评测失败」。这里展开成可读文本。 */
function fmtDetail(d) {
  if (Array.isArray(d)) {
    return d.map((e) => (e && typeof e === 'object')
      ? `${e.msg ?? JSON.stringify(e)}${e.loc ? '（' + e.loc.join('.') + '）' : ''}` : String(e)
    ).join('；');
  }
  if (d && typeof d === 'object') return JSON.stringify(d);
  return String(d ?? '');
}
async function whyFailed(r) {
  const text = await r.text().catch(() => '');
  let detail = text;
  try { detail = JSON.parse(text).detail ?? text; } catch (e) { /* 非 JSON 就原样留着 */ }
  const s = fmtDetail(detail);
  return `HTTP ${r.status} ${r.statusText}：${s.slice(0, 400) || '服务端没给出原因'}`;
}
async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw await httpError(r);
  return r.json();
}
async function postJSON(url, body) {
  const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) throw await httpError(r);
  return r.json();
}
async function delJSON(url) {
  const r = await fetch(url, { method: 'DELETE' });
  if (!r.ok) throw await httpError(r);
  return r.json();
}
/* 状态码要挂在错误对象上：模型还在后台下载时服务端回 409（稍后再试就能成），
   和 400（参数/文件真有问题）得分开 —— 前者前端要把这次切换记下来，
   下载完自动补一次，不该让用户对着「下载中」三个字反复点应用。 */
async function httpError(r) {
  const e = new Error(await whyFailed(r));
  e.status = r.status;
  return e;
}
/* 传底库之前先花 0.1 秒确认「这个窗口真的能把字节发出去」。
   内置预览窗口、嵌在应用里的网页壳子这类环境，File 句柄交给网络进程后读不到
   磁盘，请求体会整个丢掉（Content-Length: 0），几千张底库就是几十屏同一个 422。
   与其让用户看完 58 批同样的报错，不如在第一张之前就拦住并把话说清楚。 */
/* 浏览器这边挂掉的错只会写进页面 DOM，服务端日志一个字都看不见，
   用户往往只能回一句「评测失败」。失败时把原文和浏览器标识回传一份，
   下次一眼就能看出是读盘、发请求还是页面版本的问题。 */
function clientError(stage, e, extra) {
  try {
    const body = Object.assign({ stage, msg: String((e && e.message) || e || ''),
      stack: String((e && e.stack) || '').slice(0, 600), ua: navigator.userAgent }, extra || {});
    fetch('/api/clientlog', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body) }).catch(() => {});
  } catch (x) { /* 上报自己再坏就不能添乱了 */ }
}
const PNG1X1 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==';
async function preflight() {
  const bin = atob(PNG1X1);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  let j = {};
  try {
    const sig = typeof AbortSignal !== 'undefined' && AbortSignal.timeout
      ? AbortSignal.timeout(8000) : undefined;
    const r = await fetch('/api/preflight', { method: 'POST', signal: sig,
      headers: { 'Content-Type': 'image/png' }, body: new Blob([u8], { type: 'image/png' }) });
    j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
  } catch (e) {
    throw new Error(`上传自检没通过（${e.message}）。请确认服务还在跑，然后用 Safari（或 Chrome）打开 `
      + '打开 http://127.0.0.1:8617 再试');
  }
  if (!j.bytes) throw new Error('这个窗口发不出文件内容（服务端只收到 0 字节）。请改用 Safari（或 Chrome）打开 '
    + '直接打开 http://127.0.0.1:8617 重新选择图片，别在应用内置的预览窗口里上传；本机目录请用「导入本机目录」；'
    + '或者把底库放进一个文件夹，用旁边的「导入本机目录」，完全不走浏览器上传');
  return true;
}

/* ---------- 拖拽 / 选择：保留相对路径 ---------- */
/* 把 <根目录>/<人员>/<图片> 归一化成 <人员>/<图片> 交给后端认人。
   平铺在根目录的图（比如随手放的合影）不认人，避免把根目录名当成真实标签；
   只有一层时（直接拖某一个人的文件夹）保留该文件夹名作为人员。 */
function normalizeRels(items) {
  const nested = items.some((x) => x.rel.split('/').filter(Boolean).length >= 3);
  if (!nested) return items;
  return items.map((x) => ({ file: x.file, rel: x.rel.split('/').filter(Boolean).slice(1).join('/') }));
}
function fromInput(input) {
  return normalizeRels([...input.files]
    .filter((f) => /\.(jpe?g|png|webp|bmp)$/i.test(f.name))
    .map((f) => ({ file: f, rel: f.webkitRelativePath || f.name, src: 'pick' })));
}
async function walk(entry, prefix, out) {
  if (entry.isFile) {
    const f = await new Promise((res, rej) => entry.file(res, rej));
    if (/\.(jpe?g|png|webp|bmp)$/i.test(f.name)) out.push({ file: f, rel: prefix + entry.name });
    return out;
  }
  const reader = entry.createReader();
  let batch;
  do {
    batch = await new Promise((res, rej) => reader.readEntries(res, rej));
    for (const e of batch) await walk(e, prefix + entry.name + '/', out);
  } while (batch.length);
  return out;
}
function bindDrop(node, target, onDone) {
  let depth = 0;
  node.addEventListener('dragenter', (e) => { e.preventDefault(); if (++depth) node.classList.add('hot'); });
  node.addEventListener('dragover', (e) => e.preventDefault());
  node.addEventListener('dragleave', () => { if (--depth <= 0) { depth = 0; node.classList.remove('hot'); } });
  node.addEventListener('drop', async (e) => {
    e.preventDefault(); depth = 0; node.classList.remove('hot');
    const items = [...(e.dataTransfer.items || [])].map((it) => it.webkitGetAsEntry?.()).filter(Boolean);
    let out = [];
    if (items.length) for (const it of items) out = await walk(it, '', out);
    else out = [...e.dataTransfer.files].filter((f) => /\.(jpe?g|png|webp|bmp)$/i.test(f.name)).map((f) => ({ file: f, rel: f.name }));
    if (!out.length) return toast('没找到图片文件');
    out.forEach((o) => { o.src = 'drop'; });
    S[target] = normalizeRels(out); onDone();
  });
}

/* ---------- 环境 ---------- */
/* 顶栏 pill 整块取自 S.env，单独拆出来是给 20 秒轮询用的：有批次在跑时
   不宜顺带重画底库列表（用户可能正停在某一页上），但 pill 必须跟着变新。 */
function renderEnvPills() {
  const e = S.env;
  if (!e || !Array.isArray(e.models)) return;   // 轮询拿到半截结果时别把顶栏刷没了
  const m = e.machine || {};
  const ready = e.models.filter((x) => x.exists).length, tot = e.models.length;
  /* 文件在不等于文件没问题：sha256 对不上（下到一半断过网、文件被人换过）时
     推理要么起不来要么算出错乱的分数，pill 不能还是一片绿「已下载」。 */
  const broken = e.models.filter((x) => x.exists && x.sha256_ok === false).length;
  const busy = e.models.filter((x) => x.fetching);
  const failed = e.models.filter((x) => x.fetch_state === 'failed');
  /* 下载中就把 pill 换成琥珀色带进度：180 MB 在板子的网速下要几分钟，
     只显示「0/4 已下载」看着就是卡死了，人会去反复点注册、反复重启服务。 */
  const modelVal = `${ready}/${tot} 已下载`
    + (broken ? ` · ${broken} 个校验失败` : '')
    + (busy.length ? ` · ${busy.map((x) => `下载中 ${x.key} ${fPct(x)}`).join(' ')}` : '')
    + (failed.length && !busy.length ? ` · ${failed.length} 个下载失败` : '');
  const modelCls = busy.length ? 'warn' : (ready === tot && !broken && !failed.length ? 'ok' : 'bad');
  const pills = [
    ['设备', `${m.system || ''} ${m.machine || ''}`, 'ok'],
    ['CPU 核', m.cpu_count, 'ok'],
    ['推理', e.backend?.key === 'onnx' ? '本机 ONNX' : (e.backend?.error ? '未就绪' : '远程'),
      e.backend?.error ? 'bad' : 'ok'],
    ['onnxruntime', e.backend?.onnxruntime || '—', 'ok'],
    ['模型', modelVal, modelCls],
    ['底库', e.gallery ? `${e.gallery.persons} 人 / ${e.gallery.templates} 张` : '空',
      (e.gallery?.templates || 0) > 0 ? 'ok' : 'bad'],
  ];
  $('#env').innerHTML = pills.map(([l, v, cls]) =>
    `<span class="pill ${cls}">${l} <b>${esc(v)}</b></span>`).join('');
}
/* 下载进度：服务端算不出总尺寸（远端不报 content-length）时退到「已下多少 MB」，
   总比空着强，也看得出来网速有没有死。 */
const fPct = (x) => (x.fetch_pct === null || x.fetch_pct === undefined
  ? `${Math.floor((x.fetch_bytes || 0) / 1048576)}MB` : `${x.fetch_pct}%`);

/* 模型那一行下面的说明：当前识别器口径 + 下载进度 / 失败原因。
   单拆一个函数是因为 2 秒一轮的下载轮询也要刷它，但不能再顺手重画下拉框
   （用户可能正开着下拉框挑模型）。 */
function renderModelNote(e) {
  e = e || S.env;
  if (!e || !Array.isArray(e.models)) return;
  const cur = e.models.find((x) => x.key === e.state.rec);
  const bits = [];
  if (cur) {
    bits.push(`当前识别器：${cur.note}${cur.mrall_far1e6 ? ` · 公开榜 FAR=1e-6 通过率 ${cur.mrall_far1e6}%` : ''}`
      + (cur.threshold ? ` · 建议阈值 ${cur.threshold}` : '')
      + (cur.exists && cur.sha256_ok === false ? ' · 模型文件校验失败，正在重新下载' : ''));
  }
  const busy = e.models.filter((x) => x.fetching);
  if (busy.length) {
    bits.push('模型自动下载中：' + busy.map((x) => `${x.key} ${fPct(x)}`
      + (x.fetch_total ? `/${Math.floor(x.fetch_total / 1048576)}MB · ${x.fetch_mbps}MB/s` : '')).join('，'));
  }
  const failed = e.models.filter((x) => x.fetch_state === 'failed');
  if (failed.length && !busy.length) {
    bits.push('下载失败：' + failed.map((x) => `${x.key}（${x.fetch_error || '原因见服务端日志'}）`
      + (x.fetch_retry_in ? `，${x.fetch_retry_in} 秒后重试` : '')).join('；')
      + (e.auto_fetch === false ? '；自动下载已被 FACE_AUTO_FETCH=0 关掉，手动跑 bash models/fetch.sh'
                                : '；稍后自动重试，也可以手动 bash models/fetch.sh'));
  }
  $('#model-note').textContent = bits.join('　');
}
async function loadEnv() {
  S.env = await getJSON('/api/env');
  const e = S.env;
  renderEnvPills();
  /* 扫描面板的默认档、抽样张数上下限、格数与并发上限全跟着这份 env 重算：
     第一次画面板时 env 还没回来，不补这一步，按钮会停在按兜底上限判出的可用性上，
     默认勾选也还是前端抄的那一份。 */
  if (typeof applySweepEnv === 'function') applySweepEnv();

  const rec = $('#rec-model'), det = $('#det-model');
  rec.innerHTML = e.models.filter((x) => x.kind === 'recognition')
    .map((x) => `<option value="${x.key}" ${x.key === e.state.rec ? 'selected' : ''}>${x.key}${mdlTag(x)}</option>`).join('');
  det.innerHTML = e.models.filter((x) => x.kind === 'detection')
    .map((x) => `<option value="${x.key}" ${x.key === e.state.det ? 'selected' : ''}>${x.key}${mdlTag(x)}</option>`).join('');
  renderModelNote(e);
  $('#threads').value = e.state.threads || '';
  $('#remote-url').value = e.state.remote_url || '';
  $('#bk-kind').value = e.state.backend_kind || 'onnx';
  await refreshGallery();   // 底库列表按页取，几千人不卡界面
}
/* 下拉框里每个模型后面的状态尾巴。正在下载的要写进度：几千张底库拖进来才
   告一句「模型没下载」太晚了，选模型那一步就该看见它还在下。 */
const mdlTag = (x) => (x.fetching ? `（下载中 ${fPct(x)}）`
  : !x.exists ? (x.fetch_state === 'failed' ? '（下载失败）' : '（未下载）')
  : x.sha256_ok === false ? '（校验失败）' : '');

/* 换模型后旧批次不能再当结论看：批次的 backend 和底库向量都是按当时的识别模型算的，
   结果卡继续挂着会让人觉得刚切的模型跑出了那个精度。两边都折算成「识别模型标识」来比。 */
const modelKeyOf = (bk) => (!bk ? '' : bk.recognition || `${bk.key}|${bk.base_url || ''}`);
async function applyBackend(body, note) {
  /* /api/backend 失败现在一律走 HTTPException，postJSON 直接抛错到调用方的 catch；
     以前那句 if (r.error) 是早先返回 {"error":…} 时代的遗留，永远不成立。
     注意 /api/env 里的 backend.error 还活着（那儿是「后端起不来也要把环境吐出来」），别跟着删。 */
  const r = await postJSON('/api/backend', body);
  S.pendingApply = null;
  toast(note || '后端已切换'); await loadEnv();
  /* 下拉里可能已经躺着别的模型跑出来的批次，切完必须重新取一遍列表 */
  try { await loadRuns(); } catch (x) { /* 列表没刷出来不影响切换本身 */ }
  const now = modelKeyOf(r.backend), was = modelKeyOf(S.run && S.run.backend);
  if (now && was && now !== was) {
    S.run = null;
    $('#card-result').hidden = true;
    clearResult();
    toast('换模型需重新注册底库后重跑：旧模型的批次结果已收起，想回看从「历史批次」里打开', 10000);
  }
  return r;
}
$('#apply-backend').onclick = async () => {
  const body = { rec: $('#rec-model').value, det: $('#det-model').value,
    threads: $('#threads').value ? +$('#threads').value : null,
    backend_kind: $('#bk-kind').value, remote_url: $('#remote-url').value.trim() };
  try {
    await applyBackend(body);
  } catch (e) {
    if (e.status === 409) {
      /* 409 = 模型还在后台下，不是切坏了。把这次请求存下来，
         下载完由轮询那一头自动补发，省掉「等几分钟再回来点一次应用」。 */
      S.pendingApply = body;
      toast('模型还在下载，下载完成后自动切换：' + e.message, 9000);
    } else {
      S.pendingApply = null;
      toast('切换失败：' + e.message, 7000);
    }
  }
};

/* ---------- 底库 ---------- */
const stemOf = (n) => String(n).replace(/\.[^.]+$/, '');
/* 人数预估必须和后端 infer_person 同规则，否则界面上的「识别出 N 人」会骗人，
   注册完才发现几千张图被归成了一个人。 */
function personIds(items, mode, delim) {
  const parent = (r) => {
    const p = String(r).split('/').filter(Boolean);
    return p.length >= 2 ? p[p.length - 2] : '';
  };
  if (mode === 'name') return items.map((x) => stemOf(x.file.name).trim());
  if (mode === 'path') return items.map((x) => parent(x.rel));
  /* 这条必须跟后端 infer_person 的 r"^(.*?)[_\-\.\s]*(\d{1,})$" 一模一样：
     分隔符可省、位数不限。以前这儿还是老规则（要求有分隔符、最多 6 位），
     后端修过前端没跟上，于是 张三20261106.jpg 前端算一个人、后端并成另一个，
     待注册区写着「识别出 120 人」，注册完 #g-count 只剩 80 —— 预估骗人比不预估更糟。
     纯数字的文件名剥完是空串，这里如实留空（后端同样返回 None），
     调用方 filter(Boolean) 会把它挡掉，不会凭空多出一个人。 */
  if (mode === 'id_num') return items.map((x) => stemOf(x.file.name).replace(/[_\-. ]*\d+$/, '').trim());
  if (mode === 'prefix') return items.map((x) => (stemOf(x.file.name).split(delim || '__')[0] || '').trim());
  return [];
}
function renderGalleryPend() {
  const n = S.gallerySel.length;
  const people = new Set(personIds(S.gallerySel, val('#g-mode', 'name'), val('#g-delim', '__')).filter(Boolean));
  const how = people.size ? `识别出 ${people.size} 人` : '没识别出人员 ID，检查上面的规则';
  $('#g-pend').textContent = n ? `已选 ${n} 张（${how}）— 点“开始注册”` : '';
  $('#g-pend').onclick = null;
  if (n && !$('#g-start')) {
    const b = document.createElement('button');
    b.id = 'g-start'; b.className = 'primary'; b.textContent = '开始注册';
    /* 这个按钮是动态造的、以前从不禁用：上传自检（preflight）最多要等 8 秒，
       这期间再点一次就开出第二个注册会话，两个 pollJob 抢同一条进度条。
       进来先按住，跑完（含各种提前 return）再松开。 */
    b.onclick = async () => {
      b.disabled = true;
      try { await startEnroll(); } finally { if (b.isConnected) b.disabled = false; }
    };
    $('#g-pend').before(b);
  } else if (!n && $('#g-start')) $('#g-start').remove();
}
/* 底库动辄几千张，一次把整批字节压进一个请求并不划算：板子上只有几 GB 内存，
   一次网络抖动还得整批重来。这里改成「一张一个请求」平行推 4 路，
   服务端排一条队列串行做推理，有界队列顺带当节流阀：
   浏览器发得快就等一等，不会在内存里堆几百 MB，进度按总量汇总。 */
const PUSH_PARALLEL = 4;

/* 一张一个请求，请求体就是这张图的原始字节，没有 multipart 表单要解析。
   multipart 的 body 一旦被发出去过就不可重放，重试时经常发出 0 字节的空包，
   服务端只能回 422「没收到 files」，几千张底库换来满屏同一个错，
   最后收成「成功 0 个模板」。Blob 随时能重发，重试才真的有用。
   但 Safari 拿到读不动的照片会把请求一直挂着——请求头发出去了、正文一个字节都不来，
   没有截止钟的话页面就永远停在「评测中」，所以每次上传都自带超时。 */
const UPLOAD_TIMEOUT = 20000;
async function uploadRaw(url, bytes, type, tries = 4, timeoutMs = UPLOAD_TIMEOUT) {
  let last = '';
  for (let i = 0; i < tries; i++) {
    if (i) await new Promise((r) => setTimeout(r, 600 * i));
    const ctl = typeof AbortController === 'function' ? new AbortController() : null;
    const timer = ctl ? setTimeout(() => ctl.abort(), timeoutMs) : 0;
    try {
      const blob = bytes instanceof Blob ? bytes : new Blob([bytes], { type: type || 'application/octet-stream' });
      if (!blob.size) throw new Error('图片内容是空的（0 字节）');
      const send = fetch(url, { method: 'POST', body: blob, signal: ctl ? ctl.signal : undefined });
      /* 只靠 AbortSignal 不够：浏览器抽风时连 abort 都不理，请求就一直挂着。
         自己再摆一只计时器同时等，谁先到算谁，保证这一步一定有结果。 */
      let wtimer = 0;
      const alarm = new Promise((_, rej) => {
        wtimer = setTimeout(() => rej(Object.assign(
          new Error(`__abort__`), { name: 'AbortError' })), timeoutMs);
      });
      let r;
      try { r = await Promise.race([send, alarm]); } finally { clearTimeout(wtimer); }
      if (!r.ok) { last = await whyFailed(r); continue; }
      return await r.json();
    } catch (e) {
      if (e && e.name === 'AbortError') {
        throw new Error(`上传卡住了（${Math.round(timeoutMs / 1000)} 秒没响应）`
          + '：浏览器发不出这张图的正文，通常是它读不到磁盘上的原始照片');
      }
      last = (e && e.message) ? e.message : String(e);
    } finally {
      if (timer) clearTimeout(timer);
    }
  }
  throw new Error(last || '上传失败');
}

/* 读一张图依次试三条路：常规读 → FileReader → 把文件转成对象地址再读回来。
   三条都读不到才退化成「把文件本身交给 fetch 直发」（上面带超时，不会卡死）。
   Safari 从相册、iCloud、网盘拿到的照片常常只有名字和大小、正文读不出来，
   硬传只会让请求一直挂着，不如早点认输并把下一步说清楚。 */
/* 每条读法都必须配截止钟：Safari 拿到读不动的照片（iCloud 未下载、网盘按需加载、
   权限没给）时 arrayBuffer() 既不返回也不报错，四条上传流水线会一起堵死在读取上，
   按钮永远停在「评测中」——和真的卡死看起来一模一样。 */
/* 四条读法各自的截止钟。第一条是 99% 情况下真正干活的那条，要给足时间：
   四路并发挤在一起时它可能十几秒才读完，给 5 秒会把「慢」误判成「读不到」，
   然后一路退到后面三条本来就未必读得动的路上，最后冤枉照片坏了。 */
const READ_MS = [20000, 8000, 8000, 12000];
const READ_TIMEOUT = READ_MS[0];
function withTimeout(p, label, timeoutMs) {
  let timer = 0;
  const bomb = new Promise((_, rej) => {
    timer = setTimeout(() => rej(new Error(
      `${label}超时（${Math.round(timeoutMs / 1000)} 秒没读到内容，多半是浏览器读不到磁盘上的原图）`)), timeoutMs);
  });
  return Promise.race([Promise.resolve().then(() => p), bomb])
    .finally(() => clearTimeout(timer));
}

const READ_WAYS = [
  [(f) => (typeof f.arrayBuffer === 'function'
    ? f.arrayBuffer() : Promise.reject(new Error('这个浏览器不支持直接读文件内容'))), '常规读'],
  [(f) => new Promise((res, rej) => {
    const rd = new FileReader();
    rd.onload = () => res(rd.result);
    rd.onerror = () => rej(new Error('读失败（' + ((rd.error && rd.error.name) || '未知原因') + '）'));
    try { rd.readAsArrayBuffer(f); } catch (e) { rej(new Error(String((e && e.message) || e))); }
  }), '备用读'],
  [(f) => {
    const u = URL.createObjectURL(f);
    return fetch(u).then((r) => {
      if (!r.ok) throw new Error('读失败 HTTP ' + r.status);
      return r.arrayBuffer();
    }).finally(() => URL.revokeObjectURL(u));
  }, '对象地址读'],
  /* 第四条路：让图片解码器去读（走的是和解码器同一条内部通道，
     和 blob 读盘不是同一条路），读到了再重编码成 JPEG 字节。
     Safari 偶尔会出现「blob 读不动但解码器能解码」的情况，多一次机会。 */
  [(f) => {
    if (typeof createImageBitmap !== 'function') return Promise.reject(new Error('这个浏览器不支持解码读'));
    return createImageBitmap(f).then((bm) => {
      const cv = document.createElement('canvas');
      cv.width = bm.width; cv.height = bm.height;
      cv.getContext('2d').drawImage(bm, 0, 0);
      bm.close && bm.close();
      return new Promise((res, rej) => cv.toBlob((bl) => {
        if (!bl) return rej(new Error('重编码失败'));
        bl.arrayBuffer().then(res, rej);
      }, 'image/jpeg', 0.92));
    });
  }, '解码读'],
];
async function readOne(it) {
  const nm = (it.rel || (it.file && it.file.name) || '').split('/').pop() || '未命名';
  const f = it.file;
  if (!f || typeof f.size !== 'number') throw new Error('不是图片文件');
  if (!f.size) throw new Error('文件是 0 字节');
  const out = { name: nm, rel: it.rel ?? nm, type: f.type || 'image/jpeg' };
  const whys = [];
  for (const [wi, way] of READ_WAYS.entries()) {
    try {
      const buf = await withTimeout(way[0](f), way[1], READ_MS[wi] || READ_TIMEOUT);
      if (buf && buf.byteLength) return Object.assign(out, { bytes: buf });
      whys.push(way[1] + '读出 0 字节');
    } catch (e) { whys.push(way[1] + '失败：' + String((e && e.message) || e)); }
  }
  return Object.assign(out, { bytes: f, fallback: whys.join('；') || '三种读法都没读到内容' });
}
/* Safari 的 fetch 失败只会给一句 "Load failed"，Chrome 给的是 net::ERR_xxx，
   用户看不懂，也不知道该干什么。这里统一翻成一句人话。 */
function plainErr(msg) {
  const m = String(msg || '');
  if (/Load failed|NetworkError|Failed to fetch|net::ERR_|ERR_UPLOAD_FILE_CHANGED/i.test(m))
    return `浏览器没能把这个请求发出去（浏览器只回了 ${m}，正文一个字节都没发走）`;
  return m;
}
const fmtMB = (b) => (b / 1048576).toFixed(0) + ' MB';
async function startEnroll() {
  if (!S.gallerySel.length) return;
  /* 检查必须排在清空选择之前：否则页面版本不匹配时直接 return，
     用户刚拖进来的几千张图就白选了。 */
  if (!requireUi(['#g-mode', '#g-delim', '#g-maxside', '#g-maxfaces', '#g-maxtpl'])) return;
  const items = S.gallerySel.slice();
  const totalBytes = items.reduce((a, x) => a + ((x.file && x.file.size) || 0), 0);
  if (!totalBytes) {
    /* 大小全是 0：浏览器只拿到了占位符（内置预览窗口、iCloud 未下载、网盘按需加载）。
       这时候发出去的一定是空请求体，先把话说清楚再停。 */
    toast(`${items.length} 张图片大小都是 0，浏览器没真正读到文件。请改用 Safari 打开 `
      + 'http://127.0.0.1:8617 重新拖入，或先在访达里把这些照片下载到本地；'
      + '底库就在本机时，用「导入本机目录」完全不经过浏览器上传', 12000);
    return;
  }
  try { await preflight(); } catch (e) { toast(e.message, 12000); return; }
  S.gallerySel = []; S.unreadable = []; renderGalleryPend();
  $('#g-prog').hidden = false; $('#g-bar').style.width = '1%';
  $('#g-progtxt').textContent = `共 ${items.length} 张 ${fmtMB(totalBytes)}，开始逐张上传…`;
  const opts = () => ({ person_mode: val('#g-mode', 'name'), delimiter: val('#g-delim', '__'),
    max_side: +val('#g-maxside', '1280') || 1280, max_faces: +val('#g-maxfaces', '1'),
    max_templates: +val('#g-maxtpl', '0') || 0 });
  const acc = { done: 0, ok: 0, bad: [], badTotal: 0, gallery: { persons: 0, templates: 0 } };
  const left = [];
  let id = '';
  try {
    id = (await postJSON('/api/gallery/open', { total: items.length, ...opts() })).job_id;
  } catch (e) {
    /* 会话没开成就一张都别发：图片原样退回待注册，
       用户按提示改好再点一次就行，别把刚拖进来的几千张丢掉。 */
    $('#g-progtxt').textContent = '注册没开始：' + e.message;
    toast('注册没开始：' + e.message, 10000);
    S.gallerySel = items; renderGalleryPend();
    return;
  }
  const enc = encodeURIComponent;
  const pushed = new Set(), dropped = new Set();
  let next = 0, lastErr = '', sameErr = 0, stop = false;
  async function pump() {
    while (!stop) {
      const i = next++;
      if (i >= items.length) return;
      const it = items[i];
      const nm = (it.rel || (it.file && it.file.name) || '').split('/').pop() || '未命名';
      let one;
      try {
        one = await readOne(it);
      } catch (e) {
        /* 本地读不出来的图片退回待注册也没意义（iCloud 没下载、文件已被移走），
           单独记进「需复查」，结束时一次报清楚 */
        S.unreadable.push({ file: nm, reason: '本地读不出来（iCloud 未下载或已被移动）' });
        continue;
      }
      try {
        const r = await uploadRaw('/api/gallery/push?job=' + id + '&rel=' + enc(one.rel || nm)
          + '&name=' + enc(one.name), one.bytes, one.type, one.fallback ? 2 : 4, one.fallback ? 10000 : UPLOAD_TIMEOUT);
        pushed.add(i);
        $('#g-bar').style.width = Math.max(1, Math.round(((r.done || 0) / items.length) * 100)) + '%';
        $('#g-progtxt').textContent = `已送出 ${r.recved}/${items.length} 张 · `
          + `服务端已注册 ${r.done}/${items.length} 张 · ${r.ok} 个模板`;
      } catch (e) {
        const msg = String((e && e.message) || e);
        dropped.add(i);
        left.push(it);
        acc.bad.push({ file: nm, reason: msg });
        /* 连着三张一模一样的错 = 环境问题（这个窗口发不出内容、页面版本不对），
           不是网络抖动，剩下的别再多发几千个请求去刷同一句报错 */
        sameErr = msg === lastErr ? sameErr + 1 : 1;
        lastErr = msg;
        if (sameErr >= 3 && !pushed.size) stop = true;
      }
    }
  }
  await Promise.all(Array.from({ length: PUSH_PARALLEL }, () => pump()));
  /* 提前收工或中断路过：没来得及发、也没有明确失败的图片全部退回，不能默默丢掉 */
  for (let i = 0; i < items.length; i++) {
    if (!pushed.has(i) && !dropped.has(i)) { dropped.add(i); left.push(items[i]); }
  }
  try { await postJSON('/api/gallery/close?job=' + id, {}); }
  catch (e) { /* 服务端可能已经自己收尾了（超时/出错），按任务号等结果就行 */ }
  try {
    await pollJob(id, acc, items.length);
  } catch (e) {
    clientError('gallery_close', e);
    acc.bad.push({ file: '注册收尾', reason: String((e && e.message) || e) });
  }
  $('#g-bar').style.width = '100%';
  $('#g-progtxt').textContent = `结束：成功 ${acc.ok} 个模板 / 送出 ${pushed.size} 张`
    + (left.length ? ` · ${left.length} 张没发出` : '')
    + (S.unreadable.length ? ` · ${S.unreadable.length} 张本地读不出来` : '');
  await loadEnv();
  if (left.length) {
    toast(`${left.length} 张没注册上，已放回待注册列表，点「开始注册」续传`, 7000);
    S.gallerySel = left; renderGalleryPend();
  } else {
    toast(`底库完成：${acc.gallery.persons} 人 / ${acc.gallery.templates} 模板`);
  }
  /* 读不出来的图片和发送失败的图片要一起列出来：只靠一闪而过的 toast，
     几千张里少了几十个人根本发现不了。 */
  const issues = acc.bad.concat(S.unreadable);
  if (issues.length) {
    $('#g-pend').innerHTML = `需复查 ${issues.length} 条：<br>` + issues.slice(0, 12).map((b) =>
      `<code>${esc(b.file)}</code> ${esc(b.reason)}`).join('<br>');
  }
  /* 跑完就把进度条收掉：一根永远停在 100% 的横杆挂在这儿，下一次动手时
     看着像还在跑，而结论上面的 toast 和「需复查」列表里都已经给全了。 */
  $('#g-prog').hidden = true; $('#g-bar').style.width = '0%';
}
/* 几千张底库走浏览器上传，字节要在内存里过两遍，还容易撞上「当前窗口读不到磁盘」
   这类怪问题。底库和评测机通常在同一台机器上（部署到 RK3588 时更是如此），
   让服务端直接读目录最快最稳，浏览器也不占内存。 */
async function importDir() {
  if (!requireUi(['#g-path', '#g-import', '#g-mode', '#g-delim', '#g-maxside', '#g-maxfaces', '#g-maxtpl'])) return;
  const btn = $('#g-import');
  /* disabled 只挡得住鼠标点击：焦点在 #g-path 里按回车照样会走到这里，
     两次 importDir 会抢同一条 #g-prog，第二次的报错还会盖掉第一次的进度。 */
  if (btn.disabled) return;
  const path = val('#g-path', '').trim();
  if (!path) return toast('先把底库目录的完整路径填进来：在访达里选中文件夹按 Cmd+Option+C 就是「拷贝路径」', 9000);
  btn.disabled = true; btn.textContent = '扫描中…';
  $('#g-prog').hidden = false; $('#g-bar').style.width = '2%';
  $('#g-progtxt').textContent = `正在扫描 ${path} …`;
  const acc = { done: 0, ok: 0, bad: [], badTotal: 0, gallery: { persons: 0, templates: 0 } };
  try {
    const r = await postJSON('/api/gallery/enroll_dir', { path,
      person_mode: val('#g-mode', 'name'), delimiter: val('#g-delim', '__'),
      max_side: +val('#g-maxside', '1280') || 1280, max_faces: +val('#g-maxfaces', '1'),
      max_templates: +val('#g-maxtpl', '0') || 0 });
    $('#g-progtxt').textContent = `${r.path} 里有 ${r.found} 张图片，开始注册…`;
    await pollJob(r.job_id, acc, r.found);
    /* 收尾那句也用 badTotal：服务端把明细截到 200 条，坏了 312 张时进度条上写的是
       「异常 312」，跑完却变成「需复查 200 条」，前后自己打自己。 */
    $('#g-progtxt').textContent = `导入完成：${r.found} 张 → 成功 ${acc.ok} 个模板`
      + (acc.badTotal ? ` · 需复查 ${acc.badTotal} 条` : '');
    toast(`底库导入完成：${acc.gallery.persons} 人 / ${acc.gallery.templates} 模板`, 6000);
    if (acc.bad.length) $('#g-pend').innerHTML =
      `需复查 ${acc.badTotal || acc.bad.length} 条（服务端最多回 ${acc.bad.length} 条明细，这里列前 ${Math.min(12, acc.bad.length)} 条）：<br>`
      + acc.bad.slice(0, 12).map((b) => `<code>${esc(b.file)}</code> ${esc(b.reason)}`).join('<br>');
  } catch (e) {
    $('#g-progtxt').textContent = '导入失败：' + e.message;
    toast('导入失败：' + e.message, 10000);
  }
  btn.disabled = false; btn.textContent = '导入本机目录';
  /* 收尾把进度条收回：停在 100%（或 2%）的横杆一直挂着，
     下一次操作开始前看着像还在跑，而结论 toast 里已经说清了。 */
  $('#g-prog').hidden = true; $('#g-bar').style.width = '0%';
  await loadEnv();
}

function pollJob(id, acc, total) {
  return new Promise((resolve, reject) => {
    let miss = 0;
    const t = setInterval(async () => {
      let j;
      try { j = await getJSON('/api/jobs/' + id); miss = 0; }
      catch (e) {
        /* 服务端在闷头跑推理时偶尔回不过来一次，立刻 reject 会把整批上传判死 */
        if (++miss < 8) return;
        clearInterval(t);
        return reject(new Error(`连续 ${miss} 次查不到任务进度：${e.message}`));
      }
      /* 异常条数只能读 bad_total：服务端把 j.bad 截断到 200 条，坏了三百张时
         这里少报一百条，和「评测本机目录」那条路报的数字对不上。 */
      const done = acc.done + j.done, ok = acc.ok + (j.ok || 0);
      const bad = acc.bad.length + (j.bad_total ?? j.bad?.length ?? 0);
      acc.badTotal = bad;            // 收尾那句要用它，acc.bad 有 200 条上限
      $('#g-bar').style.width = Math.round((done / (total || 1)) * 100) + '%';
      $('#g-progtxt').textContent = `${done}/${total} · 成功 ${ok} 模板${bad ? ` · 异常 ${bad}` : ''}`;
      if (j.status === 'done' || j.status === 'error') {
        clearInterval(t);
        if (j.status === 'error') return reject(new Error(j.error));
        acc.done += j.done; acc.ok = ok;
        acc.bad = acc.bad.concat(j.bad || []);
        acc.gallery = j.gallery || acc.gallery;
        resolve(acc);
      }
    }, 600);
  });
}
/* ---------- 底库列表：服务端分页 ----------
   几千个人一次性渲染会把浏览器卡住（每个格子还要拉一张缩略图），
   所以每屏只画当前这一页，翻页和搜索都重新向服务端要一页数据。 */
/* 搜索框 debounce、loadEnv、翻页都会调这里。以前一句 `if (S.gloading) return`
   就把新查询丢了、也不补发，列表能一直停在半路那个旧关键词上。
   现在把最新一份参数记下来，等当前这次跑完补发一次。 */
let gNext = null;
async function refreshGallery() {
  if (S.gloading) { gNext = { page: S.gp, size: S.gs, q: S.gq }; return; }
  S.gloading = true;
  const req = { page: S.gp, size: S.gs, q: S.gq };
  let d = null;
  try {
    const url = (pg) => `/api/gallery/list?page=${pg}&size=${req.size}&q=${encodeURIComponent(req.q)}`;
    d = await getJSON(url(req.page));
    /* 删人、清空之后当前页可能已经没数据了：回到第 1 页重取，别把人留在空白页上 */
    if (!d.items.length && d.total && d.page > 1) { S.gp = 1; req.page = 1; d = await getJSON(url(1)); }
    renderGalleryPage(d);
    await refreshPersonNames();
  } catch (e) {
    toast('底库列表读取失败：' + e.message);
  } finally {
    S.gloading = false;
    const n = gNext; gNext = null;
    if (n && (!d || n.page !== req.page || n.size !== req.size || n.q !== req.q)) {
      S.gp = n.page; S.gs = n.size; S.gq = n.q;
      await refreshGallery();
    }
  }
}
/* 空底库时只收起翻页控件：提示文字 #g-pagestat 就挂在这一行里，
   整条 hidden 掉会把「底库还是空的」一起藏了，空库界面看着像一句话都没说。 */
function setGalleryNav(on) {
  ['#g-search', '#g-pageinfo', '#g-prev', '#g-next'].forEach((s) => {
    const el = $(s); if (el) el.hidden = !on;
  });
  const lab = $('#g-size') && $('#g-size').closest('label');
  if (lab) lab.hidden = !on;
}
function renderGalleryPage(d) {
  if (!requireUi(['#g-pager', '#g-pageinfo', '#g-pagestat', '#g-more', '#g-count', '#g-persons'])) return;
  const st = d.stats || {};
  $('#g-count').textContent = st.templates
    ? `${st.persons} 人 / ${st.templates} 模板` : '未注册';
  // 搜索框就在这条里：只要库里有人就得留着，否则搜不到东西时用户没法清空关键词
  $('#g-pager').hidden = false; setGalleryNav(!!st.persons);
  $('#g-pageinfo').textContent = `${d.total ? d.page : 0} / ${d.total_pages}`;
  $('#g-prev').disabled = d.page <= 1;
  $('#g-next').disabled = d.page >= d.total_pages;
  const from = (d.page - 1) * d.size + 1;
  const head = !d.total
    ? (d.q ? `「${d.q}」没搜到匹配的人` : '底库还是空的，先把人员照片拖进来')
    : `${d.q ? `「${d.q}」命中 ${d.total} 人 · ` : ''}本页第 ${from}–${from + d.items.length - 1} 人`;
  $('#g-pagestat').textContent = head
    + (st.db_bytes ? ` · 库文件 ${fmtBytes(st.db_bytes)}` : '');
  /* 每个格子带一个「只删这个人」的小叉：底库里混进一个错标的人（把路人注册成了张三），
     以前只有整库清空，几千张得全部重注册。删除按人名走，所以名字放 data-* 里让
     事件处理器原样取回，别再去反解已经拼进 HTML 的文本。 */
  $('#g-persons').innerHTML = d.items.map((p) =>
    `<div class="pcell ${p.bad ? 'warn' : ''}">${p.thumb ? `<img loading="lazy" src="/api/thumb/${esc(p.thumb)}">` : ''}<b>${esc(p.person)}</b> · ${p.templates} 张${p.bad ? ` · <span style="color:#d29922">${p.bad} 张待查</span>` : ''}`
    + `<button class="pdel small" data-del="${esc(p.person)}" data-n="${p.templates}"`
    + ` title="只删除「${esc(p.person)}」：他的 ${p.templates} 张底库模板和头像一起删掉，别的人不动">✕</button></div>`).join('');
  const rest = d.total - d.items.length;
  $('#g-more').hidden = rest <= 0;
  $('#g-more').textContent = rest > 0
    ? `底库共 ${d.total} 人，本页只显示 ${d.items.length} 人，翻页或用搜索框定位某个人` : '';
}
/* 探针真值的人名清单：只取名，几千人也才几十 KB，挂在 <datalist> 上由浏览器
   边打字边过滤，比给每条人脸塞一个几千项的 <select> 快得多。 */
async function refreshPersonNames() {
  const d = await getJSON('/api/gallery/persons?names=1');
  S.persons = d.names || [];
  const dl = $('#person-list');
  if (dl) dl.innerHTML = `<option value="${esc(UNK_LABEL)}"></option>`
    + S.persons.map((x) => `<option value="${esc(x)}"></option>`).join('');
}
$('#g-prev').onclick = () => { if (S.gp > 1) { S.gp -= 1; refreshGallery(); } };
$('#g-next').onclick = () => { S.gp += 1; refreshGallery(); };
$('#g-size').onchange = () => { S.gs = +$('#g-size').value || 80; S.gp = 1; refreshGallery(); };
let gSearchTimer = 0;
$('#g-search').oninput = (e) => {
  const q = e.target.value;
  clearTimeout(gSearchTimer);
  gSearchTimer = setTimeout(() => { S.gq = q.trim(); S.gp = 1; refreshGallery(); }, 300);
};
/* 单个人删除。格子是每次渲染重贴的 innerHTML，所以处理器挂在容器上（挂在按钮上
   得每页重新绑，翻页时就丢了）。删完重取本页：人在不在这一页、当前页会不会因此
   变成空页，都交给 refreshGallery 里那套「空页退回第 1 页」的逻辑兜。 */
const gpBox = $('#g-persons');
if (gpBox) gpBox.onclick = async (e) => {
  const b = e.target.closest && e.target.closest('[data-del]');
  if (!b || b.disabled) return;
  const who = b.dataset.del;
  if (!confirm(`只删除「${who}」这一个人？他的 ${b.dataset.n || '全部'} 张底库模板和头像一起删掉，无法撤销。`)) return;
  b.disabled = true;                      // 等回信期间按住，免得连点触发第二次删除
  try {
    const d = await delJSON('/api/gallery/person/' + encodeURIComponent(who));
    toast(d.templates_removed
      ? `已删除「${who}」：${d.templates_removed} 个模板、${d.thumbs_removed || 0} 张头像，`
        + `释放 ${fmtBytes(d.freed_bytes || 0)}`
      : `「${who}」名下已经没有模板了`);
  } catch (err) {
    /* 404 是最常见的一种：这个人已经被另一个标签页删掉了。
       按钮留着没用，直接刷一次列表让他看见真实状态。 */
    b.disabled = false;
    return toast(`删除「${who}」失败：` + err.message, 9000);
  }
  S.persons = [];
  await loadEnv();      // 顶栏的人数、真值下拉里的名字都跟底库绑着，一起刷新
};
/* 清空要连缩略图和数据库一起删，所以把释放的字节数报出来：
   只说「已清空」而磁盘上还躺着几百 MB，用户下一轮就会来问为什么没腾出空间 */
$('#g-clear').onclick = async () => {
  if (!confirm('确认清空整个底库？几千张底库要重新注册，无法撤销'
    + '（缩略图与底库数据库 gallery.db 一并删除）。')) return;
  try {
    /* 走统一的 delJSON：裸 fetch 那句 .json().detail 撞上 422 时 detail 是个数组，
       提示会变成「清空失败：[object Object]」，而 422 恰恰是最该看懂的那类错。 */
    const d = await delJSON('/api/gallery');
    toast(d.freed_bytes
      ? `底库已清空，删掉 ${d.thumbs_removed || 0} 张缩略图，释放 ${fmtBytes(d.freed_bytes)}`
      : '底库已清空');
  } catch (e) {
    return toast('清空失败：' + e.message);
  }
  S.gp = 1; S.gq = ''; S.persons = [];
  if ($('#g-search')) $('#g-search').value = '';
  await loadEnv();
  $('#g-persons').innerHTML = ''; $('#g-count').textContent = '未注册';
  setGalleryNav(false);
  if ($('#g-more')) $('#g-more').hidden = true;
  if ($('#person-list')) $('#person-list').innerHTML = '';
};
$('#g-pick-dir').onclick = () => $('#file-g-dir').click();
$('#g-pick-files').onclick = () => $('#file-g-files').click();
$('#file-g-dir').onchange = (e) => { S.gallerySel = fromInput(e.target); renderGalleryPend(); };
$('#file-g-files').onchange = (e) => { S.gallerySel = fromInput(e.target); renderGalleryPend(); };
bindDrop($('#g-drop'), 'gallerySel', renderGalleryPend);
$('#g-import').onclick = importDir;
$('#g-path').onkeydown = (e) => { if (e.key === 'Enter') importDir(); };
$('#g-mode').onchange = renderGalleryPend;
$('#g-delim').onchange = renderGalleryPend;

/* ---------- 探针 ---------- */

/* 探针图解码长边固定 1280，界面上没有控件，也**故意不去读第 2 节的 #g-maxside**
   （那个控件写着「注册时长边上限」，管的是底库注册，跟探针无关）。
   为什么可以定死：检测输入最大只有 640（#p-detsize），4K 原图先缩到 1280
   再进 640 的检测器，和直接 4K→640 的结果几乎一样，缩到 1280 只是
   把「一张 4K 原图的解码字节要同时在内存里存几份」按住 —— 并发 8 路时
   那才是真正会打爆板子内存的东西。服务端的校验范围是 128~8192。 */
const PROBE_MAX_SIDE = 1280;
function renderProbePend() {
  const n = S.probeSel.length;
  $('#p-pend').textContent = n ? `已选 ${n} 张，点“开始评测”` : '';
}
$('#p-pick-dir').onclick = () => $('#file-p-dir').click();
$('#p-pick-files').onclick = () => $('#file-p-files').click();
$('#file-p-dir').onchange = (e) => { S.probeSel = fromInput(e.target); renderProbePend(); };
$('#file-p-files').onchange = (e) => { S.probeSel = fromInput(e.target); renderProbePend(); };
bindDrop($('#p-drop'), 'probeSel', renderProbePend);

/* 评测参数的控件清单和读取方式只留一份。原先「开始评测」和「评测本机目录」
   各自抄了一份 params 字面量，多一个参数就要改两处，漏一处就是两条路口径
   不一致（吞吐按并发算、延迟按串行报就是这么来的）。 */
const P_UI = ['#p-idmode', '#p-detsize', '#p-topk', '#p-thr', '#p-mode',
  '#p-allfaces', '#p-minface', '#p-workers'];
function probeParams() {
  return {
    det_size: +val('#p-detsize', '320'), topk: +val('#p-topk', '5'),
    threshold: +val('#p-thr', '0.4078'), probe_mode: val('#p-mode', 'auto'),
    all_faces: +val('#p-allfaces', '1'), min_face: +val('#p-minface', '0') || 0,
    /* 并发路数读不到就退回 1（串行）：控件缺失时 +val() 会给出 NaN，
       JSON.stringify 把 NaN 写成 null，报错的是服务端而不是这里，更难查。 */
    workers: +val('#p-workers', '1') || 1,
  };
}

/* 探针和底库走同一套「一张一个请求，请求体就是图片原始字节」。
   整批 multipart 只要有一张读不出来就整批断流，几十张探针换来一句
   「没收到图片文件」；拆开传之后失败只影响那一张，进度也看得见。 */
$('#p-run').onclick = async () => {
  if (!S.probeSel.length) return toast('先选择探针图片');
  if (!requireUi(P_UI)) return;
  const n = S.probeSel.length, pmax = (S.env && S.env.probe_max) || 500;
  if (n > pmax) return toast(`一次最多评测 ${pmax} 张探针图，当前选了 ${n} 张。`
    + '要么分批评，要么用下面「评测本机目录」，不经过浏览器上传', 12000);
  const bytes = S.probeSel.reduce((a, x) => a + ((x.file && x.file.size) || 0), 0);
  if (!bytes) return toast(`${n} 张图片大小都是 0，浏览器没真正读到文件内容。`
    + '常见于照片还没从 iCloud/网盘下载到本地，或是从预览窗口拖出来的占位符；'
    + '请先在访达里确认照片能直接打开，再用 Safari 重新拖入。'
    + '照片就在这台机器上时，用下面「评测本机目录」完全不经过浏览器上传', 15000);
  const btn = $('#p-run'); btn.disabled = true; btn.textContent = '评测中…';
  const opts = { total: S.probeSel.length, person_mode: val('#p-idmode', 'name'),
    delimiter: val('#g-delim', '__'), max_side: PROBE_MAX_SIDE, params: probeParams() };
  const items = S.probeSel.slice();
  S.unreadable = [];
  $('#p-pend').textContent = '';
  const back = () => { btn.disabled = false; btn.textContent = '开始评测'; };
  /* lost / fb 必须声明在 try 外面：catch 是 try 的兄弟块，看不见 try 里的 const，
     之前失败的提示就是死在这里——用户只看到「评测失败」四个字。 */
  const lost = [], fb = [];
  try { await preflight(); } catch (e) { toast(e.message, 12000); back(); return; }
  try {
    const op = await postJSON('/api/probe/open', opts);
    const enc = encodeURIComponent;
    const nameOf = (i) => (items[i].rel || (items[i].file && items[i].file.name) || '')
      .split('/').pop() || '未命名';
    let next = 0, sent = 0, lastErr = '', sameErr = 0, stop = false;
    const lostIdx = [];   // 这一轮没送出去的张，稍后单发补送
    const bad = {};       // 下标 → 最后一次的错因，最后统一向用户交代

    /* 送一张：先读再发。这里只报结果，不判「这张是不是坏了」——
       四条读法同时失败很可能只是四路并发把读盘通道挤爆了，隔一会儿
       单发一次常常就好了，所以判死刑必须等到补送也失败之后。 */
    async function trySend(i) {
      const nm = nameOf(i);
      let one;
      try {
        one = await readOne(items[i]);
      } catch (e) {
        return { nm, why: String((e && e.message) || e), fbWhy: '' };
      }
      const fbWhy = one.fallback || '';
      try {
        await uploadRaw('/api/probe/push?id=' + op.upload_id + '&rel=' + enc(one.rel || nm)
          + '&name=' + enc(one.name), one.bytes, one.type,
          fbWhy ? 2 : 4, fbWhy ? 10000 : UPLOAD_TIMEOUT);
        if (fbWhy) fb.push({ file: nm, reason: fbWhy });
        sent++;
        btn.textContent = `上传 ${sent}/${items.length}…`;
        return { nm, why: '', fbWhy };
      } catch (e) {
        return { nm, why: String((e && e.message) || e), fbWhy };
      }
    }
    async function pump() {
      while (!stop) {
        const i = next++;
        if (i >= items.length) return;
        const r = await trySend(i);
        if (!r.why) continue;
        bad[i] = r;
        if (lostIdx.indexOf(i) < 0) lostIdx.push(i);
        sameErr = r.why === lastErr ? sameErr + 1 : 1;
        lastErr = r.why;
        if (sameErr >= 3 && !sent) stop = true;
      }
    }
    btn.textContent = `上传 0/${items.length}…`;
    /* 上传阶段同样要动静态：某几张卡在读盘上时，按钮数字不动看着就是死机，
       每秒把「已传几张 / 等了多久」刷出来，用户至少知道卡在哪。 */
    const u0 = Date.now();
    const utick = setInterval(() => {
      btn.textContent = `上传 ${sent}/${items.length} · ${Math.round((Date.now() - u0) / 1000)} 秒…`;
    }, 1000);
    /* 总闸：整段上传再套一层截止钟。单张截止钟靠 fetch 认 AbortSignal，
       浏览器抽风时（连 abort 都不理）那张上的闸也白搭，按钮就永远停在
       「上传 0/N · N 秒…」。总闸到点先停下所有流水线：一张都没传上去就明确报错，
       传上去了一部分就先把这部分评完，别让几十分钟的活儿整个作废。 */
    let stalled = false;
    const stallMs = Math.min(90000 + items.length * 1200, 1800000);
    let stallTimer = 0;
    try {
      await Promise.race([
        Promise.all(Array.from({ length: Math.min(4, items.length) }, () => pump())),
        new Promise((_, rej) => {
          stallTimer = setTimeout(() => rej(Object.assign(new Error('stall'), { stall: true })), stallMs);
        }),
      ]);
    } catch (e) {
      if (!e || !e.stall) throw e;
      stalled = true; stop = true;
    } finally {
      clearTimeout(stallTimer);
      clearInterval(utick);
    }
    /* 补送：首轮没送出去的，等并发停下来后一张一张重发两轮。三次实测每次卡住的
       都不是同一张（080531 / 081206 / 081713），文件在访达里都打得开——
       并发读才是嫌疑人，单发几乎都能救回来。服务端按文件名去重，重发不会多算一张。 */
    const wait = (ms) => new Promise((r) => setTimeout(r, ms));
    const retryBy = Date.now() + 90000;
    for (let round = 1; round <= 2 && lostIdx.length && Date.now() < retryBy; round++) {
      const again = [];
      for (const i of lostIdx) {
        if (Date.now() > retryBy) { again.push(i); continue; }
        btn.textContent = `补送第 ${round} 轮 ${again.length + 1}/${lostIdx.length}…`;
        await wait(900 * round);
        const r = await trySend(i);
        if (r.why) { again.push(i); bad[i] = r; }
        else {
          delete bad[i];
          clientError('probe_retry_ok', { message: '补送成功' }, { file: r.nm, round });
        }
      }
      lostIdx.length = 0;
      lostIdx.push(...again);
    }
    lost.length = 0;
    for (const i of lostIdx) {
      lost.push(nameOf(i));
      const r = bad[i];
      if (!r) continue;
      S.unreadable.push({ file: nameOf(i),
        reason: r.fbWhy ? `读盘失败：${r.fbWhy}；直接发出也失败：${r.why}` : '本地读不出来：' + r.why });
    }
    if (stalled && sent) {
      toast(`上传卡住了：${sent}/${items.length} 张已传好，剩下 ${items.length - sent} 张这次先跳过，`
        + '直接评测已传上来的这些（下次建议用「评测本机目录」）', 12000);
    }
    if (!sent) {
      const whys = [...new Set([lastErr].concat(fb.map((x) => x.reason))
        .concat(S.unreadable.map((x) => x.reason)).filter(Boolean))];
      throw new Error(`${items.length} 张探针图一张都没传上去`
        + (stalled ? `（等了 ${Math.round((Date.now() - u0) / 1000)} 秒还是零）` : '')
        + (lost.length ? `（卡住的有：${lost.slice(0, 3).join('、')}${lost.length > 3 ? ' 等' : ''}）` : '')
        + (whys.length
            ? `，原因：${whys.slice(0, 3).map(plainErr).join(' / ')}`
            : '（页面没给出具体原因）')
        + '。四条读法都试过、又单独补送过两轮还是这样，'
        + '所以不是模型的问题，是浏览器读不到这些原始照片。'
        + (items.every((x) => x.src === 'drop')
            ? '这批是拖进来的：Safari 对拖进来的文件读盘权限特别容易失效，'
              + '请改用旁边的「选择文件 / 选择文件夹」按钮重新选一次（按钮选的文件权限一直在）。'
            : '请在访达里先确认这些照片能直接打开（iCloud/网盘没下载下来的就先把文件下载到本地）。')
        + '照片在本机就用下面「评测本机目录」（填照片文件夹的完整路径，完全不经过浏览器上传）；'
        + '要走浏览器的话，Safari 需在「设置 ▸ 网站 ▸ 文件访问」里允许读取本地文件，'
        + '或换 Chrome / Edge 再试');
    }
    /* 整批评要几十秒到几十分钟（底库越大越久），不给动静的看着就像卡死，
       这里每秒把已经等了多久打在按钮上。 */
    const t0 = Date.now();
    btn.textContent = '比对中 0 秒…';
    const tick = setInterval(() => {
      btn.textContent = `比对中 ${Math.round((Date.now() - t0) / 1000)} 秒…`;
    }, 1000);
    try {
      S.run = await postJSON('/api/probe/close?id=' + op.upload_id, {});
    } finally {
      clearInterval(tick);
    }
    $('#card-result').hidden = false;
    /* 被跳过的几张只写在一闪而过的 toast 里，用户回头就找不到「到底漏了哪张」，
       只能整批重来。这里把它钉在待办区，连同下一步怎么做。 */
    $('#p-pend').innerHTML = lost.length
      ? `本批 ${items.length} 张里有 ${lost.length} 张浏览器读不出来，已跳过：<br>`
        + lost.slice(0, 12).map((x) => `<code>${esc(x)}</code>`).join('、')
        + (lost.length > 12 ? ' 等' : '')
        + '<br>这类失败多为并发读盘挤到了，已自动单发补送过两轮仍不行；'
          + '再点一次「开始评测」往往就全上来了，反复出现就改用下面「评测本机目录」。'
      : '';
    /* renderRun 会再打一次 /api/runs/<批次> 去分页取明细。那一次失败（服务端正忙、
       网络抖一下）跟评测成本身无关：批次已经跑完、已经落盘、已经进了历史下拉，
       而它和评测挤在同一个 try 里，落到下面的 catch 就变成一句「评测失败：HTTP 500」
       —— 明明几分钟的活干完了，却让人以为要重跑。所以这里单独包一层，说清是哪一步坏的。 */
    try {
      await renderRun(S.run);
    } catch (e) {
      toast('评测已完成，只是明细没加载出来：' + e.message
        + '。批次已经存进「历史批次」，在那儿重新打开就行，不用重跑', 14000);
    }
    await loadRuns();
    toast(`评测完成，用时 ${S.run.elapsed_s}s`
      + (lost.length ? ` · ${lost.length} 张没传上来，本批只评了 ${sent} 张` : ''),
      lost.length ? 8000 : 2600);
  } catch (e) {
    clientError('probe_run', e, { n, bytes, lost: lost.slice(0, 5), fb: fb.length });
    S.unreadable = S.unreadable.concat(fb);
    $('#p-pend').innerHTML = '评测失败：' + esc(e.message)
      + (S.unreadable.length
        ? `<br>读不出来的图片：<br>` + S.unreadable.slice(0, 12)
            .map((x) => `<code>${esc(x.file)}</code> ${esc(x.reason)}`).join('<br>')
        : '');
    toast('评测失败：' + e.message, 12000);
  }
  back();
};

/* 探针也留一条「服务端直接读本机目录」的路：摄像头一批几百上千张时，
   浏览器逐张推上去要几百次请求，任何一次读不到文件都整批白干；
   评测机和照片在同一台机器上（部署到 RK3588 就是这样）时读盘又快又稳。 */
async function probeFromDir() {
  if (!requireUi(['#p-path', '#p-run-dir'].concat(P_UI))) return;
  const path = val('#p-path', '').trim();
  if (!path) return toast('先把探针图片目录的完整路径填进来：在访达里选中文件夹按 Cmd+Option+C 就是「拷贝路径」', 9000);
  const btn = $('#p-run-dir');
  const back = () => { btn.disabled = false; btn.textContent = '评测本机目录'; };
  btn.disabled = true; btn.textContent = '评测中…';
  $('#p-prog').hidden = false; $('#p-bar').style.width = '2%';
  $('#p-progtxt').textContent = `正在扫描 ${path} …`;
  try {
    /* 目录模式是服务端自己读盘，不经过浏览器，没有「这个窗口发不出文件」
       这一类毛病，所以不用先跑上传自检（自检要挑一张图试传，反而多等一次）。 */
    const r = await postJSON('/api/probe/run_dir', {
      path, person_mode: val('#p-idmode', 'name'), delimiter: val('#g-delim', '__'),
      max_side: PROBE_MAX_SIDE,
      params: probeParams() });
    const total = r.found || 0;
    $('#p-progtxt').textContent = `${r.path} 里有 ${total} 张（${Math.round((r.total_bytes || 0) / 1048576)} MB），开始评测…`;
    let job = null;
    await new Promise((resolve, reject) => {
      let miss = 0;
      const t = setInterval(async () => {
        let j;
        try { j = await getJSON('/api/jobs/' + r.job_id); miss = 0; }
        catch (e) {
          /* 服务端在闷头跑推理时偶尔回不过来一次，立刻判死会白费几分钟 */
          if (++miss < 10) return;
          clearInterval(t);
          return reject(new Error(`连续 ${miss} 次查不到进度：${e.message}`));
        }
        const d = j.done || 0;
        const prep = Math.min(j.prepared || 0, total);
        $('#p-bar').style.width = Math.max(2, Math.round((d / (total || 1)) * 100)) + '%';
        $('#p-progtxt').textContent = j.status === 'done'
          ? `${total} 张都评完了，正在取回结果…`
          : `已评完 ${d}/${total} 张 · 正在识别第 ${prep} 张`
            + (j.bad_total ? ` · ${j.bad_total} 张读不出来` : '');
        if (j.status === 'done' || j.status === 'error') {
          clearInterval(t);
          if (j.status === 'error') return reject(new Error(j.error || '服务端没给出原因'));
          job = j; resolve();
        }
      }, 600);
    });
    /* 一万张的明细不会随任务一起回（那是几十 MB），只给批次号，
       明细由 openRun 按页去 /api/runs/<批次> 取。 */
    if (!job || !job.run_id) throw new Error('评测结束了但没拿到批次号，请刷新页面看历史批次');
    /* openRun 自己把失败吞成一句三秒的 toast，评测本身是成功的，
       所以这里必须问一句开没开起来：没开起来就别再喊「评测完成」把空白明细盖过去。 */
    const opened = await openRun(job.run_id, false);
    $('#p-bar').style.width = '100%';
    const bad = job.bad_total || 0;
    $('#p-progtxt').textContent = `${total} 张评测完成，用时 ${job.elapsed_s}s`
      + (bad ? ` · ${bad} 张读不出来已跳过` : '')
      + (opened ? '' : ' · 明细没打开，去「历史批次」点开它');
    const s = job.summary || {};
    /* 坏图和读不出来是两件事，得分开报：坏图是读进来了但解不开 / 推理炸了
       （summary.n_error，已按张从检出率和 Rank-1 分母里剔掉），读不出来是连字节
       都没拿到（bad_total，压根没进评测）。只报后者的话，一批几十张损坏的 JPEG
       评完只看得见一句「评测完成」，人以为这批真跑完了。 */
    toast((opened ? '评测完成：' : '评测完成，但结果没打开：')
      + `${s.n ?? total} 张 / ${s.n_faces ?? '?'} 张脸，用时 ${job.elapsed_s}s`
      + (s.n_error ? ` · ${s.n_error} 张坏图` : '')
      + (bad ? ` · ${bad} 张读不出来` : '')
      + (opened ? '' : ` · 批次 ${job.run_id} 还在，去「历史批次」里点开即可`), 8000);
    try { localStorage.setItem('facebench.ppath', path); } catch (x) { /* 隐私模式存不了就算了 */ }
    await loadRuns();
  } catch (e) {
    $('#p-progtxt').textContent = '评测失败：' + e.message;
    $('#p-pend').textContent = '评测失败：' + e.message;
    toast('评测失败：' + e.message, 14000);
  }
  back();
  /* 同上：进度条永远停在 100% 会被当成还在跑，成败都由 toast 和结果卡交代了 */
  $('#p-prog').hidden = true; $('#p-bar').style.width = '0%';
}
$('#p-run-dir').onclick = probeFromDir;
/* ---------- 并发 / 线程扫描（RK3588 选型） ----------
   选型真正要回答的是「这块板子该开几路、每个模型给几个核」，而这两个数互相牵制：
   路数换吞吐、线程数换单帧延迟。只跑一次评测给得出一个点，换块板子就得重猜；
   把两个维度摊成一张表，才能直接挑出「延迟够用的最低配置」。 */
/* 并发档最高给到 12：服务端 SWEEP_WORKERS_MAX 就是 12，而它特意留到 12 是为了让人
   看见「过了拐点之后吞吐怎么掉下去」—— 只扫到 8 的话，拐点右边是平的还是塌的看不出来，
   而这正是选型最想要的信息。线程档到 8 为止：RK3588 就 8 个核，再多一档只会互相抢核。 */
const SWEEP_W = [1, 2, 3, 4, 6, 8, 12];
const SWEEP_W_ON = [1, 2, 4];      // 默认勾选档跟服务端 SWEEP_GRID_WORKERS 一致
const SWEEP_T = [0, 1, 2, 4, 8];
const SWEEP_T_ON = [0, 4];
/* 0 是「不指定、用 onnxruntime 自己的默认线程数」。这一档必须留在候选里并显示成「默认」：
   它是装完机什么都不调时的真实状态，拿一个用户根本不会去设的参数当基线，整张表的结论没意义。 */
/* 上限以服务端 /api/env 下发的为准。以前在这儿抄了一份常量：服务端为小板子把上限
   调低时本地不知道，会把一格都没超的选择判成「超过上限」，或者反过来放行一个必被 400
   的请求，用户只看到按钮灰着不知道为什么。这份 24 只当 env 还没回来时的兜底。 */
const SWEEP_MAX_CELLS = 24;
const posNum = (v, d) => (Number.isFinite(+v) && +v > 0 ? +v : d);
const swLim = () => (S.env && S.env.sweep) || {};
const sweepMaxCells = () => posNum(swLim().max_cells, SWEEP_MAX_CELLS);
const sweepWorkersMax = () => posNum(swLim().workers_max, 12);
const thrLabel = (t) => (t === 0 ? '默认（onnxruntime 自定）' : `${t} 线程`);
let sweepJob = null, sweepBusy = false, sweepTimer = null;
/* 用户有没有手动勾过档位。/api/env 回来之后要把默认档对齐服务端下发的那一份，
   但人已经自己勾过了就不能再替他重画 —— 手刚点完按钮自己被弹回去，比数字旧一点更烦人。 */
let sweepTouched = false;
const pickedChips = (sel) => Array.from($(sel).children)
  .filter((b) => b.dataset.on === '1').map((b) => +b.dataset.v);
function chips(el, values, on, label) {
  /* 不引第三方控件：一排小按钮自己标选中态，选中才填色，风格跟旁边的按钮一致 */
  el.innerHTML = values.map((v) => {
    const sel = on.includes(v);
    /* aria-pressed 是给读屏的：选中态以前只写内联背景色，读屏念出来永远是
       「1 路 按钮」，听不出这一档到底勾没勾 —— 一排长得一样的按钮里只能靠颜色区分。 */
    return `<button class="small" data-v="${v}" data-on="${sel ? '1' : ''}"`
      + ` aria-pressed="${sel ? 'true' : 'false'}" style="margin:0 4px 0 0`
      + `${sel ? ';background:#1f6feb;border-color:#1f6feb;color:#fff' : ''}">${esc(label(v))}</button>`;
  }).join('');
  el.querySelectorAll('button').forEach((b) => {
    b.onclick = () => {
      const sel = b.dataset.on !== '1';
      b.dataset.on = sel ? '1' : '';
      b.setAttribute('aria-pressed', sel ? 'true' : 'false');
      sweepTouched = true;
      b.style.background = sel ? '#1f6feb' : '';
      b.style.borderColor = sel ? '#1f6feb' : '';
      b.style.color = sel ? '#fff' : '';
      sweepHint();
    };
  });
}
/* 扫描面板的默认档和上下限跟着服务端走。/api/env 早就下发了 grid_workers /
   grid_threads / default_images / min_images / max_images / latency_ms，本意就是
   别让前端再抄一份常量：服务端为小板子把默认档改成 1/2 时，这里还按 1/2/4 勾，
   扫出来的表跟它自己给的建议档对不上，而且一声不吭。今天两边数值恰好全对，
   所以这一步是防漂移，不是修一个正在发生的错。 */
function applySweepEnv() {
  const s = swLim();
  if (!sweepTouched && Array.isArray(s.grid_workers) && Array.isArray(s.grid_threads)
      && $('#s-workers') && $('#s-threads')) {
    const w = s.grid_workers.filter((v) => SWEEP_W.includes(v));
    const t = s.grid_threads.filter((v) => SWEEP_T.includes(v));
    if (w.length && t.length) {
      chips($('#s-workers'), SWEEP_W, w, (v) => `${v} 路`);
      chips($('#s-threads'), SWEEP_T, t, thrLabel);
    }
  }
  /* min / max 一律按服务端改；value 只在用户没动过时跟着换 —— 拿 HTML 里写死的
     defaultValue 比一下就知道动没动，别把人填了一半的数盖掉。 */
  const sync = (sel, lo, hi, def) => {
    const el = $(sel);
    if (!el) return;
    if (Number.isFinite(hi)) el.max = String(hi);
    if (Number.isFinite(lo)) el.min = String(lo);
    if (Number.isFinite(def) && +el.value === +el.defaultValue) el.value = String(def);
  };
  sync('#s-limit', s.min_images, s.max_images, s.default_images);
  sync('#s-lat', null, null, s.latency_ms);
  sweepHint();
}
function sweepHint() {
  const el = $('#s-hint');
  if (!el) return;
  const w = pickedChips('#s-workers'), t = pickedChips('#s-threads'), n = w.length * t.length;
  const max = sweepMaxCells(), wmax = sweepWorkersMax();
  /* 服务端把并发上限调低时（内存小的板子），本地那一排候选里可能还剩着几档发不出去：
     与其让人点了才吃 400，不如在这里就点名是哪几档。 */
  const over = w.filter((v) => v > wmax);
  el.textContent = !n ? '并发路数和推理线程至少各选一个。'
    : over.length
      ? `并发路数 ${over.join(' / ')} 路超过服务端上限 ${wmax} 路（每路都占一份全图字节和一个推理线程），少勾这几个。`
      : n > max
        ? `要跑 ${w.length}×${t.length}=${n} 格，超过服务端单次上限 ${max} 格（每格都得把整批探针重跑一遍），少勾几个。`
        : `将跑 ${n} 格。扫描期间服务端会临时把推理线程数切成当前档、跑完自动还原：`
          + '这期间别同时点第 1 节的「应用」或发起评测，用的会是扫描那一档。';
  if (!sweepBusy) { const b = $('#s-run'); if (b) b.disabled = !n || n > max || !!over.length; }
}
function sweepCell(c, rec) {
  if (!c) return '<td class="dim">排队中</td>';
  if (c.error) return `<td style="color:#f85149;text-align:left">${esc(c.error)}</td>`;
  const p50 = (c.total_ms || {}).p50;
  const hit = !!rec && c.threads === rec.threads && c.workers === rec.workers;
  /* 建议格只加一层底色再加一颗星：这是全表唯一要人记住的一格，但也别糊成大色块。 */
  return `<td${hit ? ' style="background:rgba(46,160,67,.14)"' : ''}>`
    + `<b>${c.throughput_faces_per_s ?? '—'}</b> 脸/秒${hit ? ' ★' : ''}<br>`
    + `<span class="dim">${c.throughput_fps ?? '—'} 图/秒 · P50 ${p50 == null ? '—' : Math.round(p50) + 'ms'}</span></td>`;
}
function renderSweep(j) {
  const g = j.grid || { workers: [], threads: [] };
  const cells = j.cells || [];
  const by = {};
  cells.forEach((c) => { by[`${c.threads}|${c.workers}`] = c; });
  const rec = (j.recommend || {}).cell;
  $('#s-table').innerHTML = `<tr><th>线程 \\ 并发</th>${g.workers.map((w) => `<th>${w} 路</th>`).join('')}</tr>`
    + g.threads.map((t) => `<tr><td>${esc(thrLabel(t))}</td>`
      + `${g.workers.map((w) => sweepCell(by[`${t}|${w}`], rec)).join('')}</tr>`).join('');
  $('#s-out').hidden = !cells.length;
  const okc = cells.filter((c) => !c.error);
  /* 口径写在表头就不用每格解释一遍，但绝不能写死成「整批墙钟」：
     服务端在拿不到 wall_s 时会退回 1000/P50 折算，那种口径下并发越高的格子越虚高，
     表头要是替它撒了谎，整张表的选型结论就是反的。所以照格子自己带的 basis 说。 */
  const bs = [...new Set(okc.map((c) => c.throughput_basis))];
  const basis = bs.length === 1 && bs[0] === 'wall' ? '整批墙钟（含排队与并行）'
    : bs.length === 1 && bs[0] === 'p50' ? '按 P50 折算（并发下会虚高，只能横向比不能当真实吞吐）'
      : bs.length > 1 ? '各格不一致（以服务端 throughput_basis 为准）' : '未回传';
  $('#s-meta').textContent = `${j.n_images ?? '—'} 张等间隔抽样 · 底库 ${okc.length ? (okc[0].gallery ?? '?') : '?'} 模板`
    + ` · 本机 ${j.cpu_count ?? '?'} 核 · 吞吐口径：${basis}`
    + ` · ${g.threads.length}×${g.workers.length}=${g.threads.length * g.workers.length} 格`;
  if (j.status !== 'done') return;
  const r = j.recommend || {}, c = r.cell, el = $('#s-rec');
  const p50 = (v) => ((v || {}).total_ms || {}).p50;
  if (!okc.length) {
    /* 后台线程整体崩掉时服务端只写 job.error、状态照样是 done，
       这时候顶层那句「session 载入失败」比第一格的报错有用得多，优先报它。 */
    el.textContent = `没有一格跑成功，给不出建议：${j.error || ((cells[0] || {}).error) || '服务端没说明原因'}`;
  } else if (c) {
    /* 建议、依据、说明三段拼一句人话。服务端在「全都不达标」那一路把 basis 和 note
       返回成了同一个字符串（note 就是 basis 本身，尾巴上接着「：最快的一格 P50 仍要…」），
       照原样拼两段会把同一句话说两遍；那一路早期版本还会让 note 以冒号开头，
       直接串起来会挤出「。：」。所以每段先去头尾标点、再去掉重复的那一段，然后用句号连。 */
    const clean = (x) => String(x || '').replace(/^[：；、，,\s]+/, '').replace(/[。；]+$/, '').trim();
    /* 比正文而不是比拼好的句子：加了「依据：」前缀之后两段就不相等了，
       拿去重后的正文比才拦得住服务端那份 basis === note。 */
    const why = clean(r.basis), also = clean(r.note);
    const segs = [`建议：${thrLabel(c.threads)} × ${c.workers} 路并发 —— 单图 P50 `
      + `${p50(c) == null ? '—' : Math.round(p50(c)) + 'ms'}，吞吐 ${c.throughput_faces_per_s ?? '—'} 脸/秒`,
      why ? '依据：' + why : '', also && also !== why ? also : ''];
    let txt = segs.filter(Boolean).join('。') + '。';
    const bt = j.best_throughput;
    /* 纯吞吐最高那格通常并发也最高：把它单独说清楚，用户才知道「建议」是在拿延迟换吞吐。 */
    if (bt && (bt.threads !== c.threads || bt.workers !== c.workers)) {
      txt += `（纯吞吐最高的是 ${thrLabel(bt.threads)} × ${bt.workers} 路 = ${bt.throughput_faces_per_s ?? '—'} 脸/秒`
        + `，但它单图 P50 ${p50(bt) == null ? '—' : Math.round(p50(bt)) + 'ms'}）`;
    }
    if (j.error) txt += `（注意：扫描线程中途异常退出 —— ${j.error}，上面这些格是退出前跑完的）`;
    el.textContent = txt;
  } else {
    el.textContent = r.note || r.basis || '跑完了，但服务端没给出建议。';
  }
  $('#s-apply').hidden = !(c && okc.length);
}
function pollSweep(id) {
  return new Promise((resolve, reject) => {
    let miss = 0;
    /* 一格都要在那一档线程数下把整批图重跑一遍，慢格几十秒很正常，轮询别太密。
        cells 是逐格追加的，所以每轮都重画一次表：先出来的格子先看见。 */
    sweepTimer = setInterval(async () => {
      let j;
      try { j = await getJSON('/api/jobs/' + id); miss = 0; }
      catch (e) {
        if (++miss < 10) return;
        clearInterval(sweepTimer); sweepTimer = null;
        return reject(new Error(`连续 ${miss} 次查不到扫描进度：${e.message}`));
      }
      sweepJob = j;
      const total = j.total || 1, done = j.done || 0;
      const bad = (j.cells || []).filter((x) => x.error).length;
      $('#s-bar').style.width = Math.max(1, Math.round((done / total) * 100)) + '%';
      $('#s-progtxt').textContent = `${done}/${total} 格 · 成功 ${done - bad} 格${bad ? ` · 失败 ${bad} 格` : ''}`;
      renderSweep(j);
      if (j.status === 'done' || j.status === 'error') {
        clearInterval(sweepTimer); sweepTimer = null;
        if (j.status === 'error') return reject(new Error(j.error || '扫描任务异常结束'));
        return resolve(j);
      }
    }, 900);
  });
}
async function startSweep() {
  if (!requireUi(['#s-path', '#s-limit', '#s-lat', '#s-workers', '#s-threads', '#s-run',
    '#s-hint', '#s-prog', '#s-bar', '#s-progtxt', '#s-out', '#s-table', '#s-meta', '#s-rec', '#s-apply'])) return;
  const btn = $('#s-run');
  if (btn.disabled) return;    // 一次扫描要几分钟，重复点会开出两个任务抢同一张表
  const path = val('#s-path', '').trim();
  if (!path) return toast('先填探针图目录的完整路径：扫描必须用真图，服务端造不出人脸', 9000);
  const workers = pickedChips('#s-workers'), threads = pickedChips('#s-threads');
  if (!workers.length || !threads.length) return toast('并发路数和推理线程至少各选一个');
  sweepBusy = true;
  btn.disabled = true; btn.textContent = '扫描中…';
  $('#s-prog').hidden = false; $('#s-bar').style.width = '1%';
  $('#s-progtxt').textContent = '排队中…';
  try {
    /* params 一定要带上。选型表每一格都是把同一批探针真跑一遍，检测输入尺寸、
       合影是不是每张脸都识别这些直接决定吞吐 —— 以前只发 path/workers/threads/limit，
       服务端只能退回 DEFAULTS，于是把 #p-detsize 从 320 改成 160 再扫，
       整张表一个数都不变，看着像「这块板子跟 det_size 无关」。
       识别口径（person_mode/delimiter/max_side）同理，跟「评测本机目录」保持同一套。 */
    const j0 = await postJSON('/api/sweep', { path, workers, threads,
      limit: +val('#s-limit', 200) || 200, latency_target_ms: +val('#s-lat', 120) || 120,
      person_mode: val('#p-idmode', 'name'), delimiter: val('#g-delim', '__'),
      max_side: PROBE_MAX_SIDE, params: probeParams() });
    $('#s-meta').textContent = `已排队 ${j0.cells} 格 · 抽 ${j0.n_images} 张 · ${j0.path}`;
    try { localStorage.setItem('facebench.spath', path); } catch (x) { /* 隐私模式存不了就算了 */ }
    await pollSweep(j0.job_id);
    toast('扫描完成，选型表已刷新');
  } catch (e) {
    toast('扫描失败：' + e.message, 14000);
    clientError('sweep', e, { path });
  } finally {
    clearInterval(sweepTimer); sweepTimer = null;
    sweepBusy = false;
    btn.disabled = false; btn.textContent = '开始扫描';
    sweepHint();   // 按当前勾选把按钮的可用状态还回去
    $('#s-prog').hidden = true; $('#s-bar').style.width = '0%';
  }
}
/* 浏览器拿着「新 app.js + 旧 index.html」时这几个控件还不存在，
   直接 .onclick 会抛在脚本中段，把后面的第 5 节一起带崩，所以先确认在不在。 */
if ($('#s-run')) $('#s-run').onclick = startSweep;
if ($('#s-apply')) $('#s-apply').onclick = () => {
  const c = ((sweepJob || {}).recommend || {}).cell;
  const el = $('#threads');
  if (!c || !el) return;
  /* 线程数是在建 onnxruntime session 时定死的，这里只能把建议填进第 1 节，
     生效还得点「应用」（那会重建后端）。0＝不指定，留空就是控件上写的「自动」。 */
  el.value = c.threads === 0 ? '' : c.threads;
  /* 并发路数也得一起填：评测真正取的是第 1 节那个 #p-workers，以前只填了线程数，
     照建议扫完点一下这颗按钮，评测还是按原来的路数串行跑，而提示语写着
     「页面上没有单独的并发路数控件」—— 那个控件一直就在第 1 节，指错地方了。
     填之前先确认下拉里有这一档：扫描档是 1/2/3/4/6/8/12，直接赋一个没有的值
     浏览器会静默变成空选中，取数时退回 1 路，等于悄悄换了个配置去复现选型表。 */
  const w = $('#p-workers');
  const has = !!w && Array.from(w.options).some((o) => +o.value === +c.workers);
  if (has) w.value = String(c.workers);
  toast(`已把建议档「${thrLabel(c.threads)} × ${c.workers} 路」填进第 1 节，线程数要点「应用」才生效`
    + (has ? ''
       : `。但并发路数下拉里没有 ${c.workers} 这一档，只填了线程数 —— `
         + `评测仍按 ${w && w.value ? w.value : 1} 路跑，要复现这一档请先在第 1 节补上它`), 12000);
};
if ($('#s-workers') && $('#s-threads')) {
  chips($('#s-workers'), SWEEP_W, SWEEP_W_ON, (v) => `${v} 路`);
  chips($('#s-threads'), SWEEP_T, SWEEP_T_ON, thrLabel);
  sweepHint();
}
function kpi(label, v, cls = '') {
  return `<div class="kpi ${cls}"><div class="v">${v}</div><div class="l">${label}</div></div>`;
}
/* 吞吐必须把口径写在脸上。服务端有两种算法：拿到整批墙钟就是真吞吐（含排队、含并行），
   拿不到就退回 1000/P50 折算 —— 磁盘上的老批次没有 wall_s，走的正是后者，而并发越高
   折算值越虚高。两种口径的数字长得一模一样时，跨批次横比会直接把结论说反。 */
function kpiThroughput(s) {
  const basis = s.throughput_basis === 'wall' ? '整批墙钟'
    : s.throughput_basis === 'p50' ? '按 P50 折算（老批次无墙钟，并发下偏高）' : '口径未知';
  return kpi(`吞吐 图 / 脸 每秒 · ${basis} · ${s.workers || 1} 路并发`,
    `${s.throughput_fps || '—'} / ${s.throughput_faces_per_s || '—'}`);
}
/* 合影同框放行拆成四项之后，只有「认错人」和「放进非底库人员」才需要人去看：
   以前只看聚合的 extra_accepted，旁人全认对也照样闪黄灯，闪几次这个灯就没人信了。
   老报告里没有这四项，退回原来的聚合口径，别报出一串 0。 */
function kpiExtra(s) {
  const head = '合影同框被放行';
  if (s.extra_identified_correct === undefined) {
    return kpi(head, `${s.extra_accepted} / ${s.extra_faces}`, s.extra_accepted ? 'lo' : 'hi');
  }
  const bit = (n, t) => (n ? `${n} ${t}` : '');
  const detail = [bit(s.extra_identified_correct, '认对'), bit(s.extra_identified_wrong, '认错'),
    bit(s.extra_nonmember_accepted, '非成员'), bit(s.extra_unlabeled_accepted, '未标注')]
    .filter(Boolean).join(' / ');
  const bad = (s.extra_identified_wrong || 0) + (s.extra_nonmember_accepted || 0);
  return kpi(`${head}${detail ? `（${detail}）` : ''}`,
    `${s.extra_accepted} / ${s.extra_faces}`, bad ? 'bad' : s.extra_accepted ? 'lo' : 'hi');
}
function kpiHeaders(r) {
  const s = r.summary, L = s.latency || {}, k = r.params.topk;
  const cls = (v, lo) => (v === null || v === undefined ? '' : v >= .9 ? 'hi' : v >= lo ? 'lo' : 'bad');
  return [
    /* 分母里含「有真值但一张脸都没检到」的那些图（n_rank1_nodetect，已算没认对）：
       不写出来的话，检测漏得越狠 Rank-1 越难看，但人会以为是人认错了而不是脸没找到。
       键名以前写的是 n_truth_nodetect —— 服务端根本没这个键，取回来是 undefined，
       于是这半句在任何批次上都不显示，自检只盯了服务端侧的键名，两边对不上也没人报错
       （t29 现在会拿真实 summary 逐个核对这里引用的键名）。 */
    kpi(`主体 Rank-1（${s.n_truth_named ?? 0} 图有真值${s.n_truth_unknown ? ` / ${s.n_truth_unknown} 非底库` : ''}`
      + `${s.n_rank1_nodetect ? ` · 含 ${s.n_rank1_nodetect} 张没检到脸已算失败` : ''}）`, pct(s.rank1), cls(s.rank1, .7)),
    kpi(`人脸级 Rank-1（${s.n_faces_labeled ?? 0} 张脸）`, pct(s.face_rank1), cls(s.face_rank1, .7)),
    kpi(`Recall@${k}（图 / 脸）`, `${pct(s[`recall@${k}`])} / ${pct(s[`face_recall@${k}`])}`),
    kpi(`人脸检出率${s.n_fallback_faces ? `（另 ${s.n_fallback_faces} 张没检出，按整图硬比）` : ''}`, pct(s.detect_rate), s.detect_rate >= .98 ? 'hi' : 'lo'),
    /* 坏图既不在 usable 里、也不占检出率的分母：坏 30 张时检出率照样能是 100%，
       整排数字看着干干净净，其实这批根本不该拿去定阈值，所以单独占一格。 */
    kpi('坏图 / 读不出（已从分母剔除）', s.n_error || 0, (s.n_error || 0) ? 'bad' : ''),
    kpi('图级 误识 / 漏识', `${s.false_accept} / ${s.false_reject}`, s.false_accept ? 'bad' : 'hi'),
    /* FAR 直接挂在同一格里：口径表里单列了「误放率 FAR」，界面上却只给两个计数，
       想知道放行率还得自己拿计算器除一次。分母只能是非底库脸
       （far_estimated 就是这个口径），别的脸混进去就是 1250% 那种数。 */
    kpi(`非底库 正确拒绝 / 误放行（${s.n_unknown_faces ?? 0} 张脸 · FAR `
      + `${s.n_unknown_faces ? pct(s.far_estimated) : '—'}）`,
      `${s.unknown_faces_correct_reject ?? 0} / ${s.unknown_faces_false_accept ?? 0}`,
      s.unknown_faces_false_accept ? 'bad' : 'hi'),
    kpiExtra(s),
    /* 这两个数不是一套口径，标签不能共用「检出人脸数」：n_faces 数的是进了识别环节的脸，
       合影档切成「只识别最大的一张」时服务端已经把每张图砍到一张；
       avg_faces_per_image 走的是检出数 faces_found，砍之前就定下来了。
       100 张每张检出 3 脸的合影在那一档下会显示「300 → 100 / 3.0」，
       读起来像「一共检出 100 张脸、平均每张 3 张」，自相矛盾。标签按各自口径写。 */
    kpi('已识别脸数 / 检出均张', `${s.n_faces} / ${s.avg_faces_per_image}`),
    kpi('单图耗时 P50', ms(L.total_ms?.p50), L.total_ms?.p50 < 80 ? 'hi' : 'lo'),
    kpi('P95 耗时', ms(L.total_ms?.p95)),
    kpiThroughput(s),
  ].join('');
}
/* KPI 和它下面那行「这批数据本身有什么毛病」永远一起换：改真值、拖阈值都只刷 KPI
   的话，真值打架的提示会留在上一批的说法上。 */
function renderKpis() {
  if (!S.run) return;
  $('#kpis').innerHTML = kpiHeaders(S.run);
  renderTruthNote();
}
function renderTruthNote() {
  const el = $('#truth-note');
  if (!el) return;
  const n = ((S.run || {}).summary || {}).n_truth_conflict_faces || 0;
  el.textContent = n
    ? `注意：有 ${n} 张合影的图级真值和主体脸级真值不一致（多半是手工给旁人的脸改过人名）。`
      + '上面「主体 Rank-1」按图级真值算，复核这批时以脸级为准。'
    : '';
  el.hidden = !n;
}
/* 阈值建议文案：开批次和拖完滑杆两处都要画，抽出来免得改了这边忘了那边。
   拖滑杆时服务端会顺手按当前底库重出一份（曲线只跟底库有关），底库在两次评测之间
   又被注册过的时候，旧报告里那份曲线已经对不上现在的底库了。 */
function renderThrReport(rep) {
  const tr = rep || {};
  /* 样本来源要如实说：纯底库互配（同场合照片，偏乐观）和「底库+本批带真值探针」
     （真实场景分数）的可信度不是一回事，同一句「底库自身分布建议」把两种口径混成一句。 */
  const who = tr.basis === 'gallery+probe' ? '实测分布（底库＋本批真值）建议' : '底库自身分布建议';
  const bits = [];
  if (tr.at_far_1e6) bits.push(`FAR≤1e-6 时阈值 ${tr.at_far_1e6.t}（通过率 ${pct(tr.at_far_1e6.tar)}）`);
  /* 只有平衡点、没有 1e-6 档也要报出来：以前这种情况整条建议塌成一句 note，
     屏幕上什么都不推荐，用户以为没算出来，其实曲线和好阈值都在。 */
  if (tr.best) bits.push(`平衡点 ${tr.best.t}（通过率 ${pct(tr.best.tar)}）`);
  $('#thr-advice').textContent = bits.length
    ? `${who}：${bits.join('；')}。${tr.note || ''}`
    : (tr.note || '');
}
async function renderRun(r) {
  const L = (r.summary || {}).latency || {};
  renderKpis();
  const rows = [['检测 SCRFD', L.detect_ms], ['对齐 warp', L.align_ms],
    ['提特征 ArcFace（单脸）', L.embed_ms], ['1:N 检索（单脸）', L.search_ms], ['单图合计', L.total_ms]];
  $('#lat').innerHTML = `<tr><th>阶段</th><th>P50</th><th>P95</th><th>P99</th><th>均值</th></tr>` + rows.map(([n, v]) => v && v.n
    ? `<tr><td>${n}</td><td>${v.p50 ?? '—'}</td><td>${v.p95 ?? '—'}</td><td>${v.p99 ?? '—'}</td><td>${v.mean ?? '—'}</td></tr>` : '').join('');
  renderGoneNote(r);
  $('#thr-slider').value = r.params.threshold;
  /* 数字框一起回填：只挪滑杆的话，打开一个阈值 0.30 的旧批次后框里还留着上次
     手敲的 0.55，此时点「开始评测」，新批次跑的是 0.55 而不是屏幕上那条线。 */
  $('#p-thr').value = r.params.threshold;
  $('#thr-val').textContent = Number(r.params.threshold).toFixed(3);
  renderThrReport(r.threshold_report);
  drawThr((r.threshold_report || {}).curve || [], r.params.threshold);
  drawScale(r.scale_curve || []);
  S.page = { offset: 0, only: 'hit', total: 0, matched: 0 };
  await loadRecords(0, 'hit');
  renderConfusions();
  $('#export').onclick = () => location.href = `/api/export/${r.run_id}`;
}
/* 复核图缺失要主动说出来。服务端在 load_runs_from_disk 里就标了 images_gone，
   但一直没人显示它：探针图被手工清掉、或者这批来自并发扫描（那种批次本来就不存复核图）时，
   屏幕上只剩一片裂图图标，看着像评测跑坏了，其实是图被主动丢弃、指标完全正常。 */
function renderGoneNote(r) {
  const el = $('#gone-note');
  if (!el) return;      // 老 HTML 里没这个节点，缺了就跳过，别把整张结果卡带崩
  const gone = !!r.images_gone;
  el.hidden = !gone;
  if (gone) el.textContent = '这个批次的探针原图已不在磁盘上（多半是 data/probes/ 被手工清理过，也可能这批当初就没保存复核图）。'
    + '指标、改标、导出报告都照常可用，只有缩略图显示不出来；要重看照片得重新跑一次评测。';
}

/* 易混淆对读的是 summary.confusions：改真值、拖阈值都会重算它，
   只在 renderRun 里画一次的话，标完错人、拖完阈值之后对表还挂着上一轮的旧误识。 */
function renderConfusions() {
  const c = (S.run && S.run.summary && S.run.summary.confusions) || [];
  $('#conf').innerHTML = c.length
    ? c.map(([k, v]) => `<span>${esc(k)} ×${v}</span>`).join('')
    : '<span>无误识</span>';
}
const UNK = '__unknown__';
const VERDICT = {
  correct: ['正确', 'ok'], wrong: ['误识', 'bad'], reject: ['漏识', 'warn'],
  correct_unknown: ['正确拒绝', 'ok'],
  /* 标签用词和筛选项、汇总卡对齐：筛选项写的是「没检出脸」，汇总卡写的是
     「坏图 / 读不出」，这里以前写「未检出 / 出错」，同一件事三套叫法，
     对着一堆徽章核记录的人只会怀疑这几个是不是不同的口径。 */
  no_face: ['没检出脸', 'warn'], no_truth: ['无真值', 'info'],
  error: ['没评成', 'bad'],
};
const UNK_LABEL = '非底库人员';
const truthLabel = (t, raw) => (t === UNK ? UNK_LABEL + (raw ? `(${raw})` : '') : t || '');
function badge(v) {
  const t = VERDICT[v] || ['—', 'info'];
  return `<span class="badge ${t[1]}">${t[0]}</span>`;
}
function faceBadge(f) {
  if (!f.truth) return '<span class="badge info">未标注</span>';
  if (f.truth === UNK) return badge(f.rank1_correct ? 'correct_unknown' : 'wrong');
  return badge(f.rank1_correct ? 'correct' : f.accepted ? 'wrong' : 'reject');
}
/* 输入框里写的是「人话」，接口要的是内部值：留空＝未标注，
   「非底库人员」＝UNKNOWN，其余按输入的人名处理（从下拉里选或直接打字都行）。 */
function truthFromInput(v) {
  const s = String(v || '').trim();
  if (!s) return null;
  if (s === UNK || s.startsWith(UNK_LABEL)) return UNK;
  return s;
}
/* ---------- 明细分页 ----------
   一万张的批次整包吐给浏览器是几十 MB，页面直接卡死。服务端已支持按页取，
   这里只管游标和「只看 xxx」的筛选项；缩略图按全局序号 _i 取，翻到第几页都不串图。 */
const REC_PAGE = 150;
/* 顺序跟服务端 _FACETS 对齐。error 必须在：坏图的 verdict 就是 error，
   而 noface 特意把 error 排除了（坏图也是一张脸都没有，混在一起两边都没法看），
   少这一项的话读不出来的那批图在界面上根本筛不出来。 */
const FACETS = [['hit', '比对成功'], ['miss', '未比对成功'], ['all', '全部'], ['bad', '判错'],
  ['risk', '可疑放行'], ['unlabeled', '未标注'], ['multi', '多人合影'], ['noface', '没检出脸'],
  ['error', '出错/坏图'], ['ok', '判对']];
function renderPager() {
  const p = S.page, fa = (S.run || {}).facets || {}, sel = $('#p-filter');
  sel.innerHTML = FACETS.map(([k, t]) => `<option value="${k}"${k === p.only ? ' selected' : ''}>${t} ${fa[k] ?? ''}</option>`).join('');
  const matched = p.matched || 0;
  const from = matched ? Math.min(p.offset + 1, matched) : 0;
  const to = Math.min(p.offset + REC_PAGE, matched);
  $('#p-showing').textContent = `第 ${from}–${to} 条 / 命中 ${matched} 条（本批共 ${p.total || 0} 张）`;
  $('#p-prev').disabled = p.offset <= 0;
  $('#p-next').disabled = to >= matched;
}
/* 连点「下一页」、或者一边拖滑杆一边翻页时，请求到达的顺序和发出去的顺序不是一回事：
   晚到的旧响应会把 S.page 的游标拽回去（看着像下一页按不动），还会拿旧的那半页
   盖掉刚画好的明细。跟阈值滑杆一样发一个号，只认最新一次的响应。 */
let recSeq = 0;
async function loadRecords(off = S.page.offset, only = S.page.only) {
  if (!S.run) return;
  const seq = ++recSeq;
  /* 先把游标落到状态里再发请求。以前只在响应回来后才赋值，请求一失败这两个字段
     就留在上一次的值上：下拉框已经换成「出错/坏图」，S.page.only 还是「比对成功」，
     下面那句「命中 N 条」是旧筛选项的数，点「下一页」翻的也是旧筛选项。 */
  S.page = { ...S.page, offset: Math.max(0, off), only: only || 'all' };
  const out = await getJSON(`/api/runs/${S.run.run_id}`
    + `?offset=${Math.max(0, off)}&limit=${REC_PAGE}&only=${encodeURIComponent(only || 'all')}`);
  if (seq !== recSeq) return;   // 已经有更新的一次在路上，这份旧明细直接丢掉
  S.run.summary = out.summary; S.run.facets = out.facets;
  S.page = { offset: out.offset, only: out.only, total: out.total, matched: out.matched };
  renderPager();
  renderRecords(out);
}
$('#p-filter').onchange = () => loadRecords(0, $('#p-filter').value).catch((e) => toast(e.message));
$('#p-prev').onclick = () => loadRecords(Math.max(0, S.page.offset - REC_PAGE)).catch((e) => toast(e.message));
$('#p-next').onclick = () => loadRecords(S.page.offset + REC_PAGE).catch((e) => toast(e.message));
/* ---------- 明细：比对成功出对照图，未比对成功只出清单 ----------
   逐张展开到每张脸的表格信息太密，眼睛扫不动。改成两组：
   比对成功的把「探针照片」和「底库留档照」并排放，一眼就能判断对不对；
   没比对上的只有文字，想知道为什么没过时再翻质量提示。
   标注入口都留着（识别率只能靠人工标注算出来）。 */
function renderRecords(r) {
  const recs = r.records, fa = r.facets || {}, TH = `/api/probe/thumb/${r.run_id}`;
  const okf = (f) => Boolean(f.accepted && f.top1_person);
  const nm = (x) => esc(String(x.name || '').split('/').pop());
  /* 单人脸图和合影主体改「整图真值」（图级 Rank-1 才算得出来）；
     合影里的旁人只改那一张脸，免得把主体的真值顶掉。 */
  const lvl = (x, f) => (x.faces.length > 1 && !f.is_subject ? f.idx : '');
  /* verdict 是服务端早就算好的图级结论（正确/误识/漏识/正确拒绝/无真值…），
     VERDICT/badge 这套标签以前没人调用：一片绿勾里根本看不出哪几张其实判错了。
     合影旁人那张卡片改的是那一张脸，只能贴脸级标签，不能套整图的结论。 */
  const tag = (x, f) => (lvl(x, f) === '' ? badge(x.verdict) : faceBadge(f));
  /* 真值是「非底库人员」时服务端另回传原始文件夹名（路人甲、cam1 这种），
     以前第二参没人传，界面只显示「非底库人员」，复核不到到底是哪批被算成了库外。 */
  const rawOf = (x, f) => (lvl(x, f) === '' ? x.truth_raw : '');
  const hits = [], misses = [];
  recs.forEach((x, i) => {
    const faces = x.faces || [];
    if (faces.some(okf)) hits.push([x, i, faces]);
    const no = faces.filter((f) => !okf(f));
    if (no.length || !faces.length) misses.push([x, i, no]);
  });
  const card = (x, i, f) => {
    const multi = x.faces.length > 1, L = lvl(x, f), p = f.top1_person;
    const src = multi && f.thumb ? `${TH}/${esc(f.thumb)}` : `${TH}/${x._i ?? i}.jpg`;
    const q = f.quality && !f.quality.ok ? esc((f.quality.reasons || []).join('、')) : '';
    return `<div class="pair${f.truth ? ' la' : ''}">
      <div class="prow">
        <div class="pp"><span class="pl">探针照片${multi ? ` 第${f.idx + 1}张脸` : ''}</span>
          <img class="pimg" loading="lazy" src="${src}"
            onerror="this.style.opacity=.25;this.onerror=null;">
          <span class="pf" title="${esc(x.name)}">${nm(x)}</span></div>
        <div class="pmid"><span class="arrow">→</span>
          <span class="sc">${f.top1_score ?? '—'}</span>
          ${f.truth_rank ? `<span class="tag">第${f.truth_rank}位</span>` : ''}</div>
        <div class="pp"><span class="pl">底库留档</span>
          <img class="pimg ref" loading="lazy" title="${esc(p)}"
            src="/api/person/thumb?person=${encodeURIComponent(p)}"
            onerror="this.style.opacity=.25;this.onerror=null;">
          <span class="pf" title="${esc(p)}">${esc(p)}</span></div>
      </div>
      <div class="pmeta">
        <div class="fquick">${tag(x, f)}
          <button class="qbtn ok" data-quick="top1" data-i="${i}" data-faces="${L}" data-p="${esc(p)}"
            title="对照过了，就是同一个人">✓ 就是TA</button>
          <button class="qbtn no" data-quick="unk" data-i="${i}" data-faces="${L}" data-p=""
            title="这人其实不在底库里，算一次误识">✗ 不在底库</button>
        </div>
        <input class="tsel" data-truth="1" data-i="${i}" data-faces="${L}" list="person-list"
          spellcheck="false" placeholder="也可以手动填这人是谁…" value="${esc(truthLabel(f.truth, rawOf(x, f)))}">
        ${q ? `<span class="qwarn">照片质量：${q}</span>` : ''}
      </div></div>`;
  };
  const mrow = (x, i, faces) => {
    if (!faces.length) return `<div class="mrow"><span class="mn" title="${esc(x.name)}">${nm(x)}</span>
      ${badge(x.verdict === 'error' ? 'error' : 'no_face')}
      ${x.error ? `<span class="qwarn">${esc(x.error)}</span>` : ''}</div>`;
    return faces.map((f) => {
      const L = lvl(x, f);
      const q = f.quality && !f.quality.ok ? esc((f.quality.reasons || []).join('、')) : '';
      return `<div class="mrow${f.truth ? ' la' : ''}">
        <span class="mn" title="${esc(x.name)}">${nm(x)}</span>
        ${faceBadge(f)}
        ${x.faces.length > 1 ? `<span class="tag">第${f.idx + 1}张脸</span>` : ''}
        <span class="dim">最像 ${esc(f.top1_person || '底库没有相似的人')} ${f.top1_score ?? '—'}，没过阈值${q ? ` · 质量 ${q}` : ''}</span>
        ${f.truth ? `<span class="tag">真值 ${esc(truthLabel(f.truth, rawOf(x, f)))}</span>` : ''}
        ${f.top1_person ? `<button class="qbtn ok" data-quick="top1" data-i="${i}" data-faces="${L}" data-p="${esc(f.top1_person)}"
          title="这人其实在底库里，只是分数没够">其实在底库</button>` : ''}
      </div>`;
    }).join('');
  };
  /* 「本页」这两个数以前按图数却互斥不了：一张合影只要既有命中的脸又有没命中的脸，
     就同时进了 hits 和 misses，于是能写出「本页 150 张：比对成功 40 · 未比对成功 130」
     这种两数之和 170 > 150 的话；而右半边「全批」那两个数是服务端按互斥口径算的
     （app/server.py 里 miss = not any(命中)），左右并排就直接打架。这里改成和服务端同一套：
     命中 = 至少一张脸过阈值，未比对成功 = 一张都没过，两数相加恒等于本页张数。
     合影里没命中的那几张脸仍然照常列在下半区（那是复核要看的），只是不再重复占一个「图」的名额。 */
  const nMissImgs = recs.length - hits.length;
  /* 下面两段标题数的是真正画出来的东西：上半区按脸出卡、下半区按脸出清单项
     （整张图一张脸都没检到时只有一行，按 1 计），拿 hits.length 当卡片数的话，
     「比对成功 40 组」底下画出 61 张卡是常事。 */
  const nCards = hits.reduce((n, [, , fs]) => n + fs.filter(okf).length, 0);
  const nRows = misses.reduce((n, [, , no]) => n + (no.length || 1), 0);
  $('#p-tag').textContent = `本页 ${recs.length} 张：比对成功 ${hits.length} · 未比对成功 ${nMissImgs}`
    + (fa.hit === undefined ? ''
      : ` ｜ 全批 比对成功 ${fa.hit} · 未比对成功 ${fa.miss} · 判错 ${fa.bad} · 可疑放行 ${fa.risk}`);
  const P = hits.map(([x, i, faces]) => faces.filter(okf).map((f) => card(x, i, f)).join('')).join('');
  const M = misses.map(([x, i, no]) => mrow(x, i, no)).join('');
  $('#records').innerHTML =
    (P ? `<div class="g-h ok">比对成功 ${nCards} 张脸 · 探针照片 ↔ 底库留档</div><div class="pairs">${P}</div>` : '')
    + (M ? `<div class="g-h no">未比对成功 ${nRows} 条</div><div class="mlist">${M}</div>` : '')
    + (P || M ? '' : '<div class="dim pad">这一页没有符合筛选项的记录，换个筛选项或翻翻页看看。</div>');
  const applyTruth = async (i, facesAttr, truth) => {
    const rec = recs[+i];
    if (!rec) return;
    const body = { run_id: r.run_id, name: rec.name, truth };
    if (facesAttr !== '' && facesAttr != null) body.face_idx = +facesAttr;
    try {
      const out = await postJSON('/api/probe/relabel', body);
      S.run.summary = out.summary; S.run.facets = out.facets;
      renderKpis();
      renderConfusions();   // 改真值直接改写误识对表，光刷 KPI 对表还是旧的
      /* 真值就是阈值分布的原料，标完一条就得连阈值建议一起刷新，
         否则把真值一条条补全之后，那一栏还停在「样本不足」。 */
      if (out.threshold_report) {
        S.run.threshold_report = out.threshold_report;
        renderThrReport(out.threshold_report);
        drawThr(out.threshold_report.curve || [], S.run.params.threshold);
      }
      await loadRecords();
    } catch (e) { toast(e.message); }
  };
  $('#records').querySelectorAll('[data-truth]').forEach((el) => {
    el.onchange = () => applyTruth(el.dataset.i, el.dataset.faces, truthFromInput(el.value));
  });
  $('#records').querySelectorAll('[data-quick]').forEach((el) => {
    el.onclick = () => applyTruth(el.dataset.i, el.dataset.faces,
        el.dataset.quick === 'unk' ? UNK : (el.dataset.p || null));
  });
}
/* ---------- 历史批次 ---------- */
const fmtBytes = (b) => (b >= 1048576 ? (b / 1048576).toFixed(1) + ' MB'
  : b >= 1024 ? Math.round(b / 1024) + ' KB' : (b || 0) + ' B');
/* 按钮上直接写清要删几个批次，省得用户以为「清空」只清下拉框 */
function syncRunClear(list) {
  const btn = $('#run-clear');
  if (!btn) return;
  btn.disabled = !list.length;
  btn.textContent = list.length ? `清空 ${list.length} 个历史批次` : '清空历史批次';
  /* 顺手管一下「删除本批」：屏幕上那一批要是已经不在列表里了（刚清空掉、
     或者在另一个标签页里删掉了），按钮就得跟着灭，否则点下去删的是个不存在的批次。 */
  const del = $('#run-del');
  if (del && !(S.run && list.some((x) => x.run_id === S.run.run_id))) del.disabled = true;
}
async function loadRuns() {
  const list = await getJSON('/api/runs');
  const sel = $('#run-pick');
  /* 下拉框整份重建后浏览器会退回第一项，而结果卡还停在原来那一批：
     切模型、跑完一次目录评测都会悄悄把下拉框指到最新批次，屏幕上的指标却是旧批次，
     下一次点「打开批次」开的是另一个批次，读的人以为看的还是同一份。 */
  const cur = (S.run && S.run.run_id) || sel.value;
  sel.innerHTML = list.length
    ? list.map((x) => {
      const d = new Date((x.created_at || 0) * 1000);
      const t = x.created_at ? `${d.getMonth() + 1}/${d.getDate()} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}` : '';
      const s = x.summary || {};
      /* run_id 是服务端生成的十六进制，眼下过 esc() 看不出差别，但它一旦带上别的字符
         （改用批次名、或者被别的工具导进来），这里就是第一个破口的地方。 */
      const rid = esc(x.run_id);
      return `<option value="${rid}">${rid} ${t} · ${s.n || '?'}图/${s.n_faces || '?'}脸 · 主体 ${pct(s.rank1)}</option>`;
    }).join('')
    : '<option value="">— 无 —</option>';
  if (cur && list.some((x) => x.run_id === cur)) sel.value = cur;
  syncRunClear(list);
  return list;
}
/* 结果区清场。换模型、清空历史、删单批、打开新批次之前都要做同一件事，
   以前这四个地方各抄一份两行，抄漏一行就留下上一批的残渣：KPI 已经换成新批次、
   下面的比对结果还是旧批次的，读的人会把旧明细当成刚跑出来的结论。 */
function clearResult() {
  $('#kpis').innerHTML = ''; $('#records').innerHTML = '';
  $('#conf').innerHTML = ''; $('#p-tag').textContent = '';
  const showing = $('#p-showing');
  if (showing) showing.textContent = '';
}
async function openRun(id, scroll = true) {
  if (!id) return false;
  try {
    S.run = await getJSON(`/api/runs/${id}?limit=1`);
    $('#run-pick').value = id;
    /* 先把屏幕上的旧批次擦干净再渲染新的。renderRun 里的明细要再打一次
       /api/runs/<批次>，那一次失败的话下面的比对结果就没人动过，
       于是结果卡弹出来：上面 KPI 是刚点开的批次、下面写着「第 1–150 条」的
       还是上一个批次，很容易当成新数字读掉。 */
    clearResult();
    $('#card-result').hidden = false;
    await renderRun(S.run);
    /* 打开成功才允许删：按钮禁用态跟着「当前显示的是哪一批」走，不跟下拉框的
       选中值走 —— 只换了下拉没点打开时，屏幕上看的还是上一批的指标，
       这时删的也必须是屏幕上那一批，不能删一个没显示过的。 */
    const del = $('#run-del');
    if (del) del.disabled = false;
    if (scroll) $('#card-result').scrollIntoView({ behavior: 'smooth', block: 'start' });
    return true;
  } catch (e) { toast('打开失败：' + e.message); return false; }
}
$('#run-open').onclick = () => openRun($('#run-pick').value);
/* 报告与探针图是成对删的（报告引用的就是 data/probes/<批次>/ 里的裁剪图，
   单独留报告会变成界面上再也删不掉的孤儿），所以确认文案把删掉的文件逐个列清，
   别让人以为「清空」只清下拉框；底库不在删除范围内。 */
$('#run-clear').onclick = async () => {
  const sel = $('#run-pick');
  const n = sel.options.length;
  if (!n || !sel.options[0].value) return toast('没有历史批次可清空');
  if (!confirm(`确认清空 ${n} 个历史批次？每批的整图缩略图、对齐人脸截图、meta.jsonl 和报告 JSON 会连同磁盘上的批次残留一起删掉，无法撤销（底库不受影响；正在评测中的批次会自动保留，不会把跑到一半的评测删废）。`)) return;
  const btn = $('#run-clear');
  btn.disabled = true; btn.textContent = '清空中…';
  try {
    const out = await delJSON('/api/runs');
    S.run = null;
    $('#card-result').hidden = true;
      clearResult();
    const list = await loadRuns();
    /* 正在评测的批次服务端会跳过（删它等于废掉这一批），必须报出来：
       不然按了「清空 5 个历史批次」却还剩一个在列表里，看着像清空没生效。 */
    const skip = (out.skipped || []).length;
    const tail = skip ? ` · ${skip} 个正在评测已保留` : '';
    toast(out.failed && out.failed.length
      ? `已清空 ${out.cleared} 个批次，${out.failed.length} 个没删掉：${out.failed.slice(0, 3).join('、')}${tail}`
      : `已清空 ${out.cleared} 个历史批次，释放 ${fmtBytes(out.freed_bytes)}${tail}`);
    if (list.length) openRun(list[0].run_id, false);
  } catch (e) {
    toast('清空失败：' + e.message);
    loadRuns().catch(() => { btn.disabled = false; });
  }
};
/* 删掉当前这一批（DELETE /api/runs/<批次>）。以前只有「清空全部」，
   阈值设错或选错目录跑歪一次，就得把旁边几个好批次一起删掉重跑。
   报告与探针图由服务端成对删，这里只负责删完把下拉和结果区收干净；
   服务端报了 failed 就说明文件还在，按钮留着让人重试，不假装删掉了。 */
const runDel = $('#run-del');
if (runDel) runDel.onclick = async () => {
  const rid = (S.run || {}).run_id || $('#run-pick').value;
  if (!rid) return toast('先打开一个批次');
  if (!confirm(`确认删除批次 ${rid}？它的报告 JSON 与整图缩略图、对齐人脸截图、meta.jsonl 会一起从磁盘删掉，无法撤销（别的批次和底库不受影响）。`)) return;
  runDel.disabled = true; runDel.textContent = '删除中…';
  try {
    const out = await delJSON(`/api/runs/${encodeURIComponent(rid)}`);
    if (out.failed && out.failed.length) {
      toast(`批次 ${rid} 没删干净：${out.failed.slice(0, 3).join('、')}（底库和其他批次没动）`, 12000);
      runDel.disabled = false;
    } else {
      S.run = null;
      $('#card-result').hidden = true;
      clearResult();
      const list = await loadRuns();
      toast(`已删除批次 ${rid}，释放 ${fmtBytes(out.freed_bytes)}`);
      if (list.length) openRun(list[0].run_id, false);
    }
  } catch (e) {
    toast('删除失败：' + e.message);
    runDel.disabled = false;
  } finally {
    runDel.textContent = '删除本批';
  }
};

/* ---------- 阈值滑杆 ---------- */
/* 一路拖过去会连发好几次重判：服务端排队顺序不保证和发出顺序一致，
   晚回来的旧响应会把新 KPI 盖回去，所以每次点号自增、只认最新那一次的响应。
   原来这段没有 try/catch，重判失败连一句提示都没有，看着就是滑杆坏了。 */
let thrSeq = 0;
async function rethreshold(t) {
  if (!S.run) return;                 // 批次被删掉/清空了，没有可重判的对象
  const seq = ++thrSeq;
  /* 批次号先抓在手里：下面两次 await 之间用户可能删掉这一批，S.run 会变成 null，
     那时候连报错都要读 S.run.run_id，于是 catch 里再抛一次，
     屏幕上只剩一句「阈值重判失败」和一条没人接的 Uncaught (in promise)。 */
  const rid = S.run.run_id;
  try {
    const out = await postJSON('/api/rethreshold', { run_id: rid, threshold: t });
    if (seq !== thrSeq || !S.run) return;   // 更新的一次在路上，或批次已经没了
    S.run.summary = out.summary; S.run.facets = out.facets;
    S.run.params.threshold = out.threshold ?? t;
    /* 数字框和滑杆是同一个阈值的两个入口：拖到 0.6 之后数字框还写着 0.4078，
       谁再从那儿改一次，改的就是个根本没在用的数。 */
    $('#p-thr').value = S.run.params.threshold;
    renderKpis();
    renderConfusions();
    await loadRecords();
    if (seq !== thrSeq || !S.run) return;
    /* 拿服务端按当前底库重出的那份刷新阈值曲线和建议；没回（模型被移走等）就沿用旧的，
       别把已经画好的画布刷空。 */
    if (out.threshold_report) {
      S.run.threshold_report = out.threshold_report;
      renderThrReport(out.threshold_report);
    }
    drawThr((S.run.threshold_report || {}).curve || [], t);
  } catch (e) {
    if (seq !== thrSeq) return;      // 过期那次的失败不该覆盖新阈值下的界面
    toast('阈值重判失败：' + e.message, 9000);
    clientError('rethreshold', e, { threshold: t, run_id: rid });
  }
}
$('#thr-slider').oninput = () => {
  const t = +$('#thr-slider').value;
  $('#thr-val').textContent = t.toFixed(3);
  if (!S.run) return;
  clearTimeout(window._tt);
  window._tt = setTimeout(() => rethreshold(t), 220);
};
/* 数字框是同一个阈值的第二个入口，以前只挂了滑杆的事件：在这儿敲一个 0.55 按 Tab，
   KPI、阈值曲线、明细一个字都不变，看着像这个框坏了；更糟的是它写着的数会被
   「开始评测」原样带走，下一批评测悄悄用的是一个用户从没让它生效过的判定线。 */
$('#p-thr').onchange = () => {
  const box = $('#p-thr');
  let t = +box.value;
  if (!Number.isFinite(t)) t = 0.4078;   // 和 probeParams 的兜底值同一个数
  t = Math.min(1, Math.max(0, t));
  box.value = t;
  $('#thr-slider').value = t;
  $('#thr-val').textContent = t.toFixed(3);
  if (!S.run) return;        // 没有打开批次时它就是「下一批评测的阈值」，改完就够了
  clearTimeout(window._tt);
  window._tt = setTimeout(() => rethreshold(t), 220);
};

/* ---------- 极简 canvas 折线图 ---------- */
function setup(cv) {
  const r = cv.getBoundingClientRect(), dpr = devicePixelRatio || 1;
  cv.width = r.width * dpr; cv.height = cv.getAttribute('height') * dpr;
  const c = cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0);
  return [c, r.width, +cv.getAttribute('height')];
}
function axes(c, w, h, pad, xr, yr) {
  c.clearRect(0, 0, w, h); c.strokeStyle = '#232b36'; c.fillStyle = '#8b98a5';
  c.font = '10px system-ui'; c.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (h - pad.t - pad.b) * i / 4;
    c.beginPath(); c.moveTo(pad.l, y); c.lineTo(w - pad.r, y); c.stroke();
    c.fillText((yr[1] - (yr[1] - yr[0]) * i / 4).toFixed(2), 4, y + 3);
  }
  for (let i = 0; i <= 5; i++) {
    const x = pad.l + (w - pad.l - pad.r) * i / 5;
    c.fillText((xr[0] + (xr[1] - xr[0]) * i / 5).toFixed(2), x - 10, h - 4);
  }
}
function line(c, pts, w, h, pad, xr, yr, color) {
  c.strokeStyle = color; c.lineWidth = 1.8; c.beginPath();
  pts.forEach(([x, y], i) => {
    const px = pad.l + (w - pad.l - pad.r) * (x - xr[0]) / (xr[1] - xr[0] || 1);
    const py = pad.t + (h - pad.t - pad.b) * (1 - (y - yr[0]) / (yr[1] - yr[0] || 1));
    i ? c.lineTo(px, py) : c.moveTo(px, py);
  });
  c.stroke();
}
/* 换批次后曲线可能是空的（这批一张脸都没检出、或底库只有一人取不到 impostor 对），
   直接 return 会把上一批的曲线原样留在画布上，看着像新批次跑出了一样的结果。 */
function blankChart(cv, text) {
  const [c, w, h] = setup(cv);
  c.clearRect(0, 0, w, h);
  c.fillStyle = '#8b98a5'; c.font = '11px system-ui';
  c.fillText(text, 10, h / 2);
}
function drawThr(curve, thr) {
  const cv = $('#chart-thr');
  if (!curve.length) return blankChart(cv, '样本不足：这批没有可用的阈值曲线');
  const [c, w, h] = setup(cv), pad = { l: 34, r: 8, t: 10, b: 18 };
  const xs = curve.map((p) => p.t);
  const xr = [Math.min(...xs), Math.max(...xs)];
  axes(c, w, h, pad, xr, [0, 1]);
  line(c, curve.map((p) => [p.t, p.tar]), w, h, pad, xr, [0, 1], '#3fb950');
  line(c, curve.map((p) => [p.t, Math.min(1, p.far * 50)]), w, h, pad, xr, [0, 1], '#f85149');
  line(c, [[thr, 0], [thr, 1]], w, h, pad, xr, [0, 1], '#388bfd');
  c.fillStyle = '#8b98a5'; c.fillText('绿=通过率 TAR   红=误识率 FAR×50   蓝=当前阈值', pad.l + 6, 14);
}
function drawScale(curve) {
  const cv = $('#chart-scale');
  if (!curve.length) return blankChart(cv, '样本不足：这批没有底库规模曲线');
  const [c, w, h] = setup(cv), pad = { l: 40, r: 8, t: 10, b: 18 };
  const xs = curve.map((p) => p.size), ys = curve.map((p) => p.p50_ms);
  const yr = [0, Math.max(...ys, ...curve.map((p) => p.p95_ms)) * 1.2];
  axes(c, w, h, pad, [Math.min(...xs), Math.max(...xs)], yr);
  line(c, curve.map((p) => [p.size, p.p95_ms]), w, h, pad, [Math.min(...xs), Math.max(...xs)], yr, '#d29922');
  line(c, curve.map((p) => [p.size, p.p50_ms]), w, h, pad, [Math.min(...xs), Math.max(...xs)], yr, '#3fb950');
  c.fillStyle = '#8b98a5'; c.fillText('绿=P50  黄=P95  横轴=底库规模(张)', pad.l + 6, 14);
}

/* 画布的位图宽度是按当前布局算一次的，CSS 却是 width:100%：窗口一拉宽，
   阈值曲线和规模曲线就被横向拉糊、坐标标签跟着变形，
   只有再拖一次阈值滑杆才会重画。窗口尺寸不常变， debounce 松一点没关系。 */
let _rzTimer = 0;
addEventListener('resize', () => {
  clearTimeout(_rzTimer);
  _rzTimer = setTimeout(() => {
    const r = S.run;
    if (!r) return;
    const th = (r.params || {}).threshold;
    drawThr((r.threshold_report || {}).curve || [], Number.isFinite(+th) ? +th : +$('#p-thr').value);
    drawScale(r.scale_curve || []);
  }, 250);
});

/* 浏览器缓存里那份 app.js 可能比服务端老一版：旧页面拼的是 multipart 表单，
   新服务端认的是逐张原始字节，几千张底库就全变成「没收到图片文件」。
   首页每次都回源校验，版本号对不上就说明页面是旧的，直接把怎么办说清楚。 */
function checkBuild() {
  const want = S.env && S.env.build;
  const mine = (document.querySelector('meta[name="app-build"]') || {}).content || '';
  if (want && mine && mine !== want)
    toast('浏览器用的还是旧版页面，请按 Cmd+Shift+R 强制刷新后再操作，否则上传会失败', 14000);
}
loadEnv().then(checkBuild).catch((e) => toast('环境读取失败：' + e.message));
/* 探针目录每次重新粘一遍完整路径很烦，填过一次就记着，下次打开自动带回 */
try { const p0 = localStorage.getItem('facebench.ppath'); if (p0 && $('#p-path') && !$('#p-path').value) $('#p-path').value = p0; } catch (e) { /* 忽略 */ }
/* 扫描吃的就是评测那批探针图，目录顺手带过来，免得同一个长路径再粘一遍 */
try {
  const s0 = localStorage.getItem('facebench.spath') || localStorage.getItem('facebench.ppath');
  if (s0 && $('#s-path') && !$('#s-path').value) $('#s-path').value = s0;
} catch (e) { /* 忽略 */ }
loadRuns().then((list) => { if (!S.run && list.length) openRun(list[0].run_id, false); })
  .catch(() => {});
/* 第 1 节那几个控件是「改了还没点应用」的草稿，而 loadEnv 会拿服务端现状整份回填。
   底库上传动辄几分钟，期间 20 秒一次的轮询能把用户刚填好的线程数、远程地址悄悄打回原形，
   等他点「应用」时生效的却是他根本没选的那套。判断“动没动过”的基准是上一份 S.env：
   控件当前值还等于上一份 env 里的值 = 没动过，可以让它回填；不等 = 用户的草稿，按住别动。 */
const CFG_CTRL = [['#threads', (e) => (e && e.state ? e.state.threads || '' : '')],
  ['#remote-url', (e) => (e && e.state ? e.state.remote_url || '' : '')],
  ['#bk-kind', (e) => (e && e.state ? e.state.backend_kind || 'onnx' : 'onnx')],
  ['#rec-model', (e) => (e && e.state ? e.state.rec || '' : '')],
  ['#det-model', (e) => (e && e.state ? e.state.det || '' : '')]];
function draftCfg(env) {
  const out = [];
  CFG_CTRL.forEach(([sel, get]) => {
    const el = $(sel); if (!el) return;
    if (String(el.value) !== String(get(env))) out.push([sel, el.value]);
  });
  return out;
}
function restoreCfg(draft) {
  draft.forEach(([sel, v]) => { const el = $(sel); if (el) el.value = v; });
}
/* 顶栏轮询。平时 20 秒一轮，有模型在后台下载时缩到 2 秒：180 MB 在 20 秒的粒度下
   就是「0% 卡半天不动」，看着跟服务死了一个样，人会去反复点注册、反复重启服务。
   间隔要每轮现算，所以是 setTimeout 自己接自己，不是 setInterval。 */
async function pollEnv() {
  try {
    let e;
    try { e = await getJSON('/api/env'); } catch (x) { return; }   // 轮询失败下一轮再来
    const prev = S.env;
    S.env = e;
    /* 顶栏的推理后端、模型、底库计数全取自 S.env：以前有批次在跑就只换数据不重画，
       页面挂一下午，pill 上还写着切走前的模型和人数。pill 重画不要钱，照画不误；
       整份底库列表还是只在没人跑批次、而且模板数真的变了时才重取（别把用户翻到的页刷掉）。 */
    renderEnvPills();
    renderModelNote(e);
    checkBuild();
    const was = (prev && prev.fetching) || [], now = e.fetching || [];
    for (const k of was.filter((x) => !now.includes(x))) {
      /* 下完了要说一声。几百 MB 的下载没人盯着，界面一声不吭的话，
         用户永远停在「到底下好没有」和「是不是下坏了」之间。 */
      const m = (e.models || []).find((x) => x.key === k);
      if (m && m.exists && m.sha256_ok !== false) toast(`模型 ${k} 下载完成，可以用了`, 6000);
      else if (m && m.fetch_state === 'failed') {
        toast(`模型 ${k} 下载失败：${m.fetch_error || '原因见服务端日志'}`, 9000);
        clientError('model-fetch', new Error(m.fetch_error || ''), { model: k });
      }
    }
    await applyPendingBackend(e);
    if (!S.run && (e.gallery?.templates || 0) !== (prev?.gallery?.templates || 0)) {
      const draft = draftCfg(prev);
      try { await loadEnv(); } catch (x) { /* 这次补刷没成，下一轮再补 */ }
      restoreCfg(draft);
    }
  } finally {
    scheduleEnvPoll();          // 不管这轮成没成，下一轮都得排上
  }
}
/* 下载完把之前被 409 挡回去的那次「应用模型」补发掉，省得人等几分钟回来再点一次。 */
async function applyPendingBackend(e) {
  const pend = S.pendingApply;
  if (!pend || (e.fetching || []).length) return;
  const want = (e.models || []).filter((m) => m.key === pend.rec || m.key === pend.det);
  const ready = pend.backend_kind === 'remote' || !want.length
    || want.every((m) => m.exists && m.sha256_ok !== false);
  S.pendingApply = null;        // 成不成都不留悬案：下坏了的模型重试一百次也不会自己变好
  if (!ready) return;
  try {
    await applyBackend(pend, '模型下载完成，已自动切换后端');
  } catch (x) {
    toast('下载完成但自动切换失败：' + x.message, 8000);
  }
}
function scheduleEnvPoll() {
  clearTimeout(S.envTimer);
  S.envTimer = setTimeout(pollEnv, ((S.env && S.env.fetching) || []).length ? 2000 : 20000);
}
scheduleEnvPoll();
