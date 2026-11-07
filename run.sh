#!/usr/bin/env bash
# 一键起服务：建虚拟环境 -> 装依赖 -> 启动人脸评测台（macOS / Linux / RK3588-Ubuntu 通用）
# 用法:  ./run.sh                 前台运行，Ctrl-C 停止
#        PORT=9000 ./run.sh        换端口
#        SKIP_INSTALL=1 ./run.sh   跳过依赖检查直接起
#        PYTHON=python3.11 ./run.sh 指定解释器
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"
PORT="${PORT:-8617}"
PYBIN="${PYTHON:-python3}"

# ---- 1. 虚拟环境 ----
if [ -x "$VENV/bin/python" ]; then
  echo "[1/3] 复用虚拟环境 $VENV"
else
  command -v "$PYBIN" >/dev/null 2>&1 || { echo "找不到 $PYBIN，请先装 Python 3.9+"; exit 1; }
  echo "[1/3] 创建虚拟环境 $VENV（用 $PYBIN）"
  "$PYBIN" -m venv "$VENV"
fi

# ---- 2. 依赖：先看能不能直接 import，缺了才装 ----
if [ "${SKIP_INSTALL:-0}" = "1" ]; then
  echo "[2/3] 跳过依赖检查（SKIP_INSTALL=1）"
elif "$VENV/bin/python" -c 'import numpy, cv2, fastapi, uvicorn, onnxruntime' 2>/dev/null; then
  echo "[2/3] 依赖已就绪"
else
  echo "[2/3] 安装依赖（首次约 1~2 分钟）"
  if command -v uv >/dev/null 2>&1; then
    echo "     使用 uv"
    uv pip install --python "$VENV/bin/python" -r requirements.txt
  elif "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
    echo "     使用 pip"
    "$VENV/bin/python" -m pip install -q --upgrade pip
    "$VENV/bin/python" -m pip install -q -r requirements.txt
  else
    echo "     虚拟环境无 pip，尝试 ensurepip（建议改装 uv 或 python3-venv）"
    "$VENV/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true
    "$VENV/bin/python" -m pip install -q -r requirements.txt
  fi
fi

# ---- 缺模型不拦路：服务起来后自己后台下（进度在界面顶栏）----
miss=""
for m in w600k_r50.onnx det_10g.onnx; do
  [ -f "models/$m" ] || miss="$miss $m"
done
if [ -n "$miss" ]; then
  echo "     models/ 缺$miss —— 服务端起来后自己后台下载，进度看界面顶栏"
  echo "     （不想自动下：FACE_AUTO_FETCH=0，之后用 bash models/fetch.sh 手动补）"
fi

echo "[3/3] 启动评测台 → http://127.0.0.1:$PORT （局域网其他机器用本机 IP + 同端口）"
exec env PORT="$PORT" "$VENV/bin/python" -m app.server
