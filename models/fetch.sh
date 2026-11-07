#!/bin/bash
# 补齐 models/ 里缺的模型，清单与校验值以 app/config.py 为准。
# 实际逻辑在 models/fetch.py（zip 成员按 range 只抽一段 + 镜像回退），这里只是入口。
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python3
[ -x "$PY" ] || PY=python3
exec "$PY" models/fetch.py "$@"
