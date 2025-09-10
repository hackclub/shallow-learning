#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PY="$ROOT_DIR/.venv/bin/python"
else
  PY="python3"
fi

CKPT_DIR="${1:-$ROOT_DIR/checkpoints}"

exec "$PY" -m platformer render \
  --ckpt-dir "$CKPT_DIR" \
  --speed 3 \
  --watch \
  --log-rays \
  "$@"



