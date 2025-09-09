#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root (one dir up from this script)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

# Choose python (prefer project .venv)
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PY="$ROOT_DIR/.venv/bin/python"
else
  PY="python3"
fi

# Default checkpoint directory
CKPT_DIR="$ROOT_DIR/checkpoints"
mkdir -p "$CKPT_DIR"

# Optimal-ish defaults; pass any extra args to override
exec "$PY" -m platformer train \
  --generations 80 \
  --pop-size 128 \
  --elite-frac 0.1 \
  --mutation-std-start 0.08 \
  --mutation-std-end 0.01 \
  --eval-episodes 3 \
  --ckpt-dir "$CKPT_DIR" \
  --render-speed 3 \
  "$@"


