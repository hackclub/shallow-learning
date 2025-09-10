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

# Launch manual play (renderer with keyboard controls)
exec "$PY" -m platformer play --fullscreen"$@"


