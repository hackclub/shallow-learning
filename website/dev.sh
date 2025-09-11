#!/usr/bin/env bash
set -euo pipefail

# Run the website in development mode with hot reload
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v npm >/dev/null 2>&1; then
  echo "Error: npm is required but not found in PATH" >&2
  exit 1
fi

exec npm run dev



