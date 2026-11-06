#!/usr/bin/env bash
# Start the backend on http://127.0.0.1:8000. API docs are at /docs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ ! -x .venv/bin/python ]; then
  echo "error: .venv missing. Run: uv sync --group dev --group ml" >&2
  exit 2
fi
exec .venv/bin/python -m foresightbev.cli serve --host 127.0.0.1 --port "${PORT:-8000}"
