#!/usr/bin/env bash
# One command to start Tavyra: builds the frontend when it is stale, then serves the API and the
# application on http://127.0.0.1:${PORT:-8000}
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ ! -d frontend/node_modules ]; then (cd frontend && npm ci --no-fund --no-audit); fi
if [ ! -f frontend/dist/index.html ] || [ -n "$(find frontend/src frontend/index.html -newer frontend/dist/index.html -print -quit)" ]; then
  (cd frontend && npm run build)
fi
if ! ls experiments/end_to_end/runs/showcase_* >/dev/null 2>&1; then
  echo "No showcase runs stored yet. Run scripts/run_showcase.sh to create them." >&2
fi
echo "Tavyra at http://127.0.0.1:${PORT:-8000}"
exec "$ROOT/.venv/bin/python" -m foresightbev.cli serve --host 127.0.0.1 --port "${PORT:-8000}"
