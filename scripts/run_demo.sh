#!/usr/bin/env bash
# One end-to-end demo: process a normal drive and a camera-loss drive from real frames,
# store both runs, then open the application.
# Usage: scripts/run_demo.sh [--no-serve]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "error: .venv missing. Run: uv sync --group dev" >&2
  exit 2
fi
export PYTHONPATH=src
"$PY" -m foresightbev.cli showcase --scene scene-0916
"$PY" -m foresightbev.cli showcase --scene scene-0103 --fault front_camera_loss
echo "Normal drive:  http://127.0.0.1:${PORT:-8000}/#/overview?run=showcase_scene-0916"
echo "Camera loss:   http://127.0.0.1:${PORT:-8000}/#/cameras?run=showcase_scene-0103_front_camera_loss&frame=18"
if [ "${1:-}" != "--no-serve" ]; then
  exec "$ROOT/scripts/run_app.sh"
fi
