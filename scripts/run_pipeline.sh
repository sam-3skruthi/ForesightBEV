#!/usr/bin/env bash
# The complete pipeline on real available inputs, with acceptance checks.
#   scripts/run_pipeline.sh                  Cam2BEV sample
#   scripts/run_pipeline.sh scene-0061       one nuScenes scene (needs nuscenes_root)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=.venv/bin/python
SCENE="${1:-}"
"$PY" -m foresightbev.cli check > /dev/null
if [ -z "$SCENE" ]; then
  "$PY" -m foresightbev.cli prepare cam2bev
  exec "$PY" -m foresightbev.acceptance --sequence cam2bev_sample
fi
"$PY" -m foresightbev.cli prepare nuscenes --scene "$SCENE"
exec "$PY" -m foresightbev.acceptance --sequence "$SCENE" --pipeline-config configs/pipeline_nuscenes.yaml
