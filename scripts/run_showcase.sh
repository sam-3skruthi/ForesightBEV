#!/usr/bin/env bash
# Regenerate every stored showcase run: all ten mini scenes, plus the camera-loss example.
# Needs the prepared nuScenes data. Detections are cached after the first run of each scene.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src
SCENES=(scene-0103 scene-0916 scene-0061 scene-0553 scene-0655 scene-0757 scene-0796 scene-1077 scene-1094 scene-1100)
args=()
for scene in "${SCENES[@]}"; do args+=(--scene "$scene"); done
.venv/bin/python -m foresightbev.cli showcase "${args[@]}"
.venv/bin/python -m foresightbev.cli showcase --scene scene-0103 --fault front_camera_loss
