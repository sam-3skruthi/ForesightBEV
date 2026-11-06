#!/usr/bin/env bash
# Replay a prepared label sequence through two ROS 2 processes, record the output as a
# backend run, and verify it against an in-process run bit for bit.
# Usage: scripts/run_ros2_replay.sh <sequence> [pipeline_config] [stall_frame]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SEQUENCE="${1:?usage: run_ros2_replay.sh <sequence> [pipeline_config] [stall_frame]}"
CONFIG="${2:-$ROOT/configs/pipeline_default.yaml}"
STALL="${3:-}"
DATA="${FORESIGHTBEV_DATA_DIR:-$ROOT/experiments/end_to_end}"
SUFFIX="${STALL:+_stall$STALL}"
OUT="$DATA/results/ros2/replay_${SEQUENCE}${SUFFIX}.json"
PIXI="$ROOT/.tools/pixi"
rm -f "$OUT"
if [ ! -x "$PIXI" ] || [ ! -d "$ROOT/ros2/.pixi" ]; then
  echo "error: ROS 2 environment missing. See docs/project/RUN_THE_SYSTEM.md" >&2
  exit 2
fi
export PYTHONPATH="$ROOT/src" ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-43}" ROS_LOCALHOST_ONLY=1
cd "$ROOT/ros2"
"$PIXI" run python -m foresightbev.integration.ros_pipeline_node --root "$ROOT" --seed 0 --sequence "$DATA/sequences/$SEQUENCE" --pipeline-config "$CONFIG" \
  --record "$DATA/runs/ros2_${SEQUENCE}${SUFFIX}" --watchdog "${WATCHDOG_S:-30}" &
PIPELINE=$!
trap 'kill $PIPELINE 2>/dev/null || true' EXIT
"$PIXI" run python -m foresightbev.integration.ros_replay_node --sequence "$DATA/sequences/$SEQUENCE" --out "$OUT" ${STALL:+--stall-frame "$STALL"}
# Bounded wait: the node exits on the end signal, or on its own idle limit.
for _ in $(seq 1 90); do kill -0 $PIPELINE 2>/dev/null || break; sleep 1; done
kill $PIPELINE 2>/dev/null || true
wait $PIPELINE || true
if [ -z "$STALL" ]; then
  "$PIXI" run python -m foresightbev.integration.verify_ros --root "$ROOT" --result "$OUT" --sequence "$DATA/sequences/$SEQUENCE" --pipeline-config "$CONFIG"
fi
echo "output: $OUT"
