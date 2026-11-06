#!/usr/bin/env bash
# Two OS processes, real ROS 2 DDS topics, lockstep closed loop.
# Usage: scripts/run_ros2_closed_loop.sh [scenario] [seed]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCENARIO="${1:-occluded_pedestrian}"
SEED="${2:-100}"
OUT="$ROOT/experiments/end_to_end/results/ros2/${SCENARIO}_s${SEED}.json"
PIXI="$ROOT/.tools/pixi"
if [ ! -x "$PIXI" ] || [ ! -d "$ROOT/ros2/.pixi" ]; then
  echo "error: ROS 2 environment missing. See docs/project/RUN_THE_SYSTEM.md" >&2
  exit 2
fi
export PYTHONPATH="$ROOT/src"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export ROS_LOCALHOST_ONLY=1
cd "$ROOT/ros2"
"$PIXI" run python -m foresightbev.integration.ros_pipeline_node --root "$ROOT" --seed "$SEED" &
PIPELINE=$!
trap 'kill $PIPELINE 2>/dev/null || true' EXIT
"$PIXI" run python -m foresightbev.integration.ros_sim_node --root "$ROOT" --scenario "$SCENARIO" --seed "$SEED" --out "$OUT"
# Bounded wait: the node exits on the end signal, or on its own idle limit.
for _ in $(seq 1 90); do kill -0 $PIPELINE 2>/dev/null || break; sleep 1; done
kill $PIPELINE 2>/dev/null || true
wait $PIPELINE || true
"$PIXI" run python -m foresightbev.integration.verify_ros --root "$ROOT" --result "$OUT"
echo "output: $OUT"
