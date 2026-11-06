#!/usr/bin/env bash
# Round-trip real ROS 2 messages over DDS and check every convention we rely on.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/ros2"
PYTHONPATH="$ROOT/src" ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-44}" ROS_LOCALHOST_ONLY=1 "$ROOT/.tools/pixi" run python -m foresightbev.integration.ros_selftest
