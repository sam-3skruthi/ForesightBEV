#!/usr/bin/env bash
# Complete backend verification. Writes a log and results/verification/summary.json.
# Run on an otherwise idle machine. Does not stop at a failure, so every result is reported.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT=experiments/end_to_end/results/verification
mkdir -p "$OUT"
LOG="$OUT/verification.log"
: > "$LOG"
export PYTHONPATH="$ROOT/src"
SUMMARY=()
step() {
  local name="$1"; shift
  local start=$SECONDS
  echo "=== $name" | tee -a "$LOG"
  if "$@" >> "$LOG" 2>&1; then local result=pass; else local result=FAIL; fi
  echo "    $result ($((SECONDS - start)) s)" | tee -a "$LOG"
  SUMMARY+=("{\"step\":\"$name\",\"result\":\"$result\",\"seconds\":$((SECONDS - start))}")
}
step unit_integration_api_tests .venv/bin/python -m pytest tests -q
step config_and_assets .venv/bin/python -m foresightbev.cli check
step dataset_validation .venv/bin/python -m foresightbev.cli validate-nuscenes
step ros2_message_selftest scripts/run_ros2_selftest.sh
step acceptance_cam2bev_real_sample scripts/run_pipeline.sh
step acceptance_nuscenes_scene_0553 scripts/run_pipeline.sh scene-0553
step acceptance_nuscenes_night_scene_1100 scripts/run_pipeline.sh scene-1100
step acceptance_streampetr_scene_0103 .venv/bin/python -m foresightbev.acceptance --sequence scene-0103 --pipeline-config configs/pipeline_streampetr.yaml
step acceptance_showcase_scene_0916 .venv/bin/python -m foresightbev.acceptance --sequence scene-0916 --pipeline-config configs/pipeline_showcase.yaml
step streampetr_port_verification .venv/bin/python -m foresightbev.evaluation.streampetr_check
step ros2_sim_closed_loop_pedestrian scripts/run_ros2_closed_loop.sh occluded_pedestrian 101
step ros2_sim_closed_loop_static scripts/run_ros2_closed_loop.sh static_obstacle 102
step ros2_sim_closed_loop_lead scripts/run_ros2_closed_loop.sh lead_brake 103
step ros2_replay_showcase_scene_0916 scripts/run_ros2_replay.sh scene-0916 "$ROOT/configs/pipeline_showcase.yaml"
step ros2_replay_showcase_stall_recovery scripts/run_ros2_replay.sh scene-0916 "$ROOT/configs/pipeline_showcase.yaml" 15
step real_log_safety_study .venv/bin/python -m foresightbev.evaluation.safety_real
for trial in $(seq 1 "${WATCHDOG_TRIALS:-5}"); do
  step "ros2_watchdog_trial_$trial" env WATCHDOG_S=5 ROS_DOMAIN_ID=$((50 + trial)) scripts/run_ros2_replay.sh cam2bev_sample "" 0
  step "ros2_watchdog_result_$trial" .venv/bin/python -c "import json,sys; d=json.load(open('experiments/end_to_end/results/ros2/replay_cam2bev_sample_stall0.json'))['decisions']; sys.exit(0 if len(d)==1 and d[0]['status']=='fallback' and 'PERCEPTION_MISSING' in d[0]['reasons'] else 1)"
done
step simulator_results_reproduce .venv/bin/python scripts/check_final_reproduces.py
step nuscenes_heldout_decisions_reproduce bash -c ".venv/bin/python -m foresightbev.cli eval-nuscenes --split held_out --name nuscenes_heldout_check --force > /dev/null && .venv/bin/python -c \"import json; a=json.load(open('experiments/end_to_end/results/nuscenes_heldout/decisions.json')); b=json.load(open('experiments/end_to_end/results/nuscenes_heldout_check/decisions.json')); raise SystemExit(0 if a==b else 1)\""
step release_audit_all_stored_runs .venv/bin/python scripts/audit_release.py
step frontend_typecheck_and_build bash -c "cd frontend && npm run build"
printf '[%s]\n' "$(IFS=,; echo "${SUMMARY[*]}")" > "$OUT/summary.json"
