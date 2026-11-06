"""Compare a ROS 2 run with the same inputs processed in-process.

Run in the same environment as the ROS nodes, so only the transport differs. Compares
every decision, and for replays the occupancy tensor hash of every frame. Writes
``matches_in_process`` into the result file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from foresightbev.config import load_pipeline_config
from foresightbev.data.sequence import LabelSequence
from foresightbev.evaluation.experiments import build_mode, load_experiment, noise_from, scaled_rig
from foresightbev.integration.ros_messages import state_hash
from foresightbev.replay import build_pipeline
from foresightbev.sim.backend import BuiltinSim
from foresightbev.sim.closed_loop import run_episode
from foresightbev.sim.scenarios import build_scenario


def local_row(result) -> dict:
    d = result.decision
    return dict(frame=result.bundle.frame_index, status=d.status.value, selected=d.selected_id, reasons=[r.value for r in d.reasons], occupancy_sha256=state_hash(result.occupancy.bev_state))


def run_replay(sequence_dir: str, pipeline_config: str | None, seed: int) -> list[dict]:
    sequence = LabelSequence(sequence_dir)
    config = load_pipeline_config(pipeline_config)
    config.seed = seed
    pipeline = build_pipeline(sequence, config)
    return [local_row(pipeline.step(sequence.bundle(i), sequence.route(i))) for i in range(len(sequence))]


def run_simulation(root: Path, ros: dict) -> tuple[list[dict], float]:
    experiment = load_experiment(root / "configs/experiment_default.yaml")
    scenario = build_scenario(ros["scenario"], ros["seed"])
    sim = BuiltinSim(scenario, scaled_rig(experiment, root), noise_from(experiment), seed=ros["seed"])
    config, _ = build_mode(experiment["modes"]["filter"], experiment.get("pipeline"))
    config.seed = ros["seed"]
    rows: list[dict] = []
    run_episode(sim, config, "filter", on_frame=lambda r, _: rows.append(local_row(r)))
    return rows, float(scenario.ego.x)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--sequence")
    parser.add_argument("--pipeline-config")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root, path = Path(args.root), Path(args.result)
    ros = json.loads(path.read_text())
    if args.sequence:
        local, ego_x = run_replay(args.sequence, args.pipeline_config, args.seed), ros["ego_x"]
    else:
        local, ego_x = run_simulation(root, ros)
    keys = ("frame", "status", "selected", "reasons")
    remote = ros["decisions"]
    hashed = all(r.get("occupancy_sha256") for r in remote)
    pairs = list(zip(local, remote))
    first = next((i for i, (a, b) in enumerate(pairs) if any(a[k] != b[k] for k in keys) or (hashed and a["occupancy_sha256"] != b["occupancy_sha256"])), None)
    same = len(local) == len(remote) and first is None and abs(ego_x - ros["ego_x"]) < 1e-6
    ros.update(matches_in_process=same, occupancy_hashes_compared=hashed, in_process_frames=len(local), first_mismatch_frame=first)
    path.write_text(json.dumps(ros, indent=1))
    print(f"{path.stem}: identical={same} frames ros={len(remote)} local={len(local)} occupancy_hashes={hashed} first_mismatch={first}")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
