"""End-to-end acceptance: inputs, calibration, every stage, ROS, API, stored results.

Each step records what it checked. A step passes only on content checks, never merely
because a service started or returned 200. The run stops at the first failure.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path

import httpx
import numpy as np

from foresightbev.config import ROOT, load_pipeline_config, load_settings
from foresightbev.contracts import CellState
from foresightbev.data.sequence import LabelSequence
from foresightbev.data.validation import validate_calibration
from foresightbev.errors import ForesightError
from foresightbev.recording import RunRecorder, environment, provenance_table
from foresightbev.replay import run_sequence


class Step:
    def __init__(self, report: list, name: str):
        self.report, self.name = report, name

    def __enter__(self):
        self.start = time.perf_counter()
        self.row = dict(step=self.name, ok=False, checks={})
        self.report.append(self.row)
        print(f"[{self.name}] running", flush=True)
        return self.row

    def __exit__(self, kind, error, _):
        self.row["seconds"] = round(time.perf_counter() - self.start, 2)
        self.row["ok"] = error is None and all(v is not False for v in self.row["checks"].values())
        if error is not None:
            self.row["error"] = f"{kind.__name__}: {error}"
        print(f"[{self.name}] {'ok' if self.row['ok'] else 'FAILED ' + self.row.get('error', str(self.row['checks']))}", flush=True)
        return error is not None and issubclass(kind, (ForesightError, AssertionError, OSError, subprocess.SubprocessError, httpx.HTTPError))


def decode(payload: dict) -> np.ndarray:
    return np.frombuffer(zlib.decompress(base64.b64decode(payload["data"])), payload["dtype"]).reshape(payload["shape"])


def run_command(command: list[str], env: dict) -> str:
    done = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    if done.returncode != 0:
        raise OSError(f"{' '.join(command[:3])} exited {done.returncode}: {(done.stdout + done.stderr)[-400:]}")
    return done.stdout


def start_server(port: int, env: dict) -> subprocess.Popen:
    process = subprocess.Popen([sys.executable, "-m", "foresightbev.cli", "serve", "--port", str(port)], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/api/v1/health", timeout=1).status_code == 200:
                return process
        except httpx.HTTPError:
            time.sleep(0.2)
        if process.poll() is not None:
            raise OSError(f"server exited with code {process.returncode}")
    process.terminate()
    raise OSError("server did not become healthy")


def accept(sequence_name: str, pipeline_config: Path, *, port: int = 8765, skip_ros: bool = False) -> dict:
    settings = load_settings()
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    directory = settings.data_dir / "sequences" / sequence_name
    report: list[dict] = []
    outcome: dict = dict(sequence=sequence_name, pipeline_config=str(pipeline_config), environment=environment(), steps=report)

    with Step(report, "inputs_and_calibration") as row:
        sequence = LabelSequence(directory)
        cameras = validate_calibration(sequence.calibration)
        row["checks"] |= dict(frames=len(sequence), cameras=len(cameras), source=sequence.doc["source"], provenance=sequence.doc["provenance"], segmenter=sequence.doc.get("segmenter"))
    if not report[-1]["ok"]:
        return outcome

    run = f"accept_{sequence_name}"
    hashes: list[str] = []
    with Step(report, "pipeline_all_stages") as row:
        config = load_pipeline_config(pipeline_config)
        target = settings.data_dir / "runs" / run
        if target.exists():
            shutil.rmtree(target)
        table = provenance_table(sequence.doc["provenance"], f"{sequence.doc['source']}: {len(sequence)} frames.", sequence.doc.get("segmenter"), detector=config.detector)
        recorder = RunRecorder(target, dict(kind="acceptance", title=run, scenario=sequence_name, seed=None, mode="open_loop_replay", input=sequence.doc["source"],
                                            caveat="Acceptance replay. Open loop: decisions do not affect later frames.", provenance=table), config.grid.bev)
        tallies = dict(occupied=0, free=0, unknown=0, detections=0, tracks=0, confirmed=0, forecast_frames=0)

        def observe(result):
            state = result.occupancy.bev_state
            hashes.append(hashlib.sha256(state.tobytes()).hexdigest())
            tallies["occupied"] += int((state == CellState.OCCUPIED).sum())
            tallies["free"] += int((state == CellState.FREE).sum())
            tallies["unknown"] += int((state == CellState.UNKNOWN).sum())
            tallies["detections"] += len(result.detections)
            tallies["tracks"] += len(result.tracks)
            tallies["confirmed"] += sum(t.confirmed for t in result.tracks)
            tallies["forecast_frames"] += int(result.forecast.last_input_timestamp <= result.forecast.t_ref)
            recorder.add(result)

        decisions = run_sequence(sequence, config, on_frame=observe)
        recorder.finish(dict(frames=len(decisions), decisions=decisions))
        statuses = sorted({d["status"] for d in decisions})
        row["checks"] |= tallies | dict(decisions=len(decisions), statuses=statuses, all_three_states_present=all(tallies[k] > 0 for k in ("occupied", "free", "unknown")),
                                        one_decision_per_frame=len(decisions) == len(sequence), forecast_causal_every_frame=tallies["forecast_frames"] == len(sequence))
    if not report[-1]["ok"]:
        return outcome

    with Step(report, "pipeline_repeatable") as row:
        second = run_sequence(sequence, load_pipeline_config(pipeline_config))
        row["checks"]["second_run_has_identical_decisions"] = second == decisions

    if not skip_ros:
        with Step(report, "ros2_two_process") as row:
            run_command([str(ROOT / "scripts/run_ros2_replay.sh"), sequence_name, str(pipeline_config)], env)
            ros = json.loads((settings.data_dir / "results/ros2" / f"replay_{sequence_name}.json").read_text())
            row["checks"] |= dict(transport=ros["transport"], ros_distro=ros["ros_distro"], rmw=ros["rmw"], decisions=len(ros["decisions"]), topics=ros["topic_message_counts"],
                                  identical_to_in_process=ros["matches_in_process"], occupancy_hashes_compared=ros["occupancy_hashes_compared"],
                                  every_stage_topic_delivered=all(ros["topic_message_counts"].get(t) == len(sequence) for t in ("tracks", "detections", "forecast", "risk")))
        if not report[-1]["ok"]:
            return outcome

    server = None
    with Step(report, "backend_cold_start_and_content") as row:
        server = start_server(port, env)
        client = httpx.Client(base_url=f"http://127.0.0.1:{port}/api/v1", timeout=60)
        status = client.get("/status").json()
        last = len(sequence) - 1
        occupancy = client.get(f"/runs/{run}/frames/{last}/occupancy").json()
        served = hashlib.sha256(decode(occupancy["bev_state"]).tobytes()).hexdigest()
        forecast = client.get(f"/runs/{run}/frames/{last}/forecast").json()
        decision = client.get(f"/runs/{run}/frames/{last}/decision").json()
        evidence = np.array(decision["candidates"][0]["path"])
        times = np.arange(len(evidence)) * 0.5
        poses = np.column_stack([evidence, np.zeros(len(evidence)), np.full(len(evidence), 1.0)])
        risk = client.post(f"/runs/{run}/frames/{last}/risk", json=dict(candidates=[dict(id="check", times=times.tolist(), poses=poses.tolist())])).json()
        row["checks"] |= dict(ready=status["ready"], served_occupancy_equals_computed=served == hashes[last], forecast_causal=forecast["causal"], decision_matches=decision["status"] == decisions[last]["status"] and decision["selected"] == decisions[last]["selected"],
                              risk_endpoint_returned_decision=risk["decision"]["status"] in ("pass", "modify", "fallback"))
        if not skip_ros:
            ros_run = client.get(f"/runs/ros2_{sequence_name}/frames/{last}/occupancy").json()
            row["checks"]["ros_output_served_by_backend_equals_computed"] = hashlib.sha256(decode(ros_run["bev_state"]).tobytes()).hexdigest() == hashes[last]
        row["checks"]["missing_output_is_404"] = client.get(f"/runs/{run}/frames/99999/occupancy").status_code == 404
    if server is not None:
        server.terminate()
        report[-1]["checks"]["clean_shutdown"] = server.wait(timeout=15) in (0, -15)

    with Step(report, "stored_results") as row:
        out = settings.data_dir / "results" / "acceptance"
        out.mkdir(parents=True, exist_ok=True)
        outcome["passed"] = all(s["ok"] for s in report)
        outcome["completed_unix"] = time.time()
        path = out / f"{sequence_name}.json"
        path.write_text(json.dumps(outcome, indent=1, default=str))
        row["checks"]["written"] = str(path)
    outcome["passed"] = all(s["ok"] for s in report)
    path.write_text(json.dumps(outcome, indent=1, default=str))
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="cam2bev_sample")
    parser.add_argument("--pipeline-config", default="configs/pipeline_default.yaml")
    parser.add_argument("--skip-ros", action="store_true")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    outcome = accept(args.sequence, ROOT / args.pipeline_config, port=args.port, skip_ros=args.skip_ros)
    passed = bool(outcome.get("passed"))
    print(json.dumps([dict(step=s["step"], ok=s["ok"], seconds=s.get("seconds")) for s in outcome["steps"]], indent=1))
    print("ACCEPTANCE " + ("PASSED" if passed else "FAILED"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
