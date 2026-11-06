"""API contract, content and failure tests, against real pipeline outputs on disk."""

import base64
import json
import time
import zlib
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from foresightbev.api.app import create_app
from foresightbev.api.jobs import JobManager, build_command
from foresightbev.api.store import RunStore, decode_array, encode_array
from foresightbev.config import load_settings
from foresightbev.contracts import CalibrationBundle, CellState, Provenance
from foresightbev.data.calibration import load_rig
from foresightbev.data.sequence import LabelSequence, write_sequence
from foresightbev.pipeline import PipelineConfig
from foresightbev.recording import RunRecorder, provenance_table
from foresightbev.replay import run_sequence
from foresightbev.risk.trajectories import RouteContext
from foresightbev.sim.backend import BuiltinSim
from foresightbev.sim.scenarios import build_scenario

ROOT = Path(__file__).resolve().parents[1]
RIG = load_rig(ROOT / "configs/rig_cam2bev_1frlr.yaml")
SMALL = CalibrationBundle(tuple(c.scaled(.25) for c in RIG.cameras), "small")
SAMPLE = ROOT / "experiments/cam2bev_ipm/inputs/1_FRLR_examples"


def bundles(n=6):
    sim = BuiltinSim(build_scenario("lead_brake", 1), SMALL)
    out = []
    for _ in range(n):
        out.append(sim.observe())
        sim.t += .2
        sim.scenario.ego.x += 1.6
        for a in sim.agents:
            a.script(a, sim.t, .2)
    return out


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    data = tmp_path_factory.mktemp("data")
    write_sequence(data / "sequences/demo", "demo", bundles(), source="fixture", provenance=Provenance.SIMULATED, segmenter=None, route_for=lambda e: RouteContext(8.0))
    sequence = LabelSequence(data / "sequences/demo")
    config = PipelineConfig()
    recorder = RunRecorder(data / "runs/demo", dict(kind="replay", title="demo", scenario="demo", seed=None, mode="replay", input="fixture", caveat="x",
                                                    provenance=provenance_table("simulated", "fixture")), config.grid.bev)
    results = []

    def add(result):
        recorder.add(result)
        results.append(result)

    decisions = run_sequence(sequence, config, on_frame=add)
    recorder.finish(dict(decisions=decisions))
    (data / "results/exp").mkdir(parents=True)
    (data / "results/exp/summary.json").write_text('{"a": 1}')
    (data / "results/exp/REPORT.md").write_text("# report")
    return data, results


@pytest.fixture()
def settings(world):
    return replace(load_settings(), data_dir=world[0], nuscenes_root=None)


@pytest.fixture()
def client(settings):
    return TestClient(create_app(settings), raise_server_exceptions=False)


def decode(payload):
    return np.frombuffer(zlib.decompress(base64.b64decode(payload["data"])), payload["dtype"]).reshape(payload["shape"])


def assert_error(response, status, code):
    assert response.status_code == status, response.text
    body = response.json()["error"]
    assert body["code"] == code and body["message"] and body["request_id"] == response.headers["x-request-id"]


def test_array_encoding_round_trip():
    a = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert (decode_array(encode_array(a)) == a).all()


def test_health_and_ready_status(client):
    assert client.get("/api/v1/health").json()["status"] == "ok"
    status = client.get("/api/v1/status").json()
    assert status["ready"] and {c["name"] for c in status["checks"]} >= {"pipeline_config", "calibration", "nuscenes"}
    assert next(c for c in status["checks"] if c["name"] == "nuscenes")["ok"] is False


def test_status_reports_broken_configuration_not_ready(settings, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("safety: {max_event_score: 9.0}")
    client = TestClient(create_app(replace(settings, pipeline_config=bad)))
    status = client.get("/api/v1/status").json()
    assert not status["ready"] and "max_event_score" in next(c for c in status["checks"] if c["name"] == "pipeline_config")["detail"]


def test_config_endpoints(client):
    assert client.get("/api/v1/config/pipeline").json()["effective"]["safety"]["max_event_score"] == .1
    assert client.post("/api/v1/config/validate", json={"safety": {"max_event_score": .3}}).json()["effective"]["safety"]["max_event_score"] == .3
    assert_error(client.post("/api/v1/config/validate", json={"safety": {"nope": 1}}), 422, "invalid_configuration")


def test_datasets_and_calibration(client):
    ids = {d["id"]: d for d in client.get("/api/v1/datasets").json()}
    assert ids["demo"]["frames"] == 6 and ids["demo"]["provenance"] == "simulated"
    calibration = client.get("/api/v1/datasets/demo/calibration").json()
    assert calibration["valid"] and calibration["cameras"][0]["name"] == "front" and calibration["cameras"][0]["origin_ego"] == [1.7, 0, 1.4]
    assert_error(client.get("/api/v1/datasets/nope/calibration"), 404, "dataset_not_found")
    assert_error(client.get("/api/v1/datasets/nuscenes/scenes"), 404, "dataset_not_found")


def test_run_detail_reports_real_provenance(client):
    detail = client.get("/api/v1/runs/demo").json()
    assert detail["frames"] == 6 and detail["has_arrays"] and detail["provenance"][0]["kind"] == "simulated" and "source_sha256" in detail["environment"]


def test_occupancy_payload_equals_pipeline_output(client, world):
    result = world[1][3]
    body = client.get("/api/v1/runs/demo/frames/3/occupancy", params=dict(include_voxels=True)).json()
    assert (decode(body["bev_state"]) == result.occupancy.bev_state).all()
    assert (decode(body["voxel_state"]) == result.occupancy.voxel_state).all()
    assert body["state_legend"]["UNKNOWN"] == int(CellState.UNKNOWN) and "Not a probability" in body["reliability_semantics"]


def test_bev_payload_equals_pipeline_output(client, world):
    body = client.get("/api/v1/runs/demo/frames/2/bev").json()
    assert (decode(body["ipm_labels"]) == world[1][2].bev.labels).all() and "Not an occupancy claim" in body["note"]


def test_perception_payload_matches_pipeline(client, world):
    body = client.get("/api/v1/runs/demo/frames/5/perception").json()
    assert len(body["tracks"]) == len(world[1][5].tracks) and body["provenance"][0]["stage"] == 4


def test_forecast_payload_is_causal_and_labelled(client, world):
    body = client.get("/api/v1/runs/demo/frames/5/forecast").json()
    assert body["causal"] and body["t_ref"] == body["last_input_timestamp"] and "uncalibrated" in body["uncertainty_semantics"]
    assert np.allclose(decode(body["occupancy_prob"]), world[1][5].forecast.occupancy_prob, atol=1 / 254)


def test_decision_matches_pipeline(client, world):
    body = client.get("/api/v1/runs/demo/frames/4/decision").json()
    assert body["selected"] == world[1][4].decision.selected_id and len(body["candidates"]) == len(world[1][4].candidates)


def test_stored_forecast_round_trip_is_exact(world):
    result = world[1][5]
    rebuilt = RunStore(world[0]).forecast_volume("demo", 5)
    assert all((a == b).all() for a, b in zip(rebuilt.agent_samples, result.forecast.agent_samples))
    assert (rebuilt.static_occupied == result.forecast.static_occupied).all()


def request_for(result, ids):
    candidates = [dict(id=c.candidate_id, times=c.times.tolist(), poses=c.poses.tolist()) for c in result.candidates if c.candidate_id in ids]
    return dict(candidates=candidates, desired_speed=8.0)


def test_risk_endpoint_recomputes_the_same_scores_as_the_pipeline(client, world):
    result = world[1][5]
    ids = ("nominal", "lane+0.0_v1.0", "lane+0.0_v0.4")
    body = client.post("/api/v1/runs/demo/frames/5/risk", json=request_for(result, ids)).json()
    computed = {r["candidate_id"]: r for r in body["risks"]}
    for risk in result.risks:
        if risk.candidate_id in ids:
            assert computed[risk.candidate_id]["event_score"] == risk.event_score
            assert computed[risk.candidate_id]["static_collision"] == risk.static_collision
    assert "Not a collision probability" in body["risks"][0]["score_semantics"]


def test_risk_endpoint_rejects_a_path_through_a_tracked_vehicle(client, world):
    result = world[1][5]
    track = max((t for t in result.tracks if t.confirmed), key=lambda t: t.hits)
    ego = result.bundle.ego
    target = ego.world_to_ego(track.position + track.velocity * 3.0)
    times = np.arange(0, 3.01, .05)
    poses = np.stack([target[0] * times / 3, target[1] * times / 3, np.zeros_like(times), np.full_like(times, 8.0)], -1)
    body = client.post("/api/v1/runs/demo/frames/5/risk", json=dict(candidates=[dict(id="into_it", times=times.tolist(), poses=poses.tolist())], desired_speed=8.0)).json()
    row = body["risks"][0]
    assert not row["admissible"] and body["decision"]["status"] == "fallback" and body["decision"]["selected"] in ("comfort_stop", "emergency_stop")


def test_threshold_override_changes_decision_logic(client, world):
    payload = request_for(world[1][5], ("nominal",))
    payload["thresholds"] = dict(max_event_score=0.0)
    relaxed = dict(payload, thresholds=dict(max_event_score=1.0))
    strict_row = client.post("/api/v1/runs/demo/frames/5/risk", json=payload).json()["risks"][0]
    loose_row = client.post("/api/v1/runs/demo/frames/5/risk", json=relaxed).json()["risks"][0]
    if strict_row["event_score"] > 0 and not strict_row["static_collision"] and not strict_row["unknown_corridor_violation"]:
        assert not strict_row["admissible"] and loose_row["admissible"]


@pytest.mark.parametrize("candidate,fragment", [
    (dict(id="x", times=[0, 1], poses=[[0, 0, 0, 1]]), "same length"), (dict(id="x", times=[1, 2], poses=[[0, 0, 0, 1]] * 2), "start at 0"),
    (dict(id="x", times=[0, 0], poses=[[0, 0, 0, 1]] * 2), "strictly increase"), (dict(id="x", times=[0, 1], poses=[[0, 0, 1]] * 2), "x, y, yaw, speed"),
    (dict(id="x", times=[0, 1], poses=[[0, 0, 0, -1]] * 2), "non-negative"), (dict(id="../x", times=[0, 1], poses=[[0, 0, 0, 1]] * 2), "pattern"),
])
def test_risk_request_validation(client, candidate, fragment):
    response = client.post("/api/v1/runs/demo/frames/0/risk", json=dict(candidates=[candidate]))
    assert_error(response, 422, "invalid_request")
    assert fragment in response.json()["error"]["message"]


def test_unknown_request_field_is_rejected(client):
    assert_error(client.post("/api/v1/runs/demo/frames/0/risk", json=dict(candidates=[], extra=1)), 422, "invalid_request")


def test_missing_run_frame_and_output_are_404(client, world):
    assert_error(client.get("/api/v1/runs/nope"), 404, "run_not_found")
    assert_error(client.get("/api/v1/runs/demo/frames/99/decision"), 404, "output_not_stored")
    assert_error(client.get("/api/v1/runs/demo/frames/99/occupancy"), 404, "output_not_stored")


def test_traversal_names_are_rejected(client):
    for path in ("/api/v1/runs/..%2F..%2Fetc", "/api/v1/experiments/..%2Fx/files/y", "/api/v1/jobs/..%2Fx"):
        assert client.get(path).status_code in (404, 422)


def test_corrupt_stored_output_is_a_clear_error(world, settings, tmp_path):
    import shutil
    data = tmp_path / "copy"
    shutil.copytree(world[0], data)
    (data / "runs/demo/arrays/00001.npz").write_bytes(b"junk")
    client = TestClient(create_app(replace(settings, data_dir=data)), raise_server_exceptions=False)
    assert client.get("/api/v1/runs/demo/frames/1/occupancy").status_code in (422, 500)
    (data / "runs/demo/manifest.json").write_text("{bad")
    assert_error(client.get("/api/v1/runs/demo"), 422, "invalid_dataset")


def test_unexpected_error_is_a_logged_500_without_details(settings, tmp_path, monkeypatch):
    app = create_app(replace(settings, data_dir=tmp_path))
    monkeypatch.setattr(app.state.store, "list", lambda: 1 / 0)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/api/v1/runs")
    assert_error(response, 500, "internal_error")
    assert "division" not in response.text
    logged = client.get("/api/v1/logs", params=dict(level="ERROR")).json()
    assert any("ZeroDivisionError" in e.get("exception", "") for e in logged)


def test_logs_and_diagnostics(client):
    client.get("/api/v1/health")
    assert any(e.get("path") == "/api/v1/health" for e in client.get("/api/v1/logs").json())
    assert client.get("/api/v1/diagnostics").json()["runs"] == 1
    assert_error(client.get("/api/v1/logs", params=dict(level="LOUD")), 422, "invalid_request")


def test_experiments_endpoints(client):
    assert client.get("/api/v1/experiments").json()[0]["name"] == "exp"
    assert client.get("/api/v1/experiments/exp/files/summary.json").json() == {"a": 1}
    assert client.get("/api/v1/experiments/exp/files/REPORT.md").json()["text"] == "# report"
    assert_error(client.get("/api/v1/experiments/exp/files/none.json"), 404, "dataset_not_found")


def test_openapi_covers_required_capabilities(client):
    paths = client.get("/openapi.json").json()["paths"]
    for path in ("/api/v1/health", "/api/v1/status", "/api/v1/datasets", "/api/v1/datasets/{dataset}/calibration", "/api/v1/runs/{run}/frames/{frame}/bev", "/api/v1/runs/{run}/frames/{frame}/occupancy",
                 "/api/v1/runs/{run}/frames/{frame}/perception", "/api/v1/runs/{run}/frames/{frame}/forecast", "/api/v1/runs/{run}/frames/{frame}/risk", "/api/v1/runs/{run}/frames/{frame}/decision",
                 "/api/v1/jobs", "/api/v1/experiments", "/api/v1/logs", "/api/v1/diagnostics", "/api/v1/uploads/{upload}/finalize"):
        assert path in paths


def test_legacy_routes_for_frozen_frontend_still_work(settings, monkeypatch):
    from foresightbev.api import app as module
    monkeypatch.setattr(module, "DATA", settings.data_dir)
    client = TestClient(module.create_app(settings))
    assert client.get("/api/runs").status_code == 200 and client.get("/api/runs/demo/timeline").status_code == 200


# Ingestion, using the real sample images when they are present.
@pytest.fixture()
def fresh(settings, tmp_path):
    return TestClient(create_app(replace(settings, data_dir=tmp_path, max_upload_bytes=4_000_000)), raise_server_exceptions=False)


def png(camera="front", shape=(604, 964)):
    source = SAMPLE / f"{camera}.png"
    if source.exists():
        return source.read_bytes()
    return cv2.imencode(".png", np.zeros((*shape, 3), np.uint8))[1].tobytes()


def test_upload_rejects_bad_images(fresh):
    assert_error(fresh.put("/api/v1/uploads/u/frames/0/front", content=b"junk"), 422, "invalid_dataset")
    assert_error(fresh.put("/api/v1/uploads/u/frames/0/front", content=b""), 422, "invalid_dataset")
    assert_error(fresh.put("/api/v1/uploads/u/frames/0/side", content=png()), 422, "invalid_dataset")
    wrong = cv2.imencode(".png", np.zeros((10, 10, 3), np.uint8))[1].tobytes()
    assert_error(fresh.put("/api/v1/uploads/u/frames/0/front", content=wrong), 422, "invalid_dataset")
    assert_error(fresh.put("/api/v1/uploads/u/frames/0/front", content=b"x" * 5_000_000), 422, "invalid_dataset")


def test_upload_finalize_validates_counts_and_time(fresh):
    assert_error(fresh.post("/api/v1/uploads/none/finalize", json=dict(source="x", provenance="simulated", frames=[dict(timestamp=0, x=0, y=0, yaw=0)])), 404, "dataset_not_found")
    assert fresh.put("/api/v1/uploads/u/frames/0/front", content=png()).status_code == 201
    frames = [dict(timestamp=0, x=0, y=0, yaw=0)] * 2
    assert_error(fresh.post("/api/v1/uploads/u/finalize", json=dict(source="x", provenance="simulated", frames=frames)), 422, "invalid_dataset")


@pytest.mark.skipif(not SAMPLE.exists(), reason="real sample is absent")
def test_uploaded_real_frame_reaches_retrievable_pipeline_outputs(settings, tmp_path, monkeypatch):
    monkeypatch.setenv("FORESIGHTBEV_DATA_DIR", str(tmp_path))
    client = TestClient(create_app(replace(settings, data_dir=tmp_path)))
    for camera in RIG.names:
        response = client.put(f"/api/v1/uploads/real/frames/0/{camera}", content=png(camera))
        assert response.status_code == 201 and response.json()["label_fractions"]["ROAD"] > .1
    assert client.post("/api/v1/uploads/real/finalize", json=dict(source="Cam2BEV example", provenance="simulated", frames=[dict(timestamp=0, x=0, y=0, yaw=0)])).status_code == 201
    job = client.post("/api/v1/jobs", json=dict(kind="pipeline", params=dict(sequence="real"))).json()
    for _ in range(600):
        state = client.get(f"/api/v1/jobs/{job['id']}").json()
        if state["status"] != "running":
            break
        time.sleep(.1)
    assert state["status"] == "succeeded", client.get(f"/api/v1/jobs/{job['id']}/log").json()
    occupancy = client.get("/api/v1/runs/replay_real/frames/0/occupancy").json()
    states = decode(occupancy["bev_state"])
    assert (states == CellState.FREE).mean() > .2 and (states == CellState.OCCUPIED).any() and (states == CellState.UNKNOWN).any()
    assert client.get("/api/v1/runs/replay_real/frames/0/decision").json()["status"] in ("modify", "fallback")


# Jobs
def manager(tmp_path, activity=lambda: []):
    return JobManager(tmp_path, activity)


def test_every_job_kind_builds_a_fixed_command():
    assert build_command("evaluation", dict(name="n"))[3:5] == ["foresightbev.cli", "eval"] or "eval" in build_command("evaluation", dict(name="n"))
    with pytest.raises(ValueError):
        build_command("rm -rf", {})


def test_job_failure_is_recorded_with_its_output(tmp_path, monkeypatch):
    jobs = manager(tmp_path)
    monkeypatch.setattr("foresightbev.api.jobs.build_command", lambda k, p: ["/bin/sh", "-c", "echo boom; exit 3"])
    job = jobs.submit("sample", {})
    for _ in range(100):
        state = jobs.get(job["id"])
        if state["status"] != "running":
            break
        time.sleep(.02)
    assert state["status"] == "failed" and state["returncode"] == 3 and state["error"] == "boom"


def test_only_one_job_at_a_time(tmp_path, monkeypatch):
    from foresightbev.errors import JobConflict
    jobs = manager(tmp_path)
    monkeypatch.setattr("foresightbev.api.jobs.build_command", lambda k, p: ["/bin/sleep", "5"])
    job = jobs.submit("sample", {})
    with pytest.raises(JobConflict):
        jobs.submit("sample", {})
    assert jobs.cancel(job["id"])["status"] == "cancelled"
    for _ in range(100):
        if jobs.get(job["id"])["finished_unix"]:
            break
        time.sleep(.02)
    assert jobs.get(job["id"])["status"] == "cancelled"
    with pytest.raises(JobConflict):
        jobs.cancel(job["id"])


def test_heavy_job_refuses_when_an_external_run_is_active(tmp_path):
    from foresightbev.errors import JobConflict
    with pytest.raises(JobConflict, match="active"):
        manager(tmp_path, lambda: ["123 foresightbev.cli eval"]).submit("evaluation", dict(name="n"))


def test_interrupted_job_is_marked_lost_after_restart(tmp_path):
    folder = tmp_path / "old"
    folder.mkdir()
    (folder / "job.json").write_text(json.dumps(dict(id="old", kind="sample", params={}, status="running", pid=2_000_000_000, command=[], created_unix=1, started_unix=1, finished_unix=None, returncode=None, error=None)))
    assert manager(tmp_path).get("old")["status"] == "lost"


def test_job_request_rejects_unsafe_values(client):
    assert_error(client.post("/api/v1/jobs", json=dict(kind="pipeline", params=dict(sequence="../x"))), 422, "invalid_request")
    assert_error(client.post("/api/v1/jobs", json=dict(kind="evaluation", params=dict(name="a", config="/etc/passwd"))), 422, "invalid_request")
    assert_error(client.post("/api/v1/jobs", json=dict(kind="shell", params={})), 422, "invalid_request")
    assert_error(client.post("/api/v1/jobs", json=dict(kind="pipeline", params={})), 422, "invalid_configuration")


def test_unknown_job_is_404(client):
    assert_error(client.get("/api/v1/jobs/none"), 404, "dataset_not_found")
    assert_error(client.delete("/api/v1/jobs/none"), 404, "dataset_not_found")
