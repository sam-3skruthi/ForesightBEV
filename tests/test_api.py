import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from foresightbev.api import app as module
    run = tmp_path / "runs/demo"
    (run / "frames").mkdir(parents=True)
    (run / "bev").mkdir()
    (run / "manifest.json").write_text(json.dumps(dict(title="demo", kind="simulation", frames=1, summary={"collided": False})))
    (run / "frames/00000.json").write_text(json.dumps(dict(frame=0, t=0.0, ego=dict(speed=3.0),
                                                           decision=dict(status="pass", selected="nominal", reasons=["NOMINAL_ADMISSIBLE"]))))
    (run / "bev/00000.png").write_bytes(b"\x89PNG")
    results = tmp_path / "results/e"
    results.mkdir(parents=True)
    for name in ("summary", "experiment", "environment"):
        (results / f"{name}.json").write_text("{}")
    (results / "REPORT.md").write_text("# r")
    monkeypatch.setattr(module, "DATA", tmp_path)
    return TestClient(module.app)


def test_lists_only_real_stored_runs(client):
    assert client.get("/api/health").json()["runs"] == 1
    assert [r["id"] for r in client.get("/api/runs").json()] == ["demo"]


def test_frame_and_timeline_round_trip(client):
    assert client.get("/api/runs/demo/frames/0").json()["decision"]["selected"] == "nominal"
    assert client.get("/api/runs/demo/timeline").json()[0]["status"] == "pass"


def test_missing_things_are_404_and_never_invented(client):
    assert client.get("/api/runs/nope").status_code == 404
    assert client.get("/api/runs/demo/frames/9").status_code == 404
    assert client.get("/api/evaluations/nope").status_code == 404
    assert client.get("/api/runs/demo/images/secret/0").status_code == 404


def test_path_traversal_is_rejected(client):
    assert client.get("/api/runs/..%2F..%2Fetc").status_code == 404
    assert client.get("/api/evaluations/..").status_code == 404


def test_image_and_evaluation(client):
    assert client.get("/api/runs/demo/images/bev/0").headers["content-type"] == "image/png"
    assert client.get("/api/evaluations").json() == ["e"]
    assert client.get("/api/evaluations/e/report").text == "# r"
    assert client.get("/api/evaluations/e").json()["calibration"] is None


def test_api_is_read_only(client):
    assert client.post("/api/runs").status_code == 405
