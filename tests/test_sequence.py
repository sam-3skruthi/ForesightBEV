import json
from pathlib import Path

import numpy as np
import pytest

from foresightbev.contracts import CalibrationBundle, CameraStatus, EgoState, Provenance
from foresightbev.data.calibration import load_rig
from foresightbev.data.sequence import LabelSequence, route_from_ego, write_sequence
from foresightbev.data.synchronization import synchronize
from foresightbev.errors import DatasetError, DatasetNotFound, SynchronizationError
from foresightbev.pipeline import PipelineConfig
from foresightbev.replay import run_sequence
from foresightbev.risk.trajectories import RouteContext
from foresightbev.sim.backend import BuiltinSim
from foresightbev.sim.scenarios import build_scenario

RIG = load_rig(Path(__file__).resolve().parents[1] / "configs/rig_cam2bev_1frlr.yaml")
SMALL = CalibrationBundle(tuple(c.scaled(.25) for c in RIG.cameras), "small")


def bundles(n=3):
    sim = BuiltinSim(build_scenario("lead_brake", 0), SMALL)
    out = []
    for _ in range(n):
        out.append(sim.observe())
        sim.t += .2
    return out


@pytest.fixture()
def seq(tmp_path):
    write_sequence(tmp_path / "s", "demo", bundles(), source="fixture", provenance=Provenance.SIMULATED, segmenter=None, route_for=lambda e: RouteContext(8.0))
    return LabelSequence(tmp_path / "s")


def test_labels_and_metadata_round_trip_exactly(seq):
    original = bundles()
    for i, bundle in enumerate(seq):
        assert (bundle.frames["front"].labels == original[i].frames["front"].labels).all()
        assert bundle.ego == original[i].ego
    assert seq.route(1).desired_speed == 8.0 and np.allclose(seq.calibration.cameras[0].K, SMALL.cameras[0].K)


def test_replay_is_deterministic_and_matches_across_runs(seq):
    assert run_sequence(seq, PipelineConfig()) == run_sequence(seq, PipelineConfig())


def test_unreadable_label_file_becomes_dropped_camera(seq):
    (seq.directory / "labels/00001/rear.png").write_bytes(b"junk")
    bundle = seq.bundle(1)
    assert bundle.frames["rear"].status == CameraStatus.DROPPED and bundle.frames["rear"].labels is None


def test_missing_sequence_and_bad_schema_fail_clearly(tmp_path, seq):
    with pytest.raises(DatasetNotFound):
        LabelSequence(tmp_path / "none")
    doc = json.loads((seq.directory / "sequence.json").read_text())
    (seq.directory / "sequence.json").write_text(json.dumps(doc | {"schema_version": 99}))
    with pytest.raises(DatasetError, match="schema"):
        LabelSequence(seq.directory)


def test_non_increasing_times_are_rejected(seq):
    doc = json.loads((seq.directory / "sequence.json").read_text())
    doc["frames"][1]["timestamp"] = doc["frames"][0]["timestamp"]
    (seq.directory / "sequence.json").write_text(json.dumps(doc))
    with pytest.raises(DatasetError, match="increase"):
        LabelSequence(seq.directory)


def test_empty_sequence_is_rejected(tmp_path):
    with pytest.raises(DatasetError):
        write_sequence(tmp_path / "e", "e", [], source="x", provenance=Provenance.SIMULATED, segmenter=None)


def test_route_from_ego_is_causal_and_bounded():
    route = route_from_ego(EgoState(0, 0, 0, 0, 10.0, 0.5))
    assert route.desired_speed == 10.0 and route.curvature == pytest.approx(.05)
    assert route_from_ego(EgoState(0, 0, 0, 0, 0.0, 5.0)).curvature == pytest.approx(.2)
    assert route_from_ego(EgoState(0, 0, 0, 0, 1.0)).desired_speed == 3.0


T = np.array([0.0, 0.5, 1.0])


def test_sync_matches_within_tolerance():
    out = synchronize({"a": T, "b": T + .02}, "a", .05)
    assert out[1].indices == {"a": 1, "b": 1} and out[1].skew["b"] == pytest.approx(.02)


def test_sync_marks_out_of_tolerance_camera_as_missing():
    out = synchronize({"a": T, "b": T + .2}, "a", .05)
    assert out[0].indices["b"] is None and out[0].skew["b"] is None


def test_sync_never_reuses_a_frame():
    out = synchronize({"a": np.array([0.0, 0.01]), "b": np.array([0.0])}, "a", .05)
    assert out[0].indices["b"] == 0 and out[1].indices["b"] is None


@pytest.mark.parametrize("streams,reference,tolerance", [({"a": T}, "z", .1), ({"a": T}, "a", 0), ({"a": T[::-1]}, "a", .1)])
def test_sync_rejects_bad_input(streams, reference, tolerance):
    with pytest.raises(SynchronizationError):
        synchronize(streams, reference, tolerance)
