"""Pipeline boundary behaviour: validation, staleness, missing perception, config."""

from pathlib import Path

import numpy as np
import pytest

from foresightbev import config as C
from foresightbev.contracts import (
    CalibrationBundle, CameraCalibration, CameraFrame, CameraStatus, DecisionStatus as DS, EgoState, FrameBundle, Provenance,
    ReasonCode as R,
)
from foresightbev.data.calibration import load_rig
from foresightbev.data.validation import validate_bundle, validate_calibration
from foresightbev.errors import CalibrationError, ConfigurationError, DatasetError
from foresightbev.pipeline import ForesightPipeline
from foresightbev.risk.trajectories import RouteContext, generate_candidates
from foresightbev.sim.backend import BuiltinSim
from foresightbev.sim.scenarios import build_scenario

ROOT = Path(__file__).resolve().parents[1]
RIG = load_rig(ROOT / "configs/rig_cam2bev_1frlr.yaml")
SMALL = CalibrationBundle(tuple(c.scaled(.25) for c in RIG.cameras), "small")
ROUTE = RouteContext(8.0)


@pytest.fixture()
def bundle():
    return BuiltinSim(build_scenario("free_road", 0), SMALL).observe()


def test_valid_bundle_passes(bundle):
    validate_bundle(bundle)


def test_declared_lost_camera_is_valid(bundle):
    bundle.frames["front"] = CameraFrame("front", 0.0, None, CameraStatus.DROPPED)
    validate_bundle(bundle)


@pytest.mark.parametrize("change,message", [
    (lambda b: setattr(b.frames["front"], "labels", np.zeros((3, 3), np.uint8)), "do not match calibration"),
    (lambda b: setattr(b.frames["front"], "labels", b.frames["front"].labels.astype(np.int32)), "uint8"),
    (lambda b: setattr(b.frames["front"], "labels", np.full_like(b.frames["front"].labels, 200)), "not a SemanticClass"),
    (lambda b: setattr(b.frames["front"], "labels", None), "segmentation"),
    (lambda b: b.frames.__setitem__("ghost", CameraFrame("ghost", 0.0, None, CameraStatus.DROPPED)), "no calibration"),
])
def test_malformed_camera_input_is_rejected_by_name(bundle, change, message):
    change(bundle)
    with pytest.raises(DatasetError, match=message):
        validate_bundle(bundle)


@pytest.mark.parametrize("ego", [EgoState(float("nan"), 0, 0, 0), EgoState(0.0, float("inf"), 0, 0), EgoState(0.0, 0, 0, 0, -1.0), EgoState(5.0, 0, 0, 0)])
def test_bad_ego_state_is_rejected(bundle, ego):
    with pytest.raises(DatasetError):
        validate_bundle(FrameBundle("s", 0, 0.0, bundle.frames, bundle.calibration, ego, Provenance.SIMULATED))


def test_pipeline_rejects_malformed_input_rather_than_deciding(bundle):
    pipeline = ForesightPipeline(SMALL)
    bundle.frames["front"].labels = np.zeros((3, 3), np.uint8)
    with pytest.raises(DatasetError):
        pipeline.step(bundle, ROUTE)


def test_calibration_direction_error_is_caught():
    cam = RIG.cameras[0]
    inverted = CameraCalibration("bad", cam.K, cam.T_ego_cam, cam.width, cam.height)
    with pytest.raises(CalibrationError):
        validate_calibration(CalibrationBundle((inverted,), "x"))


def test_calibration_report_has_one_row_per_camera():
    rows = validate_calibration(RIG)
    assert [r["name"] for r in rows] == list(RIG.names)
    assert rows[0]["horizontal_fov_deg"] == pytest.approx(120, abs=.2)


def test_late_decision_is_stale_through_the_real_pipeline(bundle):
    result = ForesightPipeline(SMALL).step(bundle, ROUTE, now=bundle.timestamp + 2.0)
    assert result.decision.status == DS.FALLBACK and R.PERCEPTION_STALE in result.decision.reasons


def test_timely_decision_is_not_stale(bundle):
    result = ForesightPipeline(SMALL).step(bundle, ROUTE, now=bundle.timestamp + 0.1)
    assert R.PERCEPTION_STALE not in result.decision.reasons


def test_decision_time_before_frame_is_rejected(bundle):
    with pytest.raises(DatasetError):
        ForesightPipeline(SMALL).step(bundle, ROUTE, now=-1.0)


def test_missing_perception_reaches_fallback_through_the_pipeline():
    decision = ForesightPipeline(SMALL).decide_without_perception(EgoState(0, 0, 0, 0, 8.0), ROUTE)
    assert decision.status == DS.FALLBACK and R.PERCEPTION_MISSING in decision.reasons and decision.selected.is_fallback


def test_all_cameras_lost_forces_fallback(bundle):
    for name in list(bundle.frames):
        bundle.frames[name] = CameraFrame(name, bundle.timestamp, None, CameraStatus.DROPPED)
    result = ForesightPipeline(SMALL).step(bundle, ROUTE)
    assert result.decision.status == DS.FALLBACK and R.CAMERA_COVERAGE_LOST in result.decision.reasons


def test_detector_and_forecaster_are_replaceable(bundle):
    calls = []
    pipeline = ForesightPipeline(SMALL, detector=lambda occupancy, bundle: calls.append(occupancy.timestamp) or [])
    assert pipeline.step(bundle, ROUTE).detections == [] and calls == [bundle.timestamp]


def test_curved_route_follows_the_arc_and_zero_curvature_is_unchanged():
    ego = EgoState(0, 0, 0, 0, 10.0)
    arc = generate_candidates(ego, RouteContext(10.0, curvature=0.05), [])[0]
    end = arc.poses[-1]
    assert end[2] == pytest.approx(0.05 * 30, abs=.02) and end[1] == pytest.approx((1 - np.cos(1.5)) / .05, abs=.3)
    straight = generate_candidates(ego, RouteContext(10.0), [])[0]
    assert np.allclose(straight.poses[:, 1], 0)


def test_default_pipeline_config_file_loads_and_changes_values():
    config = C.load_pipeline_config(ROOT / "configs/pipeline_default.yaml")
    assert config.safety.max_event_score == .1 and config.forecast.n_samples == 32
    assert C.load_pipeline_config(ROOT / "configs/pipeline_nuscenes.yaml").occupancy.max_camera_skew_s == .1


def test_config_threshold_is_actually_used():
    config = C.pipeline_config_from_dict({"safety": {"max_event_score": 0.5}})
    assert config.safety.max_event_score == .5


@pytest.mark.parametrize("values,message", [
    ({"safety": {"max_event_scor": .1}}, "unknown setting"), ({"nope": 1}, "unknown setting"),
    ({"safety": {"max_event_score": "high"}}, "must be float"), ({"safety": {"max_event_score": 7.0}}, r"\[0, 1\]"),
    ({"forecast": {"n_samples": 0}}, "n_samples"), ({"nominal_mode": "fast"}, "nominal_mode"), ({"safety": 3}, "mapping"),
])
def test_bad_pipeline_config_fails_clearly(values, message):
    with pytest.raises(ConfigurationError, match=message):
        C.pipeline_config_from_dict(values)


def test_missing_config_file_fails_clearly(tmp_path):
    with pytest.raises(ConfigurationError, match="not found"):
        C.load_pipeline_config(tmp_path / "x.yaml")
    (tmp_path / "bad.yaml").write_text("a: [")
    with pytest.raises(ConfigurationError, match="cannot parse"):
        C.load_pipeline_config(tmp_path / "bad.yaml")


def test_settings_precedence_and_unknown_keys(tmp_path, monkeypatch):
    file = tmp_path / "b.yaml"
    file.write_text("nuscenes_root: /from/file\nlog_level: debug\n")
    assert C.load_settings(file).nuscenes_root == Path("/from/file") and C.load_settings(file).log_level == "DEBUG"
    monkeypatch.setenv("FORESIGHTBEV_NUSCENES_ROOT", "/from/env")
    assert C.load_settings(file).nuscenes_root == Path("/from/env")
    file.write_text("typo: 1\n")
    with pytest.raises(ConfigurationError, match="unknown backend"):
        C.load_settings(file)
