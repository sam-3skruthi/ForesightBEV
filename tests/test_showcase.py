"""Final-build backend: ground fit, box fusion, model velocity, conflict timing, and the showcase API."""

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from foresightbev.api.app import create_app
from foresightbev.config import load_settings
from foresightbev.contracts import BEVGrid, CellState, Detection, EgoState, OccupancyFrame, Provenance, SemanticClass as C, TrackState, VoxelGrid
from foresightbev.data.calibration import load_rig
from foresightbev.geometry.projection import build_lookup
from foresightbev.perception.ground import GroundConfig, detection_bottoms, fit_plane
from foresightbev.pipeline import fuse_detections
from foresightbev.risk.evaluator import RiskEvaluator
from foresightbev.risk.trajectories import RouteContext, parametric_candidate
from foresightbev.tracking.tracker import Tracker, TrackerConfig

ROOT = Path(__file__).resolve().parents[1]
GRID = BEVGrid(-20, 60, -15, 15, 0.25)
VGRID = VoxelGrid(GRID)
RIG = load_rig(ROOT / "configs/rig_cam2bev_1frlr.yaml")


def occupancy(state=CellState.FREE):
    bev = np.full(GRID.shape, int(state), np.uint8)
    return OccupancyFrame(0.0, VGRID, np.zeros(VGRID.shape, np.uint8), np.zeros(VGRID.shape, np.uint8), bev,
                          np.zeros(GRID.shape, np.uint8), np.zeros(GRID.shape, np.float32), np.ones(GRID.shape, np.uint8), {}, "test", Provenance.SIMULATED)


def box(x, y, velocity=None):
    return Detection(0.0, C.VEHICLE, np.array([x, y, 0.8]), 4.6, 1.9, 1.6, 0.9, 1, yaw=0.0, velocity_ego=velocity)


def track(x, y, vx=0.0):
    return TrackState(1, 0.0, C.VEHICLE, np.array([x, y], float), np.array([vx, 0.0]), np.eye(4) * 0.1, 4.6, 1.9, 1.6, 10, 10, 0, True)


class TestGroundPlane:
    def rows(self, a=0.3, b=0.02, c=-0.01):
        xy = np.array([(10, 5), (20, -6), (30, 3), (15, -9), (40, 8), (8, -3), (25, 10), (35, -8)], float)
        return np.column_stack([xy, a + b * xy[:, 0] + c * xy[:, 1]])

    def test_recovers_tilt_and_offset(self):
        a, b, c = fit_plane(self.rows(), GroundConfig(ridge_offset=0.01, ridge_slope=0.01))
        assert (a, b, c) == (pytest.approx(0.3, abs=.02), pytest.approx(0.02, abs=.002), pytest.approx(-0.01, abs=.002))

    def test_few_samples_stay_flat(self):
        assert fit_plane(self.rows()[:3]) is None

    def test_outlier_is_rejected(self):
        points = self.rows(0, 0, 0)
        points[0, 2] = 4.0
        a, b, c = fit_plane(points)
        assert abs(a) < .1 and abs(b) < .01 and abs(c) < .01

    def test_slope_is_bounded(self):
        fitted = fit_plane(self.rows(0, .3, 0), GroundConfig(ridge_slope=0.0, ridge_offset=0.0, inlier_m=50))
        assert fitted[1] == pytest.approx(GroundConfig().max_slope)

    def test_uses_only_confident_boxes_of_listed_classes(self):
        rows = [dict(center_ego=[10, 0, 1], height=2, score=.9, label="car"), dict(center_ego=[10, 0, 1], height=2, score=.1, label="car"), dict(center_ego=[10, 0, 1], height=2, score=.9, label="barrier")]
        assert detection_bottoms(rows).tolist() == [[10, 0, 0]]

    def test_raised_ground_moves_the_projection(self):
        flat = build_lookup(RIG.cameras[0], VGRID)
        raised = build_lookup(RIG.cameras[0], VGRID, (0.4, 0.0, 0.0))
        shared = flat.ground_valid & raised.ground_valid
        assert shared.any() and (flat.ground_v[shared] != raised.ground_v[shared]).any()


class TestBoxFusion:
    def test_marks_box_occupied_in_a_separate_layer(self):
        occ = occupancy()
        fuse_detections(occ, [box(10, 0)])
        assert occ.model_occupied.any()
        assert (occ.bev_state[occ.model_occupied] == CellState.OCCUPIED).all()
        assert (occ.bev_state[~occ.model_occupied] == CellState.FREE).all()

    def test_never_touches_invalid_cells(self):
        occ = occupancy(CellState.INVALID)
        fuse_detections(occ, [box(10, 0)])
        assert not occ.model_occupied.any() and (occ.bev_state == CellState.INVALID).all()

    def test_leaves_unknown_outside_boxes(self):
        occ = occupancy(CellState.UNKNOWN)
        fuse_detections(occ, [box(10, 0)])
        assert (occ.bev_state[~occ.model_occupied] == CellState.UNKNOWN).all()


class TestModelVelocity:
    def run(self, use, supplied):
        tracker = Tracker(TrackerConfig(use_model_velocity=use))
        out = []
        for i in range(4):
            t = i * .5
            out = tracker.update([Detection(t, C.VEHICLE, np.array([10 + 8 * t, 1.0]), 4.6, 1.9, 1.5, 1, 1, velocity_ego=np.array([8.0, 0.0]) if supplied else None)], EgoState(t, 0, 0, 0))
        return out[0].velocity[0]

    def test_speeds_up_velocity_convergence(self):
        assert abs(self.run(True, True) - 8) < abs(self.run(False, True) - 8)

    def test_detections_without_velocity_still_track(self):
        assert self.run(True, False) == pytest.approx(self.run(False, False))


class TestConflictAndPlan:
    ego = EgoState(0, 0, 0, 0, 10)
    route = RouteContext(desired_speed=10, lanes=(0.0,), home_lane=0.0)
    horizons = np.arange(.5, 3.01, .5)

    def metrics(self, tracks):
        return RiskEvaluator().conflict_metrics(parametric_candidate(self.ego, self.route, 10, 0), tracks, self.ego, self.horizons)

    def test_stopped_car_ahead_has_a_time_to_conflict(self):
        clearance, ttc, closest = self.metrics([track(20, 0)])
        assert clearance == 0 and closest == 1 and 1 <= ttc <= 2

    def test_car_in_far_lane_has_clearance_and_no_conflict(self):
        clearance, ttc, _ = self.metrics([track(20, 12)])
        assert ttc is None and clearance > 5

    def test_no_tracks_is_not_reported_as_safe_distance(self):
        assert self.metrics([]) == (None, None, None)

    def test_user_plan_reaches_speed_and_offset(self):
        plan = parametric_candidate(self.ego, self.route, 6, 2.0)
        assert plan.poses[-1, 1] == pytest.approx(2.0, abs=.2) and plan.poses[-1, 3] == pytest.approx(6, abs=.5)


settings = load_settings()
RUN = "showcase_scene-0916"
FAULT = "showcase_scene-0103_front_camera_loss"
stored = pytest.mark.skipif(not (settings.data_dir / "runs" / RUN / "manifest.json").exists() or not (settings.data_dir / "runs" / FAULT / "manifest.json").exists(), reason="showcase runs not generated")


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(settings)) as c:
        yield c


@stored
class TestShowcaseApi:
    def test_sequences_carry_split_status_and_runs(self, client):
        rows = {s["name"]: s for s in client.get("/api/v1/sequences").json()}
        assert len(rows) == 10
        assert rows["scene-0916"]["split"] == "mini_val" and rows["scene-0916"]["runs"]
        assert rows["scene-0061"]["detector_saw_scene_in_training"] and not rows["scene-0916"]["detector_saw_scene_in_training"]

    def test_timeline_matches_stored_frames(self, client):
        rows = client.get(f"/api/v1/runs/{RUN}/timeline").json()
        assert [r["frame"] for r in rows] == list(range(len(rows))) and len(rows) == 41
        assert {r["status"] for r in rows} <= {"pass", "modify", "fallback"}

    def test_record_is_causal_and_has_every_layer_of_the_chain(self, client):
        r = client.get(f"/api/v1/runs/{RUN}/frames/20/record").json()
        for key in ("cameras", "state_fractions", "detections", "tracks", "forecast", "forecast_paths", "candidates", "decision", "diagnostics", "coverage"):
            assert key in r
        assert r["forecast"]["last_input"] <= r["forecast"]["t_ref"]
        assert len(r["cameras"]) == 6

    def test_real_camera_image_is_a_jpeg(self, client):
        response = client.get(f"/api/v1/runs/{RUN}/frames/0/image/CAM_FRONT")
        assert response.status_code == 200 and response.content[:2] == b"\xff\xd8"

    @pytest.mark.parametrize("layer", ["bev", "bev_original", "coverage", "ipm", "forecast"])
    def test_layers_are_png(self, client, layer):
        response = client.get(f"/api/v1/runs/{RUN}/frames/0/layer/{layer}")
        assert response.status_code == 200 and response.content[:4] == b"\x89PNG"

    def test_per_camera_layers(self, client):
        for layer in ("labels", "projection"):
            assert client.get(f"/api/v1/runs/{RUN}/frames/0/layer/{layer}/CAM_FRONT").status_code == 200

    def test_unknown_layer_and_camera_fail_informatively(self, client):
        assert client.get(f"/api/v1/runs/{RUN}/frames/0/layer/nope").status_code == 404
        assert client.get(f"/api/v1/runs/{RUN}/frames/0/image/CAM_NOPE").status_code == 404
        assert client.get(f"/api/v1/runs/{RUN}/frames/9999/record").status_code == 404

    def test_lost_camera_has_no_image_and_says_why(self, client):
        response = client.get(f"/api/v1/runs/{FAULT}/frames/18/image/CAM_FRONT")
        assert response.status_code == 404
        assert "CAM_FRONT" in response.json()["error"]["message"]

    def test_lost_camera_never_adds_free_space(self, client):
        healthy = client.get(f"/api/v1/runs/{FAULT}/frames/13/record").json()["state_fractions"]
        lost = client.get(f"/api/v1/runs/{FAULT}/frames/18/record").json()
        assert lost["cameras"]["CAM_FRONT"]["status"] != "ok"
        assert lost["state_fractions"]["FREE"] < healthy["FREE"]

    def test_track_histories_exist(self, client):
        rows = client.get(f"/api/v1/runs/{RUN}/tracks").json()
        assert rows and all(len(r["points"]) >= 1 for r in rows)

    def test_user_plan_is_evaluated_with_the_run_configuration(self, client):
        body = client.post(f"/api/v1/runs/{RUN}/frames/20/plan", json=dict(target_speed=6, lateral_offset=1)).json()
        ids = [r["candidate_id"] for r in body["risks"]]
        assert "user_plan" in ids and len(ids) > 1 and body["path"]
        assert body["decision"]["status"] in ("pass", "modify", "fallback")

    def test_plan_validation(self, client):
        assert client.post(f"/api/v1/runs/{RUN}/frames/20/plan", json=dict(target_speed=99)).status_code == 422

    def test_evaluation_keeps_real_and_synthetic_apart(self, client):
        data = client.get("/api/v1/evaluation").json()
        assert data["real"]["showcase"]["pooled"]["frames"] == 81
        assert "simulat" in data["synthetic"]["label"].lower() or "synthetic" in data["synthetic"]["label"].lower()
        assert "closed_loop" not in data["real"]

    def test_system_does_not_claim_integrations_that_never_ran(self, client):
        rows = {i["name"]: i["status"] for i in client.get("/api/v1/system").json()["integrations"]}
        assert rows["CARLA"] != "verified" and rows["Autoware"] != "verified"

    def test_config_inspection_rejects_traversal(self, client):
        assert client.get("/api/v1/config/pipeline/pipeline_showcase.yaml").status_code == 200
        assert client.get("/api/v1/config/pipeline/..%2Fsecret.yaml").status_code in (404, 422)
