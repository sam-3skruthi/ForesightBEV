import numpy as np
import pytest

from foresightbev.contracts import CalibrationBundle, CellState, SemanticClass as C
from foresightbev.data.calibration import load_rig
from foresightbev.evaluation.experiments import label_candidates, rollout_agents
from foresightbev.geometry.projection import project
from foresightbev.pipeline import ForesightPipeline, PipelineConfig
from foresightbev.risk.evaluator import RiskConfig
from foresightbev.risk.trajectories import RouteContext
from foresightbev.sim.backend import BuiltinSim
from foresightbev.sim.closed_loop import run_episode
from foresightbev.sim.render import CameraRenderer, NoiseConfig, apply_noise
from foresightbev.sim.scenarios import FAMILIES, build_scenario
from foresightbev.sim.world import Agent, EgoVehicle, RoadLayout, obb_corners, obb_distance, obb_overlap
from pathlib import Path

RIG = load_rig(Path(__file__).resolve().parents[1] / "configs/rig_cam2bev_1frlr.yaml")
SMALL = CalibrationBundle(tuple(c.scaled(.25) for c in RIG.cameras), "small")


def test_renderer_agrees_with_projection():
    cam = SMALL.cameras[0]
    box = Agent("b", C.VEHICLE, 15.0, 0.0, 0.0, 4.0, 2.0, 1.5)
    labels = CameraRenderer(cam).render(EgoVehicle(), RoadLayout(), [box])
    u, v, _ = project(cam, np.array([[13.0, 0.0, 0.75]]))
    assert labels[int(v[0]), int(u[0])] == C.VEHICLE
    assert labels[5, 100] == C.SKY and labels[-1, 120] == C.ROAD


def test_nearer_box_occludes_farther_box():
    cam = SMALL.cameras[0]
    far = Agent("far", C.PERSON, 25.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    near = Agent("near", C.VEHICLE, 10.0, 0.0, 0.0, 4.0, 3.0, 2.0)
    labels = CameraRenderer(cam).render(EgoVehicle(), RoadLayout(), [far, near])
    assert not (labels == C.PERSON).any()


def test_renderer_follows_ego_pose():
    cam = SMALL.cameras[0]
    box = Agent("b", C.VEHICLE, 15.0, 0.0, 0.0, 4.0, 2.0, 1.5)
    a = CameraRenderer(cam).render(EgoVehicle(), RoadLayout(), [box])
    b = CameraRenderer(cam).render(EgoVehicle(x=5.0), RoadLayout(), [Agent("b", C.VEHICLE, 20.0, 0.0, 0.0, 4.0, 2.0, 1.5)])
    assert (a == b).all()


def test_noise_is_off_by_default_and_seeded():
    labels = np.zeros((40, 40), np.uint8)
    labels[10:20, 10:20] = C.VEHICLE
    assert (apply_noise(labels, NoiseConfig(), np.random.default_rng(0)) == labels).all()
    noise = NoiseConfig(boundary_jitter_px=2, object_dropout=.5, spurious_blob_rate=1)
    assert (apply_noise(labels, noise, np.random.default_rng(1)) == apply_noise(labels, noise, np.random.default_rng(1))).all()


def test_sat_overlap_and_distance():
    a = obb_corners(0, 0, 0, 4, 2)
    assert obb_overlap(a, obb_corners(3, 0, 0, 4, 2))
    assert not obb_overlap(a, obb_corners(5, 0, 0, 4, 2))
    assert not obb_overlap(obb_corners(0, 0, np.pi / 4, 4, .5), obb_corners(1.5, -1.5, np.pi / 4, 4, .5))
    assert obb_distance(a, obb_corners(6, 0, 0, 4, 2)) == pytest.approx(2.0, abs=.05)


@pytest.mark.parametrize("name", list(FAMILIES))
def test_scenarios_are_reproducible(name):
    a, b = build_scenario(name, 3), build_scenario(name, 3)
    assert [(x.x, x.y, x.speed) for x in a.agents] == [(x.x, x.y, x.speed) for x in b.agents]
    assert [x.length for x in a.layout.buildings] == [x.length for x in b.layout.buildings]


def test_camera_dropout_reaches_pipeline_as_dropped():
    s = build_scenario("free_road", 0)
    s.dropouts = [("front", 0.0, 1.0)]
    bundle = BuiltinSim(s, SMALL).observe()
    assert bundle.frames["front"].labels is None


def test_pipeline_never_receives_ground_truth_objects():
    """The pipeline's bundle carries images and declared metadata only."""
    bundle = BuiltinSim(build_scenario("lead_brake", 0), SMALL).observe()
    allowed = {"sequence_id", "frame_index", "timestamp", "frames", "calibration", "ego", "source"}
    assert set(vars(bundle)) == allowed


def test_future_rollout_does_not_mutate_the_simulator():
    sim = BuiltinSim(build_scenario("lead_brake", 0), SMALL)
    before = [(a.x, a.speed) for a in sim.agents]
    list(rollout_agents(sim.agents, 0.0, 40, .05))
    assert before == [(a.x, a.speed) for a in sim.agents]


def test_closed_loop_produces_occupancy_from_rendered_cameras():
    """Only rendered labels reach the pipeline, and it still finds the lead vehicle."""
    s = build_scenario("lead_brake", 1)
    sim = BuiltinSim(s, SMALL, NoiseConfig(boundary_jitter_px=1))
    states = []
    run_episode(sim, PipelineConfig(), "test", on_frame=lambda r, _: states.append(r.occupancy.bev_state), max_seconds=2.0)
    assert any((x == CellState.OCCUPIED).any() for x in states) and sim.t > 1.0


def test_labels_flag_collision_course_and_not_a_clear_path():
    s = build_scenario("static_obstacle", 0)
    s.agents[0].x = 20.0
    sim = BuiltinSim(s, SMALL)
    pipeline = ForesightPipeline(SMALL, PipelineConfig())
    result = pipeline.step(sim.observe(), sim.route())
    labels = label_candidates(result, sim, RiskConfig())
    assert labels["lane+0.0_v1.0"] and not labels["emergency_stop"]


def test_filter_avoids_collision_where_nominal_alone_collides():
    """Regression on one fixed seed. Aggregate rates belong to the Stage 9 report."""
    from foresightbev.evaluation.experiments import build_mode, load_experiment, noise_from, scaled_rig
    root = Path(__file__).resolve().parents[1]
    cfg = load_experiment(root / "configs/experiment_default.yaml")
    outcome = {}
    for mode in ("filter", "idm_no_filter"):
        sim = BuiltinSim(build_scenario("occluded_pedestrian", 101), scaled_rig(cfg, root), noise_from(cfg), seed=101)
        config, _ = build_mode(cfg["modes"][mode])
        config.seed = 101
        outcome[mode] = run_episode(sim, config, mode).collided
    assert outcome == {"filter": False, "idm_no_filter": True}


def test_real_sample_runs_full_pipeline_when_present():
    from foresightbev.data.cam2bev_sample import load_sample
    sample = Path(__file__).resolve().parents[1] / "experiments/cam2bev_ipm/inputs/1_FRLR_examples"
    if not sample.exists():
        pytest.skip("local Cam2BEV sample is absent")
    result = ForesightPipeline(RIG, PipelineConfig()).step(load_sample(sample, RIG), RouteContext(8.0))
    assert result.detections and result.decision.selected is not None
    assert result.forecast.last_input_timestamp <= result.forecast.t_ref
