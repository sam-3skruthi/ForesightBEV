"""Stage 9 experiments: closed-loop ablations, event calibration and model quality.

All data here comes from the built-in simulator rendered through the real Cam2BEV
rig calibration. Results describe these scripted scenarios only. Seed ranges for
calibration and test are disjoint, and scenario parameters derive from the seed.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import yaml

from foresightbev.contracts import CalibrationBundle
from foresightbev.data.calibration import load_rig
from foresightbev.errors import ConfigurationError
from foresightbev.evaluation import metrics as M
from foresightbev.forecast.forecaster import footprint_polygon, is_stationary, rasterize, track_to_ego
from foresightbev.pipeline import PipelineConfig
from foresightbev.risk.calibration import RiskCalibrator
from foresightbev.risk.evaluator import RiskConfig
from foresightbev.sim.backend import BuiltinSim
from foresightbev.sim.closed_loop import run_episode
from foresightbev.sim.render import NoiseConfig
from foresightbev.sim.scenarios import FAMILIES, build_scenario
from foresightbev.sim.world import obb_corners, obb_overlap


def load_experiment(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def scaled_rig(cfg: dict, root: Path) -> CalibrationBundle:
    rig = load_rig(root / cfg["rig"])
    factor = cfg["image_scale"]
    return CalibrationBundle(tuple(c.scaled(factor) for c in rig.cameras), f"{rig.rig_name}_x{factor}")


def noise_from(cfg: dict) -> NoiseConfig:
    return NoiseConfig(**cfg.get("noise", {}))


def build_mode(mode: dict, overrides: dict | None = None) -> tuple[PipelineConfig, bool]:
    from foresightbev.config import pipeline_config_from_dict

    config = pipeline_config_from_dict({**(overrides or {}), **{"nominal_mode": mode.get("nominal", "idm"), "filter_enabled": mode.get("filter", True)}})
    if mode.get("unknown_as_free"):
        config.risk.treat_unknown_as_free = True
    return config, bool(mode.get("oracle_tracks"))


def _episode_job(job: tuple) -> dict:
    cfg, root, name, seed, mode_name, calibrator_path = job
    mode = cfg["modes"][mode_name]
    scenario = build_scenario(name, seed)
    if mode.get("front_dropout"):
        start = float(mode["front_dropout"][0])
        scenario.dropouts = [("front", start, start + float(mode["front_dropout"][1]))]
    sim = BuiltinSim(scenario, scaled_rig(cfg, Path(root)), noise_from(cfg), seed=seed)
    config, oracle = build_mode(mode, cfg.get("pipeline"))
    config.seed = seed
    calibrator = RiskCalibrator.load(calibrator_path) if mode.get("calibrated") and calibrator_path else None
    tracking = TrackingAccumulator() if mode.get("measure_perception") else None
    forecast = ForecastAccumulator(config.forecast) if mode.get("measure_perception") else None

    def observer(result, simulator):
        if tracking:
            tracking.add(result, simulator)
            if result.bundle.frame_index % 5 == 0:
                forecast.add(result, simulator)

    episode = run_episode(sim, config, mode_name, calibrator, oracle, observer)
    out = dataclasses.asdict(episode)
    if tracking:
        out["tracking"], out["forecast"] = tracking.summary(), forecast.summary()
    return out


def _worker_setup() -> None:
    import cv2

    cv2.setNumThreads(1)


def run_pool(function, jobs: list, workers: int) -> list:
    """Process pool with single-threaded numeric workers."""
    from multiprocessing import get_context

    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    with get_context("spawn").Pool(workers, initializer=_worker_setup) as pool:
        return list(pool.imap_unordered(function, jobs, chunksize=1))


def rollout_agents(agents, t0: float, steps: int, dt: float):
    """Future agent states for offline labels only. Never passed to the pipeline."""
    future = [copy.copy(a) for a in agents]
    t = t0
    for _ in range(steps):
        t += dt
        for agent in future:
            if agent.script is not None:
                agent.script(agent, t, dt)
        yield t - t0, future


def label_candidates(result, sim: BuiltinSim, config: RiskConfig, dt: float = 0.1, horizon: float = 3.0) -> dict[str, bool]:
    """Ground-truth event per candidate: its inflated footprint meets a real agent.

    Uses the same inflated footprint as the event definition, and ignores overlap
    that already exists at t=0, matching the evaluator. Agent scripts do not react
    to the ego, so this counterfactual is well defined here.
    """
    ego = result.bundle.ego
    length = config.ego_length + 2 * config.safety_margin
    width = config.ego_width + 2 * config.safety_margin
    start = obb_corners(ego.x, ego.y, ego.yaw, length, width)
    agents = [a for a in sim.agents if abs(a.x - ego.x) < 80]
    labels = {c.candidate_id: False for c in result.candidates}
    world = {c.candidate_id: (ego.ego_to_world(c.poses[:, :2]), c.poses[:, 2] + ego.yaw) for c in result.candidates}
    times = result.candidates[0].times
    for tau, future in rollout_agents(agents, sim.t, int(round(horizon / dt)), dt):
        index = min(int(np.searchsorted(times, tau)), len(times) - 1)
        boxes = [obb_corners(a.x, a.y, a.yaw, a.length, a.width) for a in future]
        for candidate_id, (xy, yaw) in world.items():
            if labels[candidate_id]:
                continue
            ego_box = obb_corners(xy[index, 0], xy[index, 1], yaw[index], length, width)
            labels[candidate_id] = any(obb_overlap(ego_box, b) and not obb_overlap(start, b) for b in boxes)
    return labels


def _calibration_job(job: tuple) -> dict:
    cfg, root, name, seed = job
    scenario = build_scenario(name, seed)
    sim = BuiltinSim(scenario, scaled_rig(cfg, Path(root)), noise_from(cfg), seed=seed)
    config, _ = build_mode(cfg["modes"]["filter"], cfg.get("pipeline"))
    config.seed = seed
    rows = []

    def observer(result, simulator):
        if result.bundle.frame_index % 2:
            return
        labels = label_candidates(result, simulator, config.risk)
        for risk in result.risks:
            rows.append((float(risk.event_score), float(labels[risk.candidate_id])))

    run_episode(sim, config, "calibration", on_frame=observer)
    return dict(scenario=name, seed=seed, rows=rows)


def run_calibration(cfg: dict, root: Path, out: Path, workers: int) -> dict:
    fit_seeds, test_seeds = cfg["calibration"]["fit_seeds"], cfg["calibration"]["test_seeds"]
    jobs = [(cfg, str(root), name, seed) for name in FAMILIES for seed in range(*fit_seeds)]
    jobs += [(cfg, str(root), name, seed) for name in FAMILIES for seed in range(*test_seeds)]
    results = run_pool(_calibration_job, jobs, workers)

    def collect(seeds):
        rows = [r for item in results if seeds[0] <= item["seed"] < seeds[1] for r in item["rows"]]
        return np.array(rows, float).reshape(-1, 2)

    fit, test = collect(fit_seeds), collect(test_seeds)
    if set(range(*fit_seeds)) & set(range(*test_seeds)):
        raise ConfigurationError(f"calibration fit {fit_seeds} and test {test_seeds} overlap")
    status = (f"isotonic on built-in simulator seeds {fit_seeds[0]}-{fit_seeds[1] - 1}, "
              "valid for these scripted synthetic scenarios only")
    calibrator = RiskCalibrator.fit(fit[:, 0], fit[:, 1], status)
    out.mkdir(parents=True, exist_ok=True)
    calibrator.save(out / "risk_calibrator.json")
    raw, labels = test[:, 0], test[:, 1]
    calibrated = np.clip(calibrator.apply(raw), 0, 1)
    rng = np.random.default_rng(0)
    episodes = [np.array(item["rows"], float).reshape(-1, 2) for item in results if test_seeds[0] <= item["seed"] < test_seeds[1] and item["rows"]]
    raw_groups = [(e[:, 0], e[:, 1]) for e in episodes]
    calibrated_groups = [(np.clip(calibrator.apply(e[:, 0]), 0, 1), e[:, 1]) for e in episodes]
    report = dict(
        event="inflated ego footprint overlaps a true agent within 3 s, excluding overlap at t=0",
        positive_class="event occurred under the counterfactual candidate",
        fit_rows=int(len(fit)), test_rows=int(len(test)), fit_seeds=fit_seeds, test_seeds=test_seeds,
        test_base_rate=float(labels.mean()),
        raw=dict(brier=M.brier(raw, labels), nll=M.nll(raw, labels), ece_10_equal_width=M.ece(raw, labels),
                 ece_interval=M.cluster_bootstrap(raw_groups, M.ece, rng), auroc=M.auroc(raw, labels),
                 auroc_interval=M.cluster_bootstrap(raw_groups, M.auroc, rng), reliability=M.reliability(raw, labels)),
        calibrated=dict(brier=M.brier(calibrated, labels), nll=M.nll(calibrated, labels),
                        ece_10_equal_width=M.ece(calibrated, labels), ece_interval=M.cluster_bootstrap(calibrated_groups, M.ece, rng), auroc=M.auroc(calibrated, labels),
                        reliability=M.reliability(calibrated, labels)),
        fitted_on="frames visited by the uncalibrated filter, so filter_calibrated sees a slightly different frame distribution",
        note="Rows within an episode are correlated, so effective sample size is below test_rows.",
    )
    (out / "calibration.json").write_text(json.dumps(report, indent=1))
    return report


class TrackingAccumulator:
    """Greedy nearest matching of confirmed tracks to agents within range."""

    def __init__(self, match_m: float = 3.0, max_range: float = 40.0):
        self.match_m, self.max_range = match_m, max_range
        self.position_errors: list[float] = []
        self.velocity_errors: list[float] = []
        self.nees: list[float] = []
        self.gt_frames = self.false_tracks = self.id_switches = 0
        self.assigned: dict[str, int] = {}

    def add(self, result, sim: BuiltinSim) -> None:
        ego = result.bundle.ego
        tracks = [t for t in result.tracks if t.confirmed and t.misses == 0]
        used: set[int] = set()
        for agent in sim.agents:
            if np.hypot(agent.x - ego.x, agent.y - ego.y) > self.max_range:
                continue
            self.gt_frames += 1
            best, distance = None, self.match_m
            for track in tracks:
                d = float(np.hypot(*(track.position - (agent.x, agent.y))))
                if track.track_id not in used and track.semantic == agent.semantic and d < distance:
                    best, distance = track, d
            if best is None:
                continue
            used.add(best.track_id)
            self.position_errors.append(distance)
            velocity = best.velocity - (agent.vx, agent.vy)
            self.velocity_errors.append(float(np.hypot(*velocity)))
            error = np.array([agent.x, agent.y, agent.vx, agent.vy]) - np.concatenate([best.position, best.velocity])
            self.nees.append(float(error @ np.linalg.solve(best.covariance + np.eye(4) * 1e-9, error)))
            previous = self.assigned.get(agent.name)
            self.id_switches += previous is not None and previous != best.track_id
            self.assigned[agent.name] = best.track_id
        self.false_tracks += len(tracks) - len(used)

    def summary(self) -> dict:
        return dict(gt_frames=self.gt_frames, matched=len(self.position_errors), false_tracks=self.false_tracks,
                    id_switches=self.id_switches, position_error_sum=float(sum(self.position_errors)),
                    velocity_error_sum=float(sum(self.velocity_errors)), nees_sum=float(sum(self.nees)))


class ForecastAccumulator:
    """Dynamic-agent footprint overlap per horizon, against true future positions."""

    def __init__(self, forecast_config):
        self.config = forecast_config
        count = int(round(forecast_config.horizon_s / forecast_config.step_s))
        self.intersection = {m: np.zeros(count) for m in ("forecast", "persistence")}
        self.union = {m: np.zeros(count) for m in ("forecast", "persistence")}
        self.brier_sum = np.zeros(count)
        self.brier_cells = np.zeros(count)
        self.displacement_sum = np.zeros(count)
        self.displacement_count = np.zeros(count)

    def add(self, result, sim: BuiltinSim) -> None:
        ego = result.bundle.ego
        forecast = result.forecast
        grid = forecast.grid
        moving = [a for a in sim.agents if a.script is not None]
        if not moving:
            return
        persistence = np.zeros(grid.shape, bool)
        tracked = [t for t in result.tracks if t.confirmed]
        for track in tracked:
            position, velocity, _ = track_to_ego(track, ego)
            yaw = float(np.arctan2(velocity[1], velocity[0])) if np.hypot(*velocity) > 1 else 0.0
            persistence |= rasterize(footprint_polygon(*position, yaw, track.length, track.width), grid)
        steps = int(round(self.config.step_s / sim.physics_dt))
        k = -1
        for n, (tau, future) in enumerate(rollout_agents(sim.agents, sim.t, steps * len(forecast.horizons), sim.physics_dt), 1):
            if n % steps:
                continue
            k += 1
            truth = np.zeros(grid.shape, bool)
            for agent in future:
                corners = ego.world_to_ego(obb_corners(agent.x, agent.y, agent.yaw, agent.length, agent.width))
                truth |= rasterize(corners, grid)
            predicted = forecast.occupancy_prob[k] * ~forecast.static_occupied
            mask = predicted >= 0.5
            for name, value in (("forecast", mask), ("persistence", persistence)):
                self.intersection[name][k] += int((value & truth).sum())
                self.union[name][k] += int((value | truth).sum())
            region = truth | (predicted > 0)
            self.brier_sum[k] += float(((predicted - truth) ** 2)[region].sum())
            self.brier_cells[k] += int(region.sum())
            for track in tracked:
                position, velocity, cov = track_to_ego(track, ego)
                if is_stationary(velocity, cov, self.config):
                    velocity = velocity * 0
                best = None
                for agent in future:
                    now = ego.world_to_ego(np.array([agent.x, agent.y]))
                    d = float(np.hypot(*(position + velocity * tau - now)))
                    if best is None or d < best:
                        best = d
                if best is not None and best < 8.0:
                    self.displacement_sum[k] += best
                    self.displacement_count[k] += 1

    def summary(self) -> dict:
        return dict(intersection={k: v.tolist() for k, v in self.intersection.items()},
                    union={k: v.tolist() for k, v in self.union.items()}, brier_sum=self.brier_sum.tolist(),
                    brier_cells=self.brier_cells.tolist(), displacement_sum=self.displacement_sum.tolist(),
                    displacement_count=self.displacement_count.tolist())


def run_closed_loop(cfg: dict, root: Path, out: Path, workers: int, calibrator: Path | None) -> list[dict]:
    seeds = range(*cfg["closed_loop_seeds"])
    jobs = [(cfg, str(root), name, seed, mode, str(calibrator) if calibrator else None)
            for mode in cfg["modes"] for name in FAMILIES for seed in seeds]
    episodes = run_pool(_episode_job, jobs, workers)
    episodes.sort(key=lambda e: (e["mode"], e["scenario"], e["seed"]))
    out.mkdir(parents=True, exist_ok=True)
    (out / "episodes.json").write_text(json.dumps(episodes))
    return episodes


def summarize(episodes: list[dict]) -> dict:
    rng = np.random.default_rng(0)
    modes = sorted({e["mode"] for e in episodes})
    families = sorted({e["scenario"] for e in episodes})

    def block(rows: list[dict]) -> dict:
        def column(key):
            return np.array([float(r[key]) for r in rows])
        frames = column("frames").sum()
        return dict(
            collision_rate=M.bootstrap_mean(column("collided"), rng),
            route_completion=M.bootstrap_mean(column("route_completion"), rng),
            goal_rate=M.bootstrap_mean(column("reached_goal"), rng),
            freeze_rate=M.bootstrap_mean(column("freeze"), rng),
            mean_speed=M.bootstrap_mean(column("mean_speed"), rng),
            max_decel=M.bootstrap_mean(column("max_decel"), rng),
            rms_jerk=M.bootstrap_mean(column("rms_jerk"), rng),
            intervention_fraction=float((column("modify_frames").sum() + column("fallback_frames").sum()) / max(frames, 1)),
            fallback_fraction=float(column("fallback_frames").sum() / max(frames, 1)),
            emergency_fraction=float(column("emergency_frames").sum() / max(frames, 1)),
            episodes=len(rows))

    summary = {"modes": {}}
    for mode in modes:
        rows = [e for e in episodes if e["mode"] == mode]
        summary["modes"][mode] = dict(overall=block(rows), by_scenario={f: block([r for r in rows if r["scenario"] == f]) for f in families},
                                      pipeline_ms=float(np.mean([r["pipeline_ms"] for r in rows])))
    measured = [e for e in episodes if "tracking" in e]
    if measured:
        summary["perception"] = perception_summary(measured)
    return summary


def perception_summary(rows: list[dict]) -> dict:
    occ = {k: sum(r["occupancy_stats"][k] for r in rows) for k in rows[0]["occupancy_stats"]}
    t = {k: sum(r["tracking"][k] for r in rows) for k in rows[0]["tracking"]}
    matched = max(t["matched"], 1)
    forecast = {}
    arrays = {k: np.sum([r["forecast"][k] for r in rows], axis=0) for k in ("brier_sum", "brier_cells", "displacement_sum", "displacement_count")}
    for name in ("forecast", "persistence"):
        i = np.sum([r["forecast"]["intersection"][name] for r in rows], axis=0)
        u = np.sum([r["forecast"]["union"][name] for r in rows], axis=0)
        forecast[name] = (i / np.maximum(u, 1)).tolist()
    return dict(
        occupancy=dict(false_free_rate=occ["false_free"] / max(occ["gt_occ"], 1), gt_marked_occupied=occ["occ_on_gt"] / max(occ["gt_occ"], 1),
                       gt_marked_unknown_or_outside=occ["unknown_on_gt"] / max(occ["gt_occ"], 1),
                       occupied_precision=occ["pred_occ_true"] / max(occ["pred_occ"], 1)),
        tracking=dict(recall=t["matched"] / max(t["gt_frames"], 1), mean_position_error_m=t["position_error_sum"] / matched,
                      mean_velocity_error_mps=t["velocity_error_sum"] / matched, id_switches=t["id_switches"],
                      false_track_frames=t["false_tracks"], mean_nees_4dof=t["nees_sum"] / matched,
                      nees_note="A consistent 4-state filter has mean NEES near 4."),
        forecast=dict(horizons_s=[(i + 1) * 0.25 for i in range(len(forecast["forecast"]))], agent_iou=forecast,
                      brier_on_relevant_cells=(arrays["brier_sum"] / np.maximum(arrays["brier_cells"], 1)).tolist(),
                      mean_displacement_error_m=(arrays["displacement_sum"] / np.maximum(arrays["displacement_count"], 1)).tolist(),
                      method_note="agent_iou compares forecast footprints and frozen tracked footprints with true future boxes."),
    )
