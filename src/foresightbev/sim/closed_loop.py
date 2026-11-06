"""Closed-loop runner: observe, run pipeline, apply decision, repeat."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from foresightbev.contracts import DecisionStatus, ReasonCode
from foresightbev.pipeline import FrameResult, ForesightPipeline, PipelineConfig
from foresightbev.risk.calibration import RiskCalibrator
from foresightbev.sim.backend import BuiltinSim

CONTROL_DT = 0.2


@dataclass
class Episode:
    scenario: str
    seed: int
    mode: str
    collided: bool
    collided_with: str | None
    reached_goal: bool
    progress_m: float
    route_completion: float
    duration_s: float
    min_gap_m: float
    frames: int
    pass_frames: int
    modify_frames: int
    fallback_frames: int
    emergency_frames: int
    stopped_frames: int
    max_decel: float
    rms_jerk: float
    mean_speed: float
    freeze: bool
    pipeline_ms: float
    occupancy_stats: dict = field(default_factory=dict)


def run_episode(sim: BuiltinSim, config: PipelineConfig, mode: str, calibrator: RiskCalibrator | None = None,
                oracle: bool = False, on_frame=None, max_seconds: float | None = None) -> Episode:
    scenario = sim.scenario
    pipeline = ForesightPipeline(sim.calibration, config, calibrator)
    counts = {DecisionStatus.PASS: 0, DecisionStatus.MODIFY: 0, DecisionStatus.FALLBACK: 0}
    emergency = stopped = frames = 0
    speeds: list[float] = []
    ms: list[float] = []
    stats = OccupancyAccumulator()
    limit = max_seconds or scenario.duration_s
    while sim.t < limit - 1e-6:
        bundle = sim.observe()
        kwargs = {}
        if oracle:
            kwargs["oracle_tracks"] = sim.oracle_tracks()
        result = pipeline.step(bundle, sim.route(), **kwargs)
        ms.append(sum(result.timings_ms.values()))
        stats.add(result, sim)
        if on_frame is not None:
            on_frame(result, sim)
        decision = result.decision
        counts[decision.status] += 1
        emergency += ReasonCode.FALLBACK_EMERGENCY_STOP in decision.reasons
        stopped += int(scenario.ego.speed < 0.2)
        frames += 1
        speeds.append(scenario.ego.speed)
        sim.apply(decision.selected, CONTROL_DT)
        if sim.collided_with is not None or scenario.ego.x >= scenario.goal_x:
            break
    speeds_a = np.array(speeds + [scenario.ego.speed])
    accel = np.diff(speeds_a) / CONTROL_DT
    jerk = np.diff(accel) / CONTROL_DT if len(accel) > 1 else np.zeros(1)
    progress = float(scenario.ego.x)
    ideal = min(scenario.goal_x, scenario.desired_speed * limit)
    tail = np.array(speeds[-15:])
    freeze = sim.collided_with is None and len(tail) == 15 and (tail < 0.2).all() and scenario.ego.x < scenario.goal_x
    return Episode(scenario.name, scenario.seed, mode, sim.collided_with is not None, sim.collided_with,
                   bool(scenario.ego.x >= scenario.goal_x), progress, float(min(1.0, progress / ideal)), sim.t,
                   float(sim.min_gap), frames, counts[DecisionStatus.PASS], counts[DecisionStatus.MODIFY],
                   counts[DecisionStatus.FALLBACK], emergency, stopped, float(max(0.0, -accel.min())) if len(accel) else 0.0,
                   float(np.sqrt((jerk**2).mean())), float(np.mean(speeds)), bool(freeze), float(np.mean(ms)), stats.summary())


class OccupancyAccumulator:
    """Per-episode occupancy counts against simulator ground truth."""

    def __init__(self) -> None:
        self.counts = dict(gt_occ=0, false_free=0, unknown_on_gt=0, occ_on_gt=0, pred_occ=0, pred_occ_true=0)

    def add(self, result: FrameResult, sim: BuiltinSim) -> None:
        from foresightbev.contracts import CellState as S
        state = result.occupancy.bev_state
        truth = sim.ground_truth_occupancy(result.occupancy.grid.bev)
        truth &= state != S.INVALID
        c = self.counts
        c["gt_occ"] += int(truth.sum())
        c["false_free"] += int((truth & (state == S.FREE)).sum())
        c["unknown_on_gt"] += int((truth & np.isin(state, [S.UNKNOWN, S.OUTSIDE_FOV])).sum())
        c["occ_on_gt"] += int((truth & (state == S.OCCUPIED)).sum())
        c["pred_occ"] += int((state == S.OCCUPIED).sum())
        c["pred_occ_true"] += int((truth & (state == S.OCCUPIED)).sum())

    def summary(self) -> dict:
        return dict(self.counts)
