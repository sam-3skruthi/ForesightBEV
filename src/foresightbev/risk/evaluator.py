"""Stage 6 trajectory-event evaluation against a ForecastVolume.

Event: the ego footprint, swept over each horizon interval, overlaps a forecast
occupied cell within the horizon. ``event_score`` is the fraction of whole-scene
forecast samples in which this happens. Because each sample is a joint rollout,
spatial and temporal correlation within an agent is kept, and no voxel independence
is assumed. The score inherits the constant-velocity model and is not a collision
probability unless a calibrator was fitted and is applied.

Unknown policy: at the next replan the ego must still be able to stop before any point that a
hypothetical agent could reach from unobserved space in time. The hypothetical
agent moves at ``phantom_speed``, so the lateral search band around the path widens
with arrival time. Occlusion behind a moving tracked agent is exempt, since that
agent is itself clearing the space. It is a stated policy, not a derived guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from foresightbev.contracts import ForecastVolume, TrackState, TrajectoryCandidate, TrajectoryRisk, EgoState
from foresightbev.forecast.forecaster import footprint_polygon, polygon_cells, track_to_ego
from foresightbev.risk.calibration import RiskCalibrator

EVENT = "ego_footprint_swept_overlap_with_forecast_occupied_cell_within_horizon"


@dataclass
class RiskConfig:
    ego_length: float = 4.6
    ego_width: float = 2.0
    safety_margin: float = 0.3
    substep_s: float = 0.1
    min_unknown_margin: float = 0.3
    max_unknown_margin: float = 2.5
    phantom_speed: float = 1.5
    unknown_min_cells: int = 6
    blind_zone_x: float = 4.5
    blind_zone_y: float = 3.0
    path_step_m: float = 1.0
    stop_reaction_s: float = 0.2
    stop_decel: float = 4.0
    treat_unknown_as_free: bool = False
    moving_track_speed: float = 2.0
    replan_s: float = 0.4


def stopping_distance(speed, cfg: RiskConfig):
    return speed * cfg.stop_reaction_s + speed**2 / (2 * cfg.stop_decel)


class RiskEvaluator:
    def __init__(self, config: RiskConfig | None = None, calibrator: RiskCalibrator | None = None):
        self.config = config or RiskConfig()
        self.calibrator = calibrator

    def _swept_cells(self, candidate: TrajectoryCandidate, forecast: ForecastVolume) -> list[tuple[np.ndarray, np.ndarray]]:
        cfg = self.config
        grid = forecast.grid
        length, width = cfg.ego_length + 2 * cfg.safety_margin, cfg.ego_width + 2 * cfg.safety_margin
        step = max(1, int(round(cfg.substep_s / (candidate.times[1] - candidate.times[0]))))
        out = []
        previous = 0.0
        # Space the ego already occupies at t=0 is excluded: no candidate can change
        # it, so it must not count against any of them. Events cover new ground only.
        start = candidate.poses[0]
        current = polygon_cells(footprint_polygon(start[0], start[1], start[2], length, width), grid)
        for horizon in forecast.horizons:
            index = np.nonzero((candidate.times > previous) & (candidate.times <= horizon + 1e-9))[0][::step]
            mask = np.zeros(grid.shape, bool)
            for pose in candidate.poses[index]:
                mask[polygon_cells(footprint_polygon(pose[0], pose[1], pose[2], length, width), grid)] = True
            mask[current] = False
            out.append(np.nonzero(mask))
            previous = horizon
        return out

    def _corridor_unobserved(self, forecast: ForecastVolume, tracks: list[TrackState], ego: EgoState) -> np.ndarray:
        """Unobserved cells relevant to the stopping rule."""
        cfg = self.config
        X, Y = forecast.grid.cell_centers()
        mask = forecast.unobserved & ~((np.abs(X) <= cfg.blind_zone_x) & (np.abs(Y) <= cfg.blind_zone_y))
        azimuth, distance = np.arctan2(Y, X), np.hypot(X, Y)
        for track in tracks:
            position, velocity, _ = track_to_ego(track, ego)
            if not track.confirmed or np.hypot(*velocity) < cfg.moving_track_speed:
                continue
            r = float(np.hypot(*position))
            half = np.arctan2(max(track.length, track.width) / 2 + 0.5, max(r, 1.0))
            delta = np.angle(np.exp(1j * (azimuth - np.arctan2(position[1], position[0]))))
            mask &= ~((np.abs(delta) <= half) & (distance >= r - track.length))
        return mask

    def _distance_to_unknown(self, candidate: TrajectoryCandidate, unobserved: np.ndarray, forecast: ForecastVolume) -> float:
        cfg = self.config
        poses = candidate.poses
        steps = np.hypot(*np.diff(poses[:, :2], axis=0).T)
        path = np.concatenate([[0.0], np.cumsum(steps)])
        end = float(path[-1])
        length = cfg.ego_length
        speeds = poses[:, 3]
        arrival = np.concatenate([[0.0], np.cumsum(steps / np.maximum((speeds[:-1] + speeds[1:]) / 2, 0.5))])
        final_speed = max(float(speeds[-1]), 0.5)
        distance = 0.0
        reach = stopping_distance(float(poses[:, 3].max()), cfg) + end + cfg.path_step_m
        while distance <= reach:
            if distance <= end:
                index = min(int(np.searchsorted(path, distance)), len(path) - 1)
                x, y, yaw = poses[index, :3]
                time_there = float(arrival[index])
            else:
                x, y, yaw = poses[-1, :3]
                x, y = x + (distance - end) * np.cos(yaw), y + (distance - end) * np.sin(yaw)
                time_there = float(arrival[-1]) + (distance - end) / final_speed
            margin = float(np.clip(cfg.phantom_speed * time_there, cfg.min_unknown_margin, cfg.max_unknown_margin))
            width = cfg.ego_width + 2 * margin
            ix, iy = polygon_cells(footprint_polygon(x, y, yaw, length, width), forecast.grid)
            if len(ix) == 0 or int(unobserved[ix, iy].sum()) >= cfg.unknown_min_cells:
                return distance
            distance += cfg.path_step_m
        return np.inf

    def evaluate(self, candidate: TrajectoryCandidate, forecast: ForecastVolume, unobserved: np.ndarray) -> TrajectoryRisk:
        cfg = self.config
        horizons = forecast.horizons
        K = len(horizons)
        swept = self._swept_cells(candidate, forecast)
        hits = np.zeros((forecast.n_samples, K), bool)
        static_time = None
        for k, (ix, iy) in enumerate(swept):
            if len(ix) == 0:
                continue
            hits[:, k] = forecast.agent_samples[k][:, ix, iy].any(axis=1)
            if static_time is None and forecast.static_occupied[ix, iy].any():
                static_time = float(horizons[k])
        cumulative = np.logical_or.accumulate(hits, axis=1).mean(axis=0)
        if static_time is not None:
            cumulative[horizons >= static_time] = 1.0
        score = float(cumulative[-1])
        first = np.nonzero(cumulative > 0)[0]

        poses = candidate.poses
        path = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(poses[:, :2], axis=0).T))])
        all_cells = np.concatenate([c[0] for c in swept]) if swept else np.empty(0, int)
        if cfg.treat_unknown_as_free:
            exposure, distance, violation = 0.0, np.inf, False
        else:
            cells_y = np.concatenate([c[1] for c in swept])
            exposure = float(unobserved[all_cells, cells_y].mean()) if len(all_cells) else 1.0
            distance = self._distance_to_unknown(candidate, unobserved, forecast)
            # The plan is replaced at the next cycle, so the ego only has to keep the
            # option of stopping short of that point from where it will be by then.
            limit = candidate.times <= cfg.replan_s + 1e-9
            violation = bool((path[limit] + stopping_distance(poses[limit, 3], cfg)).max() > distance)

        calibrated, status = None, "uncalibrated_model_frequency"
        if self.calibrator is not None:
            calibrated, status = float(self.calibrator.apply(score)), self.calibrator.status
        return TrajectoryRisk(candidate.candidate_id, EVENT, score, calibrated, status, cumulative,
                              static_time is not None, float(horizons[first[0]]) if len(first) else None,
                              exposure, distance, float(stopping_distance(poses[0, 3], cfg)), violation)

    def conflict_metrics(self, candidate: TrajectoryCandidate, tracks: list[TrackState], ego: EgoState, horizons: np.ndarray) -> tuple[float | None, float | None, int | None]:
        """Gap to mean constant-velocity agent paths: clearance, time to conflict, closest track.

        The gap is measured in the ego heading frame against each agent's half-diagonal, so it
        is a conservative approximation. Time to conflict is the first horizon with a zero gap.
        It is a kinematic quantity under constant velocity, not a probability.
        """
        active = [t for t in tracks if t.confirmed]
        if not active:
            return None, None, None
        cfg = self.config
        poses = np.stack([np.interp(horizons, candidate.times, candidate.poses[:, j]) for j in range(3)], -1)
        best, conflict, closest = np.inf, None, None
        for track in active:
            position, velocity, _ = track_to_ego(track, ego)
            future = position + velocity[None] * horizons[:, None]
            d = future - poses[:, :2]
            c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
            along, across = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
            radius = float(np.hypot(track.length, track.width)) / 2
            gap = np.maximum(np.abs(along) - cfg.ego_length / 2 - radius, np.abs(across) - cfg.ego_width / 2 - radius)
            gap = np.maximum(gap, 0.0)
            if float(gap.min()) < best:
                best, closest = float(gap.min()), track.track_id
            zero = np.nonzero(gap <= 0.0)[0]
            if len(zero) and (conflict is None or horizons[zero[0]] < conflict):
                conflict = float(horizons[zero[0]])
        return best, conflict, closest

    def evaluate_all(self, candidates: list[TrajectoryCandidate], forecast: ForecastVolume,
                     tracks: list[TrackState], ego: EgoState) -> list[TrajectoryRisk]:
        unobserved = self._corridor_unobserved(forecast, tracks, ego)
        out = []
        for candidate in candidates:
            risk = self.evaluate(candidate, forecast, unobserved)
            risk.min_clearance_m, risk.time_to_conflict_s, risk.closest_track_id = self.conflict_metrics(candidate, tracks, ego, forecast.horizons)
            out.append(risk)
        return out
