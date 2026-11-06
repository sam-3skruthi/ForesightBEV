"""Stage 5 baseline: Monte Carlo constant-velocity forecast of tracked footprints.

This is a tracking-based forecast, not learned 4D occupancy. Whole-agent joint
samples are drawn from each Kalman posterior, plus white-acceleration noise, and
rolled out. Static occupied cells persist. Keeping joint samples lets Stage 6
evaluate trajectory events without assuming independent voxels.

Spread comes from the tracker's motion model. It is uncalibrated until Stage 9
measures it. The persistence baseline is exposed for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from foresightbev.contracts import (
    BEVGrid, CellState, EgoState, ForecastVolume, OccupancyFrame, Provenance, TrackState,
)


@dataclass
class ForecastConfig:
    horizon_s: float = 3.0
    step_s: float = 0.25
    n_samples: int = 32
    accel_std: float = 0.3
    stationary_speed: float = 1.0
    stationary_sigmas: float = 2.0
    max_ambiguous_speed: float = 6.0
    include_unconfirmed: bool = False
    footprint_inflation_m: float = 0.1
    track_exclusion_margin_m: float = 0.75


def footprint_polygon(x: float, y: float, yaw: float, length: float, width: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    half = np.array([[length, width], [length, -width], [-length, -width], [-length, width]]) / 2
    return np.stack([x + c * half[:, 0] - s * half[:, 1], y + s * half[:, 0] + c * half[:, 1]], -1)


def polygon_cells(polygon: np.ndarray, grid: BEVGrid) -> tuple[np.ndarray, np.ndarray]:
    """Grid indices (ix, iy) covered by a convex polygon, clipped to the grid."""
    fx = (polygon[:, 0] - grid.x_min) / grid.resolution - 0.5
    fy = (polygon[:, 1] - grid.y_min) / grid.resolution - 0.5
    x0, y0 = int(np.floor(fx.min())), int(np.floor(fy.min()))
    x1, y1 = int(np.ceil(fx.max())) + 1, int(np.ceil(fy.max())) + 1
    if x1 <= 0 or y1 <= 0 or x0 >= grid.nx or y0 >= grid.ny:
        return np.empty(0, int), np.empty(0, int)
    window = np.zeros((x1 - x0, y1 - y0), np.uint8)
    points = np.stack([fy - y0, fx - x0], -1)
    cv2.fillConvexPoly(window, np.round(points * 16).astype(np.int32), 1, lineType=cv2.LINE_8, shift=4)
    ix, iy = np.nonzero(window)
    ix, iy = ix + x0, iy + y0
    keep = (ix >= 0) & (ix < grid.nx) & (iy >= 0) & (iy < grid.ny)
    return ix[keep], iy[keep]


def rasterize(polygon: np.ndarray, grid: BEVGrid) -> np.ndarray:
    """Fill a polygon into a boolean [nx, ny] mask."""
    mask = np.zeros(grid.shape, bool)
    mask[polygon_cells(polygon, grid)] = True
    return mask


def track_to_ego(track: TrackState, ego: EgoState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean position, velocity and 4x4 covariance rotated into the ego frame."""
    c, s = np.cos(ego.yaw), np.sin(ego.yaw)
    R = np.array([[c, s], [-s, c]])
    block = np.block([[R, np.zeros((2, 2))], [np.zeros((2, 2)), R]])
    return ego.world_to_ego(track.position), R @ track.velocity, block @ track.covariance @ block.T


def is_stationary(velocity: np.ndarray, covariance: np.ndarray, cfg: ForecastConfig) -> bool:
    """Heuristic: speed is small and indistinguishable from zero given the filter."""
    speed = float(np.hypot(*velocity))
    sigma = float(np.sqrt(np.trace(covariance[2:, 2:]) / 2))
    return speed < cfg.stationary_speed or (speed < cfg.stationary_sigmas * sigma and speed < cfg.max_ambiguous_speed)


class ConstantVelocityForecaster:
    method = "track_constant_velocity_monte_carlo_v1"

    def __init__(self, grid: BEVGrid, config: ForecastConfig | None = None, seed: int = 0):
        self.grid = grid
        self.config = config or ForecastConfig()
        self.rng = np.random.default_rng(seed)
        half_x, half_y = grid.nx * grid.resolution / 2, grid.ny * grid.resolution / 2
        self.origin_grid = BEVGrid(-half_x, half_x, -half_y, half_y, grid.resolution)

    def horizons(self) -> np.ndarray:
        cfg = self.config
        count = int(round(cfg.horizon_s / cfg.step_s))
        return (np.arange(count) + 1) * cfg.step_s

    def _static_layer(self, occupancy: OccupancyFrame, tracks: list[TrackState], ego: EgoState) -> np.ndarray:
        """Occupied cells not explained by a track persist in place, so nothing is lost."""
        static = occupancy.bev_state == CellState.OCCUPIED
        margin = self.config.track_exclusion_margin_m
        for track in tracks:
            position, velocity, _ = track_to_ego(track, ego)
            yaw = float(np.arctan2(velocity[1], velocity[0])) if np.hypot(*velocity) > 1.0 else 0.0
            extent = max(track.length, track.width) if np.hypot(*velocity) <= 1.0 else track.length
            exclude = rasterize(footprint_polygon(*position, yaw, extent + 2 * margin,
                                                  max(track.width, extent if np.hypot(*velocity) <= 1.0 else 0)
                                                  + 2 * margin), self.grid)
            static[exclude] = False
        return static

    def forecast(self, occupancy: OccupancyFrame, tracks: list[TrackState], ego: EgoState) -> ForecastVolume:
        if occupancy.timestamp > ego.timestamp + 1e-9:
            raise ValueError("occupancy input is later than the forecast reference time")
        cfg = self.config
        horizons = self.horizons()
        grid = self.grid
        active = [t for t in tracks if (t.confirmed or cfg.include_unconfirmed)]
        for track in active:
            if track.timestamp > ego.timestamp + 1e-9:
                raise ValueError(f"track {track.track_id} is later than the forecast reference time")
        static = self._static_layer(occupancy, active, ego)
        unobserved = np.isin(occupancy.bev_state, [CellState.UNKNOWN, CellState.OUTSIDE_FOV])

        K, S = len(horizons), cfg.n_samples
        agent_samples = [np.zeros((S, grid.nx, grid.ny), bool) for _ in range(K)]
        counts = np.zeros((K, grid.nx, grid.ny), np.float32)
        flow = np.zeros((K, grid.nx, grid.ny, 2), np.float32)
        for track in active:
            mean_p, mean_v, cov = track_to_ego(track, ego)
            state = np.concatenate([mean_p, mean_v])
            samples = self.rng.multivariate_normal(state, cov + np.eye(4) * 1e-9, size=S)
            accel = self.rng.normal(0.0, cfg.accel_std, size=(S, 2))
            if is_stationary(mean_v, cov, cfg):
                samples[:, 2:] = 0.0
                accel[:] = 0.0
            lag = max(0.0, ego.timestamp - track.timestamp)
            length = track.length + cfg.footprint_inflation_m
            width = track.width + cfg.footprint_inflation_m
            mean_yaw = float(np.arctan2(mean_v[1], mean_v[0])) if np.hypot(*mean_v) > 1.0 else 0.0
            # One footprint template per track, at its mean heading, translated to each
            # sample. Per-sample heading spread is deliberately ignored for speed.
            cells = np.stack(polygon_cells(footprint_polygon(0.0, 0.0, mean_yaw, length, width), self.origin_grid), -1)
            offset = cells - (self.origin_grid.nx // 2, self.origin_grid.ny // 2)
            for k, h in enumerate(horizons):
                t = h + lag
                position = samples[:, :2] + samples[:, 2:] * t + 0.5 * accel * t * t
                velocity = samples[:, 2:] + accel * t
                cx, cy = grid.to_index(position[:, 0], position[:, 1])
                ix = cx[:, None] + offset[:, 0]
                iy = cy[:, None] + offset[:, 1]
                keep = (ix >= 0) & (ix < grid.nx) & (iy >= 0) & (iy < grid.ny)
                sample_index = np.broadcast_to(np.arange(S)[:, None], ix.shape)[keep]
                ix, iy = ix[keep], iy[keep]
                agent_samples[k][sample_index, ix, iy] = True
                np.add.at(counts[k], (ix, iy), 1.0)
                np.add.at(flow[k, ..., 0], (ix, iy), velocity[sample_index, 0])
                np.add.at(flow[k, ..., 1], (ix, iy), velocity[sample_index, 1])
        probability = np.stack([agent_samples[k].mean(0) for k in range(K)]).astype(np.float32)
        probability[:, static] = 1.0
        hits = np.maximum(counts, 1)[..., None]
        return ForecastVolume(ego.timestamp, occupancy.timestamp, horizons, grid, probability, flow / hits, static,
                              unobserved, agent_samples, S, self.method,
                              "monte_carlo_from_kalman_posterior_uncalibrated_until_stage9", Provenance.CLASSICAL)


def persistence_forecast(occupancy: OccupancyFrame, horizons: np.ndarray) -> np.ndarray:
    """Static-world baseline: every occupied cell stays put at every horizon."""
    return np.repeat((occupancy.bev_state == CellState.OCCUPIED)[None].astype(np.float32), len(horizons), axis=0)
