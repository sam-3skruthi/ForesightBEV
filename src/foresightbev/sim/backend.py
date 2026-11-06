"""Stage 7 simulation interface. Built-in and CARLA backends share this protocol."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from foresightbev.contracts import (
    BEVGrid, CalibrationBundle, CameraFrame, CameraStatus, EgoState, FrameBundle, Provenance, TrackState, TrajectoryCandidate,
)
from foresightbev.risk.trajectories import RouteContext
from foresightbev.sim.render import CameraRenderer, NoiseConfig, apply_noise
from foresightbev.sim.scenarios import Scenario
from foresightbev.sim.world import Agent, obb_corners, obb_distance, obb_overlap


class SimBackend(Protocol):
    """What the closed loop needs from any simulator."""

    calibration: CalibrationBundle

    def observe(self) -> FrameBundle: ...
    def route(self) -> RouteContext: ...
    def apply(self, trajectory: TrajectoryCandidate, duration: float) -> None: ...
    def status(self) -> dict: ...


class BuiltinSim:
    """Kinematic closed-loop backend. Ground truth stays on this side of the API."""

    def __init__(self, scenario: Scenario, calibration: CalibrationBundle, noise: NoiseConfig | None = None,
                 physics_dt: float = 0.05, seed: int = 0):
        self.scenario = scenario
        self.calibration = calibration
        self.noise = noise or NoiseConfig()
        self.physics_dt = physics_dt
        self.rng = np.random.default_rng([seed, scenario.seed])
        self.renderers = [CameraRenderer(c) for c in calibration.cameras]
        self.t = 0.0
        self.frame = 0
        self.collided_with: str | None = None
        self.min_gap = np.inf
        self.ego_history: list[tuple[float, float, float, float]] = []

    @property
    def agents(self) -> list[Agent]:
        return self.scenario.agents

    def ego_state(self) -> EgoState:
        e = self.scenario.ego
        return EgoState(self.t, e.x, e.y, e.yaw, e.speed, e.yaw_rate)

    def observe(self) -> FrameBundle:
        scenario = self.scenario
        boxes = [a for a in self.agents] + scenario.layout.buildings
        frames = {}
        for renderer in self.renderers:
            name = renderer.cam.name
            lost = any(cam == name and start <= self.t < end for cam, start, end in scenario.dropouts)
            if lost:
                frames[name] = CameraFrame(name, self.t, None, CameraStatus.DROPPED)
                continue
            labels = apply_noise(renderer.render(scenario.ego, scenario.layout, boxes), self.noise, self.rng)
            frames[name] = CameraFrame(name, self.t, labels, CameraStatus.OK)
        bundle = FrameBundle(f"{scenario.name}_{scenario.seed}", self.frame, self.t, frames, self.calibration,
                             self.ego_state(), Provenance.SIMULATED)
        self.frame += 1
        return bundle

    def route(self) -> RouteContext:
        s = self.scenario
        return RouteContext(s.desired_speed, s.ego.y, s.ego.yaw, s.layout.lanes, 0.0)

    def _check_contact(self) -> None:
        e = self.scenario.ego
        ego_box = obb_corners(e.x, e.y, e.yaw, e.length, e.width)
        for agent in self.agents:
            if abs(agent.x - e.x) > 12 or abs(agent.y - e.y) > 8:
                continue
            box = obb_corners(agent.x, agent.y, agent.yaw, agent.length, agent.width)
            gap = obb_distance(ego_box, box)
            self.min_gap = min(self.min_gap, gap)
            if self.collided_with is None and obb_overlap(ego_box, box):
                self.collided_with = agent.name

    def apply(self, trajectory: TrajectoryCandidate, duration: float) -> None:
        """Execute a plan open-loop for ``duration`` by following its timed poses."""
        e = self.scenario.ego
        start = (e.x, e.y, e.yaw)
        c, s = np.cos(e.yaw), np.sin(e.yaw)
        steps = int(round(duration / self.physics_dt))
        for i in range(1, steps + 1):
            tau = i * self.physics_dt
            pose = np.array([np.interp(tau, trajectory.times, trajectory.poses[:, j]) for j in range(4)])
            previous_yaw = e.yaw
            e.x = start[0] + c * pose[0] - s * pose[1]
            e.y = start[1] + s * pose[0] + c * pose[1]
            e.yaw = start[2] + pose[2]
            e.speed = float(pose[3])
            e.yaw_rate = (e.yaw - previous_yaw) / self.physics_dt
            self.t += self.physics_dt
            for agent in self.agents:
                if agent.script is not None:
                    agent.script(agent, self.t, self.physics_dt)
            self._check_contact()
            self.ego_history.append((self.t, e.x, e.y, e.speed))

    def status(self) -> dict:
        return {"t": self.t, "collided_with": self.collided_with, "min_gap": self.min_gap,
                "ego_x": float(self.scenario.ego.x), "reached_goal": bool(self.scenario.ego.x >= self.scenario.goal_x)}

    def ground_truth_occupancy(self, grid: BEVGrid) -> np.ndarray:
        """Boolean [nx, ny] mask of every box footprint, in the current ego frame."""
        from foresightbev.forecast.forecaster import rasterize
        ego = self.ego_state()
        mask = np.zeros(grid.shape, bool)
        for box in self.agents + self.scenario.layout.buildings:
            if abs(box.x - ego.x) > 120:
                continue
            corners = ego.world_to_ego(obb_corners(box.x, box.y, box.yaw, box.length, box.width))
            mask |= rasterize(corners, grid)
        return mask

    def oracle_tracks(self) -> list[TrackState]:
        """Ground-truth agents as tracks, for the upper-bound ablation only."""
        tracks = []
        for i, a in enumerate(self.agents):
            tracks.append(TrackState(i + 1, self.t, a.semantic, np.array([a.x, a.y]), np.array([a.vx, a.vy]),
                                     np.eye(4) * 1e-4, a.length, a.width, a.height, 10, 10, 0, True))
        return tracks
