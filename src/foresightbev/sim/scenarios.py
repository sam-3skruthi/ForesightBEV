"""Scripted scenario families. Parameters are drawn from a seeded generator."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from foresightbev.contracts import SemanticClass as C
from foresightbev.sim.world import LANE_WIDTH, Agent, EgoVehicle, RoadLayout, constant_speed, make_buildings, move


@dataclass
class Scenario:
    name: str
    seed: int
    ego: EgoVehicle
    layout: RoadLayout
    agents: list[Agent]
    desired_speed: float = 10.0
    duration_s: float = 14.0
    goal_x: float = 110.0
    dropouts: list[tuple[str, float, float]] = field(default_factory=list)


def _car(name, x, y, speed, yaw=0.0, script=constant_speed) -> Agent:
    return Agent(name, C.VEHICLE, x, y, yaw, 4.6, 1.9, 1.5, speed, script, speed * np.cos(yaw), speed * np.sin(yaw))


def _braking(start: float, decel: float):
    def script(agent: Agent, t: float, dt: float) -> None:
        if t >= start:
            agent.speed = max(0.0, agent.speed - decel * dt)
        move(agent, dt)
    return script


def _cut_in(start: float, target_y: float, duration: float):
    def script(agent: Agent, t: float, dt: float) -> None:
        if start <= t < start + duration:
            remaining = target_y - agent.y
            agent.yaw = float(np.clip(np.arctan2(remaining, max(agent.speed * 1.2, 1.0)), -0.35, 0.35))
        else:
            agent.yaw = 0.0
        move(agent, dt)
    return script


def _pedestrian(start: float, speed: float):
    def script(agent: Agent, t: float, dt: float) -> None:
        agent.speed = speed if t >= start else 0.0
        move(agent, dt)
        if agent.y > 7.0:
            agent.speed = 0.0
    return script


def free_road(seed, layout, rng, ego):
    return []


def lead_brake(seed, layout, rng, ego):
    gap = float(rng.uniform(16, 28))
    return [_car("lead", gap, 0.0, float(rng.uniform(7, 10)), script=_braking(float(rng.uniform(1.5, 4)), float(rng.uniform(4, 7))))]


def cut_in(seed, layout, rng, ego):
    return [_car("cutter", float(rng.uniform(8, 16)), LANE_WIDTH, float(rng.uniform(6.5, 8.5)),
                 script=_cut_in(float(rng.uniform(1.0, 3.0)), 0.0, 3.5))]


def occluded_pedestrian(seed, layout, rng, ego):
    van_x = float(rng.uniform(32, 46))
    van = Agent("parked_van", C.VEHICLE, van_x, -2.9, 0.0, 6.0, 2.2, 2.6)
    arrival = van_x / 10.0
    ped = Agent("ped", C.PERSON, van_x + 4.0, -4.0, np.pi / 2, 0.6, 0.6, 1.75, 0.0,
                _pedestrian(float(rng.uniform(arrival - 2.2, arrival - 0.8)), float(rng.uniform(1.2, 1.8))))
    return [van, ped]


def static_obstacle(seed, layout, rng, ego):
    return [_car("stalled", float(rng.uniform(35, 55)), 0.0, 0.0, script=None)]


def dense_traffic(seed, layout, rng, ego):
    agents = [_car("lead", float(rng.uniform(18, 26)), 0.0, float(rng.uniform(7, 9)),
                   script=_braking(float(rng.uniform(3, 6)), float(rng.uniform(2, 4))))]
    x = float(rng.uniform(-12, 4))
    for i in range(4):
        agents.append(_car(f"adj{i}", x, LANE_WIDTH, float(rng.uniform(7, 10))))
        x += float(rng.uniform(14, 24))
    return agents


FAMILIES = {f.__name__: f for f in (free_road, lead_brake, cut_in, occluded_pedestrian, static_obstacle, dense_traffic)}


def build_scenario(name: str, seed: int) -> Scenario:
    rng = np.random.default_rng([abs(hash_name(name)), seed])
    layout = RoadLayout()
    layout.buildings = make_buildings(layout, rng)
    ego = EgoVehicle(speed=float(rng.uniform(8, 10)))
    return Scenario(name, seed, ego, layout, FAMILIES[name](seed, layout, rng, ego))


def hash_name(name: str) -> int:
    """Stable across processes, unlike ``hash``."""
    return sum((i + 1) * ord(ch) for i, ch in enumerate(name))
