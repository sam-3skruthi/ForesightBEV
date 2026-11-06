"""Built-in kinematic simulator, the Stage 7 local backend.

A straight multi-lane road with sidewalks, buildings and scripted agents. It is a
research fixture, not vehicle dynamics or CARLA, and results from it are synthetic.
World frame: +x along the road, +y left of travel. Ego starts at the origin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from foresightbev.contracts import SemanticClass as C

LANE_WIDTH = 3.5


@dataclass
class Agent:
    """Rectangular box. ``script(agent, t, dt)`` advances it; None means static."""

    name: str
    semantic: C
    x: float
    y: float
    yaw: float
    length: float
    width: float
    height: float
    speed: float = 0.0
    script: object = None
    vx: float = 0.0
    vy: float = 0.0


def constant_speed(agent: Agent, t: float, dt: float) -> None:
    move(agent, dt)


def move(agent: Agent, dt: float) -> None:
    previous = agent.x, agent.y
    agent.x += agent.speed * np.cos(agent.yaw) * dt
    agent.y += agent.speed * np.sin(agent.yaw) * dt
    agent.vx, agent.vy = (agent.x - previous[0]) / dt, (agent.y - previous[1]) / dt


@dataclass
class EgoVehicle:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    speed: float = 8.0
    yaw_rate: float = 0.0
    length: float = 4.6
    width: float = 2.0


@dataclass
class RoadLayout:
    lanes: tuple[float, ...] = (0.0, LANE_WIDTH)
    road_min_y: float = -1.75
    road_max_y: float = 5.25
    sidewalk_width: float = 3.5
    buildings: list[Agent] = field(default_factory=list)


def make_buildings(layout: RoadLayout, rng: np.random.Generator, x_start: float = -60, x_end: float = 300) -> list[Agent]:
    """Blocks on both sides, set back behind the sidewalk, with occasional gaps."""
    out = []
    for side in (-1, 1):
        x = x_start
        edge = (layout.road_min_y - layout.sidewalk_width) if side < 0 else (layout.road_max_y + layout.sidewalk_width)
        while x < x_end:
            length = float(rng.uniform(14, 28))
            depth = float(rng.uniform(10, 14))
            out.append(Agent(f"bldg{side}_{len(out)}", C.STATIC, x + length / 2, edge + side * depth / 2, 0.0, length,
                             depth, float(rng.uniform(6, 14))))
            x += length + float(rng.uniform(2, 6))
    return out


def ground_label(layout: RoadLayout, y: np.ndarray) -> np.ndarray:
    labels = np.full(y.shape, int(C.OTHER_GROUND), np.uint8)
    labels[(y >= layout.road_min_y - layout.sidewalk_width) & (y <= layout.road_max_y + layout.sidewalk_width)] = C.SIDEWALK
    labels[(y >= layout.road_min_y) & (y <= layout.road_max_y)] = C.ROAD
    return labels


def obb_corners(x, y, yaw, length, width) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    half = np.array([[length, width], [length, -width], [-length, -width], [-length, width]]) / 2
    return np.stack([x + c * half[:, 0] - s * half[:, 1], y + s * half[:, 0] + c * half[:, 1]], -1)


def obb_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Separating axis test for two rectangles given as 4x2 corners."""
    for poly in (a, b):
        for i in (0, 1):
            edge = poly[i + 1] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            pa, pb = a @ axis, b @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def obb_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Rectangle gap by sampled boundary points; 0 if overlapping. Good to ~5 cm."""
    if obb_overlap(a, b):
        return 0.0
    def samples(p):
        t = np.linspace(0, 1, 12, endpoint=False)[:, None, None]
        return (p + t * (np.roll(p, -1, axis=0) - p)).reshape(-1, 2)
    d = samples(a)[:, None] - samples(b)[None]
    return float(np.sqrt((d**2).sum(-1).min()))
