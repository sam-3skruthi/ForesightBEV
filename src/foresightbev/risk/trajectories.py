"""Stage 6 candidate generation: a small lattice of longitudinal and lateral plans."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from foresightbev.contracts import EgoState, TrackState, TrajectoryCandidate
from foresightbev.forecast.forecaster import track_to_ego


@dataclass
class RouteContext:
    """Local route: a straight reference along +s, declared per frame.

    ``lateral_offset`` is the ego's signed offset from the reference line and
    ``heading_error`` its yaw relative to it. ``lanes`` are lateral centres.
    """

    desired_speed: float = 10.0
    lateral_offset: float = 0.0
    heading_error: float = 0.0
    lanes: tuple[float, ...] = (0.0,)
    home_lane: float = 0.0
    curvature: float = 0.0


@dataclass
class PlannerConfig:
    horizon_s: float = 3.0
    dt: float = 0.05
    max_accel: float = 1.5
    comfort_decel: float = 2.5
    emergency_decel: float = 6.0
    lateral_time_s: float = 2.5
    min_lateral_length_m: float = 9.0
    speed_fractions: tuple[float, ...] = (1.0, 0.7, 0.4, 0.2, 0.0)
    idm_time_headway: float = 1.5
    idm_min_gap: float = 3.0
    corridor_half_width: float = 1.6
    ego_half_length: float = 2.3
    stationary_speed: float = 1.0
    lane_lookahead_m: float = 30.0
    lane_change_clear_m: float = 8.0


def idm_acceleration(speed: float, desired: float, gap: float | None, lead_speed: float, cfg: PlannerConfig) -> float:
    """Intelligent Driver Model acceleration, bounded for the nominal plan."""
    free = 1.0 - (speed / max(desired, 0.1)) ** 4
    interaction = 0.0
    if gap is not None:
        star = cfg.idm_min_gap + max(0.0, speed * cfg.idm_time_headway
                                     + speed * (speed - lead_speed) / (2 * np.sqrt(cfg.max_accel * cfg.comfort_decel)))
        interaction = (star / max(gap, 0.5)) ** 2
    return float(np.clip(cfg.max_accel * (free - interaction), -cfg.emergency_decel, cfg.max_accel))


def _route_position(position: np.ndarray, route: RouteContext) -> tuple[float, float]:
    """Ego-frame point to route coordinates (along, lateral from the reference line)."""
    c, s = np.cos(route.heading_error), np.sin(route.heading_error)
    return float(c * position[0] - s * position[1]), float(s * position[0] + c * position[1] + route.lateral_offset)


def lead_from_tracks(tracks: list[TrackState], ego: EgoState, cfg: PlannerConfig, route: RouteContext | None = None,
                     target_lane: float | None = None) -> tuple[float | None, float]:
    """Nearest confirmed track in the planned corridor, as (bumper gap, speed along x).

    Geometry is evaluated in route coordinates, because the ego may be yawed. The
    corridor spans the ego's lateral position and the target lane. During a lane
    change, a blocker still in the current lane is ignored while it is farther
    than ``lane_change_clear_m``, since the ego is moving out of its way.
    """
    route = route or RouteContext()
    lane = route.lateral_offset if target_lane is None else target_lane
    low, high = min(route.lateral_offset, lane), max(route.lateral_offset, lane)
    changing = high - low > 0.5
    best: tuple[float | None, float] = (None, 0.0)
    for track in tracks:
        if not track.confirmed:
            continue
        position, velocity, _ = track_to_ego(track, ego)
        along, lateral = _route_position(position, route)
        reach = cfg.corridor_half_width + track.width / 2
        if along <= 0 or lateral < low - reach or lateral > high + reach:
            continue
        if changing and abs(lateral - lane) > reach and along > cfg.lane_change_clear_m:
            continue
        gap = float(along - track.length / 2 - cfg.ego_half_length)
        if best[0] is None or gap < best[0]:
            best = (gap, float(velocity[0]))
    return best


def _speed_profile(times: np.ndarray, v0: float, target: float, accel: float, decel: float) -> np.ndarray:
    rate = accel if target >= v0 else -decel
    duration = (target - v0) / rate if rate else 0.0
    return np.where(times < duration, v0 + rate * times, target)


def _quintic(tau):
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def _phase(fraction: float) -> float:
    """Inverse of the quintic on [0, 1], by bisection on a monotone function."""
    low, high = 0.0, 1.0
    for _ in range(40):
        middle = (low + high) / 2
        low, high = (middle, high) if _quintic(middle) < fraction else (low, middle)
    return (low + high) / 2


def _lateral_profile(s: np.ndarray, l0: float, l1: float, length: float, lanes: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Quintic offset as a function of arc length, so no sideways motion at standstill.

    A replanned manoeuvre resumes from its current phase between the source lane
    and the target. Restarting from zero slope every frame would keep the ego
    near the start of the curve, and the plan assessed would not be the plan driven.
    """
    others = [lane for lane in lanes if abs(lane - l1) > 1e-6]
    source = min(others, key=lambda lane: abs(lane - l0)) if others else l0
    span = l1 - source
    if abs(span) < 1e-6 or abs(l1 - l0) < 1e-6:
        return np.full_like(s, l1), np.zeros_like(s)
    fraction = float(np.clip((l0 - source) / span, 0.0, 1.0))
    start = _phase(fraction)
    tau = np.clip(start + s / length, 0.0, 1.0)
    slope = span * (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / length
    return source + span * _quintic(tau), slope


def _candidate(cid, ego, route, cfg, speeds, target_lateral, nominal=False, fallback=False) -> TrajectoryCandidate:
    times = np.arange(0, cfg.horizon_s + cfg.dt / 2, cfg.dt)
    s = np.concatenate([[0.0], np.cumsum((speeds[:-1] + speeds[1:]) / 2 * cfg.dt)])
    length = max(cfg.min_lateral_length_m, float(ego.speed) * cfg.lateral_time_s)
    lateral, slope = _lateral_profile(s, route.lateral_offset, target_lateral, length, route.lanes)
    heading = np.arctan(slope)
    x_r, y_r = s, lateral - route.lateral_offset
    if route.curvature != 0.0:
        # Bend the straight reference into a constant-curvature arc; offset stays normal to it.
        bend = route.curvature * s
        arc_x, arc_y = np.sin(bend) / route.curvature, (1 - np.cos(bend)) / route.curvature
        x_r, y_r = arc_x - y_r * np.sin(bend), arc_y + y_r * np.cos(bend)
        heading = heading + bend
    c, sn = np.cos(-route.heading_error), np.sin(-route.heading_error)
    poses = np.stack([c * x_r - sn * y_r, sn * x_r + c * y_r, heading - route.heading_error, speeds], -1)
    return TrajectoryCandidate(cid, ego.timestamp, times, poses, nominal, fallback, target_lateral)


def choose_lane(tracks: list[TrackState], ego: EgoState, route: RouteContext, cfg: PlannerConfig) -> float:
    """Nearest lane to home without a stationary tracked blocker ahead."""
    blocked = set()
    for track in tracks:
        if not track.confirmed:
            continue
        position, velocity, _ = track_to_ego(track, ego)
        along, lateral = _route_position(position, route)
        if np.hypot(*velocity) >= cfg.stationary_speed or not 0.0 < along < cfg.lane_lookahead_m:
            continue
        for lane in route.lanes:
            if abs(lateral - lane) <= cfg.corridor_half_width + track.width / 2:
                blocked.add(lane)
    options = [lane for lane in route.lanes if lane not in blocked]
    return min(options, key=lambda lane: abs(lane - route.home_lane)) if options else route.home_lane


def parametric_candidate(ego: EgoState, route: RouteContext, target_speed: float, lateral: float, cfg: PlannerConfig | None = None) -> TrajectoryCandidate:
    """One user-specified plan: reach a speed and a lateral offset. Marked as the nominal."""
    cfg = cfg or PlannerConfig()
    times = np.arange(0, cfg.horizon_s + cfg.dt / 2, cfg.dt)
    speeds = _speed_profile(times, ego.speed, target_speed, cfg.max_accel, cfg.comfort_decel)
    return _candidate("user_plan", ego, route, cfg, speeds, lateral, nominal=True)


def generate_candidates(ego: EgoState, route: RouteContext, tracks: list[TrackState],
                        cfg: PlannerConfig | None = None, nominal_mode: str = "idm") -> list[TrajectoryCandidate]:
    """Build the lattice. The first candidate is always the nominal plan.

    ``nominal_mode`` selects the proposer. "idm" follows tracked leads and picks a
    lane around stationary blockers. "blind" holds desired speed in the home lane
    and ignores perception. Either way the safety filter still checks the plan.
    """
    cfg = cfg or PlannerConfig()
    times = np.arange(0, cfg.horizon_s + cfg.dt / 2, cfg.dt)
    v0, desired = ego.speed, route.desired_speed
    target_lane = route.home_lane
    if nominal_mode == "idm":
        target_lane = choose_lane(tracks, ego, route, cfg)
        gap, lead_speed = lead_from_tracks(tracks, ego, cfg, route, target_lane)
        accel = idm_acceleration(v0, desired, gap, lead_speed, cfg)
    elif nominal_mode == "blind":
        accel = cfg.max_accel if v0 < desired else 0.0
    else:
        raise ValueError(f"unknown nominal mode {nominal_mode}")
    nominal_speeds = np.clip(v0 + accel * times, 0.0, max(desired, v0))
    out = [_candidate("nominal", ego, route, cfg, nominal_speeds, target_lane, nominal=True)]
    for lane in route.lanes:
        for fraction in cfg.speed_fractions:
            target = desired * fraction
            speeds = _speed_profile(times, v0, target, cfg.max_accel, cfg.comfort_decel)
            out.append(_candidate(f"lane{lane:+.1f}_v{fraction:.1f}", ego, route, cfg, speeds, lane))
    lane = route.lateral_offset
    for name, decel in (("comfort_stop", cfg.comfort_decel), ("emergency_stop", cfg.emergency_decel)):
        speeds = np.maximum(v0 - decel * times, 0.0)
        out.append(_candidate(name, ego, route, cfg, speeds, lane, fallback=True))
    return out
