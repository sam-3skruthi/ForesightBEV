"""Pure conversions for the ROS 2 bridge. No rclpy import, so they test anywhere.

Topics (all under /foresight):
  cameras/<name>/labels   sensor_msgs/Image mono8, SemanticClass ids; height 0 = DROPPED
  ego                     nav_msgs/Odometry, world frame
  route                   std_msgs/String JSON
  occupancy               nav_msgs/OccupancyGrid (lossy: all non-free, non-occupied are -1)
  occupancy_state         sensor_msgs/Image mono8, exact CellState values
  detections, tracks      std_msgs/String JSON, ego-frame metres at the stamp
  forecast                std_msgs/String JSON: method, horizons, uncertainty semantics
  risk                    std_msgs/String JSON: per-candidate event scores
  decision                std_msgs/String JSON with reason codes and an occupancy hash
  trajectory              nav_msgs/Path, for standard tools
  trajectory_timed        std_msgs/String JSON with times and speeds, for execution

Stamps carry simulator time, so bundles match exactly in lockstep.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from foresightbev.contracts import CellState, SafetyDecision, TrajectoryCandidate
from foresightbev.risk.trajectories import RouteContext

OCCUPANCY_TO_ROS = np.zeros(256, np.int8) - 1
OCCUPANCY_TO_ROS[CellState.FREE] = 0
OCCUPANCY_TO_ROS[CellState.OCCUPIED] = 100


def to_stamp(t: float) -> tuple[int, int]:
    """Seconds to (sec, nanosec). Rounded to the nanosecond to survive float error."""
    total = int(round(t * 1e9))
    return total // 1_000_000_000, total % 1_000_000_000


def from_stamp(sec: int, nanosec: int) -> float:
    return sec + nanosec * 1e-9


def stamp_key(sec: int, nanosec: int) -> int:
    return sec * 1_000_000_000 + nanosec


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    return 0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2))


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def occupancy_grid_data(bev_state: np.ndarray) -> np.ndarray:
    """OccupancyGrid data: index ix + iy * nx, with x along the width."""
    return OCCUPANCY_TO_ROS[bev_state].T.reshape(-1)


def route_to_json(route: RouteContext) -> str:
    return json.dumps(dict(desired_speed=route.desired_speed, lateral_offset=float(route.lateral_offset),
                           heading_error=float(route.heading_error), lanes=list(route.lanes), home_lane=route.home_lane, curvature=route.curvature))


def route_from_json(text: str) -> RouteContext:
    d = json.loads(text)
    return RouteContext(d["desired_speed"], d["lateral_offset"], d["heading_error"], tuple(d["lanes"]), d["home_lane"], d.get("curvature", 0.0))


def trajectory_to_json(candidate: TrajectoryCandidate, frame: int) -> str:
    return json.dumps(dict(frame=frame, id=candidate.candidate_id, t_ref=candidate.t_ref, times=candidate.times.tolist(),
                           poses=candidate.poses.tolist()))


def trajectory_from_json(text: str) -> tuple[int, TrajectoryCandidate]:
    d = json.loads(text)
    return d["frame"], TrajectoryCandidate(d["id"], d["t_ref"], np.array(d["times"]), np.array(d["poses"]))


def state_hash(bev_state: np.ndarray) -> str:
    return hashlib.sha256(bev_state.tobytes()).hexdigest()


def decision_to_json(decision: SafetyDecision, frame: int, t: float, occupancy_sha256: str | None = None) -> str:
    return json.dumps(dict(frame=frame, t=t, status=decision.status.value, selected=decision.selected_id, reasons=[r.value for r in decision.reasons],
                           fallback=decision.fallback_action, occupancy_sha256=occupancy_sha256))


def tracks_to_json(result) -> str:
    ego = result.bundle.ego
    rows = []
    for t in result.tracks:
        position, velocity = ego.world_to_ego(t.position), ego.rotate_to_ego(t.velocity)
        rows.append(dict(id=t.track_id, cls=t.semantic.name, x=float(position[0]), y=float(position[1]), vx=float(velocity[0]), vy=float(velocity[1]), confirmed=t.confirmed, misses=t.misses))
    return json.dumps(dict(frame=result.bundle.frame_index, frame_id="base_link", tracks=rows))


def detections_to_json(result) -> str:
    return json.dumps(dict(frame=result.bundle.frame_index, frame_id="base_link", detections=[
        dict(cls=d.semantic.name, x=float(d.center_ego[0]), y=float(d.center_ego[1]), length=d.length, width=d.width, support=d.support) for d in result.detections]))


def forecast_to_json(result) -> str:
    f = result.forecast
    return json.dumps(dict(frame=result.bundle.frame_index, method=f.method, uncertainty=f.uncertainty, horizons_s=f.horizons.tolist(), t_ref=f.t_ref, last_input=f.last_input_timestamp, n_samples=f.n_samples))


def risk_to_json(result) -> str:
    return json.dumps(dict(frame=result.bundle.frame_index, semantics="event_score is a forecast sample frequency, not a collision probability", risks=[
        dict(id=r.candidate_id, event_score=r.event_score, static_collision=r.static_collision, unknown_corridor_violation=r.unknown_corridor_violation, calibration_status=r.calibration_status) for r in result.risks]))
