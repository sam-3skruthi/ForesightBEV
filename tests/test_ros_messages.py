
import numpy as np
import pytest

from foresightbev.contracts import CellState, TrajectoryCandidate
from foresightbev.integration import ros_messages as M
from foresightbev.risk.trajectories import RouteContext


@pytest.mark.parametrize("t", [0.0, 0.2, 1.4000000000000001, 12.6, 1000.05])
def test_stamp_round_trip(t):
    assert M.from_stamp(*M.to_stamp(t)) == pytest.approx(t, abs=1e-9)


def test_same_time_gives_same_key_despite_float_noise():
    assert M.stamp_key(*M.to_stamp(0.1 + 0.2)) == M.stamp_key(*M.to_stamp(0.3))


@pytest.mark.parametrize("yaw", [0.0, 0.5, -2.0, 3.0])
def test_quaternion_round_trip(yaw):
    assert M.quaternion_to_yaw(*M.yaw_to_quaternion(yaw)) == pytest.approx(yaw)


def test_occupancy_grid_layout_and_values():
    state = np.full((3, 2), int(CellState.UNKNOWN), np.uint8)
    state[1, 0] = CellState.OCCUPIED
    state[2, 1] = CellState.FREE
    data = M.occupancy_grid_data(state)
    assert data[1 + 0 * 3] == 100 and data[2 + 1 * 3] == 0 and data[0] == -1 and len(data) == 6


def test_outside_fov_and_invalid_map_to_unknown_in_lossy_grid():
    state = np.array([[CellState.OUTSIDE_FOV, CellState.INVALID]], np.uint8)
    assert (M.occupancy_grid_data(state) == -1).all()


def test_route_and_trajectory_round_trip():
    route = M.route_from_json(M.route_to_json(RouteContext(9.0, .5, .1, (0.0, 3.5), 0.0)))
    assert route == RouteContext(9.0, .5, .1, (0.0, 3.5), 0.0)
    plan = TrajectoryCandidate("x", 1.0, np.arange(3) * .1, np.arange(12.).reshape(3, 4))
    frame, back = M.trajectory_from_json(M.trajectory_to_json(plan, 7))
    assert frame == 7 and (back.poses == plan.poses).all() and back.candidate_id == "x"


def test_route_round_trip_keeps_curvature():
    route = RouteContext(9.0, 0.0, 0.0, (0.0,), 0.0, 0.05)
    assert M.route_from_json(M.route_to_json(route)).curvature == 0.05


def test_state_hash_detects_any_change():
    a = np.zeros((4, 4), np.uint8)
    b = a.copy()
    b[1, 1] = 1
    assert M.state_hash(a) == M.state_hash(a.copy()) and M.state_hash(a) != M.state_hash(b)
