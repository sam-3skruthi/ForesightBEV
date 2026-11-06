from pathlib import Path

import cv2
import numpy as np
import pytest

from foresightbev.contracts import BEVGrid, CameraCalibration, EgoState, VoxelGrid
from foresightbev.data.calibration import load_rig
from foresightbev.geometry.projection import (
    backproject_to_ground, build_lookup, project, warp_like_cam2bev,
)

ROOT = Path(__file__).resolve().parents[1]
RIG = ROOT / "configs/rig_cam2bev_1frlr.yaml"
SAMPLE = ROOT / "experiments/cam2bev_ipm/inputs/1_FRLR_examples"
DRONE = dict(fx=682.578, fy=682.578, px=482.0, py=302.0, yaw=0.0, pitch=90.0, roll=90.0,
             XCam=0.0, YCam=0.0, ZCam=50.0)
YAWS = dict(front=0.0, rear=180.0, left=90.0, right=-90.0)


@pytest.fixture(scope="module")
def rig():
    return load_rig(RIG)


def test_calibration_shapes_and_order(rig):
    assert rig.names == ("front", "rear", "left", "right")
    for cam in rig.cameras:
        assert cam.K.shape == (3, 3) and np.isfinite(cam.T_cam_ego).all()


def test_transform_direction_and_inverse(rig):
    for cam in rig.cameras:
        assert np.allclose(cam.T_cam_ego @ cam.T_ego_cam, np.eye(4), atol=1e-12)
    front = rig.cameras[0]
    assert np.allclose(front.origin_ego, [1.7, 0.0, 1.4])


def test_front_camera_sees_ahead_and_left_is_image_left(rig):
    front = rig.cameras[0]
    u, v, z = project(front, np.array([[11.7, 0.0, 0.0]]))
    assert z[0] == pytest.approx(10.0) and u[0] == pytest.approx(482.0)
    assert v[0] == pytest.approx(302 + 408.1295 * 1.4 / 10)
    u_left, *_ = project(front, np.array([[11.7, 2.0, 0.0]]))
    assert u_left[0] < 482.0


def test_left_camera_looks_along_positive_y(rig):
    left = rig.cameras[2]
    _, _, z = project(left, np.array([[0.5, 10.5, 0.0]]))
    assert z[0] == pytest.approx(10.0)


def test_ground_projection_round_trip(rig):
    pts = np.array([[8.0, 1.0, 0.0], [20.0, -6.0, 0.0], [5.0, 3.0, 0.0]])
    front = rig.cameras[0]
    u, v, _ = project(front, pts)
    back, valid = backproject_to_ground(front, u, v)
    assert valid.all() and np.allclose(back, pts, atol=1e-9)


def test_scaling_keeps_projection_consistent(rig):
    front = rig.cameras[0]
    half = front.scaled(0.5)
    p = np.array([[12.0, 2.0, 0.5]])
    u, v, _ = project(front, p)
    us, vs, _ = project(half, p)
    assert (half.width, half.height) == (482, 302)
    assert us[0] == pytest.approx(u[0] / 2) and vs[0] == pytest.approx(v[0] / 2)


def test_bad_calibration_is_rejected():
    with pytest.raises(ValueError):
        CameraCalibration("bad", np.eye(3), np.eye(4) * 2, 10, 10)
    with pytest.raises(ValueError):
        CameraCalibration("nan", np.full((3, 3), np.nan), np.eye(4), 10, 10)


def test_bev_grid_index_round_trip_and_bounds():
    grid = BEVGrid()
    X, Y = grid.cell_centers()
    ix, iy = grid.to_index(X, Y)
    assert (ix == np.arange(grid.nx)[:, None]).all() and (iy == np.arange(grid.ny)[None, :]).all()
    assert not grid.in_bounds(*grid.to_index(1000.0, 0.0))


def test_ego_world_round_trip():
    ego = EgoState(0.0, 10.0, -4.0, 0.7)
    p = np.array([[3.0, 2.0], [-1.0, 5.0]])
    assert np.allclose(ego.world_to_ego(ego.ego_to_world(p)), p)
    assert np.allclose(ego.ego_to_world(np.array([1.0, 0.0])), [10 + np.cos(0.7), -4 + np.sin(0.7)])


def test_lookup_matches_project(rig):
    grid = VoxelGrid(BEVGrid(0, 20, -5, 5, 1.0))
    look = build_lookup(rig.cameras[0], grid)
    u, v, z = project(rig.cameras[0], np.array([[10.5, 0.5, 0.0]]))
    assert look.ground_valid[10, 5] and look.ground_u[10, 5] == int(u[0]) and look.ground_v[10, 5] == int(v[0])


@pytest.mark.skipif(not SAMPLE.exists(), reason="local Cam2BEV sample is absent")
def test_ipm_stitch_reproduces_official_reference_exactly(rig):
    """Real-data regression: the productised IPM equals the official homography.png."""
    stitched = None
    for cam in rig.cameras:
        warped = warp_like_cam2bev(cv2.imread(str(SAMPLE / f"{cam.name}.png")), cam, YAWS[cam.name], DRONE)
        stitched = np.zeros_like(warped) if stitched is None else stitched
        mask = np.any(warped != 0, axis=-1)
        stitched[mask] = warped[mask]
    assert (stitched == cv2.imread(str(SAMPLE / "homography.png"))).all()
