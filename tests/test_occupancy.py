import numpy as np
import pytest

from foresightbev.contracts import (
    BEVGrid, CalibrationBundle, CameraFrame, CameraStatus, CellState, EgoState,
    FrameBundle, Provenance, SemanticClass as C, VoxelGrid,
)
from foresightbev.data.calibration import camera_from_config
from foresightbev.geometry.projection import project
from foresightbev.perception.occupancy import OccupancyEstimator

CFG = dict(fx=278.283, fy=408.1295, px=241.0, py=151.0, yaw=0.0, pitch=0.0, roll=0.0, XCam=1.7, YCam=0.0, ZCam=1.4)
GRID = VoxelGrid(BEVGrid(-5, 40, -12, 12, 0.25))


def rig(*yaws):
    cams = tuple(camera_from_config(n, dict(CFG, yaw=y), 482, 302) for n, y in yaws)
    return CalibrationBundle(cams, "test")


def scene(cam, box=None):
    """Label image: sky above horizon, road below, optional obstacle box (x0,x1,y0,y1,h)."""
    labels = np.full((cam.height, cam.width), int(C.SKY), np.uint8)
    labels[int(cam.K[1, 2]):] = C.ROAD
    if box:
        x0, x1, y0, y1, h = box
        corners = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (0, h)], float)
        u, v, z = project(cam, corners)
        assert (z > 0).all()
        labels[int(v.min()):int(np.ceil(v.max())), int(u.min()):int(np.ceil(u.max()))] = C.VEHICLE
    return labels


def bundle(r, labels, t=0.0, status=CameraStatus.OK):
    frames = {c.name: CameraFrame(c.name, t, labels[c.name], status) for c in r.cameras}
    return FrameBundle("t", 0, t, frames, r, EgoState(t, 0, 0, 0), Provenance.SIMULATED)


@pytest.fixture(scope="module")
def front_scene():
    r = rig(("front", 0.0))
    box = (20.0, 24.0, -1.0, 1.0, 1.5)
    est = OccupancyEstimator(r, GRID)
    _, occ = est.estimate(bundle(r, {"front": scene(r.cameras[0], box)}))
    return occ


def state_at(occ, x, y):
    ix, iy = occ.grid.bev.to_index(x, y)
    return CellState(occ.bev_state[ix, iy])


def test_observed_road_is_free(front_scene):
    assert state_at(front_scene, 10.0, 0.1) == CellState.FREE


def test_obstacle_contact_is_occupied(front_scene):
    assert state_at(front_scene, 20.1, 0.1) == CellState.OCCUPIED


def test_behind_obstacle_is_unknown_not_free_not_occupied(front_scene):
    assert state_at(front_scene, 30.0, 0.1) == CellState.UNKNOWN


def test_behind_camera_is_outside_fov_not_unknown(front_scene):
    assert state_at(front_scene, -4.0, 0.1) == CellState.OUTSIDE_FOV


def test_ego_body_is_invalid(front_scene):
    assert state_at(front_scene, 0.0, 0.0) == CellState.INVALID


def test_occupied_voxels_stop_near_obstacle_height(front_scene):
    ix, iy = front_scene.grid.bev.to_index(20.1, 0.1)
    assert front_scene.bev_height[ix, iy] == pytest.approx(1.5, abs=0.5)
    assert front_scene.voxel_state[-1, ix, iy] == CellState.FREE


def test_camera_order_is_respected():
    r = rig(("front", 0.0), ("rear", 180.0))
    labels = {"front": scene(r.cameras[0]), "rear": scene(r.cameras[1])}
    est = OccupancyEstimator(r, GRID)
    _, ok = est.estimate(bundle(r, labels))
    assert state_at(ok, 10, 0) == CellState.FREE and state_at(ok, -4.5, 0) in (CellState.FREE, CellState.OUTSIDE_FOV)
    frames = bundle(r, labels)
    frames.frames["front"] = CameraFrame("front", 0.0, None, CameraStatus.DROPPED)
    _, lost = est.estimate(frames)
    assert state_at(lost, 10, 0) == CellState.OUTSIDE_FOV


def test_dropped_camera_never_reads_as_free():
    r = rig(("front", 0.0))
    est = OccupancyEstimator(r, GRID)
    _, occ = est.estimate(bundle(r, {"front": None}, status=CameraStatus.DROPPED))
    assert occ.reliability["front"] == 0.0 and not (occ.bev_state == CellState.FREE).any()


def test_garbage_image_is_rejected_as_unreliable():
    r = rig(("front", 0.0))
    labels = np.zeros((302, 482), np.uint8)
    _, occ = OccupancyEstimator(r, GRID).estimate(bundle(r, {"front": labels}))
    assert occ.reliability["front"] == 0.0 and not (occ.bev_state == CellState.FREE).any()


def test_time_skewed_camera_is_excluded():
    r = rig(("front", 0.0))
    b = bundle(r, {"front": scene(r.cameras[0])})
    b.frames["front"].timestamp = -1.0
    _, occ = OccupancyEstimator(r, GRID).estimate(b)
    assert occ.reliability["front"] == 0.0


def test_second_camera_free_evidence_clears_shadow():
    """A shadow from one view is resolved by another view that sees the ground there."""
    cams = (camera_from_config("a", dict(CFG, YCam=0.0), 482, 302),
            camera_from_config("b", dict(CFG, YCam=-6.0), 482, 302))
    r = CalibrationBundle(cams, "stereo")
    box = (18.0, 20.0, -0.6, 0.6, 1.5)
    labels = {"a": scene(cams[0], box), "b": scene(cams[1], box)}
    _, single = OccupancyEstimator(CalibrationBundle(cams[:1], "a"), GRID).estimate(
        bundle(CalibrationBundle(cams[:1], "a"), {"a": labels["a"]}))
    _, both = OccupancyEstimator(r, GRID).estimate(bundle(r, labels))
    assert state_at(single, 27.0, 0.1) == CellState.UNKNOWN
    assert state_at(both, 27.0, 0.1) == CellState.FREE


def test_obstacle_beyond_grid_does_not_create_unknown_in_front_of_it():
    """A distant wall projects above the ground cells; the road before it stays free."""
    r = rig(("front", 0.0))
    far = (80.0, 90.0, -30.0, 30.0, 12.0)
    labels = scene(r.cameras[0], far)
    _, occ = OccupancyEstimator(r, GRID).estimate(bundle(r, {"front": labels}))
    assert state_at(occ, 15.0, 0.1) == CellState.FREE


def test_object_in_near_blind_zone_is_not_free():
    """An object filling the view from horizon to image bottom hides what is behind it."""
    r = rig(("front", 0.0))
    labels = scene(r.cameras[0])
    labels[140:, 180:300] = C.VEHICLE
    _, occ = OccupancyEstimator(r, GRID).estimate(bundle(r, {"front": labels}))
    assert state_at(occ, 25.0, 0.1) != CellState.FREE
