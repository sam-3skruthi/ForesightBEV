import numpy as np
import pytest

from foresightbev.contracts import BEVGrid, CalibrationBundle, CellState, EgoState, FrameBundle, CameraFrame, Provenance, SemanticClass as C, VoxelGrid
from foresightbev.data.calibration import camera_from_config
from foresightbev.perception.image_detection import ImageDetectionConfig, ImageSizeDetector, detect_in_image, merge_overlaps
from foresightbev.perception.occupancy import OccupancyConfig, OccupancyEstimator

CFG = dict(fx=600.0, fy=600.0, px=400.0, py=225.0, yaw=0.0, pitch=0.0, roll=0.0, XCam=1.5, YCam=0.0, ZCam=1.5)
CAM = camera_from_config("front", CFG, 800, 450)
PARAMS = ImageDetectionConfig()


def blob(top, left, height, width, cls=C.VEHICLE):
    labels = np.full((450, 800), int(C.ROAD), np.uint8)
    labels[top:top + height, left:left + width] = cls
    return labels


def test_person_range_comes_from_apparent_size_not_ground():
    """A 1.72 m person 60 px tall is at 17.2 m, wherever the blob sits vertically."""
    for top in (200, 150):
        d = detect_in_image(blob(top, 395, 60, 10, C.PERSON), CAM, C.PERSON, 0.0, PARAMS)
        assert len(d) == 1 and d[0].center_ego[0] == pytest.approx(1.5 + 17.2, abs=.1) and abs(d[0].center_ego[1]) < .3


def test_range_is_independent_of_ground_slope_unlike_contact():
    a = detect_in_image(blob(200, 380, 40, 40), CAM, C.VEHICLE, 0.0, PARAMS)[0]
    b = detect_in_image(blob(190, 380, 40, 40), CAM, C.VEHICLE, 0.0, PARAMS)[0]
    assert a.center_ego[0] == pytest.approx(b.center_ego[0])


def test_left_of_image_is_positive_y():
    d = detect_in_image(blob(200, 100, 40, 40), CAM, C.VEHICLE, 0.0, PARAMS)[0]
    assert d.center_ego[1] > 0


def test_crowd_blob_splits_into_people():
    d = detect_in_image(blob(200, 300, 60, 120, C.PERSON), CAM, C.PERSON, 0.0, PARAMS)
    assert len(d) == 4


def test_small_specks_are_ignored():
    assert detect_in_image(blob(200, 300, 3, 3, C.PERSON), CAM, C.PERSON, 0.0, PARAMS) == []


def test_covariance_grows_with_range():
    near = detect_in_image(blob(150, 395, 200, 40, C.PERSON), CAM, C.PERSON, 0.0, PARAMS)[0]
    far = detect_in_image(blob(200, 395, 30, 15, C.PERSON), CAM, C.PERSON, 0.0, PARAMS)[0]
    assert far.covariance[0, 0] > near.covariance[0, 0]


def test_image_edge_blob_uses_ground_contact():
    d = detect_in_image(blob(300, 0, 100, 100), CAM, C.VEHICLE, 0.0, PARAMS)
    assert len(d) == 1 and d[0].center_ego[0] < 15


def test_overlapping_cameras_merge_and_classes_do_not():
    a = detect_in_image(blob(200, 395, 60, 10, C.PERSON), CAM, C.PERSON, 0.0, PARAMS)[0]
    b = detect_in_image(blob(200, 395, 60, 10, C.PERSON), CAM, C.PERSON, 0.0, PARAMS)[0]
    v = detect_in_image(blob(200, 380, 40, 40), CAM, C.VEHICLE, 0.0, PARAMS)[0]
    v.center_ego = a.center_ego.copy()
    assert len(merge_overlaps([a, b, v], 2.0)) == 2


def test_detector_skips_lost_cameras():
    rig = CalibrationBundle((CAM,), "t")
    bundle = FrameBundle("s", 0, 0.0, {"front": CameraFrame("front", 0.0, None)}, rig, EgoState(0, 0, 0, 0), Provenance.RECORDED)
    assert ImageSizeDetector(rig)(None, bundle) == []


def scene_labels(box_row=260):
    labels = np.full((450, 800), int(C.SKY), np.uint8)
    labels[225:] = C.ROAD
    labels[200:box_row, 370:430] = C.VEHICLE
    return labels


GRID = VoxelGrid(BEVGrid(-5, 60, -12, 12, 0.25))


def state(tolerance):
    rig = CalibrationBundle((CAM,), "t")
    bundle = FrameBundle("s", 0, 0.0, {"front": CameraFrame("front", 0.0, scene_labels())}, rig, EgoState(0, 0, 0, 0), Provenance.SIMULATED)
    _, occ = OccupancyEstimator(rig, GRID, OccupancyConfig(ground_height_tolerance_m=tolerance)).estimate(bundle)
    return occ


def at(occ, x):
    ix, iy = GRID.bev.to_index(x, 0.1)
    return CellState(occ.bev_state[ix, iy])


def test_ground_tolerance_turns_uncertain_span_into_unknown():
    hit = 1.5 + 600 * 1.5 / 35
    near_contact = hit - 3.0
    assert at(state(0.0), near_contact) == CellState.FREE
    assert at(state(0.5), near_contact) == CellState.UNKNOWN


def test_zero_tolerance_keeps_contact_occupied():
    hit = 1.5 + 600 * 1.5 / 35
    assert at(state(0.5), hit + .2) == CellState.OCCUPIED


def test_free_near_the_ego_survives_tolerance():
    assert at(state(0.5), 12.0) == CellState.FREE


def test_detection_order_ignores_camera_arrival_order():
    """Regression: over ROS the frames dict is in arrival order, which must not matter."""
    rear = camera_from_config("rear", dict(CFG, yaw=180.0, XCam=-1.0), 800, 450)
    rig = CalibrationBundle((CAM, rear), "t")
    frames = {"front": CameraFrame("front", 0.0, blob(200, 380, 40, 40)), "rear": CameraFrame("rear", 0.0, blob(200, 380, 40, 40))}
    forward = FrameBundle("s", 0, 0.0, frames, rig, EgoState(0, 0, 0, 0), Provenance.RECORDED)
    reverse = FrameBundle("s", 0, 0.0, dict(reversed(list(frames.items()))), rig, EgoState(0, 0, 0, 0), Provenance.RECORDED)
    detector = ImageSizeDetector(rig)
    assert [tuple(d.center_ego) for d in detector(None, forward)] == [tuple(d.center_ego) for d in detector(None, reverse)]
