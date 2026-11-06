"""Loader behaviour on a synthetic fixture in the official schema. Real data is tested separately."""

import ast
import hashlib
from pathlib import Path

import numpy as np
import pytest

import nuscenes_fixture as fixture
from foresightbev.contracts import CameraStatus, Provenance, SemanticClass as C
from foresightbev.data import nuscenes as N
from foresightbev.data.validation import validate_calibration
from foresightbev.errors import DatasetError, DatasetNotFound
from foresightbev.geometry.projection import project


@pytest.fixture()
def root(tmp_path):
    return fixture.write(tmp_path)


@pytest.fixture()
def dataset(root):
    return N.NuScenesDataset(root)


def test_six_cameras_in_declared_order(dataset):
    assert dataset.calibration("scene-0001").names == N.CAMERAS


def test_quaternion_conventions():
    assert np.allclose(N.quaternion_to_matrix([1, 0, 0, 0]), np.eye(3))
    assert N.quaternion_yaw(fixture.yaw_q(0.7)) == pytest.approx(0.7)
    with pytest.raises(DatasetError):
        N.quaternion_to_matrix([0, 0, 0, 0])


def test_extrinsic_direction_camera_looks_forward_and_sits_at_its_translation(dataset):
    cam = dataset.calibration("scene-0001").cameras[0]
    assert np.allclose(cam.origin_ego, [1.7, 0, 1.5])
    assert np.allclose(cam.T_ego_cam[:3, 2], [1, 0, 0], atol=1e-9)
    u, v, z = project(cam, np.array([[11.7, 0.0, 0.0]]))
    assert z[0] == pytest.approx(10) and u[0] == pytest.approx(816) and v[0] == pytest.approx(491 + 1266 * 1.5 / 10)
    u_left, *_ = project(cam, np.array([[11.7, 2.0, 0.0]]))
    assert u_left[0] < 816


def test_calibration_passes_physical_validation_and_scales(dataset):
    assert len(validate_calibration(dataset.calibration("scene-0001"))) == 6
    scaled = dataset.calibration("scene-0001", 0.5).cameras[0]
    assert (scaled.width, scaled.height, scaled.K[0, 0]) == (800, 450, 633)


def test_frames_are_ordered_and_carry_images_but_no_labels(dataset):
    frames = list(dataset.frames("scene-0001"))
    assert [f.frame_index for f in frames] == [0, 1, 2] and frames[0].source == Provenance.RECORDED
    assert frames[1].timestamp - frames[0].timestamp == pytest.approx(.5)
    assert frames[0].frames["CAM_BACK"].image.shape == (900, 1600, 3) and frames[0].frames["CAM_BACK"].labels is None


def test_timestamps_are_per_camera(dataset):
    frame = next(dataset.frames("scene-0001"))
    assert frame.frames["CAM_FRONT"].timestamp != frame.frames["CAM_FRONT_LEFT"].timestamp


def test_speed_is_causal_backward_difference(dataset):
    frames = list(dataset.frames("scene-0001"))
    assert frames[0].ego.speed == 0.0
    assert frames[1].ego.speed == pytest.approx(5.0) and frames[2].ego.speed == pytest.approx(5.0)


def test_images_resize_with_scaled_calibration(dataset):
    calibration = dataset.calibration("scene-0001", .25)
    assert next(dataset.frames("scene-0001", calibration)).frames["CAM_FRONT"].image.shape == (225, 400, 3)


def test_missing_image_is_a_dropped_camera_not_a_failure(root):
    (root / "samples/CAM_BACK/0.jpg").unlink()
    frame = next(N.NuScenesDataset(root).frames("scene-0001"))
    assert frame.frames["CAM_BACK"].status == CameraStatus.DROPPED and frame.frames["CAM_BACK"].image is None


def test_corrupt_image_is_a_dropped_camera(root):
    (root / "samples/CAM_BACK/0.jpg").write_bytes(b"broken")
    assert next(N.NuScenesDataset(root).frames("scene-0001")).frames["CAM_BACK"].status == CameraStatus.DROPPED


def test_missing_root_and_table_fail_clearly(root, tmp_path):
    with pytest.raises(DatasetNotFound, match="v1.0-mini"):
        N.NuScenesDataset(tmp_path / "nowhere")
    (root / "v1.0-mini/ego_pose.json").unlink()
    with pytest.raises(DatasetError, match="ego_pose"):
        N.NuScenesDataset(root)


def test_corrupt_table_fails_clearly(root):
    (root / "v1.0-mini/sample.json").write_text("{oops")
    with pytest.raises(DatasetError, match="corrupt"):
        N.NuScenesDataset(root)


def test_unknown_scene_lists_known_scenes(dataset):
    with pytest.raises(DatasetNotFound, match="scene-0001"):
        dataset.sample_tokens("scene-9")


def test_version_is_configurable_for_full_dataset(tmp_path):
    fixture.write(tmp_path, version="v1.0-trainval")
    assert N.NuScenesDataset(tmp_path, "v1.0-trainval").scenes()[0].samples == 3


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode() + path.read_bytes())
    return digest.hexdigest()


def test_loading_never_modifies_the_dataset(root):
    before = tree_hash(root)
    dataset = N.NuScenesDataset(root)
    list(dataset.frames("scene-0001"))
    N.NuScenesGroundTruth(dataset).boxes("sample_0")
    assert tree_hash(root) == before


def test_inference_loader_does_not_read_annotations_or_lidar(root, monkeypatch):
    """Deleting those files must not affect the inference path."""
    (root / "v1.0-mini/sample_annotation.json").unlink()
    (root / "v1.0-mini/instance.json").unlink()
    dataset = N.NuScenesDataset(root)
    assert len(list(dataset.frames("scene-0001"))) == 3
    assert not hasattr(dataset, "annotations")
    with pytest.raises(DatasetError):
        N.NuScenesGroundTruth(dataset)


def test_lidar_rows_never_appear_as_cameras(dataset):
    frame = next(dataset.frames("scene-0001"))
    assert "LIDAR_TOP" not in frame.frames and all("pcd" not in r["filename"] for r in dataset.camera_rows["sample_0"].values())


def test_ground_truth_is_separate_and_parses_size_order(dataset):
    box = N.NuScenesGroundTruth(dataset).boxes("sample_1")[0]
    assert (box.semantic, box.length, box.width, box.height) == (C.VEHICLE, 4.6, 1.9, 1.5)
    assert box.yaw == pytest.approx(.1) and box.visibility == 4


def test_category_mapping():
    assert N.category_to_class("vehicle.bus.rigid") == C.VEHICLE and N.category_to_class("human.pedestrian.adult") == C.PERSON
    assert N.category_to_class("movable_object.barrier") is None


def test_pipeline_package_cannot_import_ground_truth():
    """Architecture guard over source files: inference modules never name the GT class."""
    root = Path(__file__).resolve().parents[1] / "src/foresightbev"
    guarded = [root / "pipeline.py", root / "replay.py", *root.glob("perception/*.py"), *root.glob("tracking/*.py"), *root.glob("forecast/forecaster.py"),
               *root.glob("risk/*.py"), root / "data/sequence.py"]
    for path in guarded:
        tree = ast.parse(path.read_text())
        names = {n.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for n in node.names}
        assert "NuScenesGroundTruth" not in names and "GroundTruthBox" not in names, path
