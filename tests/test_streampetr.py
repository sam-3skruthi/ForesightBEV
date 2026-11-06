import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from foresightbev import config as C
from foresightbev.contracts import CalibrationBundle, EgoState, FrameBundle, Provenance, SemanticClass as S
from foresightbev.errors import ConfigurationError, DatasetError, ModelUnavailable
from foresightbev.perception import streampetr as P
from foresightbev.perception.detection_cache import CACHE_NAME, CachedDetector
from foresightbev.perception.streampetr_runner import boxes_to_ego

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "experiments/end_to_end/models/streampetr/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth"
needs_weights = pytest.mark.skipif(not CHECKPOINT.exists(), reason="checkpoint not present")


def test_config_differs_from_baseline_only_in_detector():
    a = yaml.safe_load((ROOT / "configs/pipeline_nuscenes.yaml").read_text())
    b = yaml.safe_load((ROOT / "configs/pipeline_streampetr.yaml").read_text())
    assert a.pop("detector") == "image_size" and b.pop("detector") == "streampetr" and b.pop("detector_score_threshold") == .3
    assert a == b


def test_preprocessing_matches_official_transform():
    rgb = np.zeros((900, 1600, 3), np.uint8)
    out = P.preprocess_image(rgb)
    assert out.shape == (3, 256, 704) and out[0, 0, 0] == pytest.approx(-123.675 / 58.395)
    with pytest.raises(ValueError):
        P.preprocess_image(np.zeros((10, 10, 3), np.uint8))


def test_crop_keeps_the_bottom_of_the_image():
    rgb = np.zeros((900, 1600, 3), np.uint8)
    rgb[-100:] = 255
    out = P.preprocess_image(rgb)
    assert out[0, -1, 0] > out[0, 0, 0]


def test_intrinsics_follow_resize_and_crop():
    K = np.array([[1000.0, 0, 800], [0, 1000, 450], [0, 0, 1]])
    k = P.adjust_intrinsics(K)
    assert k[0, 0] == pytest.approx(440) and k[0, 2] == pytest.approx(352) and k[1, 2] == pytest.approx(450 * .44 - 140)


def test_inverse_sigmoid_is_clamped():
    assert torch.isfinite(P.inverse_sigmoid(torch.tensor([-1.0, 0.0, 1.0, 2.0]))).all()


def test_decode_order_and_size_convention():
    scores = torch.full((10, 10), -9.0)
    scores[3, 8] = 3.0
    boxes = torch.zeros(10, 10)
    boxes[3] = torch.tensor([1.0, 2.0, 0.5, np.log(4.6), np.log(1.9), np.log(1.5), 1.0, 0.0, 3.0, 4.0])
    d = P.decode(scores, boxes, max_num=1)
    assert P.CLASS_NAMES[d.labels[0]] == "pedestrian" and d.size[0, 0] == pytest.approx(4.6)
    assert d.yaw[0] == pytest.approx(np.pi / 2) and d.velocity[0].tolist() == [3.0, 4.0]


def test_decode_rejects_out_of_range_centres():
    scores = torch.zeros(10, 10)
    boxes = torch.zeros(10, 10)
    boxes[:, 0] = 100
    assert len(P.decode(scores, boxes, max_num=5).scores) == 0


def test_lidar_to_ego_conversion():
    """LIDAR_TOP is rotated 90 degrees about z: lidar +y is ego +x."""
    T = np.eye(4)
    T[:3, :3] = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    T[:3, 3] = [1.0, 0.0, 1.8]
    boxes = P.Boxes(np.array([[0.0, 10.0, 0.0]]), np.array([[4.6, 1.9, 1.5]]), np.array([np.pi / 2]), np.array([[0.0, 5.0]]), np.array([.9]), np.array([0]))
    row = boxes_to_ego(boxes, T)[0]
    assert row["center_ego"] == pytest.approx([11.0, 0.0, 1.8]) and row["velocity_ego"] == pytest.approx([5.0, 0.0])
    assert row["yaw_ego"] == pytest.approx(0.0, abs=1e-9) and row["label"] == "car" and row["length"] == pytest.approx(4.6)


def cache(tmp_path, rows):
    doc = dict(frames=[dict(index=0, timestamp=1.0, detections=rows)])
    (tmp_path / CACHE_NAME).write_text(json.dumps(doc))


ROW = dict(label="pedestrian", score=.5, center_ego=[3.0, 1.0, 1.0], length=.7, width=.7, height=1.7, yaw_ego=0.0, velocity_ego=[1.0, 0.0])
BUNDLE = FrameBundle("s", 0, 1.0, {}, CalibrationBundle((), "x") if False else None, EgoState(1.0, 0, 0, 0), Provenance.RECORDED)


def test_cached_detector_filters_score_and_non_agents(tmp_path):
    cache(tmp_path, [ROW, dict(ROW, score=.1), dict(ROW, label="traffic_cone")])
    out = CachedDetector(tmp_path, .3)(None, BUNDLE)
    assert len(out) == 1 and out[0].semantic == S.PERSON and out[0].provenance == Provenance.PRETRAINED and out[0].support == .5


def test_cached_detector_refuses_mismatched_time_and_missing_frame(tmp_path):
    cache(tmp_path, [ROW])
    wrong = FrameBundle("s", 0, 2.0, {}, None, EgoState(2.0, 0, 0, 0), Provenance.RECORDED)
    with pytest.raises(DatasetError, match="timestamp"):
        CachedDetector(tmp_path, .3)(None, wrong)
    with pytest.raises(DatasetError, match="no cached"):
        CachedDetector(tmp_path, .3)(None, FrameBundle("s", 7, 1.0, {}, None, EgoState(1.0, 0, 0, 0), Provenance.RECORDED))


def test_missing_cache_explains_the_fix(tmp_path):
    with pytest.raises(ConfigurationError, match="detect-streampetr"):
        CachedDetector(tmp_path, .3)


def test_config_accepts_detector_switch_and_rejects_unknown():
    assert C.pipeline_config_from_dict({"detector": "streampetr"}).detector == "streampetr"
    with pytest.raises(ConfigurationError):
        C.pipeline_config_from_dict({"detector": "magic"})


def test_missing_or_altered_checkpoint_is_refused(tmp_path):
    with pytest.raises(ModelUnavailable, match="no StreamPETR"):
        P.load_checkpoint(P.StreamPETR(), tmp_path / "x.pth")
    (tmp_path / "x.pth").write_bytes(b"not it")
    with pytest.raises(ModelUnavailable, match="hash"):
        P.load_checkpoint(P.StreamPETR(), tmp_path / "x.pth")


@needs_weights
def test_every_model_weight_loads_and_unused_are_explained():
    report = P.load_checkpoint(P.StreamPETR(), CHECKPOINT)
    assert report.loaded_tensors == 573 and report.missing == [] and report.unexpected == []
    assert len(report.unused_tensors) == 18 and all(k.startswith("img_roi_head.") or k.endswith(("code_weights", "match_costs")) for k in report.unused_tensors)


@pytest.fixture(scope="module")
def model():
    if not CHECKPOINT.exists():
        pytest.skip("checkpoint not present")
    return P.StreamPETRInference(CHECKPOINT, device="cpu")


def frame_args():
    K = np.tile(np.array([[1266.0, 0, 816], [0, 1266, 491], [0, 0, 1]]), (6, 1, 1))
    lidar2cam = np.tile(np.eye(4), (6, 1, 1))
    images = [np.random.default_rng(i).integers(0, 255, (900, 1600, 3), np.uint8) for i in range(6)]
    return images, K, lidar2cam, np.eye(4)


def test_output_shapes_and_memory_lifecycle(model):
    model.reset()
    images, K, l2c, pose = frame_args()
    boxes = model.step(images, K, l2c, pose, 0.0)
    head = model.model.pts_bbox_head
    assert boxes.center.shape[1] == 3 and boxes.size.shape[1] == 3 and boxes.velocity.shape[1] == 2 and (boxes.size > 0).all()
    assert head.memory_embedding.shape == (1, 640, 256)
    model.step(images, K, l2c, pose, 0.5)
    assert head.memory_timestamp[0, 128, 0].item() == pytest.approx(.5) and head.memory_timestamp[0, 0, 0].item() == 0
    assert head.memory_embedding.shape == (1, 640, 256)
    model.reset()
    assert head.memory_embedding is None and model.previous_pose is None


def test_memory_moves_with_ego_motion(model):
    model.reset()
    images, K, l2c, pose = frame_args()
    model.step(images, K, l2c, pose, 0.0)
    head = model.model.pts_bbox_head
    before = head.memory_reference_point[0, 0].clone()
    moved = np.eye(4)
    moved[0, 3] = 2.0
    head._prepare_memory("cpu", torch.from_numpy(np.linalg.inv(moved) @ pose).float(), .5)
    assert (head.memory_reference_point[0, 0] - before).tolist() == pytest.approx([-2.0, 0, 0], abs=1e-4)


def test_streaming_is_causal_and_rejects_time_reversal(model):
    model.reset()
    images, K, l2c, pose = frame_args()
    model.step(images, K, l2c, pose, 1.0)
    with pytest.raises(ValueError):
        model.step(images, K, l2c, pose, 1.0)
    model.reset()


def test_wrong_camera_count_is_rejected(model):
    images, K, l2c, pose = frame_args()
    with pytest.raises(ValueError):
        model.step(images[:5], K, l2c, pose, 0.0)
