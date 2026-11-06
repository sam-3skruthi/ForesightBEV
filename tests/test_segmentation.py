from pathlib import Path

import numpy as np
import pytest

from foresightbev.contracts import CameraFrame, CameraStatus, EgoState, FrameBundle, Provenance, SemanticClass as C
from foresightbev.data.calibration import load_rig
from foresightbev.errors import ModelUnavailable
from foresightbev.perception.segmentation import CITYSCAPES_TO_CLASS, SegFormerSegmenter, apply_segmentation, degrade_check, resize_labels

ROOT = Path(__file__).resolve().parents[1]
RIG = load_rig(ROOT / "configs/rig_cam2bev_1frlr.yaml")
MODEL = ROOT / "experiments/end_to_end/models/segformer-b2-cityscapes"


class Fake:
    name, provenance = "fake", Provenance.PRETRAINED

    def segment(self, images):
        return [np.full(i.shape[:2], int(C.ROAD), np.uint8) for i in images]


def bundle():
    frames = {"front": CameraFrame("front", 0.0, None, CameraStatus.OK, image=np.zeros((4, 4, 3), np.uint8)),
              "rear": CameraFrame("rear", 0.0, None, CameraStatus.DROPPED)}
    return FrameBundle("s", 0, 0.0, frames, RIG, EgoState(0, 0, 0, 0), Provenance.RECORDED)


def test_cityscapes_mapping_is_complete_and_correct():
    assert len(CITYSCAPES_TO_CLASS) == 19
    assert CITYSCAPES_TO_CLASS[0] == C.ROAD and CITYSCAPES_TO_CLASS[10] == C.SKY and CITYSCAPES_TO_CLASS[11] == C.PERSON and CITYSCAPES_TO_CLASS[13] == C.VEHICLE


def test_segmentation_fills_labels_and_lost_camera_stays_lost():
    out = apply_segmentation(bundle(), Fake())
    assert out.frames["front"].labels[0, 0] == C.ROAD and out.frames["rear"].labels is None


def test_resize_never_blends_class_ids():
    labels = np.array([[1, 7], [1, 7]], np.uint8)
    assert set(np.unique(resize_labels(labels, 9, 9))) == {1, 7}


def test_uniform_image_is_a_failed_camera():
    assert degrade_check(CameraFrame("a", 0, None, image=np.zeros((4, 4, 3), np.uint8))) == CameraStatus.DROPPED
    assert degrade_check(CameraFrame("a", 0, None, image=np.random.default_rng(0).integers(0, 255, (4, 4, 3), np.uint8))) == CameraStatus.OK


def test_missing_model_fails_with_instructions(tmp_path):
    with pytest.raises(ModelUnavailable, match="no segmentation model"):
        SegFormerSegmenter(tmp_path)


@pytest.mark.skipif(not (MODEL / "pytorch_model.bin").exists(), reason="segmentation weights not present")
def test_real_weights_load_and_produce_valid_class_ids():
    labels = SegFormerSegmenter(MODEL, device="cpu").segment([np.full((128, 256, 3), 120, np.uint8)])[0]
    assert labels.shape == (128, 256) and labels.dtype == np.uint8 and int(labels.max()) <= max(C)
