import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from foresightbev import cli
from foresightbev.contracts import CalibrationBundle, CameraStatus, SemanticClass as C
from foresightbev.data.calibration import load_rig
from foresightbev.data.cam2bev_sample import load_sample, mask_ego_body
from foresightbev.data.taxonomy import COLOR_TO_CLASS, PALETTE, rgb_to_labels

ROOT = Path(__file__).resolve().parents[1]
RIG = load_rig(ROOT / "configs/rig_cam2bev_1frlr.yaml")


def write_images(directory, skip=None, size=(604, 964)):
    for cam in RIG.cameras:
        if cam.name != skip:
            cv2.imwrite(str(directory / f"{cam.name}.png"), np.zeros((*size, 3), np.uint8))


def test_every_taxonomy_color_converts():
    for color, cls in COLOR_TO_CLASS.items():
        assert rgb_to_labels(np.array([[color]], np.uint8))[0, 0] == cls


def test_unknown_color_is_unlabeled_not_a_guess():
    assert rgb_to_labels(np.array([[[1, 2, 3]]], np.uint8))[0, 0] == C.UNLABELED


def test_palette_covers_all_classes():
    assert len(PALETTE) == len(C)


def test_missing_camera_file_fails_loudly(tmp_path):
    write_images(tmp_path, skip="rear")
    with pytest.raises(FileNotFoundError):
        load_sample(tmp_path, RIG)


def test_image_size_must_match_calibration(tmp_path):
    write_images(tmp_path, size=(100, 100))
    with pytest.raises(ValueError, match="calibration"):
        load_sample(tmp_path, RIG)


def test_corrupt_image_fails_loudly(tmp_path):
    write_images(tmp_path)
    (tmp_path / "left.png").write_bytes(b"not an image")
    with pytest.raises(FileNotFoundError):
        load_sample(tmp_path, RIG)


def test_loaded_sample_declares_simulated_source_and_all_cameras(tmp_path):
    write_images(tmp_path)
    bundle = load_sample(tmp_path, RIG)
    assert set(bundle.frames) == set(RIG.names) and bundle.source.value == "simulated"
    assert all(f.status == CameraStatus.OK for f in bundle.frames.values())


def test_ego_body_mask_removes_only_bottom_blob():
    labels = np.full((50, 100), int(C.ROAD), np.uint8)
    labels[45:, 30:70] = C.VEHICLE
    labels[10:15, 10:20] = C.VEHICLE
    out = mask_ego_body(labels)
    assert out[49, 50] == C.UNLABELED and out[12, 15] == C.VEHICLE


def test_cli_help_lists_commands(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])
    assert exit_info.value.code == 0
    text = capsys.readouterr().out
    assert all(c in text for c in ("sample", "run", "eval", "report", "serve"))


def test_cli_rejects_unknown_scenario(capsys):
    assert cli.main(["run", "--scenario", "nope"]) == 2
    assert "unknown scenario" in capsys.readouterr().err


def test_recorded_frame_is_json_with_no_ground_truth_in_the_decision_path(tmp_path):
    from foresightbev.pipeline import ForesightPipeline, PipelineConfig
    from foresightbev.recording import RunRecorder
    from foresightbev.sim.backend import BuiltinSim
    from foresightbev.sim.scenarios import build_scenario

    rig = CalibrationBundle(tuple(c.scaled(.25) for c in RIG.cameras), "small")
    sim = BuiltinSim(build_scenario("cut_in", 0), rig)
    config = PipelineConfig()
    result = ForesightPipeline(rig, config).step(sim.observe(), sim.route())
    recorder = RunRecorder(tmp_path / "run", dict(kind="t", title="t"), config.grid.bev)
    recorder.add(result)
    recorder.finish({})
    frame = json.loads((tmp_path / "run/frames/00000.json").read_text())
    assert frame["ground_truth_for_display_only"] is None
    assert json.loads((tmp_path / "run/manifest.json").read_text())["provenance"]
