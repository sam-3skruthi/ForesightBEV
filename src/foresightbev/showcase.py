"""End-to-end showcase run on a real nuScenes scene.

Stores everything the application displays, from real computation: detections, tracks,
forecasts, risk, decisions, before/after occupancy, per-camera projections and coverage.
Annotations are drawn for comparison only, and are read after the pipeline has finished
each frame. ``front_camera_loss`` is a declared fault injected into real frames.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import yaml

from foresightbev.config import ROOT, load_pipeline_config, load_settings
from foresightbev.contracts import CameraStatus, CellState
from foresightbev.data.nuscenes import NuScenesDataset, NuScenesGroundTruth
from foresightbev.data.sequence import LabelSequence
from foresightbev.data.taxonomy import PALETTE
from foresightbev.errors import ConfigurationError
from foresightbev.evaluation.nuscenes_eval import SceneScorer, eligible
from foresightbev.forecast.forecaster import rasterize
from foresightbev.recording import RunRecorder, provenance_table, state_image
from foresightbev.replay import build_pipeline, decision_row
from foresightbev.sim.world import obb_corners

FAULT_WINDOW = (14, 26)
FAULT_CAMERA = "CAM_FRONT"
COVERAGE_COLORS = np.array([(12, 18, 28), (28, 74, 92), (45, 170, 170), (150, 230, 215), (220, 250, 240), (255, 255, 255), (255, 255, 255)], np.uint8)


def scene_status(scene: str) -> dict:
    splits = yaml.safe_load((ROOT / "configs/nuscenes_splits.yaml").read_text())
    validation = scene in splits["mini_val"]
    return dict(split="mini_val" if validation else "mini_train", detector_saw_scene_in_training=not validation,
                note="Official mini_val. Previously seen in this project; preliminary." if validation else "In the detector's training data, so detection quality here is inflated.")


def fault_sequence(settings, scene: str) -> Path:
    """Copy a prepared sequence with one camera lost over a frame window."""
    source = settings.data_dir / "sequences" / scene
    target = settings.data_dir / "sequences" / f"{scene}_front_camera_loss"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    (target / "detections_streampetr.json").unlink(missing_ok=True)
    doc = json.loads((target / "sequence.json").read_text())
    doc["id"] = target.name
    for frame in doc["frames"]:
        if FAULT_WINDOW[0] <= frame["index"] < FAULT_WINDOW[1]:
            frame["cameras"][FAULT_CAMERA]["status"] = CameraStatus.DROPPED.value
            (target / "labels" / f"{frame['index']:05d}" / f"{FAULT_CAMERA}.png").unlink(missing_ok=True)
    (target / "sequence.json").write_text(json.dumps(doc, indent=1))
    return target


def ensure_detections(settings, dataset, scene: str, directory: Path, fault: bool) -> None:
    if (directory / "detections_streampetr.json").is_file():
        return
    from foresightbev.perception.streampetr import StreamPETRInference
    from foresightbev.perception.streampetr_runner import run_scene

    model = StreamPETRInference(settings.data_dir / "models/streampetr/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth")
    run_scene(dataset, scene, model, directory, fault=(FAULT_CAMERA, *FAULT_WINDOW) if fault else None)


def projection_image(look, labels: np.ndarray) -> np.ndarray:
    """One camera's labels on flat ground, RGB, forward up."""
    rgb = np.zeros((*look.ground_valid.shape, 3), np.uint8)
    rgb[...] = (11, 16, 24)
    rgb[look.ground_valid] = PALETTE[labels[look.ground_v[look.ground_valid], look.ground_u[look.ground_valid]]]
    return rgb[::-1, ::-1]


def box_rows(boxes, ego, status) -> list[dict]:
    rows = []
    for b in boxes:
        center = ego.world_to_ego(np.array([b.x, b.y]))
        rows.append(dict(name=b.instance[:8], cls=b.semantic.name if b.semantic else None, category=b.category, x=float(center[0]), y=float(center[1]), yaw=b.yaw - ego.yaw, length=b.length, width=b.width,
                         visibility=b.visibility, scored=status(b)))
    return rows


def run_showcase(scene: str, pipeline_config: Path, fault: str | None = None) -> dict:
    settings = load_settings()
    if settings.nuscenes_root is None:
        raise ConfigurationError("set nuscenes_root in configs/backend.yaml")
    dataset = NuScenesDataset(settings.nuscenes_root, settings.nuscenes_version)
    truth = NuScenesGroundTruth(dataset)
    config = load_pipeline_config(pipeline_config)
    base = settings.data_dir / "sequences" / scene
    directory = fault_sequence(settings, scene) if fault else base
    ensure_detections(settings, dataset, scene, directory, bool(fault))
    sequence = LabelSequence(directory)
    pipeline = build_pipeline(sequence, config)
    run_id = f"showcase_{scene}{'_' + fault if fault else ''}"
    final = settings.data_dir / "runs" / run_id
    # Written beside the stored run and swapped in at the end, so a viewer never sees a half-written run.
    target = final.with_name(f".{run_id}.partial")
    if target.exists():
        shutil.rmtree(target)
    status = scene_status(scene)
    info = next(s for s in dataset.scenes() if s.name == scene)
    table = provenance_table(sequence.doc["provenance"], f"nuScenes {settings.nuscenes_version} {scene}: {len(sequence)} keyframes, six cameras.", sequence.doc.get("segmenter"), detector=config.detector)
    table.insert(1, dict(stage=1, component="Fault injection", kind="injected", detail=f"{FAULT_CAMERA} removed for frames {FAULT_WINDOW[0]} to {FAULT_WINDOW[1] - 1}. Real frames with a declared artificial fault.")) if fault else None
    recorder = RunRecorder(target, dict(kind="showcase", title=run_id, scenario=scene, seed=None, mode="fault_injection" if fault else "open_loop_replay", input=f"nuScenes {settings.nuscenes_version} real cameras",
                                        caveat="Open-loop replay of logged real driving. Decisions do not affect later frames. Annotations are displayed for comparison only.", provenance=table, sequence=sequence.id,
                                        scene=dict(name=scene, description=info.description, samples=info.samples, **status, night=info.description.lower().startswith("night") or "night" in info.description.lower()),
                                        fault=dict(type=fault, camera=FAULT_CAMERA, first_frame=FAULT_WINDOW[0], last_frame=FAULT_WINDOW[1] - 1) if fault else None,
                                        pipeline_config=str(pipeline_config.relative_to(ROOT)), config=dict(detector=config.detector, ground_plane=config.ground_plane, fuse_model_boxes=config.fuse_model_boxes,
                                                                                                             max_event_score=config.safety.max_event_score, use_model_velocity=config.tracker.use_model_velocity)), config.grid.bev)
    scorer = SceneScorer(truth, scene, sequence, config)
    tokens = dataset.sample_tokens(scene)
    decisions = []
    grid = config.grid.bev
    started = time.perf_counter()
    for index in range(len(sequence)):
        bundle = sequence.bundle(index)
        result = pipeline.step(bundle, sequence.route(index))
        decisions.append(decision_row(result))
        name = f"{index:05d}"
        _, original = pipeline.occupancy.estimate(bundle)
        token = tokens[sequence.frames[index]["source_index"]]
        boxes = truth.boxes(token)
        scored = {id(b) for b, _ in eligible(boxes, bundle.ego)}
        counts = dict(after=[0, 0], before=[0, 0])
        for box, _center in eligible(boxes, bundle.ego):
            mask = rasterize(bundle.ego.world_to_ego(obb_corners(box.x, box.y, box.yaw, box.length, box.width)), grid)
            counts["after"][0] += int((mask & (result.occupancy.bev_state == CellState.FREE)).sum())
            counts["before"][0] += int((mask & (original.bev_state == CellState.FREE)).sum())
            counts["after"][1] += int(mask.sum())
            counts["before"][1] = counts["after"][1]
        diagnostics = dict(box_cells=counts["after"][1], box_cells_free_after=counts["after"][0], box_cells_free_before=counts["before"][0], annotated_scored=len(scored),
                           free_cells_after=int((result.occupancy.bev_state == CellState.FREE).sum()), free_cells_before=int((original.bev_state == CellState.FREE).sum()))
        recorder.add(result, box_rows(boxes, bundle.ego, lambda b: id(b) in scored), diagnostics)
        recorder.add_layer("bev_original", name, state_image(original.bev_state))
        recorder.add_layer("coverage", name, COVERAGE_COLORS[np.clip(result.occupancy.support_count, 0, 6)][::-1, ::-1])
        recorder.add_layer("ipm", name, PALETTE[np.clip(result.bev.labels, 0, 7)][::-1, ::-1])
        for camera, frame in bundle.frames.items():
            if frame.labels is not None:
                recorder.add_layer("projection", f"{name}_{camera}", projection_image(pipeline.occupancy.lookups[camera], frame.labels))
                recorder.add_layer("labels", f"{name}_{camera}", PALETTE[np.clip(frame.labels, 0, 7)])
        scorer.add(result)
    summary = dict(frames=len(decisions), decisions=decisions, wall_s=time.perf_counter() - started, metrics=scorer.scores.summary(), metric_note="Annotation scoring on this scene only. See the scene status for validity.")
    recorder.finish(summary)
    if final.exists():
        shutil.rmtree(final)
    target.rename(final)
    return dict(run=run_id, frames=len(decisions), statuses=sorted({d["status"] for d in decisions}), wall_s=summary["wall_s"])


