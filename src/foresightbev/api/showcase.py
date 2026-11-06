"""Application API: sequence selection, stage views, evaluation and system information.

Every value comes from stored run outputs, stored experiment results, or the dataset.
Where something does not exist, the response says so; nothing is substituted.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, Request as HttpRequest
from fastapi.responses import FileResponse

from foresightbev.api import schemas as S
from foresightbev.api.store import OutputNotStored, RunStore
from foresightbev.api.v1 import jobs, settings, store
from foresightbev.config import ROOT, Settings, load_pipeline_config
from foresightbev.contracts import EgoState, TrajectoryCandidate
from foresightbev.data.sequence import LabelSequence
from foresightbev.data.validation import validate_calibration
from foresightbev.errors import DatasetError, DatasetNotFound
from foresightbev.risk.evaluator import RiskEvaluator
from foresightbev.risk.safety_filter import PerceptionHealth, SafetyFilter
from foresightbev.risk.trajectories import RouteContext, generate_candidates, parametric_candidate

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", responses={404: {"model": S.ErrorResponse}, 422: {"model": S.ErrorResponse}})
LAYERS = {"bev": "bev", "bev_original": "bev_original", "coverage": "coverage", "ipm": "ipm", "forecast": "forecast"}
CAMERA_LAYERS = {"projection", "labels"}


def _dataset(request: HttpRequest, config: Settings):
    from foresightbev.data.nuscenes import NuScenesDataset

    if config.nuscenes_root is None or not (config.nuscenes_root / config.nuscenes_version).is_dir():
        raise DatasetNotFound("nuScenes is not configured or not found. See configs/backend.yaml.")
    cached = getattr(request.app.state, "nuscenes", None)
    if cached is None:
        cached = NuScenesDataset(config.nuscenes_root, config.nuscenes_version)
        request.app.state.nuscenes = cached
    return cached


def _run_manifest(data: RunStore, run: str) -> dict:
    return dict(data.manifest(run), id=run)


def _sequence(config: Settings, manifest: dict) -> LabelSequence:
    name = manifest.get("sequence")
    if not name:
        raise OutputNotStored(f"run {manifest['id']} has no recorded sequence")
    return LabelSequence(Path(name) if Path(name).is_absolute() else config.data_dir / "sequences" / name)


@router.get("/sequences", response_model=list[S.SequenceSummary], summary="Real driving sequences and their stored runs")
def sequences(request: HttpRequest, config: Settings = Depends(settings), data: RunStore = Depends(store)) -> list[dict]:
    dataset = _dataset(request, config)
    from foresightbev.showcase import scene_status

    runs: dict[str, list[dict]] = {}
    for doc in data.list():
        if doc.get("kind") == "showcase":
            runs.setdefault(doc["scenario"], []).append(dict(id=doc["id"], mode=doc.get("mode"), fault=doc.get("fault"), frames=doc.get("frames")))
    out = []
    for s in dataset.scenes():
        status = scene_status(s.name)
        out.append(dict(name=s.name, description=s.description, samples=s.samples, split=status["split"], detector_saw_scene_in_training=status["detector_saw_scene_in_training"], status_note=status["note"],
                        night="night" in s.description.lower(), runs=runs.get(s.name, [])))
    return out


@router.get("/runs/{run}/timeline", response_model=list[S.TimelineRow], summary="One row per frame")
def timeline(run: S.SafeName, data: RunStore = Depends(store)) -> list[dict]:
    rows = []
    for path in sorted((data.directory(run) / "frames").glob("*.json")):
        d = json.loads(path.read_text())
        rows.append(dict(frame=d["frame"], t=d["t"], speed=d["ego"]["speed"], status=d["decision"]["status"], selected=d["decision"]["selected"], reasons=d["decision"]["reasons"],
                         tracks=sum(t["confirmed"] for t in d["tracks"]), detections=len(d["detections"]), healthy_cameras=sum(c["status"] == "ok" for c in d["cameras"].values())))
    if not rows:
        raise OutputNotStored(f"run {run} has no frames")
    return rows


@router.get("/runs/{run}/frames/{frame}/record", summary="Everything stored for one frame")
def frame_record(run: S.SafeName, frame: int, data: RunStore = Depends(store)) -> dict:
    return data.frame(run, frame)


@router.get("/runs/{run}/frames/{frame}/cameras", response_model=list[S.CameraView], summary="Camera status, timing and calibration")
def camera_views(run: S.SafeName, frame: int, config: Settings = Depends(settings), data: RunStore = Depends(store)) -> list[dict]:
    manifest = _run_manifest(data, run)
    record = data.frame(run, frame)
    rows = {r["name"]: r for r in validate_calibration(_sequence(config, manifest).calibration)}
    return [dict(name=name, status=c["status"], reliability=c["reliability"], timestamp=c["timestamp"], skew_ms=c["skew_ms"], calibration=rows[name]) for name, c in record["cameras"].items()]


@router.get("/runs/{run}/frames/{frame}/image/{camera}", summary="The real camera image for this frame", responses={200: {"content": {"image/jpeg": {}}}})
def camera_image(run: S.SafeName, frame: int, camera: S.SafeName, request: HttpRequest, config: Settings = Depends(settings), data: RunStore = Depends(store)) -> FileResponse:
    manifest = _run_manifest(data, run)
    record = data.frame(run, frame)
    if camera not in record["cameras"]:
        raise DatasetNotFound(f"camera {camera} is not in this run")
    if record["cameras"][camera]["status"] != "ok":
        raise OutputNotStored(f"{camera} is {record['cameras'][camera]['status']} at frame {frame}")
    token = _sequence(config, manifest).frames[frame].get("sample_token")
    if not token:
        raise OutputNotStored(f"run {run} has no source images")
    dataset = _dataset(request, config)
    return FileResponse(dataset.root / dataset.camera_rows[token][camera]["filename"], media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@router.get("/runs/{run}/frames/{frame}/layer/{layer}", summary="BEV preview: bev, bev_original, coverage, ipm, forecast", responses={200: {"content": {"image/png": {}}}})
def layer_image(run: S.SafeName, frame: int, layer: S.SafeName, data: RunStore = Depends(store)) -> FileResponse:
    if layer not in LAYERS:
        raise DatasetNotFound(f"unknown layer {layer}. Valid: {sorted(LAYERS)}")
    path = data.directory(run) / LAYERS[layer] / f"{frame:05d}.png"
    if not path.is_file():
        raise OutputNotStored(f"run {run} has no {layer} for frame {frame}")
    return FileResponse(path, media_type="image/png")


@router.get("/runs/{run}/frames/{frame}/layer/{layer}/{camera}", summary="Per-camera view: labels, projection", responses={200: {"content": {"image/png": {}}}})
def camera_layer(run: S.SafeName, frame: int, layer: S.SafeName, camera: S.SafeName, data: RunStore = Depends(store)) -> FileResponse:
    if layer not in CAMERA_LAYERS:
        raise DatasetNotFound(f"unknown layer {layer}. Valid: {sorted(CAMERA_LAYERS)}")
    path = data.directory(run) / layer / f"{frame:05d}_{camera}.png"
    if not path.is_file():
        raise OutputNotStored(f"{camera} has no {layer} at frame {frame}")
    return FileResponse(path, media_type="image/png")


@router.get("/runs/{run}/tracks", response_model=list[S.TrackHistory], summary="Track histories across the run, world frame")
def track_histories(run: S.SafeName, data: RunStore = Depends(store)) -> list[dict]:
    tracks: dict[int, dict] = {}
    for path in sorted((data.directory(run) / "frames").glob("*.json")):
        d = json.loads(path.read_text())
        e = d["ego"]
        ego = EgoState(d["t"], e["x"], e["y"], e["yaw"])
        for t in d["tracks"]:
            if not t["confirmed"]:
                continue
            world = ego.ego_to_world(np.array([t["x"], t["y"]]))
            row = tracks.setdefault(t["id"], dict(id=t["id"], cls=t["cls"], first_frame=d["frame"], last_frame=d["frame"], points=[]))
            row["last_frame"] = d["frame"]
            row["points"].append(dict(frame=d["frame"], x=float(world[0]), y=float(world[1]), speed=float(np.hypot(t["vx"], t["vy"]))))
    return sorted(tracks.values(), key=lambda t: t["id"])


def _run_config(data: RunStore, run: str):
    path = data.manifest(run).get("pipeline_config")
    return load_pipeline_config(ROOT / path if path else None)


@router.post("/runs/{run}/frames/{frame}/plan", response_model=S.RiskResponse, summary="Evaluate a plan by target speed and lateral offset, using this run's configuration")
def evaluate_plan(run: S.SafeName, frame: int, body: S.PlanRequest, data: RunStore = Depends(store)) -> dict:
    from foresightbev.api.v1 import risk_rows

    config = _run_config(data, run)
    forecast = data.forecast_volume(run, frame)
    tracks, ego = data.tracks(run, frame)
    record = data.frame(run, frame)
    ego = dataclasses.replace(ego, speed=record["ego"]["speed"], yaw_rate=record["ego"].get("yaw_rate", 0.0))
    route = RouteContext(body.target_speed, 0.0, 0.0, (0.0, body.lateral_offset) if body.lateral_offset else (0.0,), 0.0)
    plan = parametric_candidate(ego, route, body.target_speed, body.lateral_offset, config.planner)
    stops = [c for c in generate_candidates(ego, route, [], config.planner) if c.is_fallback]
    candidates: list[TrajectoryCandidate] = [plan] + stops
    risks = RiskEvaluator(config.risk).evaluate_all(candidates, forecast, tracks, ego)
    safety = SafetyFilter(config.safety)
    healthy = sum(c["reliability"] > 0 for c in record["cameras"].values())
    decision = safety.decide(candidates, risks, PerceptionHealth(True, 0.0, healthy, len(record["cameras"])), body.target_speed)
    return dict(run=run, frame=frame, risks=risk_rows(risks, safety), decision=dict(status=decision.status.value, selected=decision.selected_id, reasons=[c.value for c in decision.reasons], fallback=decision.fallback_action),
                path=np.round(plan.poses[::10, :2], 2).tolist())


def _result(config: Settings, name: str, file: str) -> dict | None:
    path = config.data_dir / "results" / name / file
    return json.loads(path.read_text()) if path.is_file() else None


def _pooled(config: Settings, name: str) -> dict | None:
    doc = _result(config, name, "nuscenes_summary.json")
    return None if doc is None else dict(pooled=doc["pooled"], per_scene=doc["per_scene"], scenes=doc["scenes"])


@router.get("/evaluation", summary="All evaluation results, labelled by data source and validity")
def evaluation(config: Settings = Depends(settings)) -> dict:
    final = _result(config, "final", "summary.json")
    return dict(
        real=dict(
            label="Real nuScenes footage. mini_val scenes are previously seen; two scenes are preliminary.",
            showcase=_pooled(config, "showcase_mini_val"), streampetr=_pooled(config, "streampetr_mini_val"), baseline=_pooled(config, "streampetr_baseline_image_size"), heldout_classical=_pooled(config, "nuscenes_heldout_check"),
            per_class=_result(config, "streampetr", "per_class_and_sweep.json"), port_verification=_result(config, "streampetr", "port_verification.json"), occupancy_variants=_result(config, "occupancy_diagnostics", "variants.json"),
            tracking_study_mini_train=_result(config, "tracking_study", "mini_train.json"), tracking_mini_val=_result(config, "tracking_study", "mini_val.json"), forecast=_result(config, "forecast_real", "mini_val.json"), safety=_result(config, "safety_real", "mini_val.json"),
            dataset=_result(config, "datasets", "nuscenes_v1.0-mini.json")),
        synthetic=dict(label="Built-in simulator. Scripted agents, no real driving. Not safety evidence.", closed_loop=final, calibration=_result(config, "final", "calibration.json"), forecaster=_result(config, "final", "learned_forecaster.json"),
                       experiment=_result(config, "final", "experiment.json")),
        cam2bev=dict(label="One simulated Cam2BEV frame.", sample=_result(config, "final", "real_sample.json")),
        verification=_result(config, "verification", "summary.json"))


@router.get("/system", summary="Backend, ROS, model and dataset availability")
def system(request: HttpRequest, config: Settings = Depends(settings), manager=Depends(jobs)) -> dict:
    root = config.nuscenes_root
    dataset = dict(configured=root is not None, root=str(root) if root else None, version=config.nuscenes_version, tables_found=bool(root and (root / config.nuscenes_version).is_dir()))
    if dataset["tables_found"]:
        d = _dataset(request, config)
        dataset |= dict(scenes=len(d.scenes()), samples=len(d.samples), occupancy_labels=False)
    runs = sorted((config.data_dir / "results/ros2").glob("*.json"))
    verified = []
    for path in runs:
        doc = json.loads(path.read_text())
        verified.append(dict(id=path.stem, frames=len(doc.get("decisions", [])), matches_in_process=doc.get("matches_in_process"), rmw=doc.get("rmw"), ros_distro=doc.get("ros_distro"), stall_frame=doc.get("stall_frame")))
    checkpoint = config.data_dir / "models/streampetr/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth"
    models = [dict(name="SegFormer-B2 (Cityscapes)", kind="pretrained", available=(config.segmentation_model / "pytorch_model.bin").is_file(), used_in="Image labels", licence="NVIDIA research licence, non-commercial"),
              dict(name="StreamPETR R50 428q", kind="pretrained", available=checkpoint.is_file(), used_in="3D detection", licence="Unverified, treated as non-commercial"),
              dict(name="Forecast U-Net", kind="trained on simulator", available=(config.data_dir / "checkpoints/forecast_unet.pt").is_file(), used_in="Evaluated separately, not in the risk chain", licence="Project")]
    integrations = [dict(name="ROS 2 Jazzy", status="verified" if any(v["matches_in_process"] for v in verified) else "unverified", detail="Two-process DDS runs compared with in-process runs."),
                    dict(name="CARLA", status="not executed", detail="Adapter tested only against a fake module. No server on this machine."),
                    dict(name="Autoware", status="not integrated", detail="Mapping note only. Not installed.")]
    return dict(dataset=dataset, models=models, integrations=integrations, ros_runs=verified, ros_environment_present=(ROOT / "ros2/.pixi").is_dir(), active_job=manager.active, external_processes=manager.activity(),
                pipeline_configs=sorted(p.name for p in (ROOT / "configs").glob("pipeline_*.yaml")), active_pipeline_config=str(config.pipeline_config), data_dir=str(config.data_dir))


@router.get("/config/pipeline/{name}", response_model=S.ConfigCheck, summary="Read one pipeline configuration. Read-only.")
def pipeline_config_file(name: S.SafeName) -> dict:
    path = ROOT / "configs" / name
    if not name.startswith("pipeline_") or not name.endswith(".yaml") or not path.is_file():
        raise DatasetNotFound(f"no pipeline configuration {name}")
    try:
        return dict(valid=True, effective=dataclasses.asdict(load_pipeline_config(path)))
    except ValueError as error:
        raise DatasetError(str(error)) from error
