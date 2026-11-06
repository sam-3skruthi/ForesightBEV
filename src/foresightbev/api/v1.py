"""Versioned backend API. Every value comes from stored or freshly computed outputs."""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, Query, Request as HttpRequest

from foresightbev import __version__
from foresightbev.api import jobs as job_module
from foresightbev.api import logs as log_module
from foresightbev.api import schemas as S
from foresightbev.api.store import RunStore, encode_array
from foresightbev.config import ROOT, Settings, load_pipeline_config, pipeline_config_from_dict
from foresightbev.contracts import CalibrationBundle, CameraFrame, CameraStatus, CellState, EgoState, FrameBundle, Provenance, SemanticClass, TrajectoryCandidate
from foresightbev.data.calibration import load_rig
from foresightbev.data.sequence import LabelSequence, write_sequence
from foresightbev.data.taxonomy import rgb_to_labels
from foresightbev.data.validation import validate_calibration
from foresightbev.errors import ConfigurationError, DatasetError, DatasetNotFound, JobConflict
from foresightbev.recording import environment
from foresightbev.risk.evaluator import RiskEvaluator
from foresightbev.risk.safety_filter import PerceptionHealth, SafetyFilter
from foresightbev.risk.trajectories import RouteContext, generate_candidates

logger = logging.getLogger(__name__)
SCORE_SEMANTICS = "Fraction of joint forecast samples where the event occurs, under the forecaster's model. Not a collision probability."
router = APIRouter(prefix="/api/v1", responses={404: {"model": S.ErrorResponse}, 409: {"model": S.ErrorResponse}, 422: {"model": S.ErrorResponse}})
SAMPLE = ROOT / "experiments/cam2bev_ipm/inputs/1_FRLR_examples"
RIG = ROOT / "configs/rig_cam2bev_1frlr.yaml"


def settings(request: HttpRequest) -> Settings:
    return request.app.state.settings


def store(request: HttpRequest) -> RunStore:
    return request.app.state.store


def jobs(request: HttpRequest) -> job_module.JobManager:
    return request.app.state.jobs


@router.get("/health", response_model=S.Health, summary="Liveness: the process is serving requests")
def health() -> dict:
    return dict(status="ok", version=__version__)


@router.get("/status", response_model=S.SystemStatus, summary="Readiness: configuration, assets and active work")
def status(request: HttpRequest, config: Settings = Depends(settings), manager: job_module.JobManager = Depends(jobs)) -> dict:
    checks = []

    def check(name: str, function) -> None:
        try:
            checks.append(dict(name=name, ok=True, detail=str(function())))
        except Exception as error:
            logger.warning("status check %s failed: %s", name, error)
            checks.append(dict(name=name, ok=False, detail=str(error)))

    check("pipeline_config", lambda: load_pipeline_config(config.pipeline_config) and config.pipeline_config)
    check("calibration", lambda: f"{len(validate_calibration(load_rig(RIG)))} cameras")
    check("data_directory", lambda: config.data_dir if config.data_dir.is_dir() else _raise(f"missing {config.data_dir}"))
    optional = []
    optional.append(dict(name="real_sample", ok=SAMPLE.is_dir(), detail=str(SAMPLE)))
    optional.append(dict(name="segmentation_model", ok=(config.segmentation_model / "config.json").is_file(), detail=str(config.segmentation_model)))
    root = config.nuscenes_root
    optional.append(dict(name="nuscenes", ok=bool(root and (root / config.nuscenes_version).is_dir()), detail=str(root) if root else "nuscenes_root is not configured"))
    ready = all(c["ok"] for c in checks)
    return dict(ready=ready, version=__version__, checks=checks + optional, active_job=manager.active, external_processes=manager.activity(), environment=environment())


def _raise(message: str):
    raise ConfigurationError(message)


@router.get("/config/pipeline", response_model=S.ConfigCheck, summary="Effective pipeline configuration")
def get_pipeline_config(config: Settings = Depends(settings)) -> dict:
    return dict(valid=True, effective=dataclasses.asdict(load_pipeline_config(config.pipeline_config)))


@router.post("/config/validate", response_model=S.ConfigCheck, summary="Validate overrides without applying them")
def validate_config(values: dict) -> dict:
    return dict(valid=True, effective=dataclasses.asdict(pipeline_config_from_dict(values)))


def _sequences(config: Settings) -> list[LabelSequence]:
    out = []
    for path in sorted((config.data_dir / "sequences").glob("*/sequence.json")):
        try:
            out.append(LabelSequence(path.parent))
        except DatasetError as error:
            logger.warning("skipping unreadable sequence %s: %s", path.parent, error)
    return out


@router.get("/datasets", response_model=list[S.DatasetSummary], summary="Inputs available on this machine")
def list_datasets(config: Settings = Depends(settings)) -> list[dict]:
    out = [dict(id=s.id, kind="sequence", source=s.doc["source"], provenance=s.doc["provenance"], frames=len(s), cameras=list(s.calibration.names), segmenter=s.doc.get("segmenter"),
                note="Prepared label sequence.") for s in _sequences(config)]
    if SAMPLE.is_dir():
        out.append(dict(id="cam2bev_example", kind="cam2bev_sample", source="Cam2BEV 1_FRLR example", provenance="simulated", frames=1, cameras=list(load_rig(RIG).names), segmenter=None,
                        note="One frame of semantic images from a simulated dataset, with official top-down labels."))
    if config.nuscenes_root and (config.nuscenes_root / config.nuscenes_version).is_dir():
        from foresightbev.data.nuscenes import CAMERAS, NuScenesDataset

        dataset = NuScenesDataset(config.nuscenes_root, config.nuscenes_version)
        out.append(dict(id="nuscenes", kind="nuscenes", source=f"nuScenes {config.nuscenes_version}", provenance="recorded", frames=sum(s.samples for s in dataset.scenes()), cameras=list(CAMERAS),
                        segmenter=None, note="Real camera data. Needs prepare_nuscenes before pipeline runs. No occupancy labels in this format."))
    return out


@router.get("/datasets/nuscenes/scenes", response_model=list[S.SceneSummary], summary="Scenes in the configured nuScenes root")
def nuscenes_scenes(config: Settings = Depends(settings)) -> list[dict]:
    from foresightbev.data.nuscenes import NuScenesDataset

    if config.nuscenes_root is None:
        raise DatasetNotFound("nuscenes_root is not configured. Set it in configs/backend.yaml.")
    return [dict(name=s.name, description=s.description, samples=s.samples) for s in NuScenesDataset(config.nuscenes_root, config.nuscenes_version).scenes()]


def _calibration_for(dataset: str, config: Settings) -> CalibrationBundle:
    if dataset == "cam2bev_example":
        return load_rig(RIG)
    path = config.data_dir / "sequences" / dataset
    if (path / "sequence.json").is_file():
        return LabelSequence(path).calibration
    raise DatasetNotFound(f"no dataset {dataset}. See GET /api/v1/datasets.")


@router.get("/datasets/{dataset}/calibration", response_model=S.CalibrationStatus, summary="Validated calibration for a dataset")
def dataset_calibration(dataset: S.SafeName, config: Settings = Depends(settings)) -> dict:
    calibration = _calibration_for(dataset, config)
    return dict(dataset=dataset, valid=True, rig=calibration.rig_name, cameras=validate_calibration(calibration))


def _upload_dir(config: Settings, upload: str) -> Path:
    return config.data_dir / "uploads" / upload


@router.put("/uploads/{upload}/frames/{frame}/{camera}", response_model=S.FrameUploadResult, status_code=201, summary="Ingest one colour-coded semantic PNG for the Cam2BEV rig")
async def upload_frame(upload: S.SafeName, frame: int, camera: S.SafeName, request: HttpRequest, config: Settings = Depends(settings)) -> dict:
    rig = load_rig(RIG)
    cam = next((c for c in rig.cameras if c.name == camera), None)
    if cam is None:
        raise DatasetError(f"camera {camera} is not in rig {rig.rig_name}. Valid: {list(rig.names)}")
    if not 0 <= frame < 10000:
        raise DatasetError("frame must be between 0 and 9999")
    body = await request.body()
    if len(body) > config.max_upload_bytes:
        raise DatasetError(f"upload is {len(body)} bytes; the limit is {config.max_upload_bytes}")
    bgr = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR) if body else None
    if bgr is None:
        raise DatasetError("body is not a decodable image")
    if bgr.shape[:2] != (cam.height, cam.width):
        raise DatasetError(f"image is {bgr.shape[1]}x{bgr.shape[0]}, but {camera} is calibrated for {cam.width}x{cam.height}")
    labels = rgb_to_labels(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    folder = _upload_dir(config, upload) / f"{frame:05d}"
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / f"{camera}.png"), labels)
    fractions = {c.name: float((labels == c).mean()) for c in SemanticClass if (labels == c).any()}
    return dict(upload=upload, frame=frame, camera=camera, width=cam.width, height=cam.height, label_fractions=fractions)


@router.post("/uploads/{upload}/finalize", response_model=S.DatasetSummary, status_code=201, summary="Turn uploaded frames into a label sequence")
def finalize_upload(upload: S.SafeName, body: S.FinalizeRequest, config: Settings = Depends(settings)) -> dict:
    folder = _upload_dir(config, upload)
    if not folder.is_dir():
        raise DatasetNotFound(f"no upload named {upload}")
    directories = sorted(p for p in folder.iterdir() if p.is_dir())
    if len(directories) != len(body.frames):
        raise DatasetError(f"{len(directories)} frames were uploaded, but {len(body.frames)} ego states were given")
    if any(b.timestamp <= a.timestamp for a, b in zip(body.frames, body.frames[1:])):
        raise DatasetError("timestamps must strictly increase")
    rig = load_rig(RIG)
    target = config.data_dir / "sequences" / upload
    if target.exists():
        raise JobConflict(f"sequence {upload} already exists")
    provenance = Provenance(body.provenance)

    def stream():
        for index, (directory, e) in enumerate(zip(directories, body.frames)):
            frames = {}
            for cam in rig.cameras:
                path = directory / f"{cam.name}.png"
                labels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path.is_file() else None
                frames[cam.name] = CameraFrame(cam.name, e.timestamp, labels, CameraStatus.OK if labels is not None else CameraStatus.DROPPED)
            yield FrameBundle(upload, index, e.timestamp, frames, rig, EgoState(e.timestamp, e.x, e.y, e.yaw, e.speed, e.yaw_rate), provenance)

    write_sequence(target, upload, stream(), source=body.source, provenance=provenance, segmenter=None, route_for=lambda ego: RouteContext(body.desired_speed))
    sequence = LabelSequence(target)
    return dict(id=upload, kind="sequence", source=body.source, provenance=provenance.value, frames=len(sequence), cameras=list(rig.names), segmenter=None, note="Uploaded. Run the pipeline job on this sequence.")


def _run_summary(doc: dict, data: RunStore) -> dict:
    run = doc.get("id")
    return dict(id=run, kind=doc.get("kind"), title=doc.get("title"), scenario=doc.get("scenario"), seed=doc.get("seed"), mode=doc.get("mode"), frames=doc.get("frames", 0), input=doc.get("input"),
                caveat=doc.get("caveat"), has_arrays=(data.runs / run / "arrays").is_dir())


@router.get("/runs", response_model=list[S.RunSummary], summary="Recorded pipeline runs")
def list_runs(data: RunStore = Depends(store)) -> list[dict]:
    return [_run_summary(doc, data) for doc in data.list()]


@router.get("/runs/{run}", response_model=S.RunDetail, summary="One run with provenance and environment")
def get_run(run: S.SafeName, data: RunStore = Depends(store)) -> dict:
    doc = dict(data.manifest(run), id=run)
    return _run_summary(doc, data) | dict(grid={k: doc["grid"][k] for k in ("x_min", "x_max", "y_min", "y_max", "resolution", "nx", "ny")}, provenance=doc["provenance"],
                                          environment=doc["environment"], summary=doc.get("summary"))


def _grid(data: RunStore, run: str) -> dict:
    g = data.manifest(run)["grid"]
    return {k: g[k] for k in ("x_min", "x_max", "y_min", "y_max", "resolution", "nx", "ny")}


@router.get("/runs/{run}/frames/{frame}/bev", response_model=S.BevOutput, summary="Stage 2 BEV projection")
def bev_output(run: S.SafeName, frame: int, data: RunStore = Depends(store)) -> dict:
    a, record = data.arrays(run, frame), data.frame(run, frame)
    return dict(run=run, frame=frame, timestamp=record["t"], grid=_grid(data, run), method="flat_ground_ipm_nearest", ipm_labels=encode_array(a["ipm_labels"]), support_count=encode_array(a["support_count"]))


@router.get("/runs/{run}/frames/{frame}/occupancy", response_model=S.OccupancyOutput, summary="Stage 3 occupancy with explicit states and reliability")
def occupancy_output(run: S.SafeName, frame: int, include_voxels: bool = False, data: RunStore = Depends(store)) -> dict:
    a, record = data.arrays(run, frame), data.frame(run, frame)
    return dict(run=run, frame=frame, timestamp=record["t"], grid=_grid(data, run), state_legend={s.name: int(s) for s in CellState}, class_legend={c.name: int(c) for c in SemanticClass},
                state_fractions=record["state_fractions"], voxel_state_counts=record["voxel_state_counts"], camera_reliability=record["cameras"], bev_state=encode_array(a["bev_state"]),
                bev_class=encode_array(a["bev_class"]), bev_height_m=encode_array(a["bev_height"].astype(np.float32)), voxel_state=encode_array(a["voxel_state"]) if include_voxels else None)


@router.get("/runs/{run}/frames/{frame}/perception", response_model=S.PerceptionOutput, summary="Stage 4 detections and tracks")
def perception_output(run: S.SafeName, frame: int, data: RunStore = Depends(store)) -> dict:
    record = data.frame(run, frame)
    rows = [r for r in data.manifest(run)["provenance"] if r["stage"] == 4]
    return dict(run=run, frame=frame, timestamp=record["t"], detections=record["detections"], tracks=record["tracks"], provenance=rows)


@router.get("/runs/{run}/frames/{frame}/forecast", response_model=S.ForecastOutput, summary="Stage 5 forecast with explicit timing")
def forecast_output(run: S.SafeName, frame: int, data: RunStore = Depends(store)) -> dict:
    d, record = data.forecast_arrays(run, frame), data.frame(run, frame)
    f = record["forecast"]
    return dict(run=run, frame=frame, t_ref=f["t_ref"], last_input_timestamp=f["last_input"], method=f["method"], uncertainty_semantics=f["uncertainty"], n_samples=f["n_samples"], horizons_s=f["horizons"],
                occupancy_prob=encode_array(d["occupancy_prob"].astype(np.float32) / 255.0), static_occupied=encode_array(d["static_occupied"]), unobserved=encode_array(d["unobserved"]), causal=f["last_input"] <= f["t_ref"])


@router.get("/runs/{run}/frames/{frame}/decision", response_model=S.DecisionRecord, summary="Stage 6 stored decision with candidate evidence")
def decision_output(run: S.SafeName, frame: int, data: RunStore = Depends(store)) -> dict:
    record = data.frame(run, frame)
    d = record["decision"]
    return dict(run=run, frame=frame, timestamp=record["t"], status=d["status"], selected=d["selected"], nominal=d["nominal"], reasons=d["reasons"], fallback=d["fallback"], constraints=d["constraints"],
                candidates=record["candidates"])


def risk_rows(risks, safety: SafetyFilter) -> list[dict]:
    rows = []
    for r in risks:
        problems = safety.admissible(r)
        rows.append(dict(candidate_id=r.candidate_id, event_definition=r.event_definition, event_score=r.event_score, score_semantics=SCORE_SEMANTICS, calibrated_probability=r.calibrated_probability,
                         calibration_status=r.calibration_status, static_collision=r.static_collision, first_contact_time_s=r.first_contact_time, per_horizon_score=r.per_horizon_score.tolist(),
                         unknown_exposure=r.unknown_exposure, distance_to_unknown_m=None if not np.isfinite(r.distance_to_unknown) else r.distance_to_unknown, stopping_distance_m=r.stopping_distance,
                         unknown_corridor_violation=r.unknown_corridor_violation, min_clearance_m=r.min_clearance_m, time_to_conflict_s=r.time_to_conflict_s, closest_track_id=r.closest_track_id,
                         admissible=not problems, problems=[p.value for p in problems]))
    return rows


@router.post("/runs/{run}/frames/{frame}/risk", response_model=S.RiskResponse, summary="Evaluate your trajectories against this frame's stored forecast")
def evaluate_risk(run: S.SafeName, frame: int, body: S.RiskRequest, data: RunStore = Depends(store)) -> dict:
    forecast = data.forecast_volume(run, frame)
    tracks, ego = data.tracks(run, frame)
    record = data.frame(run, frame)
    path = data.manifest(run).get("pipeline_config")
    config = load_pipeline_config(ROOT / path if path else None)
    risk_config, safety_config = config.risk, config.safety
    if body.thresholds.max_event_score is not None:
        safety_config.max_event_score = body.thresholds.max_event_score
    if body.thresholds.max_perception_age_s is not None:
        safety_config.max_perception_age_s = body.thresholds.max_perception_age_s
    supplied = [TrajectoryCandidate(c.id, ego.timestamp, np.array(c.times), np.array(c.poses), is_nominal=i == 0, terminal_lateral=float(c.poses[-1][1])) for i, c in enumerate(body.candidates)]
    stops = [c for c in generate_candidates(ego, RouteContext(body.desired_speed), []) if c.is_fallback]
    candidates = supplied + stops
    risks = RiskEvaluator(risk_config).evaluate_all(candidates, forecast, tracks, ego)
    safety = SafetyFilter(safety_config)
    healthy = sum(c["reliability"] > 0 for c in record["cameras"].values())
    decision = safety.decide(candidates, risks, PerceptionHealth(True, 0.0, healthy, len(record["cameras"])), body.desired_speed)
    return dict(run=run, frame=frame, risks=risk_rows(risks, safety), decision=dict(status=decision.status.value, selected=decision.selected_id, reasons=[c.value for c in decision.reasons], fallback=decision.fallback_action))


@router.post("/jobs", response_model=S.Job, status_code=202, summary="Start pipeline, preparation, evaluation or training work")
def submit_job(body: S.JobRequest, manager: job_module.JobManager = Depends(jobs)) -> dict:
    try:
        return manager.submit(body.kind, body.params)
    except KeyError as error:
        raise ConfigurationError(f"job {body.kind} requires parameter {error}") from error


@router.get("/jobs", response_model=list[S.Job], summary="All jobs, newest first")
def list_jobs(manager: job_module.JobManager = Depends(jobs)) -> list[dict]:
    return manager.list()


def _job(manager: job_module.JobManager, job_id: str) -> dict:
    job = manager.get(job_id)
    if job is None:
        raise DatasetNotFound(f"no job {job_id}")
    return job


@router.get("/jobs/{job_id}", response_model=S.Job, summary="One job")
def get_job(job_id: S.SafeName, manager: job_module.JobManager = Depends(jobs)) -> dict:
    return _job(manager, job_id)


@router.get("/jobs/{job_id}/log", response_model=S.JobLog, summary="Job output")
def job_log(job_id: S.SafeName, lines: int = Query(200, ge=1, le=5000), manager: job_module.JobManager = Depends(jobs)) -> dict:
    _job(manager, job_id)
    return dict(id=job_id, lines=manager.log(job_id, lines))


@router.delete("/jobs/{job_id}", response_model=S.Job, summary="Cancel a running job")
def cancel_job(job_id: S.SafeName, manager: job_module.JobManager = Depends(jobs)) -> dict:
    _job(manager, job_id)
    return manager.cancel(job_id)


@router.get("/experiments", response_model=list[S.ExperimentSummary], summary="Stored evaluation result sets")
def list_experiments(config: Settings = Depends(settings)) -> list[dict]:
    out = []
    for path in sorted(p for p in (config.data_dir / "results").glob("*") if p.is_dir()):
        out.append(dict(name=path.name, has_report=(path / "REPORT.md").is_file(), files=sorted(f.name for f in path.iterdir() if f.is_file())))
    return out


@router.get("/experiments/index", summary="Append-only record of every evaluation run")
def experiment_index(config: Settings = Depends(settings)) -> list[dict]:
    path = config.data_dir / "experiments.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []


@router.get("/experiments/{name}/files/{file}", summary="One stored result file")
def experiment_file(name: S.SafeName, file: S.SafeName, config: Settings = Depends(settings)):
    path = config.data_dir / "results" / name / file
    if not path.is_file():
        raise DatasetNotFound(f"experiment {name} has no file {file}")
    if path.suffix == ".json":
        return json.loads(path.read_text())
    return dict(name=file, text=path.read_text())


@router.get("/logs", summary="Recent structured backend log entries")
def get_logs(request: HttpRequest, lines: int = Query(100, ge=1, le=2000), level: str | None = Query(None, pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")) -> list[dict]:
    return log_module.tail(request.app.state.log_path, lines, level)


@router.get("/diagnostics", summary="Counts that help explain the backend's state")
def diagnostics(request: HttpRequest, config: Settings = Depends(settings), manager: job_module.JobManager = Depends(jobs)) -> dict:
    data_dir = config.data_dir
    recent = log_module.tail(request.app.state.log_path, 1000, "ERROR")
    return dict(uptime_s=time.time() - request.app.state.started, runs=len(list((data_dir / "runs").glob("*/manifest.json"))), sequences=len(list((data_dir / "sequences").glob("*/sequence.json"))),
                jobs={s: sum(j["status"] == s for j in manager.list()) for s in ("running", "succeeded", "failed", "cancelled", "lost")}, recent_errors=recent[-5:],
                settings={k: str(v) for k, v in vars(config).items()})
