"""Typed API contracts. Requests reject unknown fields; responses state their units.

Coordinates: ``*_ego`` values are metres in the ego frame at the frame's timestamp,
x forward, y left. Velocities are m/s. Arrays are indexed as in contracts.py.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SafeName = Annotated[str, Field(pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,99}$")]
Status = Literal["pass", "modify", "fallback"]


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class Health(BaseModel):
    status: Literal["ok"]
    version: str


class Check(BaseModel):
    name: str
    ok: bool
    detail: str


class SystemStatus(BaseModel):
    ready: bool
    version: str
    checks: list[Check]
    active_job: str | None
    external_processes: list[str]
    environment: dict


class DatasetSummary(BaseModel):
    id: str
    kind: Literal["sequence", "cam2bev_sample", "nuscenes"]
    source: str
    provenance: str
    frames: int | None
    cameras: list[str]
    segmenter: str | None
    note: str


class CalibrationStatus(BaseModel):
    dataset: str
    valid: bool
    rig: str
    cameras: list[CalibrationCamera]


class SceneSummary(BaseModel):
    name: str
    description: str
    samples: int


class ArrayPayload(BaseModel):
    shape: list[int]
    dtype: str
    encoding: Literal["zlib+base64"]
    data: str


class GridInfo(BaseModel):
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    resolution: float
    nx: int
    ny: int


class SequenceRun(BaseModel):
    id: str
    mode: str | None
    fault: dict | None
    frames: int | None


class SequenceSummary(BaseModel):
    name: str
    description: str
    samples: int
    split: Literal["mini_val", "mini_train"]
    detector_saw_scene_in_training: bool
    status_note: str
    night: bool
    runs: list[SequenceRun]


class TimelineRow(BaseModel):
    frame: int
    t: float
    speed: float
    status: Status
    selected: str
    reasons: list[str]
    tracks: int
    detections: int
    healthy_cameras: int


class CalibrationCamera(BaseModel):
    name: str
    width: int
    height: int
    fx: float
    fy: float
    origin_ego: list[float]
    forward_ego: list[float]
    horizontal_fov_deg: float


class CameraView(BaseModel):
    name: str
    status: str
    reliability: float
    timestamp: float
    skew_ms: float
    calibration: CalibrationCamera


class TrackPoint(BaseModel):
    frame: int
    x: float
    y: float
    speed: float


class TrackHistory(BaseModel):
    id: int
    cls: str
    first_frame: int
    last_frame: int
    points: list[TrackPoint]


class PlanRequest(Request):
    target_speed: float = Field(ge=0, le=25, description="m/s")
    lateral_offset: float = Field(default=0.0, ge=-6, le=6, description="Metres, left positive.")


class RunSummary(BaseModel):
    id: str
    kind: str | None
    title: str | None
    scenario: str | None
    seed: int | None
    mode: str | None
    frames: int
    input: str | None
    caveat: str | None
    has_arrays: bool


class RunDetail(RunSummary):
    grid: GridInfo
    provenance: list[dict]
    environment: dict
    summary: dict | None


class BevOutput(BaseModel):
    run: str
    frame: int
    timestamp: float
    grid: GridInfo
    method: str
    ipm_labels: ArrayPayload
    support_count: ArrayPayload
    note: str = "Flat-ground IPM labels with per-cell camera support. Not an occupancy claim."


class OccupancyOutput(BaseModel):
    run: str
    frame: int
    timestamp: float
    grid: GridInfo
    state_legend: dict[str, int]
    class_legend: dict[str, int]
    state_fractions: dict[str, float]
    voxel_state_counts: dict[str, int]
    camera_reliability: dict[str, dict]
    bev_state: ArrayPayload
    bev_class: ArrayPayload
    bev_height_m: ArrayPayload
    voxel_state: ArrayPayload | None
    reliability_semantics: str = "Heuristic usable-label fraction. Not a probability, OOD score or epistemic uncertainty."


class PerceptionOutput(BaseModel):
    run: str
    frame: int
    timestamp: float
    detections: list[dict]
    tracks: list[dict]
    provenance: list[dict]


class ForecastOutput(BaseModel):
    run: str
    frame: int
    t_ref: float
    last_input_timestamp: float
    method: str
    uncertainty_semantics: str
    n_samples: int
    horizons_s: list[float]
    occupancy_prob: ArrayPayload
    static_occupied: ArrayPayload
    unobserved: ArrayPayload
    causal: bool


class DecisionRecord(BaseModel):
    run: str
    frame: int
    timestamp: float
    status: Status
    selected: str
    nominal: str | None
    reasons: list[str]
    fallback: str | None
    constraints: dict
    candidates: list[dict]


class TrajectoryInput(Request):
    id: str = Field(pattern=r"^[A-Za-z0-9_+.-]{1,64}$")
    times: list[float] = Field(min_length=2, description="Seconds from the frame time, starting at 0, strictly increasing.")
    poses: list[list[float]] = Field(description="[x, y, yaw, speed] in the ego frame: metres, radians, m/s.")

    @model_validator(mode="after")
    def check(self):
        if len(self.times) != len(self.poses):
            raise ValueError("times and poses must have the same length")
        if self.times[0] != 0 or any(b <= a for a, b in zip(self.times, self.times[1:])):
            raise ValueError("times must start at 0 and strictly increase")
        if any(len(p) != 4 for p in self.poses):
            raise ValueError("each pose must be [x, y, yaw, speed]")
        if any(not all(map(_finite, p)) for p in self.poses) or any(p[3] < 0 for p in self.poses):
            raise ValueError("poses must be finite with non-negative speed")
        return self


def _finite(value: float) -> bool:
    return value == value and abs(value) != float("inf")


class Thresholds(Request):
    max_event_score: float | None = Field(default=None, ge=0, le=1)
    max_perception_age_s: float | None = Field(default=None, gt=0)


class RiskRequest(Request):
    candidates: list[TrajectoryInput] = Field(min_length=1, max_length=32, description="The first candidate is treated as the nominal plan.")
    desired_speed: float = Field(default=10.0, gt=0, le=50)
    thresholds: Thresholds = Thresholds()


class RiskRow(BaseModel):
    candidate_id: str
    event_definition: str
    event_score: float
    score_semantics: str
    calibrated_probability: float | None
    calibration_status: str
    static_collision: bool
    first_contact_time_s: float | None
    per_horizon_score: list[float]
    unknown_exposure: float
    distance_to_unknown_m: float | None
    stopping_distance_m: float
    unknown_corridor_violation: bool
    min_clearance_m: float | None
    time_to_conflict_s: float | None
    closest_track_id: int | None
    admissible: bool
    problems: list[str]


class RiskResponse(BaseModel):
    run: str
    frame: int
    risks: list[RiskRow]
    decision: dict
    path: list[list[float]] | None = None


class JobRequest(Request):
    kind: Literal["sample", "episode", "prepare_cam2bev", "prepare_nuscenes", "pipeline", "evaluation", "evaluate_nuscenes", "train_forecaster", "showcase"]
    params: dict = {}

    @field_validator("params")
    @classmethod
    def names_are_safe(cls, value: dict) -> dict:
        import re

        for key in ("name", "sequence"):
            if key in value and not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,99}", str(value[key])):
                raise ValueError(f"{key} has invalid characters")
        for key in ("config", "pipeline_config"):
            if key in value and not re.fullmatch(r"configs/[A-Za-z0-9_.-]+\.yaml", str(value[key])):
                raise ValueError(f"{key} must be a file directly under configs/")
        return value


class Job(BaseModel):
    id: str
    kind: str
    params: dict
    status: Literal["queued", "running", "succeeded", "failed", "cancelled", "lost"]
    command: list[str]
    created_unix: float
    started_unix: float | None
    finished_unix: float | None
    returncode: int | None
    error: str | None


class JobLog(BaseModel):
    id: str
    lines: list[str]


class ExperimentSummary(BaseModel):
    name: str
    has_report: bool
    files: list[str]


class FrameUploadResult(BaseModel):
    upload: str
    frame: int
    camera: str
    width: int
    height: int
    label_fractions: dict[str, float]


class EgoInput(Request):
    timestamp: float
    x: float
    y: float
    yaw: float
    speed: float = Field(default=0.0, ge=0)
    yaw_rate: float = 0.0


class FinalizeRequest(Request):
    source: str = Field(min_length=1, max_length=100, description="Where these images came from. Stored as provenance.")
    provenance: Literal["recorded", "simulated"]
    frames: list[EgoInput] = Field(min_length=1, description="One entry per uploaded frame, in order, with strictly increasing timestamps.")
    desired_speed: float = Field(default=8.0, gt=0, le=50)


class ConfigCheck(BaseModel):
    valid: bool
    effective: dict
