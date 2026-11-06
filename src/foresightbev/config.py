"""Configuration: strict pipeline YAML plus backend settings from file and environment.

Pipeline YAML mirrors PipelineConfig sections. Unknown keys and wrong types are
errors, so a typo can never silently leave a default safety threshold in place.
"""

from __future__ import annotations

import dataclasses
import os
import typing
from dataclasses import dataclass
from pathlib import Path

import yaml

from foresightbev.contracts import BEVGrid, VoxelGrid
from foresightbev.errors import ConfigurationError
from foresightbev.pipeline import PipelineConfig

ROOT = Path(__file__).resolve().parents[2]


def _apply(target, values: dict, path: str) -> None:
    if not isinstance(values, dict):
        raise ConfigurationError(f"{path} must be a mapping")
    fields = {f.name: f for f in dataclasses.fields(target)}
    hints = typing.get_type_hints(type(target))
    for key, value in values.items():
        if key not in fields:
            raise ConfigurationError(f"unknown setting {path}.{key}. Valid: {sorted(fields)}")
        current = getattr(target, key)
        if dataclasses.is_dataclass(current):
            _apply(current, value, f"{path}.{key}")
            continue
        expected = hints[key]
        if expected is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if isinstance(current, tuple):
            if not isinstance(value, list) or len(value) == 0:
                raise ConfigurationError(f"{path}.{key} must be a non-empty list")
            value = tuple(value)
        elif type(value) is not type(current):
            raise ConfigurationError(f"{path}.{key} must be {type(current).__name__}, got {type(value).__name__}")
        setattr(target, key, value)


def validate_pipeline_config(config: PipelineConfig) -> PipelineConfig:
    s, r, f = config.safety, config.risk, config.forecast
    checks = [(0.0 <= s.max_event_score <= 1.0, "safety.max_event_score must be in [0, 1]"),
              (s.max_perception_age_s > 0, "safety.max_perception_age_s must be positive"),
              (s.min_healthy_cameras >= 1, "safety.min_healthy_cameras must be at least 1"),
              (r.stop_decel > 0, "risk.stop_decel must be positive"), (r.phantom_speed >= 0, "risk.phantom_speed must not be negative"),
              (f.n_samples >= 1, "forecast.n_samples must be at least 1"), (f.step_s > 0 and f.horizon_s >= f.step_s, "forecast horizon and step are inconsistent"),
              (config.planner.horizon_s >= f.horizon_s, "planner.horizon_s must cover the forecast horizon"),
              (config.grid.bev.resolution > 0, "grid resolution must be positive"), (config.nominal_mode in ("idm", "blind"), "nominal_mode must be idm or blind"),
              (config.detector in ("bev_clusters", "image_size", "streampetr"), "detector must be bev_clusters, image_size or streampetr"),
              (0.0 <= config.detector_score_threshold <= 1.0, "detector_score_threshold must be in [0, 1]"),
              (config.ground_plane in ("flat", "detections"), "ground_plane must be flat or detections"),
              (config.ground_plane == "flat" or config.detector == "streampetr", "ground_plane detections needs detector streampetr"),
              (not config.fuse_model_boxes or config.detector == "streampetr", "fuse_model_boxes needs detector streampetr"),
              (config.occupancy.ground_height_tolerance_m >= 0, "occupancy.ground_height_tolerance_m must not be negative")]
    for ok, message in checks:
        if not ok:
            raise ConfigurationError(message)
    return config


def pipeline_config_from_dict(values: dict | None) -> PipelineConfig:
    config = PipelineConfig()
    values = dict(values or {})
    grid = values.pop("grid", None)
    if grid is not None:
        grid = dict(grid)
        bev = BEVGrid(**grid.pop("bev", {}))
        config.grid = VoxelGrid(bev=bev, **grid)
    _apply(config, values, "pipeline")
    return validate_pipeline_config(config)


def load_pipeline_config(path: str | Path | None) -> PipelineConfig:
    if path is None:
        return validate_pipeline_config(PipelineConfig())
    path = Path(path)
    if not path.is_file():
        raise ConfigurationError(f"pipeline config not found: {path}")
    try:
        return pipeline_config_from_dict(yaml.safe_load(path.read_text()) or {})
    except (yaml.YAMLError, TypeError) as error:
        raise ConfigurationError(f"cannot parse {path}: {error}") from error


@dataclass(frozen=True)
class Settings:
    """Backend settings. Precedence: environment, then configs/backend.yaml, then defaults."""

    data_dir: Path
    nuscenes_root: Path | None
    nuscenes_version: str
    segmentation_model: Path
    pipeline_config: Path
    log_level: str
    max_upload_bytes: int


def load_settings(file: Path | None = None) -> Settings:
    file = file or Path(os.environ.get("FORESIGHTBEV_CONFIG", ROOT / "configs/backend.yaml"))
    doc: dict = {}
    if file.is_file():
        try:
            doc = yaml.safe_load(file.read_text()) or {}
        except yaml.YAMLError as error:
            raise ConfigurationError(f"cannot parse {file}: {error}") from error
    known = {"data_dir", "nuscenes_root", "nuscenes_version", "segmentation_model", "pipeline_config", "log_level", "max_upload_bytes"}
    if set(doc) - known:
        raise ConfigurationError(f"unknown backend settings {sorted(set(doc) - known)}. Valid: {sorted(known)}")

    def value(key: str, default):
        return os.environ.get(f"FORESIGHTBEV_{key.upper()}", doc.get(key, default))

    def path(key: str, default):
        raw = value(key, default)
        if raw in (None, ""):
            return None
        resolved = Path(raw).expanduser()
        return resolved if resolved.is_absolute() else ROOT / resolved

    return Settings(path("data_dir", "experiments/end_to_end"), path("nuscenes_root", None), str(value("nuscenes_version", "v1.0-mini")),
                    path("segmentation_model", "experiments/end_to_end/models/segformer-b2-cityscapes"),
                    path("pipeline_config", "configs/pipeline_default.yaml"), str(value("log_level", "INFO")).upper(),
                    int(value("max_upload_bytes", 32 * 1024 * 1024)))
