"""One-frame orchestration of Stages 2-6 behind typed contracts.

Input is a FrameBundle (camera labels, calibration, ego metadata) and a declared
route. Nothing here reads a simulator: ground truth reaches the pipeline only via
the explicitly named oracle arguments, used by Stage 9 upper-bound ablations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


from foresightbev.contracts import (
    BEVFrame, CellState, Detection, EgoState, ForecastVolume, FrameBundle, OccupancyFrame, SafetyDecision,
    TrackState, TrajectoryCandidate, TrajectoryRisk, VoxelGrid,
)
from foresightbev.data.validation import validate_bundle, validate_calibration
from foresightbev.errors import ConfigurationError, DatasetError
from foresightbev.forecast.forecaster import ConstantVelocityForecaster, ForecastConfig, footprint_polygon, rasterize
from foresightbev.interfaces import Detector, Forecaster
from foresightbev.perception.detection import DetectionConfig, detect, range_noise_coefficient
from foresightbev.perception.ground import fit_plane
from foresightbev.perception.image_detection import ImageDetectionConfig, ImageSizeDetector
from foresightbev.perception.occupancy import OccupancyConfig, OccupancyEstimator
from foresightbev.risk.calibration import RiskCalibrator
from foresightbev.risk.evaluator import RiskConfig, RiskEvaluator
from foresightbev.risk.safety_filter import PerceptionHealth, SafetyConfig, SafetyFilter
from foresightbev.risk.trajectories import PlannerConfig, RouteContext, generate_candidates
from foresightbev.tracking.tracker import Tracker, TrackerConfig


GROUND_CLASSES = ("car", "truck", "bus", "pedestrian")


def fuse_detections(occupancy: OccupancyFrame, detections: list[Detection]) -> None:
    """Mark model boxes occupied in a separate, labelled layer. Never touches INVALID cells."""
    grid = occupancy.grid.bev
    layer = np.zeros(grid.shape, bool)
    for d in detections:
        mask = rasterize(footprint_polygon(d.center_ego[0], d.center_ego[1], d.yaw or 0.0, d.length, d.width), grid) & (occupancy.bev_state != CellState.INVALID)
        layer |= mask
        occupancy.bev_class[mask] = int(d.semantic)
    occupancy.bev_state[layer] = CellState.OCCUPIED
    occupancy.model_occupied = layer


@dataclass
class PipelineConfig:
    grid: VoxelGrid = field(default_factory=VoxelGrid)
    occupancy: OccupancyConfig = field(default_factory=OccupancyConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    image_detection: ImageDetectionConfig = field(default_factory=ImageDetectionConfig)
    detector: str = "bev_clusters"
    ground_plane: str = "flat"
    fuse_model_boxes: bool = False
    detector_score_threshold: float = 0.3
    nominal_mode: str = "idm"
    filter_enabled: bool = True
    seed: int = 0


@dataclass
class FrameResult:
    bundle: FrameBundle
    bev: BEVFrame
    occupancy: OccupancyFrame
    detections: list[Detection]
    tracks: list[TrackState]
    forecast: ForecastVolume
    candidates: list[TrajectoryCandidate]
    risks: list[TrajectoryRisk]
    decision: SafetyDecision
    timings_ms: dict[str, float]
    forecast_config: ForecastConfig | None = None


class ForesightPipeline:
    """Stateful per sequence: the tracker holds past frames, never future ones.

    Detector and forecaster are replaceable, and must keep to foresightbev.interfaces.
    A learned forecaster without joint samples cannot be used here, by design.
    """

    def __init__(self, calibration, config: PipelineConfig | None = None, calibrator: RiskCalibrator | None = None,
                 detector: Detector | None = None, forecaster: Forecaster | None = None):
        self.config = config or PipelineConfig()
        cfg = self.config
        validate_calibration(calibration)
        self.occupancy = OccupancyEstimator(calibration, cfg.grid, cfg.occupancy)
        if cfg.detection.range_std_per_m2 == 0.0:
            cfg.detection.range_std_per_m2 = range_noise_coefficient(calibration)
        if detector is not None:
            self.detector = detector
        elif cfg.detector == "image_size":
            self.detector = ImageSizeDetector(calibration, cfg.image_detection)
        elif cfg.detector == "streampetr":
            raise ConfigurationError("detector streampetr reads cached model outputs. Use replay.build_pipeline with a prepared sequence.")
        else:
            self.detector = lambda occupancy, bundle: detect(occupancy, cfg.detection)
        self.tracker = Tracker(cfg.tracker)
        self.forecaster = forecaster or ConstantVelocityForecaster(cfg.grid.bev, cfg.forecast, cfg.seed)
        self.evaluator = RiskEvaluator(cfg.risk, calibrator)
        self.filter = SafetyFilter(cfg.safety)
        self.total_cameras = len(calibration.cameras)

    def step(self, bundle: FrameBundle, route: RouteContext, *, now: float | None = None, oracle_tracks: list[TrackState] | None = None,
             oracle_occupancy=None) -> FrameResult:
        """Process one frame. ``now`` is the decision time; it defaults to the frame time.

        Decisions made later than the frame carry a positive perception age, so a
        delayed bundle is judged stale by the safety filter.
        """
        cfg = self.config
        validate_bundle(bundle)
        marks = [time.perf_counter()]
        early = self.detector(None, bundle) if cfg.detector == "streampetr" else None
        plane = None
        if cfg.ground_plane == "detections" and early is not None:
            bottoms = [(d.center_ego[0], d.center_ego[1], d.bottom_z) for d in early if d.support >= cfg.detector_score_threshold and d.model_class in GROUND_CLASSES]
            plane = fit_plane(np.array(bottoms, float).reshape(-1, 3))
        bev, occupancy = self.occupancy.estimate(bundle, plane)
        occupancy.ground_plane = plane
        if oracle_occupancy is not None:
            occupancy.bev_state, occupancy.bev_class = oracle_occupancy
        if cfg.fuse_model_boxes and early is not None:
            fuse_detections(occupancy, [d for d in early if d.support >= cfg.detector_score_threshold])
        marks.append(time.perf_counter())
        all_detections = early if early is not None else self.detector(occupancy, bundle)
        detections = [d for d in all_detections if d.support >= cfg.detector_score_threshold] if early is not None else all_detections
        marks.append(time.perf_counter())
        tracks = oracle_tracks if oracle_tracks is not None else self.tracker.update(all_detections, bundle.ego)
        marks.append(time.perf_counter())
        forecast = self.forecaster.forecast(occupancy, tracks, bundle.ego)
        marks.append(time.perf_counter())
        candidates = generate_candidates(bundle.ego, route, tracks, cfg.planner, cfg.nominal_mode)
        risks = self.evaluator.evaluate_all(candidates, forecast, tracks, bundle.ego)
        marks.append(time.perf_counter())
        decided_at = bundle.timestamp if now is None else now
        if decided_at < bundle.timestamp - 1e-9:
            raise DatasetError(f"decision time {decided_at} is before frame time {bundle.timestamp}")
        healthy = sum(score > 0 for score in occupancy.reliability.values())
        health = PerceptionHealth(True, decided_at - occupancy.timestamp, healthy, len(occupancy.reliability))
        if cfg.filter_enabled:
            decision = self.filter.decide(candidates, risks, health, route.desired_speed, route.home_lane)
        else:
            decision = passthrough(candidates)
        marks.append(time.perf_counter())
        names = ("occupancy", "detection", "tracking", "forecast", "risk", "safety")
        timings = {n: 1000 * (b - a) for n, a, b in zip(names, marks, marks[1:])}
        return FrameResult(bundle, bev, occupancy, detections, tracks, forecast, candidates, risks, decision, timings, cfg.forecast)

    def decide_without_perception(self, ego: EgoState, route: RouteContext) -> SafetyDecision:
        """No frame arrived. Missing perception is never a clear road: always a fallback."""
        candidates = generate_candidates(ego, route, [], self.config.planner, self.config.nominal_mode)
        return self.filter.decide(candidates, [], PerceptionHealth(False, float("inf"), 0, self.total_cameras),
                                  route.desired_speed, route.home_lane)


def passthrough(candidates: list[TrajectoryCandidate]) -> SafetyDecision:
    """No-filter ablation: the nominal plan is executed unchecked."""
    from foresightbev.contracts import DecisionStatus

    nominal = candidates[0]
    return SafetyDecision(DecisionStatus.PASS, nominal.candidate_id, nominal, [], None, nominal.candidate_id, {})
