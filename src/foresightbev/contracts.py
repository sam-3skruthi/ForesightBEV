"""Typed contracts shared by all Tavyra stages.

Conventions (authoritative for the whole package):

* Ego frame: x forward, y left, z up, right-handed, metres, origin on the ground
  plane under the vehicle reference point (matches Cam2BEV vehicle coordinates).
* World frame: right-handed, z up, metres; yaw is counter-clockwise about +z.
* Camera optical frame: x right, y down, z forward (OpenCV).
* ``T_a_b`` maps homogeneous points expressed in frame ``b`` into frame ``a``.
* BEV arrays are indexed ``[ix, iy]``; voxel arrays ``[iz, ix, iy]``. Index 0 is at
  ``x_min`` / ``y_min`` / ``z_min``.
* Times are seconds. Forecast flow is *forward* flow: metres per second that the
  content of a cell moves, expressed in the reference ego frame at ``t_ref``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum

import numpy as np


class Provenance(str, Enum):
    """What kind of thing produced an output. Reported on every stage output."""

    TRAINED = "trained"
    PRETRAINED = "pretrained"
    CLASSICAL = "classical"
    HEURISTIC = "heuristic"
    SIMULATED = "simulated"
    RECORDED = "recorded"
    ORACLE = "oracle"
    DEFERRED = "deferred"


class CellState(IntEnum):
    """Occupancy state. These are mutually exclusive, distinct semantics.

    FREE needs positive evidence. UNKNOWN is inside camera coverage but
    unobserved (occluded or unresolved). OUTSIDE_FOV has no camera coverage at
    all. INVALID is excluded by design, such as the ego body.
    """

    INVALID = 0
    FREE = 1
    OCCUPIED = 2
    UNKNOWN = 3
    OUTSIDE_FOV = 4


class SemanticClass(IntEnum):
    UNLABELED = 0
    ROAD = 1
    SIDEWALK = 2
    OTHER_GROUND = 3
    SKY = 4
    STATIC = 5
    VEHICLE = 6
    PERSON = 7


GROUND_CLASSES = (SemanticClass.ROAD, SemanticClass.SIDEWALK, SemanticClass.OTHER_GROUND)
OBSTACLE_CLASSES = (SemanticClass.STATIC, SemanticClass.VEHICLE, SemanticClass.PERSON)
DYNAMIC_CLASSES = (SemanticClass.VEHICLE, SemanticClass.PERSON)


class CameraStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    DROPPED = "dropped"


@dataclass(frozen=True)
class CameraCalibration:
    """Pinhole camera. ``T_cam_ego`` maps ego-frame points into the optical frame."""

    name: str
    K: np.ndarray
    T_cam_ego: np.ndarray
    width: int
    height: int
    distortion: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.K.shape != (3, 3) or self.T_cam_ego.shape != (4, 4):
            raise ValueError(f"{self.name}: K must be 3x3 and T_cam_ego 4x4")
        if not (np.isfinite(self.K).all() and np.isfinite(self.T_cam_ego).all()):
            raise ValueError(f"{self.name}: calibration has non-finite values")
        rotation = self.T_cam_ego[:3, :3]
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
            raise ValueError(f"{self.name}: rotation is not orthonormal")
        if self.distortion and any(abs(c) > 0 for c in self.distortion):
            raise ValueError(f"{self.name}: distortion is not supported; undistort first")

    @property
    def T_ego_cam(self) -> np.ndarray:
        rotation, translation = self.T_cam_ego[:3, :3], self.T_cam_ego[:3, 3]
        out = np.eye(4)
        out[:3, :3] = rotation.T
        out[:3, 3] = -rotation.T @ translation
        return out

    @property
    def origin_ego(self) -> np.ndarray:
        return self.T_ego_cam[:3, 3]

    def scaled(self, factor: float) -> "CameraCalibration":
        """Calibration after resizing the image; K and size must change together."""
        K = self.K.copy()
        K[:2] *= factor
        return CameraCalibration(self.name, K, self.T_cam_ego,
                                 round(self.width * factor), round(self.height * factor))


@dataclass(frozen=True)
class CalibrationBundle:
    cameras: tuple[CameraCalibration, ...]
    rig_name: str

    def __post_init__(self) -> None:
        names = [c.name for c in self.cameras]
        if len(set(names)) != len(names) or not names:
            raise ValueError(f"camera names must be unique and non-empty: {names}")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.cameras)


@dataclass(frozen=True)
class EgoState:
    """Ego pose in the world frame plus speed, from declared vehicle metadata."""

    timestamp: float
    x: float
    y: float
    yaw: float
    speed: float = 0.0
    yaw_rate: float = 0.0

    def T_world_ego(self) -> np.ndarray:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.array([[c, -s, self.x], [s, c, self.y], [0.0, 0.0, 1.0]])

    def world_to_ego(self, xy: np.ndarray) -> np.ndarray:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        d = np.asarray(xy, float) - (self.x, self.y)
        return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)

    def rotate_to_ego(self, v: np.ndarray) -> np.ndarray:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        v = np.asarray(v, float)
        return np.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1]], -1)

    def ego_to_world(self, xy: np.ndarray) -> np.ndarray:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        p = np.asarray(xy, float)
        return np.stack([c * p[..., 0] - s * p[..., 1] + self.x,
                         s * p[..., 0] + c * p[..., 1] + self.y], -1)


@dataclass
class CameraFrame:
    """One camera at one time.

    ``labels`` holds SemanticClass ids, or None until segmented or if lost. ``image``
    optionally holds the RGB image that labels were derived from. The geometry
    stages read only ``labels``.
    """

    camera: str
    timestamp: float
    labels: np.ndarray | None
    status: CameraStatus = CameraStatus.OK
    image: np.ndarray | None = None


@dataclass
class FrameBundle:
    """Synchronised camera frames: the only perception input to the pipeline."""

    sequence_id: str
    frame_index: int
    timestamp: float
    frames: dict[str, CameraFrame]
    calibration: CalibrationBundle
    ego: EgoState
    source: Provenance


@dataclass(frozen=True)
class BEVGrid:
    x_min: float = -30.0
    x_max: float = 50.0
    y_min: float = -22.0
    y_max: float = 22.0
    resolution: float = 0.25

    @property
    def nx(self) -> int:
        return int(round((self.x_max - self.x_min) / self.resolution))

    @property
    def ny(self) -> int:
        return int(round((self.y_max - self.y_min) / self.resolution))

    @property
    def shape(self) -> tuple[int, int]:
        return self.nx, self.ny

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        xs = self.x_min + (np.arange(self.nx) + 0.5) * self.resolution
        ys = self.y_min + (np.arange(self.ny) + 0.5) * self.resolution
        return np.meshgrid(xs, ys, indexing="ij")

    def to_index(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        ix = np.floor((np.asarray(x) - self.x_min) / self.resolution).astype(int)
        iy = np.floor((np.asarray(y) - self.y_min) / self.resolution).astype(int)
        return ix, iy

    def in_bounds(self, ix, iy) -> np.ndarray:
        ix, iy = np.asarray(ix), np.asarray(iy)
        return (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)


@dataclass(frozen=True)
class VoxelGrid:
    bev: BEVGrid = field(default_factory=BEVGrid)
    z_min: float = 0.0
    z_max: float = 3.0
    dz: float = 0.5

    @property
    def nz(self) -> int:
        return int(round((self.z_max - self.z_min) / self.dz))

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.nz, self.bev.nx, self.bev.ny

    def z_centers(self) -> np.ndarray:
        return self.z_min + (np.arange(self.nz) + 0.5) * self.dz


@dataclass
class BEVFrame:
    """Stage 2 output: labels warped to the ground plane, not an occupancy claim."""

    timestamp: float
    grid: BEVGrid
    labels: np.ndarray
    per_camera_support: dict[str, np.ndarray]
    method: str
    provenance: Provenance


@dataclass
class OccupancyFrame:
    """Stage 3 output, in the ego frame at ``timestamp``.

    ``reliability`` values are heuristic per-camera health scores. They are not
    calibrated probabilities, an OOD score, or epistemic uncertainty.
    """

    timestamp: float
    grid: VoxelGrid
    voxel_state: np.ndarray
    voxel_class: np.ndarray
    bev_state: np.ndarray
    bev_class: np.ndarray
    bev_height: np.ndarray
    support_count: np.ndarray
    reliability: dict[str, float]
    method: str
    provenance: Provenance
    model_occupied: np.ndarray | None = None
    ground_plane: tuple[float, float, float] | None = None


@dataclass
class Detection:
    """Stage 4 metric detection in the ego frame.

    ``support`` is a heuristic evidence score in [0, 1], not a probability.
    """

    timestamp: float
    semantic: SemanticClass
    center_ego: np.ndarray
    length: float
    width: float
    height: float
    support: float
    n_cells: int
    provenance: Provenance = Provenance.CLASSICAL
    covariance: np.ndarray | None = None
    yaw: float | None = None
    velocity_ego: np.ndarray | None = None
    model_class: str | None = None
    bottom_z: float | None = None


@dataclass
class TrackState:
    """Kalman track in the world frame. Covariance comes from the filter's model."""

    track_id: int
    timestamp: float
    semantic: SemanticClass
    position: np.ndarray
    velocity: np.ndarray
    covariance: np.ndarray
    length: float
    width: float
    height: float
    age: int
    hits: int
    misses: int
    confirmed: bool
    history: list[tuple[float, float, float]] = field(default_factory=list)


@dataclass
class ForecastVolume:
    """Stage 5 output in the ego frame at ``t_ref``. Must not use data after t_ref."""

    t_ref: float
    last_input_timestamp: float
    horizons: np.ndarray
    grid: BEVGrid
    occupancy_prob: np.ndarray
    flow: np.ndarray
    static_occupied: np.ndarray
    unobserved: np.ndarray
    agent_samples: list[np.ndarray]
    n_samples: int
    method: str
    uncertainty: str
    provenance: Provenance


@dataclass
class TrajectoryCandidate:
    """Timestamped poses (x, y, yaw, speed) in the ego frame at ``t_ref``."""

    candidate_id: str
    t_ref: float
    times: np.ndarray
    poses: np.ndarray
    is_nominal: bool = False
    is_fallback: bool = False
    terminal_lateral: float = 0.0


class ReasonCode(str, Enum):
    NOMINAL_ADMISSIBLE = "NOMINAL_ADMISSIBLE"
    NOMINAL_EVENT_SCORE_EXCEEDED = "NOMINAL_EVENT_SCORE_EXCEEDED"
    NOMINAL_STATIC_COLLISION = "NOMINAL_STATIC_COLLISION"
    NOMINAL_UNKNOWN_CORRIDOR = "NOMINAL_UNKNOWN_CORRIDOR"
    POST_FALLBACK_HOLD = "POST_FALLBACK_HOLD"
    COVERAGE_SPEED_HOLD = "COVERAGE_SPEED_HOLD"
    PERCEPTION_MISSING = "PERCEPTION_MISSING"
    PERCEPTION_STALE = "PERCEPTION_STALE"
    CAMERA_COVERAGE_LOST = "CAMERA_COVERAGE_LOST"
    NO_ADMISSIBLE_CANDIDATE = "NO_ADMISSIBLE_CANDIDATE"
    FALLBACK_COMFORT_STOP = "FALLBACK_COMFORT_STOP"
    FALLBACK_EMERGENCY_STOP = "FALLBACK_EMERGENCY_STOP"


class DecisionStatus(str, Enum):
    PASS = "pass"
    MODIFY = "modify"
    FALLBACK = "fallback"


@dataclass
class TrajectoryRisk:
    """Stage 6 event estimate for one candidate.

    ``event_score`` is a Monte Carlo frequency under the forecast's own motion
    model. It is not a collision probability. ``calibrated_probability`` is set
    only when a fitted calibrator was applied, and ``calibration_status`` then
    states the data it was fitted on.
    """

    candidate_id: str
    event_definition: str
    event_score: float
    calibrated_probability: float | None
    calibration_status: str
    per_horizon_score: np.ndarray
    static_collision: bool
    first_contact_time: float | None
    unknown_exposure: float
    distance_to_unknown: float
    stopping_distance: float
    unknown_corridor_violation: bool
    min_clearance_m: float | None = None
    time_to_conflict_s: float | None = None
    closest_track_id: int | None = None


@dataclass
class SafetyDecision:
    status: DecisionStatus
    selected_id: str
    selected: TrajectoryCandidate
    reasons: list[ReasonCode]
    fallback_action: str | None
    nominal_id: str | None
    active_constraints: dict[str, float]
