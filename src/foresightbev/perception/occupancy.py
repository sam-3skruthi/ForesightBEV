"""Stages 2-3: IPM BEV and semantic space carving into a 3D occupancy grid.

Classical multi-view geometry, no learned weights. For every camera:

1. Ground cells are looked up in the semantic image (IPM under a flat-ground
   assumption). Along each azimuth, the nearest ground cell showing an obstacle
   label is taken as that obstacle's contact range.
2. Each voxel is classified by the ray through it. A ground or sky pixel means the
   ray passed through the voxel, so it is free. An obstacle pixel means free in
   front of the contact range, occupied in a thin band at it, and unknown behind.

Cameras are fused so that positive free evidence from any camera wins, then
occupied, then unknown. Voxels no healthy camera covers are OUTSIDE_FOV.

With ``ground_height_tolerance_m`` > 0, free evidence stops at the nearest range the
contact could really be at if the ground were that much higher than assumed. A contact
seen at flat-ground range d may truly lie at d * h / (h + dz) for camera height h. The
span between is UNKNOWN, because a wrong FREE is the dangerous error.

Limits: flat ground, and only the visible surface of an obstacle is recovered, so
this is an observation model and not a semantic occupancy network.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import minimum_filter1d

from foresightbev.contracts import (
    BEVFrame, CalibrationBundle, CameraStatus, CellState, FrameBundle, OccupancyFrame, Provenance,
    SemanticClass, VoxelGrid,
)
from foresightbev.geometry.projection import AZIMUTH_BINS, CameraLookup, build_lookup

logger = logging.getLogger(__name__)
METHOD = "semantic_space_carving_v1"
_GROUND_OR_SKY = np.zeros(256, bool)
_GROUND_OR_SKY[[SemanticClass.ROAD, SemanticClass.SIDEWALK, SemanticClass.OTHER_GROUND, SemanticClass.SKY]] = True
_OBSTACLE = np.zeros(256, bool)
_OBSTACLE[[SemanticClass.STATIC, SemanticClass.VEHICLE, SemanticClass.PERSON]] = True


@dataclass
class OccupancyConfig:
    contact_band_m: float = 0.6
    range_slack_m: float = 0.25
    min_valid_label_fraction: float = 0.5
    max_camera_skew_s: float = 0.05
    ego_half_length: float = 2.4
    ego_half_width: float = 1.1
    clearance_height_m: float = 2.0
    ground_height_tolerance_m: float = 0.0


def camera_reliability(labels: np.ndarray | None, status: CameraStatus, min_fraction: float) -> float:
    """Heuristic health score: fraction of pixels with a usable label, 0 if unusable."""
    if labels is None or status == CameraStatus.DROPPED:
        return 0.0
    usable = float(np.mean(labels != SemanticClass.UNLABELED))
    return usable if usable >= min_fraction else 0.0


class OccupancyEstimator:
    def __init__(self, calibration: CalibrationBundle, grid: VoxelGrid, config: OccupancyConfig | None = None):
        self.calibration = calibration
        self.grid = grid
        self.config = config or OccupancyConfig()
        self.lookups: dict[str, CameraLookup] = {c.name: build_lookup(c, grid) for c in calibration.cameras}
        tolerance = self.config.ground_height_tolerance_m
        self.free_fraction = {c.name: float(c.origin_ego[2] / (c.origin_ego[2] + tolerance)) for c in calibration.cameras}
        X, Y = grid.bev.cell_centers()
        cfg = self.config
        self.ego_mask = (np.abs(X) <= cfg.ego_half_length) & (np.abs(Y) <= cfg.ego_half_width)

    def _contact_range(self, look: CameraLookup, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ground_labels = labels[look.ground_v, look.ground_u]
        hit = look.ground_valid & _OBSTACLE[ground_labels] & ~self.ego_mask
        first = np.full(AZIMUTH_BINS, np.inf)
        np.minimum.at(first, look.azimuth_bin[hit], look.range_xy[hit])
        return minimum_filter1d(first, size=3, mode="wrap"), ground_labels

    def _carve_camera(self, look: CameraLookup, labels: np.ndarray):
        cfg = self.config
        first, ground_labels = self._contact_range(look, labels)
        voxel_labels = labels[look.voxel_v, look.voxel_u]
        hit_range = first[look.azimuth_bin]
        rng = look.range_xy
        is_obstacle = _OBSTACLE[voxel_labels]
        # An infinite hit range means no in-grid ground cell on this azimuth shows
        # an obstacle, so any contact lies beyond the grid and these voxels are in
        # front of it. An obstacle nearer than the image bottom still produces a
        # finite hit, because the nearest visible ground cell then shows it.
        before = rng < hit_range * self.free_fraction[look.name] - cfg.range_slack_m
        contact = rng >= hit_range - cfg.range_slack_m
        band = rng <= hit_range + cfg.contact_band_m
        free = look.voxel_valid & (_GROUND_OR_SKY[voxel_labels] | (is_obstacle & before))
        occupied = look.voxel_valid & is_obstacle & contact & band & np.isfinite(hit_range)
        return free, occupied, voxel_labels, ground_labels

    def estimate(self, bundle: FrameBundle, plane: tuple[float, float, float] | None = None) -> tuple[BEVFrame, OccupancyFrame]:
        """Estimate occupancy. ``plane`` replaces flat ground with z = a + b x + c y for this frame."""
        cfg = self.config
        lookups = self.lookups if plane is None else {c.name: build_lookup(c, self.grid, plane) for c in self.calibration.cameras}
        shape = self.grid.shape
        any_free = np.zeros(shape, bool)
        any_occupied = np.zeros(shape, bool)
        any_view = np.zeros(shape, bool)
        voxel_class = np.zeros(shape, np.uint8)
        bev_labels = np.zeros(self.grid.bev.shape, np.uint8)
        support = np.zeros(self.grid.bev.shape, np.uint8)
        reliability: dict[str, float] = {}
        per_camera_support: dict[str, np.ndarray] = {}

        for name, look in lookups.items():
            frame = bundle.frames.get(name)
            usable = frame is not None and abs(frame.timestamp - bundle.timestamp) <= cfg.max_camera_skew_s
            score = camera_reliability(frame.labels, frame.status, cfg.min_valid_label_fraction) if usable else 0.0
            reliability[name] = score
            if score == 0.0:
                logger.info("camera %s unusable at t=%.3f (present=%s)", name, bundle.timestamp, frame is not None)
                per_camera_support[name] = np.zeros(self.grid.bev.shape, bool)
                continue
            free, occupied, voxel_labels, ground_labels = self._carve_camera(look, frame.labels)
            any_free |= free
            any_occupied |= occupied
            any_view |= look.voxel_valid
            voxel_class[occupied] = voxel_labels[occupied]
            stitched = look.ground_valid & (bev_labels == 0)
            bev_labels[stitched] = ground_labels[stitched]
            support += look.ground_valid.astype(np.uint8)
            per_camera_support[name] = look.ground_valid

        voxel_state = np.full(shape, int(CellState.OUTSIDE_FOV), np.uint8)
        voxel_state[any_view] = CellState.UNKNOWN
        voxel_state[any_occupied] = CellState.OCCUPIED
        voxel_state[any_free] = CellState.FREE
        voxel_class[voxel_state != CellState.OCCUPIED] = 0
        voxel_state[:, self.ego_mask] = CellState.INVALID

        bev_state, bev_class, bev_height = self._collapse(voxel_state, voxel_class, any_view)
        bev = BEVFrame(bundle.timestamp, self.grid.bev, bev_labels, per_camera_support,
                       "flat_ground_ipm_nearest", Provenance.CLASSICAL)
        occupancy = OccupancyFrame(bundle.timestamp, self.grid, voxel_state, voxel_class, bev_state, bev_class,
                                   bev_height, support, reliability, METHOD, Provenance.CLASSICAL)
        return bev, occupancy

    def _collapse(self, state: np.ndarray, classes: np.ndarray, any_view: np.ndarray):
        """Column summary up to clearance height. Free needs every viewed voxel free."""
        zs = self.grid.z_centers()
        low = zs <= self.config.clearance_height_m
        occupied = state == CellState.OCCUPIED
        unknown = state == CellState.UNKNOWN
        out = np.full(self.grid.bev.shape, int(CellState.OUTSIDE_FOV), np.uint8)
        out[any_view.any(axis=0)] = CellState.UNKNOWN
        out[(state[0] == CellState.FREE) & ~unknown[low].any(axis=0)] = CellState.FREE
        column_occupied = occupied[low].any(axis=0)
        out[column_occupied] = CellState.OCCUPIED
        out[self.ego_mask] = CellState.INVALID
        priority = np.max(classes, axis=0)
        bev_class = np.where(column_occupied, priority, 0).astype(np.uint8)
        top = occupied.shape[0] - 1 - np.argmax(occupied[::-1], axis=0)
        height = np.where(occupied.any(axis=0), zs[top] + self.grid.dz / 2, 0.0)
        return out, bev_class, height.astype(np.float32)
