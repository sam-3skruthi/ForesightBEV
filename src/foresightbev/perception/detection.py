"""Stage 4 detection: metric agents from dynamic-class occupied BEV cells.

Classical clustering plus a heuristic box completion. Cameras see only the near
surface, so the box centre is pushed away from the camera by a class-size prior.
Touching agents merge into one detection, because inputs are semantic and carry
no instance ids.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from foresightbev.contracts import Detection, OccupancyFrame, SemanticClass


@dataclass
class DetectionConfig:
    min_cells_vehicle: int = 3
    min_cells_person: int = 1
    vehicle_size: tuple[float, float] = (4.6, 1.9)
    person_size: tuple[float, float] = (0.7, 0.7)
    side_view_min_extent_m: float = 3.0
    full_support_cells: int = 24
    merge_radius_m: float = 1.0
    vehicle_length_range: tuple[float, float] = (3.5, 9.0)
    vehicle_width_range: tuple[float, float] = (1.6, 2.6)
    tangential_std_m: float = 0.3
    min_range_std_m: float = 0.3
    range_std_per_m2: float = 0.0
    max_range_m: float = float("inf")
    min_cells_person_far: int = 1
    far_range_m: float = float("inf")


def range_noise_coefficient(calibration, pixels: float = 0.5) -> float:
    """Ground-plane range std per squared metre for a contact-row error in pixels.

    For a camera at height h, depth is d = fy h / dv, so dd = d^2 dv / (fy h). This
    is why monocular flat-ground range error grows with the square of distance.
    """
    return max(pixels / (c.K[1, 1] * c.origin_ego[2]) for c in calibration.cameras)


def detection_covariance(center: np.ndarray, cfg: DetectionConfig) -> np.ndarray:
    """2x2 ego-frame position covariance: radial from geometry, tangential fixed."""
    distance = max(float(np.linalg.norm(center)), 1e-6)
    radial = center / distance
    tangent = np.array([-radial[1], radial[0]])
    std_r = max(cfg.min_range_std_m, cfg.range_std_per_m2 * distance**2)
    return std_r**2 * np.outer(radial, radial) + cfg.tangential_std_m**2 * np.outer(tangent, tangent)


def detect(occupancy: OccupancyFrame, config: DetectionConfig | None = None) -> list[Detection]:
    cfg = config or DetectionConfig()
    grid = occupancy.grid.bev
    X, Y = grid.cell_centers()
    out: list[Detection] = []
    for semantic, minimum, size in ((SemanticClass.VEHICLE, cfg.min_cells_vehicle, cfg.vehicle_size),
                                    (SemanticClass.PERSON, cfg.min_cells_person, cfg.person_size)):
        mask = occupancy.bev_class == semantic
        if not mask.any():
            continue
        # Near-surface fragments of one vehicle are merged. Agents closer than the
        # merge radius become one detection, a limit of label-only input.
        radius = int(round(cfg.merge_radius_m / grid.resolution)) if semantic == SemanticClass.VEHICLE else 1
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        merged = cv2.dilate(mask.astype(np.uint8), kernel) > 0
        labels, count = ndimage.label(merged, structure=np.ones((3, 3), bool))
        for index in range(1, count + 1):
            cells = (labels == index) & mask
            n = int(cells.sum())
            if n < minimum:
                continue
            points = np.stack([X[cells], Y[cells]], -1).astype(np.float32)
            center = points.mean(0)
            distance = float(np.hypot(*center))
            if distance > cfg.max_range_m or (semantic == SemanticClass.PERSON and distance > cfg.far_range_m and n < cfg.min_cells_person_far):
                continue
            if n >= 3:
                (_, _), (a, b), _ = cv2.minAreaRect(points)
                long_side, short_side = max(a, b) + grid.resolution, min(a, b) + grid.resolution
            else:
                long_side = short_side = grid.resolution
            depth_prior = size[1] if long_side >= cfg.side_view_min_extent_m else size[0]
            direction = center / max(float(np.linalg.norm(center)), 1e-6)
            completed = center + direction * max(0.0, depth_prior - short_side) / 2.0
            if semantic == SemanticClass.VEHICLE:
                length = float(np.clip(max(long_side, size[0]), *cfg.vehicle_length_range))
                width = float(np.clip(short_side, *cfg.vehicle_width_range))
            else:
                length, width = size
            center_ego = completed.astype(float)
            out.append(Detection(occupancy.timestamp, semantic, center_ego, length, width,
                                 float(occupancy.bev_height[cells].max()), min(1.0, n / cfg.full_support_cells), n,
                                 covariance=detection_covariance(center_ego, cfg)))
    return out
