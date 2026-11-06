"""Stage 4 detection from image-space blobs, ranged by apparent size.

Flat-ground range fails on real roads: a ground height error dz at range d moves the
estimate by about d * dz / h. Apparent size does not depend on the ground: r = f * H / p,
for a class height prior H and pixel height p. That prior is a heuristic and is
declared here. Blobs have no instance ids, so a crowd is split by its aspect ratio.
A vehicle blob cut by the image edge gives only a lower bound on size, so it falls
back to the ground contact, which is reliable at the short ranges where this happens.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from foresightbev.contracts import CalibrationBundle, CameraCalibration, Detection, FrameBundle, Provenance, SemanticClass


@dataclass
class ImageDetectionConfig:
    vehicle_height_m: float = 1.65
    person_height_m: float = 1.72
    vehicle_size: tuple[float, float] = (4.6, 1.9)
    person_size: tuple[float, float] = (0.7, 0.7)
    min_height_px_vehicle: int = 8
    min_height_px_person: int = 8
    min_area_px: int = 20
    person_width_ratio: float = 0.5
    max_range_m: float = 45.0
    merge_distance_m: float = 2.0
    radial_std_fraction: float = 0.2
    tangential_std_m: float = 0.4
    edge_margin_px: int = 2
    full_support_px: int = 400


def _ray_azimuth_point(cam: CameraCalibration, u: float, depth: float) -> np.ndarray:
    """Ego-frame ground point at optical depth ``depth`` through image column ``u``."""
    x = (u - cam.K[0, 2]) / cam.K[0, 0] * depth
    pose = cam.T_ego_cam
    return (pose[:3, :3] @ np.array([x, 0.0, depth]) + pose[:3, 3])[:2]


def _contact_depth(cam: CameraCalibration, bottom: float) -> float | None:
    """Optical depth of a ground contact, assuming a level camera at its mounted height."""
    drop = bottom - cam.K[1, 2]
    return float(cam.K[1, 1] * cam.origin_ego[2] / drop) if drop > 2 else None


def _covariance(center: np.ndarray, cfg: ImageDetectionConfig) -> np.ndarray:
    distance = max(float(np.linalg.norm(center)), 1.0)
    radial = center / distance
    tangent = np.array([-radial[1], radial[0]])
    std_r = max(0.5, cfg.radial_std_fraction * distance)
    return std_r**2 * np.outer(radial, radial) + cfg.tangential_std_m**2 * np.outer(tangent, tangent)


def detect_in_image(labels: np.ndarray, cam: CameraCalibration, semantic: SemanticClass, timestamp: float, cfg: ImageDetectionConfig) -> list[Detection]:
    mask = (labels == semantic).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    person = semantic == SemanticClass.PERSON
    height_prior = cfg.person_height_m if person else cfg.vehicle_height_m
    minimum = cfg.min_height_px_person if person else cfg.min_height_px_vehicle
    size = cfg.person_size if person else cfg.vehicle_size
    out = []
    for index in range(1, count):
        left, top, width, height, area = stats[index]
        if area < cfg.min_area_px or height < minimum:
            continue
        cut = top <= cfg.edge_margin_px or top + height >= cam.height - cfg.edge_margin_px or left <= cfg.edge_margin_px or left + width >= cam.width - cfg.edge_margin_px
        contact = _contact_depth(cam, float(top + height))
        if cut:
            if contact is None:
                continue
            depth = contact
        else:
            depth = float(cam.K[1, 1] * height_prior / height)
        parts = max(1, int(round(width / (cfg.person_width_ratio * height)))) if person else 1
        for part in range(parts):
            u = left + width * (part + 0.5) / parts
            center = _ray_azimuth_point(cam, u, depth)
            if float(np.hypot(*center)) <= cfg.max_range_m:
                direction = center / max(float(np.linalg.norm(center)), 1e-6)
                completed = center + direction * (size[0] / 2 if not person else 0.0)
                out.append(Detection(timestamp, semantic, completed, size[0], size[1], height_prior, min(1.0, float(area) / cfg.full_support_px), int(area), Provenance.HEURISTIC, _covariance(completed, cfg)))
    return out


def merge_overlaps(detections: list[Detection], limit: float) -> list[Detection]:
    """Cameras overlap at their edges; keep the better-supported detection of a pair."""
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda d: -d.support):
        if all(d.semantic != detection.semantic or np.hypot(*(d.center_ego - detection.center_ego)) > limit for d in kept):
            kept.append(detection)
    return kept


class ImageSizeDetector:
    def __init__(self, calibration: CalibrationBundle, config: ImageDetectionConfig | None = None):
        self.cameras = {c.name: c for c in calibration.cameras}
        self.config = config or ImageDetectionConfig()

    def __call__(self, occupancy, bundle: FrameBundle) -> list[Detection]:
        out: list[Detection] = []
        # Calibration order, never arrival order: detection order sets track order,
        # which sets the Monte Carlo draws, so it must not depend on message timing.
        for name in self.cameras:
            frame = bundle.frames.get(name)
            if frame is None or frame.labels is None:
                continue
            for semantic in (SemanticClass.VEHICLE, SemanticClass.PERSON):
                out.extend(detect_in_image(frame.labels, self.cameras[name], semantic, bundle.timestamp, self.config))
        return merge_overlaps(out, self.config.merge_distance_m)
