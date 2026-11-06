"""Semantic camera renderer: per-pixel ray casting through real rig calibration.

Output has the same form as the real Cam2BEV sample and as a CARLA semantic camera,
so the pipeline runs unchanged on all three. Noise models are declared and off by
default: boundary jitter, missed small objects and spurious blobs.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from foresightbev.contracts import CameraCalibration, SemanticClass as C
from foresightbev.sim.world import Agent, EgoVehicle, RoadLayout, ground_label


@dataclass
class NoiseConfig:
    boundary_jitter_px: int = 0
    object_dropout: float = 0.0
    spurious_blob_rate: float = 0.0


class CameraRenderer:
    def __init__(self, cam: CameraCalibration, max_range: float = 90.0):
        self.cam = cam
        self.max_range = max_range
        uu, vv = np.meshgrid(np.arange(cam.width) + 0.5, np.arange(cam.height) + 0.5)
        d = np.stack([(uu - cam.K[0, 2]) / cam.K[0, 0], (vv - cam.K[1, 2]) / cam.K[1, 1], np.ones_like(uu)], -1)
        pose = cam.T_ego_cam
        self.dirs = (d @ pose[:3, :3].T).reshape(-1, 3)
        self.grid_dirs = self.dirs.reshape(cam.height, cam.width, 3)
        self.origin = pose[:3, 3].copy()
        z = self.dirs[:, 2]
        self.ground_t = np.where(z < -1e-6, -self.origin[2] / np.where(z < -1e-6, z, -1.0), np.inf)
        self.ground_ego = self.origin[:2] + self.dirs[:, :2] * np.where(np.isfinite(self.ground_t), self.ground_t, 0.0)[:, None]

    def render(self, ego: EgoVehicle, layout: RoadLayout, boxes: list[Agent]) -> np.ndarray:
        cam = self.cam
        c, s = np.cos(ego.yaw), np.sin(ego.yaw)
        world_y = ego.y + s * self.ground_ego[:, 0] + c * self.ground_ego[:, 1]
        labels = np.where(np.isfinite(self.ground_t), ground_label(layout, world_y), int(C.SKY)).astype(np.uint8)
        best = self.ground_t.copy().reshape(cam.height, cam.width)
        labels = labels.reshape(cam.height, cam.width)
        for box in boxes:
            dx, dy = box.x - ego.x, box.y - ego.y
            bx, by = c * dx + s * dy, -s * dx + c * dy
            reach = np.hypot(box.length, box.width) / 2 + self.max_range
            if bx * bx + by * by > reach * reach:
                continue
            yaw = box.yaw - ego.yaw
            window = self._screen_window(bx, by, yaw, box)
            if window is None:
                continue
            r0, r1, c0, c1 = window
            dirs = self.grid_dirs[r0:r1, c0:c1]
            local_best = best[r0:r1, c0:c1]
            local_labels = labels[r0:r1, c0:c1]
            cb, sb = np.cos(yaw), np.sin(yaw)
            ox, oy = self.origin[0] - bx, self.origin[1] - by
            px, py = cb * ox + sb * oy, -sb * ox + cb * oy
            dxb = cb * dirs[..., 0] + sb * dirs[..., 1]
            dyb = -sb * dirs[..., 0] + cb * dirs[..., 1]
            dzb = dirs[..., 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                ax1, ax2 = (-box.length / 2 - px) / dxb, (box.length / 2 - px) / dxb
                ay1, ay2 = (-box.width / 2 - py) / dyb, (box.width / 2 - py) / dyb
                az1, az2 = (0.0 - self.origin[2]) / dzb, (box.height - self.origin[2]) / dzb
            near = np.maximum.reduce([np.minimum(ax1, ax2), np.minimum(ay1, ay2), np.minimum(az1, az2)])
            far = np.minimum.reduce([np.maximum(ax1, ax2), np.maximum(ay1, ay2), np.maximum(az1, az2)])
            hit = (near <= far) & (far > 0) & (near < local_best)
            local_best[hit] = near[hit]
            local_labels[hit] = int(box.semantic)
        return labels

    def _screen_window(self, bx: float, by: float, yaw: float, box: Agent):
        """Image rows and columns the box can touch, or None if it is behind the camera."""
        cam = self.cam
        c, s = np.cos(yaw), np.sin(yaw)
        hl, hw = box.length / 2, box.width / 2
        corners = np.array([[bx + c * a - s * b, by + s * a + c * b, z]
                            for a in (-hl, hl) for b in (-hw, hw) for z in (0.0, box.height)])
        rotation, translation = cam.T_cam_ego[:3, :3], cam.T_cam_ego[:3, 3]
        p = corners @ rotation.T + translation
        z = p[:, 2]
        if (z <= 0.05).all():
            return None
        if (z <= 0.05).any():
            return 0, cam.height, 0, cam.width
        u = cam.K[0, 0] * p[:, 0] / z + cam.K[0, 2]
        v = cam.K[1, 1] * p[:, 1] / z + cam.K[1, 2]
        c0, c1 = max(int(np.floor(u.min())) - 1, 0), min(int(np.ceil(u.max())) + 2, cam.width)
        r0, r1 = max(int(np.floor(v.min())) - 1, 0), min(int(np.ceil(v.max())) + 2, cam.height)
        return None if c0 >= c1 or r0 >= r1 else (r0, r1, c0, c1)


def apply_noise(labels: np.ndarray, noise: NoiseConfig, rng: np.random.Generator) -> np.ndarray:
    """Segmentation-like errors, so downstream stages are not fed perfect masks."""
    out = labels
    if noise.boundary_jitter_px > 0:
        size = 2 * int(rng.integers(0, noise.boundary_jitter_px + 1)) + 1
        kernel = np.ones((size, size), np.uint8)
        for cls in (C.VEHICLE, C.PERSON):
            mask = (out == cls).astype(np.uint8)
            if size > 1 and mask.any():
                changed = cv2.erode(mask, kernel) if rng.random() < 0.5 else cv2.dilate(mask, kernel)
                road = out == C.ROAD
                out = out.copy()
                out[(mask == 1) & (changed == 0)] = int(C.ROAD)
                out[(changed == 1) & road] = int(cls)
    if noise.object_dropout > 0 or noise.spurious_blob_rate > 0:
        out = out.copy()
        mask = np.isin(out, [C.VEHICLE, C.PERSON]).astype(np.uint8)
        count, components, stats, _ = cv2.connectedComponentsWithStats(mask)
        for index in range(1, count):
            if rng.random() < noise.object_dropout:
                out[components == index] = int(C.ROAD)
        if rng.random() < noise.spurious_blob_rate:
            h, w = out.shape
            cx, cy = int(rng.integers(0, w)), int(rng.integers(h // 2, h))
            cv2.circle(out, (cx, cy), int(rng.integers(1, 3)), int(C.STATIC), -1)
    return out
