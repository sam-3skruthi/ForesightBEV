"""Stage 2 geometry: projection, IPM, and precomputed camera-to-grid lookups."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from foresightbev.contracts import CameraCalibration, VoxelGrid

AZIMUTH_BINS = 1440


def project(cam: CameraCalibration, points_ego: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project ...x3 ego points. Returns continuous pixel (u, v) and optical depth."""
    rotation, translation = cam.T_cam_ego[:3, :3], cam.T_cam_ego[:3, 3]
    p = points_ego @ rotation.T + translation
    z = p[..., 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = cam.K[0, 0] * p[..., 0] / safe + cam.K[0, 2]
    v = cam.K[1, 1] * p[..., 1] / safe + cam.K[1, 2]
    return u, v, z


def backproject_to_ground(cam: CameraCalibration, u, v) -> tuple[np.ndarray, np.ndarray]:
    """Intersect pixel rays with z=0. Returns ego points and a valid-ray mask."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    d = np.stack([(u - cam.K[0, 2]) / cam.K[0, 0], (v - cam.K[1, 2]) / cam.K[1, 1],
                  np.ones_like(u)], -1)
    pose = cam.T_ego_cam
    rays = d @ pose[:3, :3].T
    origin = pose[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -origin[2] / rays[..., 2]
    valid = rays[..., 2] < -1e-9
    return origin + rays * t[..., None], valid


def cam2bev_output_shape(drone: dict) -> tuple[tuple[int, int], tuple[float, float]]:
    shape = (int(2 * drone["py"]), int(2 * drone["px"]))
    dx = shape[1] / drone["fx"] * drone["ZCam"]
    dy = shape[0] / drone["fy"] * drone["ZCam"]
    return shape, (shape[0] / dy, shape[1] / dx)


def cam2bev_ipm(cam: CameraCalibration, drone: dict) -> np.ndarray:
    """The analytical ground-plane homography that Cam2BEV ``ipm.py`` computes."""
    shape, px_per_m = cam2bev_output_shape(drone)
    shift = (shape[0] / 2.0 + drone["YCam"] * px_per_m[0], shape[1] / 2.0 - drone["XCam"] * px_per_m[1])
    M = np.array([[1.0 / px_per_m[1], 0.0, -shift[1] / px_per_m[1]],
                  [0.0, -1.0 / px_per_m[0], shift[0] / px_per_m[0]],
                  [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    P = cam.K @ cam.T_cam_ego[:3, :4]
    return np.linalg.inv(P @ M)


def warp_like_cam2bev(image: np.ndarray, cam: CameraCalibration, yaw_deg: float, drone: dict) -> np.ndarray:
    """Warp one image the way upstream does, including its behind-camera mask."""
    shape, px_per_m = cam2bev_output_shape(drone)
    warped = cv2.warpPerspective(image, cam2bev_ipm(cam, drone), (shape[1], shape[0]), flags=cv2.INTER_NEAREST)
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    theta = np.rad2deg(np.arctan2(-rows + shape[0] / 2 - drone["YCam"] * px_per_m[0],
                                  cols - shape[1] / 2 + drone["XCam"] * px_per_m[1]))
    delta = np.abs(theta - yaw_deg)
    warped[(delta > 90) & (delta < 270)] = 0
    return warped


@dataclass
class CameraLookup:
    """Per-camera lookups into the fixed grid, so each frame is only array indexing."""

    name: str
    ground_u: np.ndarray
    ground_v: np.ndarray
    ground_valid: np.ndarray
    voxel_u: np.ndarray
    voxel_v: np.ndarray
    voxel_valid: np.ndarray
    range_xy: np.ndarray
    azimuth_bin: np.ndarray


def build_lookup(cam: CameraCalibration, grid: VoxelGrid, plane: tuple[float, float, float] | None = None) -> CameraLookup:
    """Grid-to-pixel lookups. ``plane`` (a, b, c) sets ground height z = a + b x + c y."""
    X, Y = grid.bev.cell_centers()
    height = np.zeros_like(X) if plane is None else plane[0] + plane[1] * X + plane[2] * Y
    ground = np.stack([X, Y, height], -1)
    u, v, z = project(cam, ground)
    ground_u, ground_v = np.floor(np.clip(u, -1e6, 1e6)).astype(np.int32), np.floor(np.clip(v, -1e6, 1e6)).astype(np.int32)
    ground_valid = (z > 0.05) & (ground_u >= 0) & (ground_u < cam.width) & (ground_v >= 0) & (ground_v < cam.height)

    zs = grid.z_centers()[:, None, None] + height
    voxels = np.stack([np.broadcast_to(X, grid.shape), np.broadcast_to(Y, grid.shape),
                       np.broadcast_to(zs, grid.shape)], -1)
    u, v, z = project(cam, voxels)
    voxel_u, voxel_v = np.floor(np.clip(u, -1e6, 1e6)).astype(np.int32), np.floor(np.clip(v, -1e6, 1e6)).astype(np.int32)
    voxel_valid = (z > 0.05) & (voxel_u >= 0) & (voxel_u < cam.width) & (voxel_v >= 0) & (voxel_v < cam.height)

    origin = cam.origin_ego
    dx, dy = X - origin[0], Y - origin[1]
    azimuth = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi)
    azimuth_bin = (azimuth * AZIMUTH_BINS).astype(np.int32) % AZIMUTH_BINS
    return CameraLookup(cam.name, np.clip(ground_u, 0, cam.width - 1), np.clip(ground_v, 0, cam.height - 1),
                        ground_valid, np.clip(voxel_u, 0, cam.width - 1), np.clip(voxel_v, 0, cam.height - 1),
                        voxel_valid, np.hypot(dx, dy), azimuth_bin)
