"""Run StreamPETR over nuScenes scenes. Inputs: images, calibration, ego poses, timestamps.

The fixed LIDAR_TOP extrinsic defines the model's coordinate frame. No LiDAR data is read.
The reference ego pose and time come from CAM_FRONT, not the LiDAR-timed keyframe.
"""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from foresightbev.data.nuscenes import NuScenesDataset
from foresightbev.perception.detection_cache import CACHE_NAME
from foresightbev.perception.streampetr import CAMERA_ORDER, CLASS_NAMES, Boxes, StreamPETRInference


def frame_inputs(dataset: NuScenesDataset, token: str, drop_camera: str | None = None) -> dict:
    """Images and geometry for one sample, in the model's camera order.

    ``drop_camera`` replaces one image with black, a declared fault for robustness tests.
    """
    rows = dataset.camera_rows[token]
    reference = rows["CAM_FRONT"]
    T_global_ego = dataset.pose_matrix(dataset.poses[reference["ego_pose_token"]])
    T_ego_lidar = dataset.pose_matrix(dataset.calibrated[dataset.lidar_calibration[token]])
    T_global_lidar = T_global_ego @ T_ego_lidar
    images, intrinsics, lidar2cam = [], [], []
    for channel in CAMERA_ORDER:
        row = rows[channel]
        bgr = cv2.imread(str(dataset.root / row["filename"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(row["filename"])
        images.append(np.zeros_like(bgr) if channel == drop_camera else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        calibrated = dataset.calibrated[row["calibrated_sensor_token"]]
        intrinsics.append(np.asarray(calibrated["camera_intrinsic"], float))
        T_global_cam = dataset.pose_matrix(dataset.poses[row["ego_pose_token"]]) @ dataset.pose_matrix(calibrated)
        lidar2cam.append(np.linalg.inv(T_global_cam) @ T_global_lidar)
    return dict(images=images, intrinsics=np.stack(intrinsics), lidar2cam=np.stack(lidar2cam), T_global_lidar=T_global_lidar, T_ego_lidar=T_ego_lidar, timestamp=reference["timestamp"] * 1e-6)


def boxes_to_ego(boxes: Boxes, T_ego_lidar: np.ndarray) -> list[dict]:
    """LIDAR_TOP frame to ego frame. Axes differ: LIDAR_TOP x is right, y is forward."""
    R, t = T_ego_lidar[:3, :3], T_ego_lidar[:3, 3]
    rotation = float(np.arctan2(R[1, 0], R[0, 0]))
    rows = []
    for i in range(len(boxes.scores)):
        center = R @ boxes.center[i] + t
        velocity = R[:2, :2] @ boxes.velocity[i]
        rows.append(dict(label=CLASS_NAMES[int(boxes.labels[i])], score=float(boxes.scores[i]), center_ego=center.tolist(), length=float(boxes.size[i, 0]), width=float(boxes.size[i, 1]), height=float(boxes.size[i, 2]),
                         yaw_ego=float(boxes.yaw[i] + rotation), velocity_ego=velocity.tolist()))
    return rows


def peak_rss_gb() -> float:
    """Process peak RSS. macOS reports bytes."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def run_scene(dataset: NuScenesDataset, scene: str, model: StreamPETRInference, output: Path, keep_threshold: float = 0.05, fault: tuple[str, int, int] | None = None) -> dict:
    """Write the detection cache. Keeps low scores so sweeps need no rerun.

    ``fault`` (camera, first frame, end frame) blanks one camera over a frame range.
    """
    model.reset()
    frames, seconds = [], []
    for index, token in enumerate(dataset.sample_tokens(scene)):
        inputs = frame_inputs(dataset, token, fault[0] if fault and fault[1] <= index < fault[2] else None)
        start = time.perf_counter()
        boxes = model.step(inputs["images"], inputs["intrinsics"], inputs["lidar2cam"], inputs["T_global_lidar"], inputs["timestamp"])
        if torch.backends.mps.is_available() and model.device == "mps":
            torch.mps.synchronize()
        seconds.append(time.perf_counter() - start)
        rows = [r for r in boxes_to_ego(boxes, inputs["T_ego_lidar"]) if r["score"] >= keep_threshold]
        frames.append(dict(index=index, token=token, timestamp=inputs["timestamp"], detections=rows))
    stable = seconds[3:] or seconds
    metrics = dict(mean_ms=float(1000 * np.mean(stable)), p95_ms=float(1000 * np.percentile(stable, 95)), frames=len(seconds), device=model.device, peak_rss_gb=peak_rss_gb(),
                   mps_driver_gb=float(torch.mps.driver_allocated_memory() / 1e9) if model.device == "mps" else None)
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / CACHE_NAME).write_text(json.dumps(dict(model=model.name, checkpoint_sha256=model.report.sha256, coordinates="ego frame at CAM_FRONT time", score="raw network score, uncalibrated", metrics=metrics, frames=frames)))
    return metrics
