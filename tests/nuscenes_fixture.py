"""Write a tiny dataset in the official nuScenes schema. Synthetic: for loader tests only."""

import json
from pathlib import Path

import cv2
import numpy as np

from foresightbev.data.nuscenes import CAMERAS

# Pose quaternions (w, x, y, z) for sensor-to-ego with the camera looking along +x (optical z -> ego x).
FRONT_Q = [0.5, -0.5, 0.5, -0.5]


def yaw_q(yaw):
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]


def write(root: Path, frames: int = 3, version: str = "v1.0-mini", speed: float = 5.0) -> Path:
    tables = {k: [] for k in ("scene", "sample", "sample_data", "calibrated_sensor", "sensor", "ego_pose", "sample_annotation", "instance", "category", "visibility", "log", "lidarseg")}
    tables.pop("lidarseg")
    tables["category"] = [dict(token="c_car", name="vehicle.car", description=""), dict(token="c_ped", name="human.pedestrian.adult", description=""),
                          dict(token="c_cone", name="movable_object.trafficcone", description="")]
    tables["visibility"] = [dict(token="4", level="v80-100", description="")]
    tables["instance"] = [dict(token="inst_car", category_token="c_car", nbr_annotations=frames, first_annotation_token="", last_annotation_token="")]
    for i, channel in enumerate(("LIDAR_TOP",) + CAMERAS):
        modality = "lidar" if channel == "LIDAR_TOP" else "camera"
        tables["sensor"].append(dict(token=f"sensor_{channel}", channel=channel, modality=modality))
        intrinsic = [] if modality == "lidar" else [[1266.0, 0, 816.0], [0, 1266.0, 491.0], [0, 0, 1]]
        tables["calibrated_sensor"].append(dict(token=f"cal_{channel}", sensor_token=f"sensor_{channel}", translation=[1.7, 0.0, 1.5], rotation=FRONT_Q, camera_intrinsic=intrinsic))
    t0 = 1_533_000_000_000_000
    for n in range(frames):
        t = t0 + n * 500_000
        token = f"sample_{n}"
        tables["sample"].append(dict(token=token, timestamp=t, prev=f"sample_{n-1}" if n else "", next=f"sample_{n+1}" if n + 1 < frames else "", scene_token="scene_0"))
        pose = f"pose_{n}"
        tables["ego_pose"].append(dict(token=pose, timestamp=t, translation=[100.0 + speed * n * .5, 200.0, 0.0], rotation=yaw_q(0.0)))
        for c, channel in enumerate(CAMERAS):
            filename = f"samples/{channel}/{n}.jpg"
            path = root / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), np.full((900, 1600, 3), 40 + c * 20, np.uint8))
            tables["sample_data"].append(dict(token=f"sd_{channel}_{n}", sample_token=token, ego_pose_token=pose, calibrated_sensor_token=f"cal_{channel}", timestamp=t + c * 8000,
                                              fileformat="jpg", is_key_frame=True, height=900, width=1600, filename=filename,
                                              prev=f"sd_{channel}_{n-1}" if n else "", next=f"sd_{channel}_{n+1}" if n + 1 < frames else ""))
        tables["sample_data"].append(dict(token=f"sd_lidar_{n}", sample_token=token, ego_pose_token=pose, calibrated_sensor_token="cal_LIDAR_TOP", timestamp=t, fileformat="pcd",
                                          is_key_frame=True, height=0, width=0, filename=f"samples/LIDAR_TOP/{n}.pcd.bin", prev="", next=""))
        tables["sample_annotation"].append(dict(token=f"ann_{n}", sample_token=token, instance_token="inst_car", visibility_token="4", attribute_tokens=[], translation=[120.0 + n, 201.0, 0.8],
                                                size=[1.9, 4.6, 1.5], rotation=yaw_q(0.1), prev="", next="", num_lidar_pts=10, num_radar_pts=0))
    tables["scene"] = [dict(token="scene_0", log_token="log_0", nbr_samples=frames, first_sample_token="sample_0", last_sample_token=f"sample_{frames-1}", name="scene-0001", description="fixture")]
    directory = root / version
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        (directory / f"{name}.json").write_text(json.dumps(rows))
    return root
