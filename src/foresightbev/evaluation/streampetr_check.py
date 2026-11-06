"""Verify the StreamPETR port on real frames, with negative controls.

Each control deliberately breaks one thing. A real check must then get worse, which
shows that the check measures that thing. Annotations are used for scoring only.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from foresightbev.config import load_settings
from foresightbev.data.nuscenes import NuScenesDataset, NuScenesGroundTruth
from foresightbev.perception.streampetr import CAMERA_ORDER, CLASS_NAMES, StreamPETRInference
from foresightbev.perception.streampetr_runner import boxes_to_ego, frame_inputs

MINI_VAL = ("scene-0103", "scene-0916")
GROUP = dict(car="vehicle.car", truck="vehicle.truck", bus="vehicle.bus", pedestrian="human.pedestrian", bicycle="vehicle.bicycle", motorcycle="vehicle.motorcycle")


def match_rows(rows, boxes, ego, threshold=0.3, limit=2.0):
    """Nearest same-class annotation for each confident detection, one to one."""
    pairs, used = [], set()
    for row in sorted((r for r in rows if r["score"] >= threshold and r["label"] in GROUP), key=lambda r: -r["score"]):
        best, distance = None, limit
        for i, box in enumerate(boxes):
            if i in used or not box.category.startswith(GROUP[row["label"]]):
                continue
            d = float(np.hypot(*(np.array(row["center_ego"][:2]) - ego.world_to_ego(np.array([box.x, box.y])))))
            if d < distance:
                best, distance = i, d
        if best is not None:
            used.add(best)
            pairs.append((row, boxes[best], distance))
    return pairs


def angle_error(a: float, b: float) -> float:
    return abs(float(np.angle(np.exp(1j * (a - b)))))


def summarize(pairs, confident, annotated) -> dict:
    """``pairs`` holds (detection, annotation, annotated yaw in ego frame, annotated z, distance)."""
    if not pairs:
        return dict(matches=0, confident=confident, annotated=annotated)
    long_objects = [p for p in pairs if p[1].length > 3.0 and not p[1].category.startswith("human")]
    return dict(matches=len(pairs), confident=confident, annotated=annotated, recall=len(pairs) / max(annotated, 1), precision=len(pairs) / max(confident, 1), mean_center_error_m=float(np.mean([p[4] for p in pairs])),
                mean_vertical_center_error_m=float(np.mean([abs(p[0]["center_ego"][2] - p[3]) for p in pairs])), median_length_ratio=float(np.median([p[0]["length"] / p[1].length for p in pairs])),
                median_width_ratio=float(np.median([p[0]["width"] / p[1].width for p in pairs])), median_height_ratio=float(np.median([p[0]["height"] / p[1].height for p in pairs])),
                median_yaw_error_deg=float(np.degrees(np.median([angle_error(p[0]["yaw_ego"], p[2]) for p in long_objects]))) if long_objects else None)


def run(model, dataset, truth, *, shuffle_cameras=False, reset_every_frame=False, frames_limit=None) -> dict:
    pairs, confident, annotated = [], 0, 0
    velocities = []
    for scene in MINI_VAL:
        model.reset()
        tokens = dataset.sample_tokens(scene)
        instances = {}
        for i, token in enumerate(tokens):
            for box in truth.boxes(token):
                instances.setdefault(box.instance, {})[i] = box
        for index, token in enumerate(tokens[:frames_limit]):
            inputs = frame_inputs(dataset, token)
            if reset_every_frame:
                model.reset()
            lidar2cam = inputs["lidar2cam"]
            if shuffle_cameras:
                lidar2cam = np.roll(lidar2cam, 1, axis=0)
            boxes = model.step(inputs["images"], inputs["intrinsics"], lidar2cam, inputs["T_global_lidar"], inputs["timestamp"])
            rows = boxes_to_ego(boxes, inputs["T_ego_lidar"])
            T = np.linalg.inv(inputs["T_global_lidar"] @ np.linalg.inv(inputs["T_ego_lidar"]))
            yaw0 = float(np.arctan2(inputs["T_global_lidar"][1, 0], inputs["T_global_lidar"][0, 0]) - np.arctan2(inputs["T_ego_lidar"][1, 0], inputs["T_ego_lidar"][0, 0]))
            x, y = (inputs["T_global_lidar"] @ np.linalg.inv(inputs["T_ego_lidar"]))[:2, 3]

            class Frame:
                @staticmethod
                def world_to_ego(p):
                    return (T @ np.array([p[0], p[1], 0.0, 1.0]))[:2]

            gt = [b for b in truth.boxes(token) if any(b.category.startswith(c) for c in GROUP.values()) and b.visibility >= 2 and np.hypot(*Frame.world_to_ego(np.array([b.x, b.y]))) <= 40]
            rows = [r for r in rows if np.hypot(*r["center_ego"][:2]) <= 40]
            confident += sum(r["score"] >= 0.3 and r["label"] in GROUP for r in rows)
            annotated += len(gt)
            for row, box, distance in match_rows(rows, gt, Frame):
                local_z = (T @ np.array([box.x, box.y, box.z, 1.0]))[2]
                pairs.append((row, box, box.yaw - yaw0, float(local_z), distance))
                history = instances[box.instance]
                if index - 1 in history and index + 1 in history:
                    a, b = history[index - 1], history[index + 1]
                    world_v = np.array([b.x - a.x, b.y - a.y]) / 1.0
                    c, s = np.cos(yaw0), np.sin(yaw0)
                    velocities.append(float(np.hypot(*(np.array(row["velocity_ego"]) - np.array([c * world_v[0] + s * world_v[1], -s * world_v[0] + c * world_v[1]])))))
    out = summarize(pairs, confident, annotated)
    out["mean_velocity_error_mps"] = float(np.mean(velocities)) if velocities else None
    return out


def main() -> int:
    settings = load_settings()
    dataset = NuScenesDataset(settings.nuscenes_root, settings.nuscenes_version)
    truth = NuScenesGroundTruth(dataset)
    model = StreamPETRInference(settings.data_dir / "models/streampetr/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth")
    report = dict(load=dict(checkpoint_tensors=model.report.checkpoint_tensors, loaded=model.report.loaded_tensors, unused=len(model.report.unused_tensors), missing=model.report.missing, unexpected=model.report.unexpected, sha256=model.report.sha256),
                  scenes=list(MINI_VAL), scene_status="mini_val, previously seen: 0103 was a tuning scene, 0916 was in the reported held-out set", classes=list(CLASS_NAMES), cameras=list(CAMERA_ORDER))
    report["port"] = run(model, dataset, truth)
    report["control_no_temporal_memory"] = run(model, dataset, truth, reset_every_frame=True)
    report["control_calibration_shuffled"] = run(model, dataset, truth, shuffle_cameras=True)
    out = settings.data_dir / "results/streampetr"
    out.mkdir(parents=True, exist_ok=True)
    (out / "port_verification.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
