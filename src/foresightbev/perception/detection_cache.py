"""Model detections as stored per-sequence files, the same way labels are stored.

Heavy models run once, in the PyTorch environment. The pipeline, ROS replay and
evaluation then read identical detections, and need no PyTorch. Each file records
the model, weights hash, inference coordinates and measured latency.

Frame: ego at the frame's timestamp. Scores are raw network outputs, not probabilities.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from foresightbev.contracts import Detection, FrameBundle, Provenance, SemanticClass
from foresightbev.errors import ConfigurationError, DatasetError

CACHE_NAME = "detections_streampetr.json"
# nuScenes detection class to tracker class. Barriers and cones are not agents.
CLASS_MAP = dict(car=SemanticClass.VEHICLE, truck=SemanticClass.VEHICLE, construction_vehicle=SemanticClass.VEHICLE, bus=SemanticClass.VEHICLE, trailer=SemanticClass.VEHICLE,
                 motorcycle=SemanticClass.VEHICLE, bicycle=SemanticClass.VEHICLE, pedestrian=SemanticClass.PERSON)
POSITION_STD_M = 0.7  # Prior from published translation error, not fitted to any scene here.


class CachedDetector:
    def __init__(self, sequence_directory: Path, score_threshold: float):
        path = Path(sequence_directory) / CACHE_NAME
        if not path.is_file():
            raise ConfigurationError(f"no cached detections at {path}. Run: foresightbev detect-streampetr --scene <scene>")
        try:
            self.doc = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            raise DatasetError(f"corrupt {path}: {error}") from error
        self.frames = {f["index"]: f for f in self.doc["frames"]}
        self.threshold = score_threshold

    def __call__(self, occupancy, bundle: FrameBundle) -> list[Detection]:
        frame = self.frames.get(bundle.frame_index)
        if frame is None:
            raise DatasetError(f"no cached detections for frame {bundle.frame_index}")
        if abs(frame["timestamp"] - bundle.timestamp) > 1e-6:
            raise DatasetError(f"cached detections for frame {bundle.frame_index} have a different timestamp")
        out = []
        for row in frame["detections"]:
            semantic = CLASS_MAP.get(row["label"])
            if semantic is None or row["score"] < self.threshold:
                continue
            out.append(Detection(bundle.timestamp, semantic, np.array(row["center_ego"][:2], float), row["length"], row["width"], row["height"], row["score"], 0, Provenance.PRETRAINED,
                                 np.eye(2) * POSITION_STD_M**2, row["yaw_ego"], np.array(row["velocity_ego"], float), row["label"], row["center_ego"][2] - row["height"] / 2))
        return out
