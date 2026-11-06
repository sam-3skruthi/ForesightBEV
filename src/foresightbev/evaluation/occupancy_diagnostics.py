"""Trace the annotation-derived false-free figure and compare geometry variants.

Metric naming. nuScenes mini has no occupancy labels. ``annotated_box_cells_marked_free``
is the share of BEV cells inside annotated dynamic-object footprints that the pipeline
labels observed-free. It is a spatial proxy over dynamic objects, not voxel occupancy
accuracy: static scene structure is unscored and box footprints are only rectangles.
``free_area_m2`` is the cost side: a method can always reach zero by marking nothing free.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass, field, replace

import numpy as np
import yaml

from foresightbev.config import ROOT, load_pipeline_config, load_settings
from foresightbev.contracts import CellState
from foresightbev.data.nuscenes import NuScenesDataset, NuScenesGroundTruth, quaternion_to_matrix
from foresightbev.data.sequence import LabelSequence
from foresightbev.evaluation.nuscenes_eval import MIN_VISIBILITY, eligible
from foresightbev.forecast.forecaster import footprint_polygon, rasterize
from foresightbev.perception.detection_cache import CACHE_NAME, CLASS_MAP
from foresightbev.perception.ground import detection_bottoms, fit_plane
from foresightbev.perception.occupancy import OccupancyEstimator
from foresightbev.sim.world import obb_corners

BINS = ((0, 10), (10, 20), (20, 30), (30, 40))


def fuse_boxes(state: np.ndarray, classes: np.ndarray, rows: list[dict], grid, score: float = 0.3) -> None:
    """Mark confident model boxes occupied. Model-asserted, unlike geometric occupied cells."""
    for row in rows:
        semantic = CLASS_MAP.get(row["label"])
        if semantic is None or row["score"] < score:
            continue
        x, y = row["center_ego"][:2]
        mask = rasterize(footprint_polygon(x, y, row["yaw_ego"], row["length"], row["width"]), grid)
        mask &= state != CellState.INVALID
        state[mask] = CellState.OCCUPIED
        classes[mask] = int(semantic)


@dataclass
class Tally:
    cells: Counter = field(default_factory=Counter)
    by_range: dict = field(default_factory=lambda: {b: Counter() for b in BINS})
    by_class: dict = field(default_factory=lambda: {c: Counter() for c in ("VEHICLE", "PERSON")})
    free_area: float = 0.0
    unknown_area: float = 0.0
    frames: int = 0

    def add(self, state, boxes, ego, grid) -> None:
        self.frames += 1
        X, Y = grid.cell_centers()
        near = np.hypot(X, Y) <= 40
        self.free_area += float(((state == CellState.FREE) & near).sum() * grid.resolution**2)
        self.unknown_area += float(((state == CellState.UNKNOWN) & near).sum() * grid.resolution**2)
        for box, center in boxes:
            mask = rasterize(ego.world_to_ego(obb_corners(box.x, box.y, box.yaw, box.length, box.width)), grid)
            names = {CellState.FREE: "free", CellState.OCCUPIED: "occupied", CellState.UNKNOWN: "unknown", CellState.OUTSIDE_FOV: "outside", CellState.INVALID: "invalid"}
            r = float(np.hypot(*center))
            for value, name in names.items():
                n = int((mask & (state == value)).sum())
                self.cells[name] += n
                self.by_class[box.semantic.name][name] += n
                for low, high in BINS:
                    if low <= r < high or (high == 40 and r == 40):
                        self.by_range[(low, high)][name] += n

    @staticmethod
    def rate(c: Counter) -> dict:
        total = sum(c.values())
        return dict(cells=total, free=c["free"] / total if total else None, occupied=c["occupied"] / total if total else None, unknown=(c["unknown"] + c["outside"]) / total if total else None)

    def summary(self) -> dict:
        return dict(frames=self.frames, annotated_box_cells=self.rate(self.cells), by_range={f"{a}-{b}m": self.rate(v) for (a, b), v in self.by_range.items()}, by_class={k: self.rate(v) for k, v in self.by_class.items()},
                    free_area_m2_per_frame=self.free_area / max(self.frames, 1), unknown_area_m2_per_frame=self.unknown_area / max(self.frames, 1))


def pose_tilt_plane(dataset, token) -> tuple[float, float, float]:
    """Ground plane implied if the world were level and the car tilted: tests pose correction."""
    row = dataset.camera_rows[token]["CAM_FRONT"]
    R = quaternion_to_matrix(dataset.poses[row["ego_pose_token"]]["rotation"])
    normal = R.T @ np.array([0.0, 0.0, 1.0])
    return 0.0, float(-normal[0] / normal[2]), float(-normal[1] / normal[2])


def evaluate(scenes: list[str], variants: dict[str, dict]) -> dict:
    settings = load_settings()
    dataset = NuScenesDataset(settings.nuscenes_root, settings.nuscenes_version)
    truth = NuScenesGroundTruth(dataset)
    base = load_pipeline_config(ROOT / "configs/pipeline_nuscenes.yaml")
    out = {}
    for name, v in variants.items():
        tally = Tally()
        occupancy_cfg = replace(base.occupancy, **v.get("occupancy", {}))
        for scene in scenes:
            sequence = LabelSequence(settings.data_dir / "sequences" / scene)
            cache = None
            path = sequence.directory / CACHE_NAME
            if path.is_file():
                cache = {f["index"]: f["detections"] for f in json.loads(path.read_text())["frames"]}
            if (v.get("plane") == "detections" or v.get("fuse")) and cache is None:
                raise FileNotFoundError(path)
            estimator = OccupancyEstimator(sequence.calibration, base.grid, occupancy_cfg)
            tokens = dataset.sample_tokens(scene)
            for i in range(len(sequence)):
                bundle = sequence.bundle(i)
                token = tokens[sequence.frames[i]["source_index"]]
                plane = None
                if v.get("plane") == "pose":
                    plane = pose_tilt_plane(dataset, token)
                elif v.get("plane") == "detections":
                    plane = fit_plane(detection_bottoms(cache[i]))
                _, occ = estimator.estimate(bundle, plane)
                state = occ.bev_state
                if v.get("fuse"):
                    fuse_boxes(state, occ.bev_class, cache[i], base.grid.bev)
                tally.add(state, eligible(truth.boxes(token), bundle.ego), bundle.ego, base.grid.bev)
        out[name] = tally.summary()
        s = out[name]
        print(f"{name:32s} free {s['annotated_box_cells']['free']:.3f} occ {s['annotated_box_cells']['occupied']:.3f} unk {s['annotated_box_cells']['unknown']:.3f} | free area {s['free_area_m2_per_frame']:.0f} m2 | by range {[round(r['free'], 3) if r['free'] is not None else None for r in s['by_range'].values()]}", flush=True)
    return out


VARIANTS = {
    "flat_no_tolerance": dict(occupancy=dict(ground_height_tolerance_m=0.0)),
    "current_tolerance_0.4": dict(),
    "pose_tilt_correction": dict(plane="pose"),
    "tolerance_0.8": dict(occupancy=dict(ground_height_tolerance_m=0.8)),
    "local_plane_from_detections": dict(plane="detections"),
    "current_plus_fused_boxes": dict(fuse=True),
    "local_plane_plus_fused_boxes": dict(plane="detections", fuse=True),
}


def main() -> int:
    scenes = yaml.safe_load((ROOT / "configs/nuscenes_splits.yaml").read_text())["mini_val"]
    report = dict(metric="annotated_box_cells_marked_free: annotation-derived spatial proxy on dynamic-object footprints, not voxel occupancy accuracy", scenes=scenes,
                  scene_status="mini_val, previously seen", min_visibility=MIN_VISIBILITY, variants=evaluate(scenes, VARIANTS))
    out = load_settings().data_dir / "results/occupancy_diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    (out / "variants.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
