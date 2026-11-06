"""Score Stage 2-3 on the one real multi-camera sample against official labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from foresightbev.contracts import BEVGrid, CellState, SemanticClass as C, VoxelGrid
from foresightbev.data.calibration import load_rig
from foresightbev.data.cam2bev_sample import bev_ground_truth, load_sample
from foresightbev.perception.occupancy import OccupancyEstimator

PX_PER_M = 682.578 / 50.0


def sample_gt(labels: np.ndarray, grid: BEVGrid) -> np.ndarray:
    """Resample an official 604x964 top-down image onto the BEV grid."""
    X, Y = grid.cell_centers()
    col = np.floor(482 + X * PX_PER_M).astype(int)
    row = np.floor(302 - Y * PX_PER_M).astype(int)
    inside = (col >= 0) & (col < 964) & (row >= 0) & (row < 604)
    out = np.full(grid.shape, 255, np.uint8)
    out[inside] = labels[row[inside], col[inside]]
    return out


def evaluate(sample_dir: str | Path, rig_path: str | Path, grid: BEVGrid | None = None) -> dict:
    grid = grid or BEVGrid(-35, 35, -22, 22, 0.25)
    rig = load_rig(rig_path)
    bundle = load_sample(sample_dir, rig)
    _, occ = OccupancyEstimator(rig, VoxelGrid(grid)).estimate(bundle)
    gt = sample_gt(bev_ground_truth(sample_dir), grid)
    rgb_visible = sample_gt_visibility(sample_dir, grid)

    inside = (gt != 255) & (occ.bev_state != CellState.INVALID)
    state = occ.bev_state
    gt_occ = inside & np.isin(gt, [C.VEHICLE, C.PERSON, C.STATIC])
    gt_ground = inside & np.isin(gt, [C.ROAD, C.SIDEWALK, C.OTHER_GROUND])
    pred_occ = inside & (state == CellState.OCCUPIED)
    pred_free = inside & (state == CellState.FREE)
    pred_unknown = inside & np.isin(state, [CellState.UNKNOWN, CellState.OUTSIDE_FOV])
    def rate(a, b):
        return float((a & b).sum() / max(int(b.sum()), 1))
    gt_veh = inside & (gt == C.VEHICLE)
    dynamic = pred_occ & np.isin(occ.bev_class, [C.VEHICLE, C.PERSON])
    result = {
        "input": "Cam2BEV 1_FRLR example, one frame, four simulated semantic cameras",
        "grid": [grid.nx, grid.ny, grid.resolution],
        "state_fractions": {s.name: float((state == s).mean()) for s in CellState},
        "free_precision_gt_ground": rate(gt_ground, pred_free),
        "ground_recall_as_free": rate(pred_free, gt_ground),
        "false_free_rate_gt_obstacle": rate(pred_free, gt_occ),
        "gt_obstacle_marked_occupied": rate(pred_occ, gt_occ),
        "gt_obstacle_marked_unknown": rate(pred_unknown, gt_occ),
        "occupied_precision_gt_obstacle": rate(gt_occ, pred_occ),
        "vehicle_gt_marked_occupied_as_dynamic": rate(dynamic, gt_veh),
        "visibility_agreement_all_cells": visibility_agreement(state, rgb_visible, inside),
        "visibility_agreement_excluding_static_cells": visibility_agreement(state, rgb_visible, inside & (gt != C.STATIC)),
        "note": "One frame only. Not a generalisation claim.",
    }
    return result


def sample_gt_visibility(sample_dir, grid: BEVGrid) -> np.ndarray:
    from foresightbev.data.cam2bev_sample import read_rgb
    rgb = read_rgb(Path(sample_dir) / "bev+occlusion.png")
    occluded = (rgb == (150, 150, 150)).all(-1).astype(np.uint8)
    return sample_gt(occluded, grid)


def visibility_agreement(state: np.ndarray, occluded: np.ndarray, inside: np.ndarray) -> dict:
    """Compare UNKNOWN with the official occluded pixels. Positive class: occluded."""
    valid = inside & (occluded != 255)
    truth = (occluded == 1) & valid
    pred = (state == CellState.UNKNOWN) & valid
    tp = int((truth & pred).sum())
    return {"positive_class": "officially occluded cell", "precision": tp / max(int(pred.sum()), 1),
            "recall": tp / max(int(truth.sum()), 1), "iou": tp / max(int((truth | pred).sum()), 1)}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    print(json.dumps(evaluate(root / "experiments/cam2bev_ipm/inputs/1_FRLR_examples",
                              root / "configs/rig_cam2bev_1frlr.yaml"), indent=1))
