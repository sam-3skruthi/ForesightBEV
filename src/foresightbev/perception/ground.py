"""Local ground-plane estimate from detected objects' bottoms. Camera-derived only.

Detected boxes give sparse ground-height samples: the bottom of a box stands on the
ground. A robust ridge-regularised plane z = a + b x + c y is fitted per frame. With few
samples it falls back towards flat ground. It cannot see a surface between objects, so
this is a coarse tilt and offset correction, not a road surface reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroundConfig:
    min_samples: int = 4
    ridge_offset: float = 2.0
    ridge_slope: float = 200.0
    inlier_m: float = 0.5
    iterations: int = 3
    max_slope: float = 0.06


def fit_plane(points: np.ndarray, config: GroundConfig | None = None) -> tuple[float, float, float] | None:
    """Fit from Nx3 ego-frame bottoms. None means too little evidence: stay flat."""
    cfg = config or GroundConfig()
    if len(points) < cfg.min_samples:
        return None
    x = np.column_stack([np.ones(len(points)), points[:, 0], points[:, 1]])
    y = points[:, 2]
    prior = np.diag([cfg.ridge_offset, cfg.ridge_slope, cfg.ridge_slope])
    mask = np.ones(len(points), bool)
    beta = np.zeros(3)
    for _ in range(cfg.iterations):
        if mask.sum() < cfg.min_samples:
            return None
        beta = np.linalg.solve(x[mask].T @ x[mask] + prior, x[mask].T @ y[mask])
        mask = np.abs(x @ beta - y) <= cfg.inlier_m
    return float(beta[0]), float(np.clip(beta[1], -cfg.max_slope, cfg.max_slope)), float(np.clip(beta[2], -cfg.max_slope, cfg.max_slope))


def detection_bottoms(rows: list[dict], score: float = 0.3, classes: tuple[str, ...] = ("car", "truck", "bus", "pedestrian")) -> np.ndarray:
    picked = [(r["center_ego"][0], r["center_ego"][1], r["center_ego"][2] - r["height"] / 2) for r in rows if r["score"] >= score and r["label"] in classes]
    return np.array(picked, float).reshape(-1, 3)
