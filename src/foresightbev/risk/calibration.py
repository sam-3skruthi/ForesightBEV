"""Isotonic (pool-adjacent-violators) mapping from event score to event frequency.

Trained only by ``foresightbev.evaluation.calibration`` on labelled scenarios. The
output is an empirical frequency on that data, so ``status`` always names it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def pool_adjacent_violators(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x, kind="stable")
    x, y = np.asarray(x, float)[order], np.asarray(y, float)[order]
    unique, inverse = np.unique(x, return_inverse=True)
    weight = np.bincount(inverse).astype(float)
    value = np.bincount(inverse, weights=y) / weight
    levels, weights, knots = [], [], []
    for v, w, knot in zip(value, weight, unique):
        levels.append(v); weights.append(w); knots.append([knot])
        while len(levels) > 1 and levels[-2] >= levels[-1]:
            total = weights[-2] + weights[-1]
            merged = (levels[-2] * weights[-2] + levels[-1] * weights[-1]) / total
            levels[-2:], weights[-2:] = [merged], [total]
            knots[-2:] = [knots[-2] + knots[-1]]
    xs = np.array([k for group in knots for k in (group[0], group[-1])], float)
    ys = np.array([l for l, group in zip(levels, knots) for _ in (0, 1)], float)
    return xs, ys


@dataclass
class RiskCalibrator:
    x: np.ndarray
    y: np.ndarray
    status: str

    @classmethod
    def fit(cls, scores: np.ndarray, labels: np.ndarray, status: str) -> "RiskCalibrator":
        if len(scores) == 0 or len(scores) != len(labels):
            raise ValueError("calibration needs matching, non-empty scores and labels")
        x, y = pool_adjacent_violators(np.asarray(scores), np.asarray(labels))
        return cls(x, y, status)

    def apply(self, score):
        return np.interp(score, self.x, self.y)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"x": self.x.tolist(), "y": self.y.tolist(), "status": self.status}, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "RiskCalibrator":
        doc = json.loads(Path(path).read_text())
        return cls(np.array(doc["x"]), np.array(doc["y"]), doc["status"])
