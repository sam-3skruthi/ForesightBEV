"""Stage 9 metric functions. Every aggregate states its positive class and binning."""

from __future__ import annotations

import numpy as np


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def nll(p: np.ndarray, y: np.ndarray, eps: float = 1e-4) -> float:
    q = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    """Equal-width bins on [0, 1]. Only non-empty bins are returned."""
    index = np.minimum((p * bins).astype(int), bins - 1)
    out = []
    for b in range(bins):
        mask = index == b
        if mask.any():
            out.append(dict(low=b / bins, high=(b + 1) / bins, count=int(mask.sum()),
                            mean_predicted=float(p[mask].mean()), observed_frequency=float(y[mask].mean())))
    return out


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-width bins, weighted by bin count."""
    rows = reliability(p, y, bins)
    return float(sum(r["count"] * abs(r["mean_predicted"] - r["observed_frequency"]) for r in rows) / max(len(p), 1))


def auroc(score: np.ndarray, y: np.ndarray) -> float | None:
    """Rank-based AUROC with ties averaged. None when only one class is present."""
    positives, negatives = y == 1, y == 0
    if not positives.any() or not negatives.any():
        return None
    order = np.argsort(score, kind="stable")
    sorted_scores = score[order]
    _, inverse, counts = np.unique(sorted_scores, return_inverse=True, return_counts=True)
    cumulative = np.cumsum(counts)
    average_rank = cumulative - (counts - 1) / 2
    ranks = np.empty(len(score))
    ranks[order] = average_rank[inverse]
    n_pos, n_neg = positives.sum(), negatives.sum()
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, samples: int = 2000) -> dict:
    """Mean with a percentile 95% interval, resampling episodes."""
    values = np.asarray(values, float)
    if len(values) == 0:
        return dict(mean=None, low=None, high=None, n=0)
    draws = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(1)
    return dict(mean=float(values.mean()), low=float(np.percentile(draws, 2.5)),
                high=float(np.percentile(draws, 97.5)), n=int(len(values)))


def cluster_bootstrap(groups: list[tuple[np.ndarray, np.ndarray]], metric, rng: np.random.Generator, samples: int = 500) -> dict:
    """Interval for a metric over (p, y) rows, resampling whole episodes.

    Rows within an episode are correlated, so row-level resampling would be too narrow.
    """
    indices = rng.integers(0, len(groups), size=(samples, len(groups)))
    values = []
    for draw in indices:
        p = np.concatenate([groups[i][0] for i in draw])
        y = np.concatenate([groups[i][1] for i in draw])
        value = metric(p, y)
        if value is not None:
            values.append(value)
    return dict(low=float(np.percentile(values, 2.5)), high=float(np.percentile(values, 97.5)), episodes=len(groups))


def iou(a: np.ndarray, b: np.ndarray) -> float | None:
    union = int((a | b).sum())
    return int((a & b).sum()) / union if union else None
