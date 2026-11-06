"""Stage 5 trained baseline: small U-Net forecasting agent occupancy on a coarse BEV.

Input is the last four occupancy frames, warped into the current ego frame. Output
is one logit per cell for six horizons. Future frames appear only as training
labels. It is trained on built-in simulator data only, so it is a trained synthetic
baseline and not a validated 4D occupancy model. It returns per-cell marginals and
no joint samples, so Stage 6 risk does not use it: per-cell marginals cannot
support a trajectory event without an independence assumption.
"""

from __future__ import annotations

import copy
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from foresightbev.contracts import BEVGrid, CellState, EgoState, OccupancyFrame, SemanticClass as C
from foresightbev.forecast.forecaster import footprint_polygon, rasterize, track_to_ego
from foresightbev.sim.world import obb_corners

FACTOR = 4
HISTORY = 4
CHANNELS = 4
HORIZONS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


def coarse_grid(grid: BEVGrid) -> BEVGrid:
    return BEVGrid(grid.x_min, grid.x_max, grid.y_min, grid.y_max, grid.resolution * FACTOR)


def downsample(mask: np.ndarray) -> np.ndarray:
    """Area coverage of each coarse cell, as float32."""
    nx, ny = mask.shape[0] // FACTOR, mask.shape[1] // FACTOR
    return mask[: nx * FACTOR, : ny * FACTOR].reshape(nx, FACTOR, ny, FACTOR).astype(np.float32).mean(axis=(1, 3))


def occupancy_channels(occupancy: OccupancyFrame) -> np.ndarray:
    state, classes = occupancy.bev_state, occupancy.bev_class
    return np.stack([downsample(np.isin(classes, [C.VEHICLE, C.PERSON])), downsample(state == CellState.OCCUPIED),
                     downsample(state == CellState.FREE), downsample(np.isin(state, [CellState.UNKNOWN, CellState.OUTSIDE_FOV]))])


def warp_to(channels: np.ndarray, past: EgoState, now: EgoState, grid: BEVGrid) -> np.ndarray:
    """Resample a past frame into the current ego frame by the known ego motion."""
    X, Y = grid.cell_centers()
    world = now.ego_to_world(np.stack([X, Y], -1))
    p = past.world_to_ego(world)
    map_iy = ((p[..., 1] - grid.y_min) / grid.resolution - 0.5).astype(np.float32)
    map_ix = ((p[..., 0] - grid.x_min) / grid.resolution - 0.5).astype(np.float32)
    return np.stack([cv2.remap(c, map_iy, map_ix, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0) for c in channels])


class HistoryBuffer:
    """Causal input builder. Older slots repeat the oldest frame at sequence start."""

    def __init__(self, grid: BEVGrid):
        self.grid = coarse_grid(grid)
        self.frames: deque[tuple[EgoState, np.ndarray]] = deque(maxlen=HISTORY)

    def push(self, occupancy: OccupancyFrame, ego: EgoState) -> np.ndarray:
        if self.frames and ego.timestamp <= self.frames[-1][0].timestamp:
            raise ValueError("history must be pushed in increasing time order")
        self.frames.append((ego, occupancy_channels(occupancy)))
        items = list(self.frames)
        items = [items[0]] * (HISTORY - len(items)) + items
        return np.concatenate([warp_to(c, e, ego, self.grid) for e, c in items])


def future_labels(agents, ego: EgoState, t0: float, grid: BEVGrid, dt: float = 0.05) -> np.ndarray:
    """Offline labels only: true agent footprints at each horizon, in the current ego frame."""
    future = [copy.copy(a) for a in agents]
    fine = BEVGrid(grid.x_min, grid.x_max, grid.y_min, grid.y_max, grid.resolution / FACTOR)
    out = []
    t = t0
    for horizon in HORIZONS:
        while t < t0 + horizon - 1e-9:
            t += dt
            for agent in future:
                if agent.script is not None:
                    agent.script(agent, t, dt)
        mask = np.zeros(fine.shape, bool)
        for agent in future:
            corners = ego.world_to_ego(obb_corners(agent.x, agent.y, agent.yaw, agent.length, agent.width))
            mask |= rasterize(corners, fine)
        out.append(downsample(mask) > 0.25)
    return np.stack(out)


def baseline_predictions(result) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo CV probability and frozen-track mask on the same coarse grid."""
    forecast = result.forecast
    ego = result.bundle.ego
    indices = [int(np.argmin(np.abs(forecast.horizons - h))) for h in HORIZONS]
    fine = forecast.grid
    cv = np.stack([downsample(forecast.agent_samples[k].mean(0)) for k in indices])
    mask = np.zeros(fine.shape, bool)
    for track in result.tracks:
        if track.confirmed:
            position, velocity, _ = track_to_ego(track, ego)
            yaw = float(np.arctan2(velocity[1], velocity[0])) if np.hypot(*velocity) > 1 else 0.0
            mask |= rasterize(footprint_polygon(*position, yaw, track.length, track.width), fine)
    return cv, np.repeat((downsample(mask) > 0.25)[None], len(HORIZONS), axis=0)


def build_model(in_channels: int = CHANNELS * HISTORY, out_channels: int = len(HORIZONS), width: int = 24):
    import torch
    from torch import nn

    def block(a, b):
        return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(4, b), nn.SiLU(), nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(4, b), nn.SiLU())

    class SmallUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1, self.e2, self.e3 = block(in_channels, width), block(width, 2 * width), block(2 * width, 4 * width)
            self.pool = nn.MaxPool2d(2)
            self.d2, self.d1 = block(6 * width, 2 * width), block(3 * width, width)
            self.out = nn.Conv2d(width, out_channels, 1)

        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            up2 = nn.functional.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
            d2 = self.d2(torch.cat([up2, e2], 1))
            up1 = nn.functional.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
            return self.out(self.d1(torch.cat([up1, e1], 1)))

    return SmallUNet()


def _dataset_job(job: tuple) -> dict:
    from foresightbev.evaluation.experiments import build_mode, noise_from, scaled_rig
    from foresightbev.sim.backend import BuiltinSim
    from foresightbev.sim.closed_loop import run_episode
    from foresightbev.sim.scenarios import build_scenario

    cfg, root, name, seed = job
    scenario = build_scenario(name, seed)
    sim = BuiltinSim(scenario, scaled_rig(cfg, Path(root)), noise_from(cfg), seed=seed)
    config, _ = build_mode(cfg["modes"]["filter"])
    config.seed = seed
    buffer = HistoryBuffer(config.grid.bev)
    rows = dict(x=[], y=[], cv=[], frozen=[])

    def observer(result, simulator):
        x = buffer.push(result.occupancy, result.bundle.ego)
        if result.bundle.frame_index < HISTORY - 1 or result.bundle.frame_index % 2:
            return
        cv, frozen = baseline_predictions(result)
        rows["x"].append(x.astype(np.float16))
        rows["y"].append(future_labels(simulator.agents, result.bundle.ego, simulator.t, buffer.grid))
        rows["cv"].append(cv.astype(np.float16))
        rows["frozen"].append(frozen)

    run_episode(sim, config, "dataset", on_frame=observer)
    return {k: np.stack(v) if v else None for k, v in rows.items()} | dict(seed=seed, scenario=name)


def build_dataset(cfg: dict, root: Path, seeds: tuple[int, int], out: Path, workers: int) -> Path:
    from foresightbev.evaluation.experiments import run_pool
    from foresightbev.sim.scenarios import FAMILIES

    jobs = [(cfg, str(root), name, seed) for name in FAMILIES for seed in range(*seeds)]
    results = [r for r in run_pool(_dataset_job, jobs, workers) if r["x"] is not None]
    results.sort(key=lambda r: (r["scenario"], r["seed"]))
    arrays = {k: np.concatenate([r[k] for r in results]) for k in ("x", "y", "cv", "frozen")}
    arrays["seed"] = np.concatenate([np.full(len(r["x"]), r["seed"]) for r in results])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    return out


@dataclass
class TrainConfig:
    epochs: int = 25
    batch_size: int = 32
    learning_rate: float = 1e-3
    positive_weight: float = 6.0
    seed: int = 0
    device: str = "mps"


def per_horizon(prediction: np.ndarray, truth: np.ndarray, region: np.ndarray | None = None) -> tuple[list[float], list[float]]:
    """IoU at p>=0.5, and Brier over ``region``. Models compare only on one shared region."""
    iou, brier = [], []
    for k in range(truth.shape[1]):
        p, y = prediction[:, k], truth[:, k]
        mask = p >= 0.5
        iou.append(float((mask & y).sum() / max(int((mask | y).sum()), 1)))
        cells = (y | (p > 0.01)) if region is None else region[:, k]
        brier.append(float(((p - y) ** 2)[cells].mean()) if cells.any() else 0.0)
    return iou, brier


def score_models(predictions: dict[str, np.ndarray], truth: np.ndarray) -> dict:
    """Score every model on identical frames and an identical cell region."""
    region = truth.copy()
    for prediction in predictions.values():
        region |= prediction > 0.01
    return {name: dict(zip(("iou", "brier"), per_horizon(p, truth, region))) for name, p in predictions.items()}


def train(train_path: Path, val_path: Path, test_path: Path, checkpoint: Path, config: TrainConfig | None = None) -> dict:
    import torch

    cfg = config or TrainConfig()
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if cfg.device != "mps" or torch.backends.mps.is_available() else "cpu")

    def load(path):
        with np.load(path) as d:
            return {k: d[k] for k in d.files}

    data, val, test = load(train_path), load(val_path), load(test_path)
    X = torch.from_numpy(data["x"].astype(np.float32))
    Y = torch.from_numpy(data["y"].astype(np.float32))
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)
    weight = torch.tensor(cfg.positive_weight, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)

    def predict(array):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(array), 64):
                out.append(torch.sigmoid(model(torch.from_numpy(array[i:i + 64].astype(np.float32)).to(device))).cpu().numpy())
        return np.concatenate(out)

    generator = torch.Generator().manual_seed(cfg.seed)
    history, best = [], -1.0
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(cfg.epochs):
        model.train()
        order = torch.randperm(len(X), generator=generator)
        total = 0.0
        for i in range(0, len(X), cfg.batch_size):
            batch = order[i:i + cfg.batch_size]
            loss = loss_fn(model(X[batch].to(device)), Y[batch].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss) * len(batch)
        scheduler.step()
        val_iou, _ = per_horizon(predict(val["x"]), val["y"])
        score = float(np.mean(val_iou))
        history.append(dict(epoch=epoch, train_loss=total / len(X), val_mean_iou=score))
        print(f"epoch {epoch} loss {total / len(X):.4f} val_iou {score:.4f}", flush=True)
        if score > best:
            best = score
            torch.save(dict(model=model.state_dict(), val_mean_iou=score, epoch=epoch), checkpoint)
    model.load_state_dict(torch.load(checkpoint, weights_only=True)["model"])
    models = score_models(dict(learned_unet=predict(test["x"]), constant_velocity_monte_carlo=test["cv"].astype(np.float32), frozen_tracks=test["frozen"].astype(np.float32)), test["y"])
    return dict(horizons_s=list(HORIZONS), models=models, train=dict(rows_train=len(X), rows_val=len(val["x"]), rows_test=len(test["x"]),
                                                                  seeds_train=sorted(set(data["seed"].tolist())), seeds_test=sorted(set(test["seed"].tolist())),
                                                                  parameters=sum(p.numel() for p in model.parameters()), device=str(device),
                                                                  best_val_mean_iou=best, history=history, config=vars(cfg)),
                note="All three models scored on identical held-out simulator frames at 1 m resolution, with Brier over one shared cell region. Synthetic data only.")


if __name__ == "__main__":
    import argparse

    from foresightbev.evaluation.experiments import load_experiment

    parser = argparse.ArgumentParser(description="Build data for, train and score the learned forecaster.")
    parser.add_argument("--name", default="default")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--train-seeds", type=int, nargs=2, default=(300, 330))
    parser.add_argument("--val-seeds", type=int, nargs=2, default=(330, 336))
    parser.add_argument("--test-seeds", type=int, nargs=2, default=(336, 346))
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    base = root / "experiments/end_to_end"
    data = base / "data/forecast"
    cfg = load_experiment(root / "configs/experiment_default.yaml")
    for split, seeds in (("train", args.train_seeds), ("val", args.val_seeds), ("test", args.test_seeds)):
        path = data / f"{split}.npz"
        if not path.exists():
            build_dataset(cfg, root, tuple(seeds), path, args.workers)
            print("built", path, flush=True)
    result = train(data / "train.npz", data / "val.npz", data / "test.npz", base / "checkpoints/forecast_unet.pt", TrainConfig(epochs=args.epochs))
    (base / "results" / args.name).mkdir(parents=True, exist_ok=True)
    (base / "results" / args.name / "learned_forecaster.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result["models"].items()}, indent=1))
