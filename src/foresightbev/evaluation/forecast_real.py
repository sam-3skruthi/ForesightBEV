"""Score forecasters on real mini_val annotations, on one shared 1 m BEV grid.

Compares constant velocity, frozen tracks and the simulator-trained U-Net. The U-Net has
never seen real data. Targets are annotated agent footprints at future keyframes, used
offline only. Horizons match nuScenes keyframes: 0.5 s to 3.0 s.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch
import yaml

from foresightbev.config import ROOT, load_pipeline_config, load_settings
from foresightbev.data.nuscenes import NuScenesDataset, NuScenesGroundTruth
from foresightbev.data.sequence import LabelSequence
from foresightbev.forecast.forecaster import rasterize
from foresightbev.forecast.learned import FACTOR, HORIZONS, HistoryBuffer, baseline_predictions, build_model, coarse_grid, downsample, score_models
from foresightbev.replay import run_sequence
from foresightbev.sim.world import obb_corners


def main() -> int:
    settings = load_settings()
    dataset = NuScenesDataset(settings.nuscenes_root, settings.nuscenes_version)
    truth = NuScenesGroundTruth(dataset)
    scenes = yaml.safe_load((ROOT / "configs/nuscenes_splits.yaml").read_text())["mini_val"]
    config = load_pipeline_config(ROOT / "configs/pipeline_showcase.yaml")
    fine = config.grid.bev
    rows: dict[str, list] = dict(x=[], y=[], cv=[], frozen=[])
    for scene in scenes:
        sequence = LabelSequence(settings.data_dir / "sequences" / scene)
        tokens = dataset.sample_tokens(scene)
        buffer = HistoryBuffer(fine)

        def observe(result):
            i = result.bundle.frame_index
            x = buffer.push(result.occupancy, result.bundle.ego)
            if i < 3 or i + 6 >= len(tokens):
                return
            targets = []
            for step in range(1, 7):
                mask = np.zeros(fine.shape, bool)
                for box in truth.boxes(tokens[i + step]):
                    if box.semantic is not None:
                        mask |= rasterize(result.bundle.ego.world_to_ego(obb_corners(box.x, box.y, box.yaw, box.length, box.width)), fine)
                targets.append(downsample(mask) > 0.25)
            cv, frozen = baseline_predictions(result)
            rows["x"].append(x.astype(np.float32)); rows["y"].append(np.stack(targets)); rows["cv"].append(cv.astype(np.float32)); rows["frozen"].append(frozen)

        run_sequence(sequence, config, on_frame=observe)
    data = {k: np.stack(v) for k, v in rows.items()}
    model = build_model()
    model.load_state_dict(torch.load(settings.data_dir / "checkpoints/forecast_unet.pt", weights_only=True)["model"])
    model.eval()
    with torch.no_grad():
        learned = torch.cat([torch.sigmoid(model(torch.from_numpy(data["x"][i:i + 16]))) for i in range(0, len(data["x"]), 16)]).numpy()
    models = score_models(dict(simulator_trained_unet=learned, constant_velocity_monte_carlo=data["cv"], frozen_tracks=data["frozen"].astype(np.float32)), data["y"])
    report = dict(split="mini_val, previously seen, preliminary", scenes=scenes, frames=len(data["x"]), horizons_s=list(HORIZONS), grid_m=fine.resolution * FACTOR, target="annotated agent footprints at future keyframes, offline only", models=models,
                  note="Occupancy-grid metric. The U-Net was trained only on simulator data. Motion models here use the showcase tracker.")
    out = settings.data_dir / "results/forecast_real"
    out.mkdir(parents=True, exist_ok=True)
    (out / "mini_val.json").write_text(json.dumps(report, indent=1))
    for name, value in models.items():
        print(name, "iou", [round(v, 3) for v in value["iou"]], "brier", [round(v, 3) for v in value["brier"]])
    print("frames", len(data["x"]), coarse_grid(fine).shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
