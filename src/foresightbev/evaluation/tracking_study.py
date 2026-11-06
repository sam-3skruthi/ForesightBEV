"""Compare tracker settings on cached detections. Tuning uses mini_train only.

mini_train scenes were in the detector's training data, so absolute figures there are
inflated. They are used only to compare tracker logic. mini_val is scored once, afterwards.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import yaml

from foresightbev.config import ROOT, load_settings

from foresightbev.data.nuscenes import NuScenesDataset, NuScenesGroundTruth
from foresightbev.data.sequence import LabelSequence
from foresightbev.evaluation.nuscenes_eval import Scores, SceneScorer, eligible

from foresightbev.perception.detection_cache import CachedDetector
from foresightbev.pipeline import PipelineConfig
from foresightbev.risk.evaluator import RiskConfig
from foresightbev.tracking.tracker import Tracker, TrackerConfig

BASE = TrackerConfig(max_misses=2)
VARIANTS = {
    "current": dict(),
    "model_velocity": dict(use_model_velocity=True),
    "model_velocity_low_noise": dict(use_model_velocity=True, velocity_std=0.4),
    "two_stage": dict(use_model_velocity=True, second_stage=True, birth_score=0.3),
    "two_stage_longer_coast": dict(use_model_velocity=True, second_stage=True, birth_score=0.3, max_misses=4),
    "two_stage_confirm_2": dict(use_model_velocity=True, second_stage=True, birth_score=0.3, confirm_hits=2),
    "two_stage_higher_birth": dict(use_model_velocity=True, second_stage=True, birth_score=0.4),
}


def score_variant(dataset, truth, settings, scenes: list[str], config: TrackerConfig) -> dict:
    pooled = Scores()
    for scene in scenes:
        sequence = LabelSequence(settings.data_dir / "sequences" / scene)
        scorer = SceneScorer(truth, scene, sequence, PipelineConfig(risk=RiskConfig()))
        scorer.forecast_config.stationary_sigmas = 3.0
        floor = 0.1 if config.second_stage else 0.3
        detector = CachedDetector(sequence.directory, floor)
        tracker = Tracker(config)
        for index in range(len(sequence)):
            bundle = sequence.bundle(index)
            detections = detector(None, bundle)
            tracks = tracker.update(detections, bundle.ego)
            source = sequence.frames[index]["source_index"]
            scorer.all_boxes = truth.boxes(scorer.tokens[source])
            boxes = eligible(scorer.all_boxes, bundle.ego)
            result = SimpleNamespace(bundle=bundle, tracks=tracks, detections=[d for d in detections if d.support >= 0.3])
            scorer._tracking(result, boxes, source)
            scorer._forecast(result, boxes, source)
        pooled.merge(scorer.scores)
    s = pooled.summary()
    t, f = s["tracking"], s["forecast"]
    return dict(recall=t["recall_2m"], matched=t["matched"], annotated=t["annotated"], false_track_frames=t["false_track_frames"], id_switches=t["id_switches"], id_switches_per_100=100 * t["id_switches"] / max(t["matched"], 1),
                position_error=t["mean_position_error_m"], velocity_error=t["mean_velocity_error_mps"], nees=t["mean_nees_4dof"], forecast_1s=f["constant_velocity_error_m"][1], forecast_3s=f["constant_velocity_error_m"][5],
                frozen_1s=f["frozen_error_m"][1], frozen_3s=f["frozen_error_m"][5], pairs_3s=f["pairs"][5])


def main() -> int:
    settings = load_settings()
    dataset = NuScenesDataset(settings.nuscenes_root, settings.nuscenes_version)
    truth = NuScenesGroundTruth(dataset)
    splits = yaml.safe_load((ROOT / "configs/nuscenes_splits.yaml").read_text())
    mini_val = splits["mini_val"]
    mini_train = [s.name for s in dataset.scenes() if s.name not in mini_val]
    which = sys.argv[1] if len(sys.argv) > 1 else "mini_train"
    scenes = mini_train if which == "mini_train" else mini_val
    names = list(VARIANTS) if which == "mini_train" else sys.argv[2:]
    out = {}
    for name in names:
        out[name] = score_variant(dataset, truth, settings, scenes, replace(BASE, **VARIANTS[name]))
        r = out[name]
        print(f"{name:28s} rec {r['recall']:.3f} false {r['false_track_frames']:5d} idsw {r['id_switches']:4d} ({r['id_switches_per_100']:.1f}) pos {r['position_error']:.3f} vel {r['velocity_error']:.3f} nees {r['nees']:.1f} fc {r['forecast_1s']:.2f}/{r['forecast_3s']:.2f} frz {r['frozen_1s']:.2f}/{r['frozen_3s']:.2f}", flush=True)
    target = settings.data_dir / "results/tracking_study"
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{which}.json").write_text(json.dumps(dict(scenes=scenes, split=which, note="mini_train figures are inflated: the detector trained on those scenes." if which == "mini_train" else "mini_val, previously seen, preliminary.", variants=out), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
