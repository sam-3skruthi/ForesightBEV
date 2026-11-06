"""Per-class and score-sweep tables for cached StreamPETR detections on mini_val.

Reporting only. The operating point stays at the score fixed before evaluation.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import yaml

from foresightbev.config import ROOT, load_settings
from foresightbev.data.nuscenes import NuScenesDataset, NuScenesGroundTruth
from foresightbev.data.sequence import LabelSequence
from foresightbev.evaluation.nuscenes_eval import MATCH_M, MAX_RANGE_M, MIN_VISIBILITY, eligible, excused, in_range

FINE = dict(car="vehicle.car", truck="vehicle.truck", construction_vehicle="vehicle.construction", bus="vehicle.bus", trailer="vehicle.trailer", motorcycle="vehicle.motorcycle", bicycle="vehicle.bicycle", pedestrian="human.pedestrian")
SWEEP = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def _match(predicted: list[dict], boxes: list) -> set[int]:
    pairs = sorted((float(np.hypot(*(np.array(p["center_ego"][:2]) - c))), i, j) for i, p in enumerate(predicted) for j, (_, c) in enumerate(boxes))
    a, b = set(), set()
    for d, i, j in pairs:
        if d <= MATCH_M and i not in a and j not in b:
            a.add(i); b.add(j)
    return a


def _count(rows: list[dict], target: list, all_boxes: list, ego) -> dict:
    taken = _match(rows, target)
    unmatched = [r for i, r in enumerate(rows) if i not in taken]
    return dict(annotated=len(target), predicted=len(rows) - excused(unmatched, lambda r: np.array(r["center_ego"][:2]), all_boxes, ego), matched=len(taken))


def main() -> int:
    settings = load_settings()
    dataset = NuScenesDataset(settings.nuscenes_root, settings.nuscenes_version)
    truth = NuScenesGroundTruth(dataset)
    scenes = yaml.safe_load((ROOT / "configs/nuscenes_splits.yaml").read_text())["mini_val"]
    fine = {k: Counter() for k in FINE}
    sweep = {t: Counter() for t in SWEEP}
    for scene in scenes:
        sequence = LabelSequence(settings.data_dir / "sequences" / scene)
        doc = json.loads((sequence.directory / "detections_streampetr.json").read_text())
        for frame in doc["frames"]:
            ego = sequence.bundle(frame["index"]).ego
            all_boxes = truth.boxes(frame["token"])
            boxes = eligible(all_boxes, ego)
            rows = in_range(frame["detections"], lambda r: np.array(r["center_ego"][:2]))
            for label, prefix in FINE.items():
                target = [b for b in boxes if b[0].category.startswith(prefix)]
                fine[label].update(_count([r for r in rows if r["label"] == label and r["score"] >= 0.3], target, all_boxes, ego))
                for threshold in SWEEP:
                    sweep[threshold].update(_count([r for r in rows if r["label"] == label and r["score"] >= threshold], target, all_boxes, ego))

    def score(c):
        return dict(annotated=c["annotated"], predicted=c["predicted"], matched=c["matched"], recall=c["matched"] / c["annotated"] if c["annotated"] else None,
                    precision=c["matched"] / c["predicted"] if c["predicted"] else None)

    out = settings.data_dir / "results/streampetr"
    report = dict(scenes=scenes, match_m=MATCH_M, max_range_m=MAX_RANGE_M, min_visibility=MIN_VISIBILITY, operating_score=0.3, per_class=dict((k, score(v)) for k, v in fine.items()), score_sweep_reporting_only={str(k): score(v) for k, v in sweep.items()})
    (out / "per_class_and_sweep.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=0)[:2600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
