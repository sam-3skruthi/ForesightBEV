"""Risk validation on real logs: interventions and the unknown-space policy, on mini_val.

Annotations score the outcome and are never pipeline inputs. Each frame's nominal plan is
checked against where annotated objects really were at later keyframes. Replay is open
loop, so other road users did not react to that plan: a "conflict" here means the plan
would have met the logged scene, not that a collision would have occurred. Thresholds are
the frozen ones, and this study changes none of them.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import yaml

from foresightbev.config import load_pipeline_config, load_settings
from foresightbev.data.nuscenes import NuScenesDataset
from foresightbev.data.sequence import LabelSequence
from foresightbev.evaluation.nuscenes_eval import NuScenesGroundTruth, SceneScorer
from foresightbev.replay import run_sequence

ROOT = Path(__file__).resolve().parents[3]


def study(scenes: list[str], config) -> dict:
    settings = load_settings()
    truth = NuScenesGroundTruth(NuScenesDataset(settings.nuscenes_root, settings.nuscenes_version))
    counts = Counter()
    reasons = Counter()
    selected = {}
    for scene in scenes:
        sequence = LabelSequence(settings.data_dir / "sequences" / scene)
        scorer = SceneScorer(truth, scene, sequence, config)

        def observe(result, scene=scene, scorer=scorer):
            index = sequence.frames[result.bundle.frame_index]["source_index"]
            scorer.add(result)
            nominal = next(c for c in result.candidates if c.is_nominal)
            conflict = scorer._true_contact(nominal, index, result.bundle.ego)
            intervened = result.decision.status.value != "pass"
            counts["frames"] += 1
            counts["nominal_conflicts_in_annotations"] += conflict
            counts["interventions"] += intervened
            counts["interventions_with_conflict"] += intervened and conflict
            counts["interventions_without_conflict"] += intervened and not conflict
            counts["conflicts_passed_without_intervention"] += conflict and not intervened
            reasons.update(r.value for r in result.decision.reasons)
            selected[(scene, result.bundle.frame_index)] = result.decision.selected_id

        run_sequence(sequence, config, None, observe)
        counts["logged_alarm"] += scorer.scores.logged_alarm
        counts["logged_unknown_violation"] += scorer.scores.logged_unknown
        counts["logged_frames"] += scorer.scores.logged_frames
        counts["logged_true_contact"] += scorer.scores.logged_contact
    return dict(counts=dict(counts), reasons=dict(reasons)), selected


def main() -> int:
    scenes = yaml.safe_load((ROOT / "configs/nuscenes_splits.yaml").read_text())["mini_val"]
    config = load_pipeline_config(ROOT / "configs/pipeline_showcase.yaml")
    conservative, picked = study(scenes, config)
    permissive, other = study(scenes, replace(config, risk=replace(config.risk, treat_unknown_as_free=True)))
    report = dict(status="mini_val, previously seen, preliminary", scenes=scenes, frames=conservative["counts"]["frames"],
                  note="Open-loop replay. A conflict means the nominal plan overlapped annotated objects at later keyframes as logged; other road users did not react. Thresholds are frozen and were not tuned here.",
                  unknown_conservative=conservative, unknown_as_free=permissive, frames_with_different_choice=sum(picked[k] != other[k] for k in picked))
    out = load_settings().data_dir / "results/safety_real"
    out.mkdir(parents=True, exist_ok=True)
    (out / "mini_val.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
