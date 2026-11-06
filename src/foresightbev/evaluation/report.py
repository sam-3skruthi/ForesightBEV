"""Build a Markdown report from stored Stage 9 results. It computes nothing new."""

from __future__ import annotations

import json
from pathlib import Path


def ci(block: dict, digits: int = 3) -> str:
    if block["mean"] is None:
        return "n/a"
    return f"{block['mean']:.{digits}f} [{block['low']:.{digits}f}, {block['high']:.{digits}f}]"


def render(directory: Path) -> str:
    summary = json.loads((directory / "summary.json").read_text())
    config = json.loads((directory / "experiment.json").read_text())
    environment = json.loads((directory / "environment.json").read_text())
    lines = ["# Tavyra Stage 9 report", "",
             "Generated from stored results by `foresightbev report`. **All closed-loop numbers come from the built-in "
             "kinematic simulator with scripted, non-reactive agents, rendered through the real Cam2BEV rig calibration.** "
             "They describe these scenarios only. They are not real-world safety evidence and are not comparable to "
             "published benchmarks.", "",
             f"Commit `{environment['git_commit'][:10]}` (dirty: {environment['git_dirty']}), Python {environment['python']}, "
             f"{environment['platform']}. Closed-loop seeds {config['closed_loop_seeds']}, image scale {config['image_scale']}, "
             f"noise {config.get('noise')}.", "",
             "Intervals are 95% percentile bootstrap over episodes.", "", "## Closed loop", "",
             "| Mode | Episodes | Collision rate | Route completion | Goal rate | Freeze rate | Mean max decel (m/s²) | Intervention frames | Emergency frames |",
             "|---|---:|---|---|---|---|---|---:|---:|"]
    for name, block in summary["modes"].items():
        o = block["overall"]
        lines.append(f"| {name} | {o['episodes']} | {ci(o['collision_rate'])} | {ci(o['route_completion'])} | {ci(o['goal_rate'])} | "
                     f"{ci(o['freeze_rate'])} | {ci(o['max_decel'], 2)} | {o['intervention_fraction']:.3f} | {o['emergency_fraction']:.3f} |")
    families = sorted({f for b in summary["modes"].values() for f in b["by_scenario"]})
    lines += ["", "## Collision rate by scenario", "", "| Mode | " + " | ".join(families) + " |", "|---|" + "---:|" * len(families)]
    for name, block in summary["modes"].items():
        lines.append(f"| {name} | " + " | ".join(f"{block['by_scenario'][f]['collision_rate']['mean']:.2f}" for f in families) + " |")
    lines += ["", "## Route completion by scenario", "", "| Mode | " + " | ".join(families) + " |", "|---|" + "---:|" * len(families)]
    for name, block in summary["modes"].items():
        lines.append(f"| {name} | " + " | ".join(f"{block['by_scenario'][f]['route_completion']['mean']:.2f}" for f in families) + " |")
    p = summary.get("perception")
    if p:
        o, t, f = p["occupancy"], p["tracking"], p["forecast"]
        lines += ["", "## Perception, mode `filter`", "", "Footprints against simulator ground truth, per BEV cell.", "",
                  "| Metric | Value |", "|---|---:|",
                  f"| False-free rate (true occupied cell marked FREE) | {o['false_free_rate']:.4f} |",
                  f"| True occupied cell marked OCCUPIED | {o['gt_marked_occupied']:.4f} |",
                  f"| True occupied cell marked UNKNOWN or OUTSIDE_FOV | {o['gt_marked_unknown_or_outside']:.4f} |",
                  f"| Occupied precision | {o['occupied_precision']:.4f} |", "",
                  "Only near surfaces can be OCCUPIED, so an object interior is expected to be UNKNOWN.", "",
                  "## Tracking", "", "| Metric | Value |", "|---|---:|",
                  f"| Recall within 40 m (3 m gate) | {t['recall']:.3f} |", f"| Mean position error (m) | {t['mean_position_error_m']:.2f} |",
                  f"| Mean velocity error (m/s) | {t['mean_velocity_error_mps']:.2f} |", f"| ID switches | {t['id_switches']} |",
                  f"| False track-frames | {t['false_track_frames']} |", f"| Mean NEES, 4 states (consistent filter is near 4) | {t['mean_nees_4dof']:.1f} |",
                  "", "## Forecast", "", "IoU of dynamic-agent footprints with true future boxes, on scenarios with moving agents.", "",
                  "| Horizon (s) | CV Monte Carlo IoU | Frozen tracks IoU | Brier (relevant cells) | Mean displacement error (m) |", "|---:|---:|---:|---:|---:|"]
        for i, h in enumerate(f["horizons_s"]):
            lines.append(f"| {h:.2f} | {f['agent_iou']['forecast'][i]:.3f} | {f['agent_iou']['persistence'][i]:.3f} | "
                         f"{f['brier_on_relevant_cells'][i]:.3f} | {f['mean_displacement_error_m'][i]:.2f} |")
    calibration = directory / "calibration.json"
    if calibration.exists():
        c = json.loads(calibration.read_text())
        lines += ["", "## Trajectory-event calibration", "", f"Event: {c['event']}. {c['note']}", "",
                  f"Fit rows {c['fit_rows']}, test rows {c['test_rows']}, test base rate {c['test_base_rate']:.3f}. "
                  f"Seeds fit {c['fit_seeds']}, test {c['test_seeds']}.", "",
                  "| Score | Brier | NLL | ECE (10 equal-width bins) | ECE 95% interval (episode bootstrap) | AUROC |", "|---|---:|---:|---:|---|---:|"]
        for label in ("raw", "calibrated"):
            s = c[label]
            interval = s.get("ece_interval")
            band = f"[{interval['low']:.4f}, {interval['high']:.4f}]" if interval else "not computed"
            lines.append(f"| {label} | {s['brier']:.4f} | {s['nll']:.4f} | {s['ece_10_equal_width']:.4f} | {band} | {s['auroc']:.3f} |")
    learned = directory / "learned_forecaster.json"
    if learned.exists():
        f = json.loads(learned.read_text())
        names = list(f["models"])
        t = f["train"]
        lines += ["", "## Trained forecaster against baselines", "", f"{f['note']} Train rows {t['rows_train']}, test rows {t['rows_test']}, {t['parameters']} parameters.", "",
                  "IoU at p>=0.5 by horizon:", "", "| Horizon (s) | " + " | ".join(names) + " |", "|---:|" + "---:|" * len(names)]
        for i, h in enumerate(f["horizons_s"]):
            lines.append(f"| {h} | " + " | ".join(f"{f['models'][n]['iou'][i]:.3f}" for n in names) + " |")
    real = directory / "real_sample.json"
    if real.exists():
        r = json.loads(real.read_text())
        lines += ["", "## Real multi-camera sample, Stage 2-3 only", "", f"{r['input']}. {r['note']}", "", "| Metric | Value |", "|---|---:|"]
        for key in ("free_precision_gt_ground", "ground_recall_as_free", "false_free_rate_gt_obstacle", "occupied_precision_gt_obstacle",
                    "vehicle_gt_marked_occupied_as_dynamic"):
            lines.append(f"| {key} | {r[key]:.3f} |")
        lines.append(f"| occluded recall, excluding static cells | {r['visibility_agreement_excluding_static_cells']['recall']:.3f} |")
        lines.append(f"| occluded precision, excluding static cells | {r['visibility_agreement_excluding_static_cells']['precision']:.3f} |")
    return "\n".join(lines) + "\n"
