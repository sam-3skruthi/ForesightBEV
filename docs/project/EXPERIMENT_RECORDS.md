# Experiment records

Stored in `experiments/end_to_end/results/<name>/`. `experiments/end_to_end/experiments.jsonl` is an append-only index
of runs from the integrity tooling onward. `dev_*` sets are tuning history, not results.

| Set | Data | Split | Samples | Status |
|---|---|---|---:|---|
| `final` | Built-in simulator, four cameras | Closed-loop seeds 400–414, never used in development | 90 episodes per mode, 810 total | **Reported** |
| `final/calibration.json` | Built-in simulator | Fit seeds 200–219, test seeds 230–239 | 51,350 fit rows, 25,740 test rows | **Reported** |
| `final/learned_forecaster.json` | Built-in simulator | Train seeds 300–329, val 330–335, test 336–345 | 5,581 / 1,105 / 1,869 frames | **Reported** |
| `default` | Built-in simulator | Seeds 100–109, overlap tuning seeds 100–102 | 60 episodes per mode | Development only |
| `nuscenes_heldout` | nuScenes v1.0-mini, real, six cameras | Eight held-out scenes, scored once with a frozen config | 325 frames | **Reported** |
| `streampetr_mini_val` | nuScenes mini_val, StreamPETR | Two previously seen scenes, preliminary | 81 frames, 2,459 objects | **Reported** |
| `streampetr_baseline_image_size` | nuScenes mini_val, apparent-size baseline | Same scenes and scorer | 81 frames | **Reported** |
| `streampetr/` | Port verification with controls, per-class table, score sweep | mini_val | 81 frames | **Reported** |
| `nuscenes_heldout_check` | Same as `nuscenes_heldout`, corrected scorer | Reproducibility: decisions identical | 325 frames | Verification |
| `dev_baseline`, `dev_v1`–`dev_v3` | nuScenes v1.0-mini | Development scenes 0061 and 0103 | 39–79 frames | Tuning history |
| `datasets/` | nuScenes validation report | All | 404 samples | **Reported** |
| `acceptance/` | Cam2BEV sample, scene-0553, scene-1100 | End-to-end checks | 1, 41, 40 frames | **Reported** |
| `ros2/` | Simulator and replays | ROS against in-process comparison | Per run | **Reported** |
| `verification/` | Everything | Recorded verification pass | See `summary.json` | **Reported** |

Calibration note: `final` predates the `ece_interval` field, so it has no clustered interval. Re-running calibration
produces one.

## Reproduce

```bash
.venv/bin/python -m foresightbev.cli eval --config configs/experiment_final.yaml --name final_rerun --final
```

```bash
.venv/bin/python -m foresightbev.forecast.learned --name final_rerun
```

```bash
.venv/bin/python -m foresightbev.cli prepare nuscenes
```

```bash
.venv/bin/python -m foresightbev.cli eval-nuscenes --split held_out --name nuscenes_rerun
```

Results are deterministic given the same environment. `scripts/check_final_reproduces.py` replays stored episodes and
compares them field by field. Across environments, numeric libraries can change Monte Carlo draws, so ROS comparisons
run in the ROS environment on both sides.

## Leakage review

| Risk | Control |
|---|---|
| Seed overlap between splits | `check_seed_separation` refuses overlaps at start, including forecaster seeds |
| Tuning on reported seeds | `--final` refuses development seeds. `final` uses 400–414. |
| Calibration fit and test overlap | Disjoint seeds, and a guard in `run_calibration` |
| Future data in inference | `FrameBundle` has no such field. Forecast rejects later inputs. Future agent states exist only in offline labels. |
| Annotations in inference | Separate class, AST guard, and a test that deletes the tables |
| LiDAR in inference | No code reads it |
| nuScenes tuning on reported scenes | Tuned on two scenes only; eight held out and scored once |
| Correlated rows treated as independent | Episode-level bootstrap for calibration intervals |

Residual: calibration rows come from frames visited by the uncalibrated filter, so `filter_calibrated` sees a slightly
different distribution. Simulator seeds share scenario families and code, so "held out" means unseen parameters.
