# ForesightBEV Stage 9 report

Generated from stored results by `foresightbev report`. **All closed-loop numbers come from the built-in kinematic simulator with scripted, non-reactive agents, rendered through the real Cam2BEV rig calibration.** They describe these scenarios only. They are not real-world safety evidence and are not comparable to published benchmarks.

Commit `aaef1cb631` (dirty: True), Python 3.12.15, macOS-26.6-arm64-arm-64bit. Closed-loop seeds [100, 101], image scale 0.5, noise {'boundary_jitter_px': 1, 'object_dropout': 0.02, 'spurious_blob_rate': 0.1}.

Intervals are 95% percentile bootstrap over episodes.

## Closed loop

| Mode | Episodes | Collision rate | Route completion | Goal rate | Freeze rate | Mean max decel (m/s²) | Intervention frames | Emergency frames |
|---|---:|---|---|---|---|---|---:|---:|
| blind_filter | 6 | 0.167 [0.000, 0.500] | 0.541 [0.339, 0.774] | 0.167 [0.000, 0.500] | 0.333 [0.000, 0.667] | 5.00 [3.00, 6.00] | 0.691 | 0.069 |
| blind_no_filter | 6 | 0.667 [0.333, 1.000] | 0.570 [0.331, 0.834] | 0.333 [0.000, 0.667] | 0.000 [0.000, 0.000] | 0.00 [0.00, 0.00] | 0.000 | 0.000 |
| filter | 6 | 0.167 [0.000, 0.500] | 0.534 [0.339, 0.751] | 0.167 [0.000, 0.500] | 0.333 [0.000, 0.667] | 5.00 [3.00, 6.00] | 0.188 | 0.044 |
| filter_calibrated | 6 | 0.000 [0.000, 0.000] | 0.572 [0.375, 0.795] | 0.167 [0.000, 0.500] | 0.333 [0.000, 0.667] | 5.00 [3.00, 6.00] | 0.222 | 0.067 |
| filter_front_dropout | 6 | 0.000 [0.000, 0.000] | 0.500 [0.292, 0.744] | 0.167 [0.000, 0.500] | 0.333 [0.000, 0.667] | 5.42 [4.25, 6.00] | 0.540 | 0.051 |
| filter_oracle_tracks | 6 | 0.167 [0.000, 0.500] | 0.537 [0.338, 0.762] | 0.167 [0.000, 0.500] | 0.500 [0.167, 0.833] | 4.38 [2.38, 5.96] | 0.470 | 0.025 |
| filter_unknown_as_free | 6 | 0.167 [0.000, 0.500] | 0.533 [0.339, 0.757] | 0.167 [0.000, 0.500] | 0.333 [0.000, 0.667] | 5.00 [3.00, 6.00] | 0.146 | 0.055 |
| idm_no_filter | 6 | 0.500 [0.167, 0.833] | 0.462 [0.308, 0.686] | 0.167 [0.000, 0.500] | 0.333 [0.000, 0.667] | 4.00 [2.00, 6.00] | 0.000 | 0.000 |
| idm_no_filter_front_dropout | 6 | 0.333 [0.000, 0.667] | 0.561 [0.319, 0.801] | 0.333 [0.000, 0.667] | 0.333 [0.000, 0.667] | 3.00 [1.00, 5.00] | 0.000 | 0.000 |

## Collision rate by scenario

| Mode | cut_in | dense_traffic | free_road | lead_brake | occluded_pedestrian | static_obstacle |
|---|---:|---:|---:|---:|---:|---:|
| blind_filter | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 |
| blind_no_filter | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 | 1.00 |
| filter | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 |
| filter_calibrated | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| filter_front_dropout | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| filter_oracle_tracks | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 |
| filter_unknown_as_free | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 |
| idm_no_filter | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| idm_no_filter_front_dropout | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |

## Route completion by scenario

| Mode | cut_in | dense_traffic | free_road | lead_brake | occluded_pedestrian | static_obstacle |
|---|---:|---:|---:|---:|---:|---:|
| blind_filter | 0.84 | 0.44 | 1.00 | 0.32 | 0.40 | 0.24 |
| blind_no_filter | 0.27 | 0.50 | 1.00 | 0.35 | 1.00 | 0.30 |
| filter | 0.76 | 0.49 | 1.00 | 0.30 | 0.40 | 0.25 |
| filter_calibrated | 0.76 | 0.49 | 1.00 | 0.30 | 0.64 | 0.25 |
| filter_front_dropout | 0.75 | 0.44 | 1.00 | 0.27 | 0.30 | 0.23 |
| filter_oracle_tracks | 0.78 | 0.50 | 1.00 | 0.29 | 0.40 | 0.25 |
| filter_unknown_as_free | 0.76 | 0.49 | 1.00 | 0.30 | 0.39 | 0.26 |
| idm_no_filter | 0.27 | 0.49 | 1.00 | 0.32 | 0.40 | 0.29 |
| idm_no_filter_front_dropout | 0.27 | 0.49 | 1.00 | 0.32 | 1.00 | 0.29 |

## Perception, mode `filter`

Footprints against simulator ground truth, per BEV cell.

| Metric | Value |
|---|---:|
| False-free rate (true occupied cell marked FREE) | 0.0101 |
| True occupied cell marked OCCUPIED | 0.0393 |
| True occupied cell marked UNKNOWN or OUTSIDE_FOV | 0.9507 |
| Occupied precision | 0.8853 |

Only near surfaces can be OCCUPIED, so an object interior is expected to be UNKNOWN.

## Tracking

| Metric | Value |
|---|---:|
| Recall within 40 m (3 m gate) | 0.767 |
| Mean position error (m) | 1.16 |
| Mean velocity error (m/s) | 0.82 |
| ID switches | 8 |
| False track-frames | 35 |
| Mean NEES, 4 states (consistent filter is near 4) | 50.5 |

## Forecast

IoU of dynamic-agent footprints with true future boxes, on scenarios with moving agents.

| Horizon (s) | CV Monte Carlo IoU | Frozen tracks IoU | Brier (relevant cells) | Mean displacement error (m) |
|---:|---:|---:|---:|---:|
| 0.25 | 0.349 | 0.256 | 0.290 | 1.88 |
| 0.50 | 0.337 | 0.164 | 0.260 | 1.95 |
| 0.75 | 0.328 | 0.140 | 0.231 | 2.00 |
| 1.00 | 0.316 | 0.139 | 0.212 | 2.06 |
| 1.25 | 0.315 | 0.141 | 0.191 | 1.95 |
| 1.50 | 0.313 | 0.147 | 0.174 | 2.09 |
| 1.75 | 0.304 | 0.158 | 0.159 | 2.16 |
| 2.00 | 0.291 | 0.165 | 0.145 | 2.33 |
| 2.25 | 0.274 | 0.164 | 0.134 | 2.37 |
| 2.50 | 0.255 | 0.164 | 0.126 | 2.33 |
| 2.75 | 0.239 | 0.168 | 0.117 | 2.35 |
| 3.00 | 0.223 | 0.172 | 0.109 | 2.50 |

## Trajectory-event calibration

Event: inflated ego footprint overlaps a true agent within 3 s, excluding overlap at t=0. Rows within an episode are correlated, so effective sample size is below test_rows.

Fit rows 4488, test rows 2244, test base rate 0.163. Seeds fit [200, 202], test [230, 231].

| Score | Brier | NLL | ECE (10 equal-width bins) | AUROC |
|---|---:|---:|---:|---:|
| raw | 0.0687 | 0.4813 | 0.0713 | 0.958 |
| calibrated | 0.0545 | 0.1774 | 0.0266 | 0.957 |

## Real multi-camera sample, Stage 2-3 only

Cam2BEV 1_FRLR example, one frame, four simulated semantic cameras. One frame only. Not a generalisation claim.

| Metric | Value |
|---|---:|
| free_precision_gt_ground | 0.968 |
| ground_recall_as_free | 0.760 |
| false_free_rate_gt_obstacle | 0.087 |
| occupied_precision_gt_obstacle | 0.807 |
| vehicle_gt_marked_occupied_as_dynamic | 0.245 |
| occluded recall, excluding static cells | 0.980 |
| occluded precision, excluding static cells | 0.909 |
