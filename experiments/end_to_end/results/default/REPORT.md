# ForesightBEV Stage 9 report

Generated from stored results by `foresightbev report`. **All closed-loop numbers come from the built-in kinematic simulator with scripted, non-reactive agents, rendered through the real Cam2BEV rig calibration.** They describe these scenarios only. They are not real-world safety evidence and are not comparable to published benchmarks.

Commit `aaef1cb631` (dirty: True), Python 3.12.15, macOS-26.6-arm64-arm-64bit. Closed-loop seeds [100, 110], image scale 0.5, noise {'boundary_jitter_px': 1, 'object_dropout': 0.02, 'spurious_blob_rate': 0.1}.

Intervals are 95% percentile bootstrap over episodes.

## Closed loop

| Mode | Episodes | Collision rate | Route completion | Goal rate | Freeze rate | Mean max decel (m/s²) | Intervention frames | Emergency frames |
|---|---:|---|---|---|---|---|---:|---:|
| blind_filter | 60 | 0.050 [0.000, 0.100] | 0.744 [0.674, 0.815] | 0.433 [0.317, 0.567] | 0.200 [0.100, 0.300] | 4.01 [3.38, 4.58] | 0.499 | 0.034 |
| blind_no_filter | 60 | 0.817 [0.717, 0.917] | 0.524 [0.466, 0.587] | 0.183 [0.083, 0.283] | 0.000 [0.000, 0.000] | 0.00 [0.00, 0.00] | 0.000 | 0.000 |
| filter | 60 | 0.000 [0.000, 0.000] | 0.805 [0.742, 0.864] | 0.483 [0.367, 0.600] | 0.150 [0.067, 0.250] | 4.37 [3.76, 4.92] | 0.170 | 0.020 |
| filter_calibrated | 60 | 0.000 [0.000, 0.000] | 0.799 [0.734, 0.860] | 0.467 [0.350, 0.600] | 0.150 [0.067, 0.250] | 4.25 [3.61, 4.81] | 0.172 | 0.022 |
| filter_front_dropout | 60 | 0.000 [0.000, 0.000] | 0.673 [0.611, 0.735] | 0.167 [0.083, 0.267] | 0.183 [0.100, 0.283] | 4.68 [4.27, 5.10] | 0.282 | 0.028 |
| filter_oracle_tracks | 60 | 0.000 [0.000, 0.000] | 0.905 [0.869, 0.936] | 0.483 [0.350, 0.617] | 0.017 [0.000, 0.050] | 4.13 [3.49, 4.70] | 0.174 | 0.008 |
| filter_unknown_as_free | 60 | 0.000 [0.000, 0.000] | 0.804 [0.742, 0.865] | 0.500 [0.367, 0.617] | 0.150 [0.067, 0.250] | 4.58 [3.96, 5.19] | 0.144 | 0.024 |
| idm_no_filter | 60 | 0.250 [0.150, 0.367] | 0.693 [0.617, 0.763] | 0.350 [0.233, 0.467] | 0.150 [0.067, 0.250] | 4.70 [4.06, 5.30] | 0.000 | 0.000 |
| idm_no_filter_front_dropout | 60 | 0.317 [0.200, 0.433] | 0.611 [0.539, 0.686] | 0.233 [0.133, 0.350] | 0.183 [0.083, 0.284] | 4.24 [3.54, 4.93] | 0.000 | 0.000 |

## Collision rate by scenario

| Mode | cut_in | dense_traffic | free_road | lead_brake | occluded_pedestrian | static_obstacle |
|---|---:|---:|---:|---:|---:|---:|
| blind_filter | 0.00 | 0.00 | 0.00 | 0.10 | 0.20 | 0.00 |
| blind_no_filter | 1.00 | 1.00 | 0.00 | 1.00 | 0.90 | 1.00 |
| filter | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| filter_calibrated | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| filter_front_dropout | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| filter_oracle_tracks | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| filter_unknown_as_free | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| idm_no_filter | 0.40 | 0.10 | 0.00 | 0.10 | 0.80 | 0.10 |
| idm_no_filter_front_dropout | 0.50 | 0.10 | 0.00 | 0.20 | 0.90 | 0.20 |

## Route completion by scenario

| Mode | cut_in | dense_traffic | free_road | lead_brake | occluded_pedestrian | static_obstacle |
|---|---:|---:|---:|---:|---:|---:|
| blind_filter | 0.90 | 0.52 | 1.00 | 0.51 | 0.86 | 0.67 |
| blind_no_filter | 0.34 | 0.56 | 1.00 | 0.42 | 0.44 | 0.38 |
| filter | 0.84 | 0.56 | 1.00 | 0.43 | 1.00 | 1.00 |
| filter_calibrated | 0.83 | 0.54 | 1.00 | 0.43 | 1.00 | 0.99 |
| filter_front_dropout | 0.77 | 0.55 | 1.00 | 0.76 | 0.37 | 0.58 |
| filter_oracle_tracks | 0.87 | 0.74 | 1.00 | 0.95 | 0.88 | 1.00 |
| filter_unknown_as_free | 0.83 | 0.56 | 1.00 | 0.43 | 1.00 | 1.00 |
| idm_no_filter | 0.62 | 0.53 | 1.00 | 0.59 | 0.49 | 0.94 |
| idm_no_filter_front_dropout | 0.57 | 0.53 | 1.00 | 0.54 | 0.44 | 0.58 |

## Perception, mode `filter`

Footprints against simulator ground truth, per BEV cell.

| Metric | Value |
|---|---:|
| False-free rate (true occupied cell marked FREE) | 0.0113 |
| True occupied cell marked OCCUPIED | 0.0403 |
| True occupied cell marked UNKNOWN or OUTSIDE_FOV | 0.9484 |
| Occupied precision | 0.8681 |

Only near surfaces can be OCCUPIED, so an object interior is expected to be UNKNOWN.

## Tracking

| Metric | Value |
|---|---:|
| Recall within 40 m (3 m gate) | 0.798 |
| Mean position error (m) | 1.00 |
| Mean velocity error (m/s) | 0.85 |
| ID switches | 131 |
| False track-frames | 672 |
| Mean NEES, 4 states (consistent filter is near 4) | 27.6 |

## Forecast

IoU of dynamic-agent footprints with true future boxes, on scenarios with moving agents.

| Horizon (s) | CV Monte Carlo IoU | Frozen tracks IoU | Brier (relevant cells) | Mean displacement error (m) |
|---:|---:|---:|---:|---:|
| 0.25 | 0.389 | 0.280 | 0.248 | 1.79 |
| 0.50 | 0.377 | 0.178 | 0.228 | 1.88 |
| 0.75 | 0.363 | 0.152 | 0.211 | 1.95 |
| 1.00 | 0.348 | 0.152 | 0.196 | 2.05 |
| 1.25 | 0.333 | 0.156 | 0.181 | 2.10 |
| 1.50 | 0.316 | 0.164 | 0.169 | 2.17 |
| 1.75 | 0.293 | 0.172 | 0.158 | 2.28 |
| 2.00 | 0.271 | 0.177 | 0.148 | 2.38 |
| 2.25 | 0.252 | 0.178 | 0.138 | 2.47 |
| 2.50 | 0.235 | 0.173 | 0.129 | 2.55 |
| 2.75 | 0.227 | 0.167 | 0.120 | 2.61 |
| 3.00 | 0.228 | 0.167 | 0.112 | 2.69 |

## Trajectory-event calibration

Event: inflated ego footprint overlaps a true agent within 3 s, excluding overlap at t=0. Rows within an episode are correlated, so effective sample size is below test_rows.

Fit rows 51350, test rows 25740, test base rate 0.096. Seeds fit [200, 220], test [230, 240].

| Score | Brier | NLL | ECE (10 equal-width bins) | AUROC |
|---|---:|---:|---:|---:|
| raw | 0.0774 | 0.5787 | 0.0784 | 0.917 |
| calibrated | 0.0491 | 0.1665 | 0.0041 | 0.917 |

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
