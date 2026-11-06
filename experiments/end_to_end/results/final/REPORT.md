# ForesightBEV Stage 9 report

Generated from stored results by `foresightbev report`. **All closed-loop numbers come from the built-in kinematic simulator with scripted, non-reactive agents, rendered through the real Cam2BEV rig calibration.** They describe these scenarios only. They are not real-world safety evidence and are not comparable to published benchmarks.

Commit `aaef1cb631` (dirty: True), Python 3.12.15, macOS-26.6-arm64-arm-64bit. Closed-loop seeds [400, 415], image scale 0.5, noise {'boundary_jitter_px': 1, 'object_dropout': 0.02, 'spurious_blob_rate': 0.1}.

Intervals are 95% percentile bootstrap over episodes.

## Closed loop

| Mode | Episodes | Collision rate | Route completion | Goal rate | Freeze rate | Mean max decel (m/s²) | Intervention frames | Emergency frames |
|---|---:|---|---|---|---|---|---:|---:|
| blind_filter | 90 | 0.067 [0.022, 0.122] | 0.799 [0.748, 0.852] | 0.500 [0.400, 0.600] | 0.133 [0.067, 0.200] | 4.14 [3.67, 4.58] | 0.439 | 0.038 |
| blind_no_filter | 90 | 0.778 [0.689, 0.856] | 0.558 [0.507, 0.610] | 0.222 [0.144, 0.311] | 0.000 [0.000, 0.000] | 0.00 [0.00, 0.00] | 0.000 | 0.000 |
| filter | 90 | 0.011 [0.000, 0.033] | 0.795 [0.747, 0.840] | 0.411 [0.311, 0.511] | 0.133 [0.067, 0.211] | 4.50 [4.01, 4.94] | 0.183 | 0.022 |
| filter_calibrated | 90 | 0.011 [0.000, 0.033] | 0.798 [0.754, 0.843] | 0.400 [0.300, 0.500] | 0.133 [0.067, 0.211] | 4.50 [4.04, 4.93] | 0.184 | 0.023 |
| filter_front_dropout | 90 | 0.011 [0.000, 0.033] | 0.670 [0.622, 0.718] | 0.156 [0.078, 0.233] | 0.144 [0.078, 0.222] | 4.65 [4.33, 4.99] | 0.293 | 0.023 |
| filter_oracle_tracks | 90 | 0.000 [0.000, 0.000] | 0.906 [0.880, 0.931] | 0.489 [0.389, 0.589] | 0.011 [0.000, 0.033] | 3.98 [3.53, 4.42] | 0.182 | 0.008 |
| filter_unknown_as_free | 90 | 0.011 [0.000, 0.033] | 0.819 [0.773, 0.863] | 0.478 [0.367, 0.589] | 0.122 [0.056, 0.189] | 4.65 [4.18, 5.09] | 0.184 | 0.023 |
| idm_no_filter | 90 | 0.233 [0.156, 0.322] | 0.714 [0.655, 0.774] | 0.344 [0.256, 0.444] | 0.144 [0.078, 0.222] | 4.84 [4.34, 5.28] | 0.000 | 0.000 |
| idm_no_filter_front_dropout | 90 | 0.311 [0.211, 0.411] | 0.622 [0.566, 0.680] | 0.211 [0.133, 0.300] | 0.256 [0.167, 0.344] | 4.00 [3.40, 4.60] | 0.000 | 0.000 |

## Collision rate by scenario

| Mode | cut_in | dense_traffic | free_road | lead_brake | occluded_pedestrian | static_obstacle |
|---|---:|---:|---:|---:|---:|---:|
| blind_filter | 0.00 | 0.00 | 0.00 | 0.27 | 0.07 | 0.07 |
| blind_no_filter | 1.00 | 1.00 | 0.00 | 1.00 | 0.67 | 1.00 |
| filter | 0.00 | 0.00 | 0.00 | 0.00 | 0.07 | 0.00 |
| filter_calibrated | 0.00 | 0.00 | 0.00 | 0.00 | 0.07 | 0.00 |
| filter_front_dropout | 0.00 | 0.00 | 0.00 | 0.00 | 0.07 | 0.00 |
| filter_oracle_tracks | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| filter_unknown_as_free | 0.00 | 0.00 | 0.00 | 0.07 | 0.00 | 0.00 |
| idm_no_filter | 0.47 | 0.07 | 0.00 | 0.00 | 0.80 | 0.07 |
| idm_no_filter_front_dropout | 0.53 | 0.07 | 0.00 | 0.20 | 0.80 | 0.27 |

## Route completion by scenario

| Mode | cut_in | dense_traffic | free_road | lead_brake | occluded_pedestrian | static_obstacle |
|---|---:|---:|---:|---:|---:|---:|
| blind_filter | 0.92 | 0.54 | 1.00 | 0.50 | 0.96 | 0.87 |
| blind_no_filter | 0.34 | 0.58 | 1.00 | 0.45 | 0.59 | 0.39 |
| filter | 0.82 | 0.58 | 1.00 | 0.53 | 0.96 | 0.88 |
| filter_calibrated | 0.82 | 0.57 | 1.00 | 0.52 | 0.96 | 0.91 |
| filter_front_dropout | 0.79 | 0.57 | 0.99 | 0.68 | 0.51 | 0.49 |
| filter_oracle_tracks | 0.86 | 0.75 | 1.00 | 0.94 | 0.90 | 0.99 |
| filter_unknown_as_free | 0.84 | 0.63 | 1.00 | 0.61 | 1.00 | 0.84 |
| idm_no_filter | 0.60 | 0.56 | 1.00 | 0.71 | 0.51 | 0.91 |
| idm_no_filter_front_dropout | 0.56 | 0.57 | 1.00 | 0.61 | 0.50 | 0.49 |

## Perception, mode `filter`

Footprints against simulator ground truth, per BEV cell.

| Metric | Value |
|---|---:|
| False-free rate (true occupied cell marked FREE) | 0.0113 |
| True occupied cell marked OCCUPIED | 0.0404 |
| True occupied cell marked UNKNOWN or OUTSIDE_FOV | 0.9482 |
| Occupied precision | 0.8664 |

Only near surfaces can be OCCUPIED, so an object interior is expected to be UNKNOWN.

## Tracking

| Metric | Value |
|---|---:|
| Recall within 40 m (3 m gate) | 0.784 |
| Mean position error (m) | 0.99 |
| Mean velocity error (m/s) | 0.87 |
| ID switches | 216 |
| False track-frames | 1057 |
| Mean NEES, 4 states (consistent filter is near 4) | 28.3 |

## Forecast

IoU of dynamic-agent footprints with true future boxes, on scenarios with moving agents.

| Horizon (s) | CV Monte Carlo IoU | Frozen tracks IoU | Brier (relevant cells) | Mean displacement error (m) |
|---:|---:|---:|---:|---:|
| 0.25 | 0.368 | 0.267 | 0.249 | 1.88 |
| 0.50 | 0.354 | 0.172 | 0.227 | 2.01 |
| 0.75 | 0.342 | 0.149 | 0.207 | 2.07 |
| 1.00 | 0.328 | 0.147 | 0.190 | 2.17 |
| 1.25 | 0.314 | 0.149 | 0.175 | 2.27 |
| 1.50 | 0.301 | 0.155 | 0.162 | 2.34 |
| 1.75 | 0.286 | 0.161 | 0.150 | 2.39 |
| 2.00 | 0.265 | 0.161 | 0.140 | 2.42 |
| 2.25 | 0.245 | 0.161 | 0.131 | 2.52 |
| 2.50 | 0.231 | 0.162 | 0.123 | 2.60 |
| 2.75 | 0.222 | 0.165 | 0.115 | 2.70 |
| 3.00 | 0.220 | 0.169 | 0.108 | 2.75 |

## Trajectory-event calibration

Event: inflated ego footprint overlaps a true agent within 3 s, excluding overlap at t=0. Rows within an episode are correlated, so effective sample size is below test_rows.

Fit rows 51350, test rows 25740, test base rate 0.096. Seeds fit [200, 220], test [230, 240].

| Score | Brier | NLL | ECE (10 equal-width bins) | ECE 95% interval (episode bootstrap) | AUROC |
|---|---:|---:|---:|---|---:|
| raw | 0.0774 | 0.5787 | 0.0784 | not computed | 0.917 |
| calibrated | 0.0491 | 0.1665 | 0.0041 | not computed | 0.917 |

## Trained forecaster against baselines

All three models scored on identical held-out simulator frames at 1 m resolution, with Brier over one shared cell region. Synthetic data only. Train rows 5581, test rows 1869, 269430 parameters.

IoU at p>=0.5 by horizon:

| Horizon (s) | learned_unet | constant_velocity_monte_carlo | frozen_tracks |
|---:|---:|---:|---:|
| 0.5 | 0.756 | 0.461 | 0.262 |
| 1.0 | 0.723 | 0.433 | 0.226 |
| 1.5 | 0.685 | 0.398 | 0.234 |
| 2.0 | 0.658 | 0.359 | 0.248 |
| 2.5 | 0.644 | 0.330 | 0.252 |
| 3.0 | 0.619 | 0.327 | 0.247 |

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
