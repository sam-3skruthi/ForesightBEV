# Implementation status

Updated 2026-11-05. Local only: nothing is committed or published. Evidence paths are under `experiments/end_to_end/results/`.

**Summary.** Tavyra runs end to end on real camera footage, from six cameras to a traceable safety decision, and in
closed loop in simulation. ROS 2 output is bit-identical to in-process runs, and a nine-workspace application shows every
stored result from the typed API. This is a working research prototype. Detection, tracking and false-free space improved
sharply on two preliminary validation scenes. The safety filter is not validated on real driving: real logs contain almost
no conflicts, and most interventions there had no annotated conflict behind them.

## Stage status

| Stage | Implemented | Tested | Integrated | Partial | Deferred |
|---|---|---|---|---|---|
| 0 Literature, REA, gap | Prior audits and synthesis kept. No new audits. | n/a | Informs design | Prior-art comparison open | REA 05, OccDriver, SURE-Plan, Drive-OccWorld comparison |
| 1 Data, cameras, calibration | nuScenes loader, Cam2BEV loader, uploads, label sequences, sync, validation, pretrained segmentation | Yes, including real data | Yes | Sync utility is tested but unused: keyframes are already grouped by the dataset | Sweeps at 12 Hz, full nuScenes |
| 2 Geometry, BEV | Projection, flat-ground IPM, grid lookups | Yes; exact match to official Cam2BEV output | Yes | Flat ground fails on real roads, see below | Learned lifting, ground estimation |
| 3 Occupancy, reliability | 3D semantic space carving; free, occupied, unknown, outside-FOV, invalid; heuristic camera health | Yes | Yes | Observational only. Real false-free rate is high. | Learned occupancy, OOD, epistemic uncertainty |
| 4 Detection, tracking | StreamPETR through a config switch, with BEV-cluster and apparent-size baselines. Kalman tracker. | Yes | Yes, through cached detections | Preliminary real validation on two scenes only. Baselines stay weak. | Larger validation, night, using model velocity in the tracker |
| 5 Forecasting | Monte Carlo constant velocity in the risk chain. Trained U-Net, evaluated separately. | Yes | Constant velocity: yes. U-Net: no. | Constant velocity loses to frozen tracks on real data. | Learned forecast with joint samples, real training data |
| 6 Risk, safety | Candidates, swept events, unknown policy, calibrator, filter, fallback, reason codes, API evaluation | Yes | Yes | Too few real conflicts to validate. Calibration is simulator-only. | Real-data calibration, formal analysis |
| 7 CARLA, ROS 2, Autoware | Built-in simulator. ROS 2 nodes, replay, watchdog, self-test. CARLA adapter. | ROS 2: yes, live. CARLA: fake module only. | ROS 2: yes | CARLA never executed | CARLA run, Autoware integration |
| 8 FastAPI, React | Typed API v1, jobs, ingestion, logs, diagnostics. Nine-workspace React application, API-backed, inspected in a browser. | API: yes, with content checks | Yes | Dashboard does not use `/api/v1` | Frontend redesign, after your review |
| 9 Evaluation | Ablations, calibration, tracking, forecast and occupancy metrics, nuScenes scoring, leak guards, reports, records | Yes | Yes | No clustered interval in stored `final` calibration | Statistical power beyond these samples |

## What is what

| Component | Kind |
|---|---|
| SegFormer-B2 | Pretrained on Cityscapes, not retrained, cross-dataset here |
| U-Net forecaster | Trained, simulator data only, not in the risk chain |
| StreamPETR R50 428q | Pretrained on nuScenes train and nuImages, not retrained, PyTorch inference port. Weights licence unverified, treated as non-commercial. |
| Occupancy, projection, tracker, constant-velocity forecast, candidate lattice | Classical |
| BEV-cluster and apparent-size detectors (baselines), camera health, safety filter rules, stationary test | Heuristic |
| Risk calibrator | Fitted on simulator data, valid for those scenarios only |
| Built-in simulator | Simulated, scripted, agents never react |
| CARLA adapter | Unexecuted |
| Autoware | Not integrated; mapping note only |
| Occ3D labels, LiDAR | Not present in use |

## Results: real footage (nuScenes v1.0-mini)

Eight held-out scenes, 325 frames, config frozen on two development scenes. Annotations used for scoring only. Matching
within 2 m, range 40 m, visibility at least 40%. Ten scenes verify a pipeline; they do not support generalisation claims.

| Metric | Value |
|---|---:|
| Vehicle recall / precision | 0.214 / 0.210 (3,711 annotated) |
| Person recall / precision | 0.251 / 0.361 (1,710 annotated) |
| Detection position error when matched | 0.93 m at 0–15 m, 1.17 m at 15–30 m |
| Track recall within 2 m | 0.163 |
| ID switches, false track-frames | 151, 1,654 |
| Mean track position, velocity error | 0.96 m, 2.46 m/s |
| Mean NEES (consistent is near 4) | 34.4 |
| Annotated box cells marked FREE | 0.361 |
| Forecast error at 1 s / 3 s, constant velocity | 2.67 m / 6.76 m |
| Forecast error at 1 s / 3 s, frozen tracks | 2.27 m / 5.02 m |
| Decisions: pass / modify / fallback | 113 / 76 / 136 |
| Alarm rate on the human driver's path (276 frames, 0 true contacts) | 0.529 |
| Mean pipeline time, single process | 130–500 ms per frame |

Range varies by scene: alarm rate 0.00 (0553, 1100) to 1.00 (1077), and box cells marked FREE 0.22 (0553) to 0.63 (1100).
Night scenes 1077, 1094 and 1100 show the highest false-free rates, as expected from a daytime-trained model.

Main cause: flat-ground range. Annotations show ground height spreading about ±0.5 m around the assumed plane. At a
1.5 m camera height, that moves a contact at 30 m by about 10 m.

## Results: pretrained detector on real footage (mini_val)

StreamPETR R50 428q, pretrained, inference only, against the apparent-size baseline. Same scenes, same labels, same
pipeline configuration, same safety thresholds, same scorer.

**Status of this data.** Official `mini_val` (scene-0103, scene-0916): 81 frames, 2,459 annotated objects within 40 m. Both are
previously seen here: 0103 was a tuning scene, 0916 was in the reported held-out set. Two scenes are preliminary
validation only. Both are daytime. Nothing here supports a claim about generalisation.

| Metric, 2 m match, score cutoff 0.3 | Baseline | StreamPETR |
|---|---:|---:|
| Vehicle recall / precision (1,528 annotated) | 0.205 / 0.311 | **0.902 / 0.806** |
| Person recall / precision (931 annotated) | 0.292 / 0.459 | **0.831 / 0.734** |
| Detection error, 0–15 m | 0.91 m (316) | **0.39 m** (739) |
| Detection error, 15–30 m | 1.03 m (240) | **0.57 m** (1,118) |
| Detection error, 30–40 m | 1.19 m (30) | **0.70 m** (296) |
| Track recall | 0.193 | **0.797** |
| ID switches per 100 matched | 18.1 (86) | **9.8** (192) |
| False track-frames | 559 | 419 |
| Track position / velocity error | 0.94 m / 2.16 m/s | **0.53 m / 0.50 m/s** |
| Mean NEES (consistent is near 4) | 22.2 | **3.3** |
| Forecast error at 1 s / 3 s, constant velocity | 2.12 / 4.99 m | **0.84 / 1.86 m** |
| Forecast error at 1 s / 3 s, frozen tracks | 1.43 / 2.91 m | 0.93 / 1.88 m |
| Forecast pairs at 1 s / 3 s | 527 / 436 | 1,835 / 1,470 |
| Annotated box cells marked FREE | 0.303 | 0.303 (unchanged, as occupancy is unchanged) |
| Decisions: pass / modify / fallback | 17 / 9 / 55 | 69 / 5 / 7 |
| Alarms on the human path (68 frames, 0 true contacts) | 52 (0.765) | 6 (0.088) |
| Pipeline, cached detections | 269 ms | 382 ms |
| Detector inference on M5 | Part of pipeline time | **288–292 ms per frame**, p95 about 320 ms, 1.3 GB MPS, 1.1 GB peak RSS |

Counts in brackets are matched objects or raw switches. Live latency with StreamPETR adds its inference time to segmentation
and pipeline time. Forecast pairs differ because each detector matches different objects, so errors are not over identical sets.

| Class at cutoff 0.3 | Annotated | Recall | Precision |
|---|---:|---:|---:|
| Car | 1,237 | 0.926 | 0.814 |
| Truck | 70 | 0.971 | 0.958 |
| Bus | 29 | 1.000 | 0.967 |
| Pedestrian | 931 | 0.831 | 0.734 |
| Motorcycle | 163 | 0.693 | 0.796 |
| Bicycle | 29 | 0.793 | 0.383 |

Fine-class matching is same-class. No construction vehicles or trailers were annotated in range.

Score sweep over those classes, reported only and not selected from: 0.2 gives recall 0.929, precision 0.476. 0.3 gives
0.876, 0.779. 0.5 gives 0.735, 0.945. 0.7 gives 0.617, 0.980.

Scoring rule: predictions beyond 40 m are excluded, and an unmatched prediction on an annotated low-visibility object is
not a false positive. Matching runs first, so this cannot lower recall. The rule applies equally to both detectors.
Re-scoring the earlier held-out set this way keeps its recall (0.21, 0.25) and raises precision from 0.21 to 0.24 and
0.36 to 0.40. Its 325 decisions are identical, and the table above in this document keeps the original scoring.

**Read this carefully.** Detection and tracking improved substantially. That does not make the safety filter ready:

- 68 frames and zero contacts cannot establish safety. A low alarm rate on safe human driving says nothing about missed hazards.
- False-free cells are unchanged at 30%, because occupancy still rests on flat-ground geometry.
- Night is untested on this detector. All night scenes are in `mini_train`, excluded from headline evaluation.
- The closed loop has not been rerun: the simulator produces semantic images, which this RGB detector cannot read.
- Mini-scale results are not comparable to StreamPETR's published val scores: different data size, metric and scoring.

Separate real-world and closed-loop validation is still required before any claim about the filter.

## Results: final showcase configuration (mini_val)

`configs/pipeline_showcase.yaml`: StreamPETR, local ground plane from detections, model boxes fused into a labelled layer,
model velocity in the tracker. 81 frames, scene-0103 and scene-0916, previously seen, preliminary. Safety thresholds match
`pipeline_nuscenes.yaml`.

| Measure | Result |
|---|---|
| Vehicle recall, precision (2 m) | .902, .806 |
| Person recall, precision (2 m) | .831, .734 |
| Track recall, ID switches per 100 matched, velocity error | .786, 7.7, .197 m/s |
| Annotated box cells marked free (proxy) | .303 before, .066 after |
| Pipeline time, cached perception excluded | about 450 ms |
| Decisions | 73 pass, 5 modify, 3 fallback |
| Human path alarms | 1 of 68 frames, 0 true contacts |

The proxy is annotation-derived and covers dynamic-object footprints only. It is not voxel accuracy.

Occupancy corrections compared, same scenes:

| Variant | Marked free | Free m² per frame |
|---|---|---|
| Flat ground, no tolerance | .550 | 1511 |
| Previous baseline | .303 | 1240 |
| Pose tilt | .312 | 1255 |
| Height tolerance .8 | .192 | 1083 |
| Local plane | .199 | 1146 |
| Baseline with model boxes | .082 | 1188 |
| Local plane with model boxes | .066 | 1113 |

Interventions against annotations (`safety_real/mini_val.json`). A conflict means the nominal plan overlapped annotated
objects at later keyframes as logged. Replay is open loop, so other road users did not react.

| Unknown policy | Conflicts | Interventions | With conflict | Without conflict | Conflicts passed |
|---|---|---|---|---|---|
| Conservative (used) | 3 | 8 | 2 | 6 | 1 |
| Unknown as free | 3 | 6 | 2 | 4 | 1 |

Three conflicts cannot support a rate. On these scenes the conservative unknown policy added two interventions and caught
nothing extra, so its value on real data is not shown.

Forecast on real occupancy, 63 frames at 1 m cells, IoU at 1 s and 3 s: constant velocity .362, .297; frozen tracks .378,
.317; simulator-trained U-Net .154, .116. The U-Net does not transfer and stays out of the risk path. Constant velocity
beats frozen on position error (.78 and 1.66 m against .94 and 1.86 m) but not on this occupancy measure, likely because most
agents here are parked.

The tracker logic was chosen on mini_train and scored once on mini_val. Two-stage association was rejected because it
tripled false tracks.

Fault run: `scene-0103` with the front camera removed for frames 14 to 25, a declared artificial fault on real frames. It
exposed that the filter let the plan speed up during the loss, so the showcase configuration now forbids that with
`hold_speed_on_coverage_loss`. The setting came from reviewing the fault, not from any score, and the simulator
configuration keeps it off, so frozen simulator results are unchanged.

## Results: simulator (held-out seeds 400–414)

90 episodes per mode across six scripted families. 95% bootstrap intervals. Four rendered cameras, declared label noise.

| Mode | Collision rate | Route completion | Freeze rate |
|---|---|---|---|
| Blind nominal, no filter | 0.778 [0.689, 0.856] | 0.558 [0.507, 0.610] | 0.000 |
| IDM nominal, no filter | 0.233 [0.156, 0.322] | 0.714 [0.655, 0.774] | 0.144 |
| IDM with filter | 0.011 [0.000, 0.033] | 0.795 [0.747, 0.840] | 0.133 |
| With calibrated scores | 0.011 [0.000, 0.033] | 0.798 [0.754, 0.843] | 0.133 |
| Unknown treated as free | 0.011 [0.000, 0.033] | 0.819 [0.773, 0.863] | 0.122 |
| Oracle tracks (upper bound) | 0.000 [0.000, 0.000] | 0.906 [0.880, 0.931] | 0.011 |
| Filter, front camera lost 1.5 s | 0.011 [0.000, 0.033] | 0.670 [0.622, 0.718] | 0.144 |
| No filter, front camera lost 1.5 s | 0.311 [0.211, 0.411] | 0.622 [0.566, 0.680] | 0.256 |

- **The unknown-space policy has no measured benefit here.** Collisions are equal, and it costs about 0.02 route completion. Hypothesis H1's safety value is not shown.
- **Calibration improves probabilities but not decisions.** ECE fell from 0.078 to 0.004 on held-out seeds, AUROC 0.917 either way. Closed-loop outcomes are unchanged.
- **Perception costs route completion.** Oracle tracks reach 0.906 against 0.795.
- **Tracker covariance is inconsistent** in simulation too: NEES 28.3.
- Trained U-Net against baselines on held-out seeds, agent IoU at 1 s / 3 s: 0.723 / 0.619, constant velocity 0.433 / 0.327, frozen tracks 0.226 / 0.247. Brier on a shared region at 3 s: 0.047, 0.082, 0.191.

## Verification on frozen code

Latest complete run, final showcase build: 29 of 29 steps passed, 352 tests. It adds showcase acceptance on scene-0916, ROS 2
replay with the showcase configuration, a stalled-camera replay, and the real-log safety study. Every workspace was also
checked in a browser across three runs, with screenshots in `docs/project/screenshots/`. The table below is from the earlier
25-step run.

`scripts/verify_backend.sh`. Log and summary: `results/verification/`. Figures for the final run are in that summary.

| Check | Result |
|---|---|
| Automated tests | 317 passed, 0 failed, 0 skipped |
| nuScenes validation | Valid, zero problems. Archive and all 13 table hashes unchanged. |
| ROS 2 message self-test | 6 of 6 conventions |
| Acceptance: Cam2BEV sample, nuScenes scene-0553, night scene-1100 | Passed |
| Acceptance: StreamPETR on scene-0103 (40 frames, 6 cameras) | Passed: ROS identical, served output identical |
| StreamPETR port verification with controls | Passed |
| ROS 2 closed loop, three simulator scenarios | Decisions and occupancy hashes identical to in-process |
| Perception watchdog over DDS | 5 of 5, after 12 of 12 and 10 of 10 in earlier runs |
| Stored simulator results reproduce | 4 of 4 sampled episodes, field by field |
| nuScenes held-out decisions reproduce | All 325 identical |
| Cold start and clean shutdown | Passed in each acceptance run |
| Frontend type check and build | Passed |

What acceptance proves: real inputs reach calibration, BEV, occupancy, detection, tracking, forecast, risk and decision;
ROS 2 and the backend carry the same bytes; results are stored. It does not prove those outputs are accurate.
Accuracy is reported above. Without the pretrained detector it is low on real footage.

## Failures found and fixed during verification

| Failure | Cause | Fix |
|---|---|---|
| Real-sample unknown fraction 72% | Contacts beyond the grid were treated as blind | Corrected azimuth logic |
| Staleness and missing-perception unreachable in the pipeline | Age was always zero; no frame meant no decision | `now` argument, `decide_without_perception` |
| Hard-coded provenance | One static table for all runs | Built per run |
| ROS decisions differed from in-process on real footage | Detection followed message arrival order | Calibration order everywhere |
| ROS run hung indefinitely | End signal lost at exit; unbounded wait | Repeated signal, idle exit, bounded wait |
| Watchdog failed 2 of 5 trials | DDS discovery race lost first messages; no fallback without ego state | Readiness handshake, stop without ego. Then 12 of 12. |
| Stale result could mask a failed trial | Old output file was read | Removed before each trial |
| Brier not comparable between forecasters | Each model used its own region | Shared region |
| Reproducibility check compared scorer output, not pipeline output | It failed after an intended scoring change | Now compares decisions |
| Low-visibility scoring lowered recall | Predictions were removed before matching | Match first, then excuse only unmatched predictions |
| Provenance said detection was heuristic for StreamPETR runs | Fixed table entry | Named per detector |
| Near-field unknown vetoed every real frame | Blind zone sized for the Cam2BEV rig | nuScenes configuration |
| Real range errors, phantom tracks | Flat-ground range; overestimated noise | Apparent-size detector, ground tolerance, measured noise |
