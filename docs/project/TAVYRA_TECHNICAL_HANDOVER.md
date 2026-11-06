# Tavyra technical handover

Tavyra is a camera-centric research prototype. It takes six recorded camera views, builds a bird's-eye picture of what is
free, occupied or unknown, finds and tracks road users, predicts where they may go, scores candidate plans, and explains a
safety decision. This document describes the code as it exists. Python imports stay under the package name `foresightbev`,
the name of the repository and workspace, so nothing breaks; Tavyra is the project name.

## 1. Data flow for one frame

```text
Six recorded camera images + calibration + recorded ego pose
  ├─ SegFormer-B2 labels per pixel (cached in the sequence)
  └─ StreamPETR 3D boxes with velocity (cached per scene)
        │
Local ground plane fit from detected box bottoms
        │
Grid lookups → semantic space carving → BEV and 3D occupancy
        │
Model boxes fused into a labelled occupied layer
        │
Kalman tracker in the world frame
        │
Monte Carlo constant-velocity forecast, 3 s ahead, 32 joint samples
        │
Candidate plans → swept event scores, unknown-space checks, clearance, TTC
        │
Safety filter → pass, modify or fallback, with reason codes
        │
Stored frame record → FastAPI → React application
```

Code: `pipeline.py:ForesightPipeline.step`. Stored runs replay cached segmentation and detections. The decision never
changes the next frame, because the car in the log already drove. This is open-loop replay.

## 2. Coordinate frames and units

| Item | Convention |
|---|---|
| Ego frame | x forward, y left, z up, metres, origin at the rear-axle ego origin of nuScenes |
| World frame | nuScenes global frame; tracks live here so ego motion does not look like agent motion |
| Camera frame | x right, y down, z forward |
| Grid | x −30 to 50 m, y −22 to 22 m, 0.25 m cells, z 0 to 3 m in 0.5 m layers |
| Array order | BEV `[ix, iy]`, voxels `[iz, ix, iy]` |
| BEV images | Row 0 is x max, column 0 is y max, so forward is up and left is left |
| Units | Metres, seconds, m/s, radians |
| Keyframes | 2 Hz. Cameras in one keyframe differ by up to about 36 ms; the limit is 100 ms in the showcase configuration |

StreamPETR works in the `LIDAR_TOP` frame, where x is right and y is forward. Only that sensor's fixed calibration is
used, as a coordinate transform. No LiDAR point is read.

## 3. Stage by stage

### Stage 0: literature and gap
Audits and synthesis live in `research/`. The candidate question was whether uncertainty in future camera-derived occupancy
can become calibrated trajectory-event probabilities. It remains a candidate. Direct comparison with OccDriver, SURE-Plan,
Drive-OccWorld and PORA is unfinished, so no novelty claim is established.

### Stage 1: data, cameras, calibration
`data/nuscenes.py` reads the official tables. It checks camera calibration and timing, and refuses malformed input.
Processed images are 800×450, with intrinsics scaled to match. Annotations load through a separate class and are used only
for scoring and display. Camera health is the fraction of usable pixel labels. It is a heuristic, not an OOD score.

Ego speed is the distance between consecutive camera-row poses divided by their time gap, so it is causal. The first frame
has no previous pose, so speed is recorded as 0 and the application shows it as unavailable.

**Important boundary:** `ego_pose` is an input to tracking, geometry and StreamPETR's temporal memory. nuScenes produced
those poses by its own localisation, which uses LiDAR with a map. Tavyra reads no LiDAR, but its ego motion is not
camera-derived. Say "camera-centric perception using recorded ego pose", not "camera-only end to end".

### Stage 2: geometry and BEV
A camera has intrinsics K and an extrinsic pose (R, t) from camera to ego. For an ego point p:

```text
X_cam = Rᵀ (p − t)
u = fx · X/Z + cx
v = fy · Y/Z + cy      valid only if Z > 0 and (u, v) lies inside the image
```

`geometry/projection.py:build_lookup` precomputes this for every ground cell and voxel. IPM here means asking, for each grid
cell, which pixel sees it, assuming that cell lies on the ground. Flat ground uses z = 0. The local plane uses
z = a + b·x + c·y. Anything above the ground smears away from the camera, which is why a parked car becomes a long streak in
the combined view.

Validation: the projection matches the official Cam2BEV output exactly on its sample.

### Stage 3: occupancy
`perception/occupancy.py` performs semantic space carving with no learned occupancy model:

1. Project ground cells into the labels. Along each azimuth, the nearest ground cell with an obstacle label gives contact range d.
2. For each voxel, read the label along its ray. Road or sky means the ray passed through, so the voxel is free. An obstacle label means free before d, occupied in a 0.6 m band at d, and unknown behind.
3. Fuse cameras: any free evidence wins, then occupied, then unknown. Cells no healthy camera sees are OUTSIDE_FOV.
4. Collapse columns up to 2 m into a BEV state.

| State | Meaning |
|---|---|
| FREE | Positive visual evidence of empty space |
| OCCUPIED | Visual or model evidence of an object |
| UNKNOWN | A camera looks there but sees no evidence, such as behind an object |
| OUTSIDE_FOV | No healthy camera covers it. Shown as no coverage |
| INVALID | The ego body |

Missing observation never becomes free. A lost camera removes its evidence, so its cone falls to unknown or no coverage,
unless another camera has evidence there.

**Depth source.** Contact range comes from the flat-ground intersection, not a depth network. If the real contact point is
higher by Δz, range is wrong, and a camera at height h sees it at d·h/(h+Δz). The height tolerance marks that span unknown.
This explains the false-free problem: ground errors move the contact outward, and "free wins" then keeps wrong free cells.

**False-free work on mini_val.** The figure is the share of annotated dynamic-box cells marked free. It is a proxy, because
nuScenes mini has no occupancy labels. Flat ground gave .550, the earlier tolerance gave .303, and the final method gives
.066 by fitting a plane to detected objects and fusing model boxes into a separate `model_occupied` layer. Free area fell
from 1240 to 1113 m² per frame.

### Stage 4: detection and tracking

**StreamPETR.** A ResNet-50 and FPN extract image features. 428 queries, 300 learned plus 128 propagated from earlier frames
in a memory queue, attend to camera features with 3D position embeddings built from intrinsics and extrinsics. Depth is
learned from image evidence and temporal context. The output is 3D boxes, scores, velocity and ten classes. Tavyra maps car,
truck, construction vehicle, bus, trailer, motorcycle and bicycle to VEHICLE, and pedestrian to PERSON. Barrier and traffic
cone are not mapped. The score cutoff is 0.3, fixed before evaluation. The port in `perception/streampetr.py` is plain
PyTorch, loading 573 needed tensors strictly under `weights_only=True`. The 16 2D-head tensors plus 2 loss-weight tensors of
591 are unused. Scores are not probabilities.

**Baselines.** `image_size` ranges image blobs by apparent size. `bev_clusters` clusters dynamic cells. Both are heuristics
and stay selectable.

**Local ground plane.** `perception/ground.py` takes bottoms of confident car, truck, bus and pedestrian boxes and fits
z = a + b·x + c·y with ridge regression and three rounds of outlier rejection. Slope is bounded to ±6%. With fewer than 4
samples it stays flat. It is a coarse tilt and offset, not a road surface.

**Tracker.** `tracking/tracker.py` is a constant-velocity Kalman filter with state [x, y, vx, vy] in the world frame:

```text
x' = F x            P' = F P Fᵀ + Q
F = [I, dt·I; 0, I]   Q from white acceleration (1.2 m/s² vehicles, 3.0 m/s² people)
S = G P Gᵀ + N      K = P Gᵀ S⁻¹
x = x + K (y − G x)    P = (I − K G) P (I − K G)ᵀ + K N Kᵀ
```

The showcase measures position (σ 0.7 m vehicles, 0.35 m people) and model velocity (σ 0.7 m/s). Association uses the
Hungarian algorithm on squared Mahalanobis distance within a χ² gate of 9.21 (2 DoF, 99%), same class only. Tracks confirm
after 3 hits for vehicles and 2 for people, and are removed after more than 2 misses, about 1 s at 2 Hz.

NEES (normalised estimation error squared) tests filter honesty. For 4 states a consistent filter averages about 4; mini_val gave 3.4.

### Stage 5: forecasting
`forecast/forecaster.py` draws 32 samples per agent from the Kalman posterior and rolls each out:

```text
p(t) = p₀ + v₀ t + ½ a t²      a ~ N(0, 0.3² m²/s⁴), fixed per sample
```

Horizons run every 0.25 s to 3 s. Agents judged stationary stay still. Static occupied cells persist. Keeping joint samples
lets Stage 6 ask whether one sampled future hits a plan, instead of assuming voxels are independent. This is tracking
extrapolation, **not a trained 4D occupancy network**. Its spread is uncalibrated model spread.

A U-Net in `forecast/learned.py` was trained on simulator data only. On simulator data its IoU is .619 to .756. On real
occupancy it gives .154 at 1 s and .116 at 3 s, so it does not transfer, and it stays out of the risk path.

### Stage 6: decision and safety
**Candidates** (`risk/trajectories.py`): a nominal plan from IDM car-following, five speed targets (100, 70, 40, 20 and 0% of
desired speed) in each lane, and comfort (2.5 m/s²) and emergency (6 m/s²) stops. On real logs there is no map: the route is
"keep current heading and curvature at max(current speed, 3 m/s)", built from present state only.

**Event score** (`risk/evaluator.py`). For each horizon, sweep the ego footprint with a 0.3 m margin and test overlap:

```text
hit(s, k) = ego footprint at horizon k overlaps sample s
score = (1/S) Σ_s max_k hit(s, k)
```

If the swept path meets static occupied cells, the score becomes 1 from that time on. The score is a model frequency, **not
a collision probability**.

**Unknown corridor.** Find how far along the plan unknown cells appear. The inspected width grows with phantom speed 1.5 m/s
times arrival time, limited to 0.3–2.5 m, to allow for something emerging. The plan fails if, within the 0.4 s replan
interval, travelled distance plus stopping distance exceeds that:

```text
stopping_distance(v) = v·0.2 s + v²/(2·4 m/s²)
```

**Clearance and TTC** use mean constant-velocity agent paths. TTC here is the first horizon at which that approximate gap
reaches zero.

**Safety filter** (`risk/safety_filter.py`):
1. Perception missing, stale (over 0.75 s), or no healthy camera: fallback.
2. Nominal admissible (no static collision, score ≤ 0.10, no unknown violation): pass.
3. Otherwise pick the lowest-cost admissible alternative: modify. Cost favours speed, staying in lane and low risk.
4. If none exists: fallback. Comfort stop, or emergency stop if comfort stop itself is unsafe.

After a fallback it forbids acceleration for 1 s. In the showcase configuration, losing a camera also forbids speeding up.
The filter gives a decision relative to its own model. It does not guarantee collision freedom.

### Stage 7: simulation and ROS 2
The built-in kinematic simulator runs scripted, non-reactive scenarios in closed loop. ROS 2 Jazzy (RoboStack) proves that
the pipeline works as separate processes over messages. Replay and simulator nodes publish camera labels, odometry and
route; the pipeline node waits on `/foresight/pipeline_ready`, assembles a frame, and publishes occupancy, tracks, forecast,
risk, path and decision. A watchdog handles incomplete frames, and `verify_ros` compares output with in-process runs,
including occupancy hashes. Topic names keep the `/foresight/` prefix.

The CARLA adapter exists and has only run against a fake module. **CARLA and Autoware have never executed.**

### Stage 8: backend and application
FastAPI serves typed routes under `/api/v1`, with structured errors and request IDs, and binds to 127.0.0.1. The React
application reads those routes and has no fixed metrics. Selection (run and frame) lives in the URL. The shown frame, its
data and its images always come from one record, so stale values cannot pair with a new frame. It only observes stored
results, and it is outside any decision path.

### Stage 9: evaluation

| Split | Scenes | Use |
|---|---|---|
| dev | 0061, 0103 | Classical tuning |
| held_out | 0553, 0655, 0757, 0796, 0916, 1077, 1094, 1100 | Classical pipeline, scored once |
| mini_val | 0103, 0916 | StreamPETR headline. Previously seen by this project, preliminary |
| mini_train | Other eight | Detector trained on these; figures inflated. Tracker variants compared here |

Matching is 2 m within 40 m. Thresholds were not tuned on mini_val. The simulator has separate seeds for tuning (100–102),
calibration, the U-Net, and final closed-loop evaluation (400–414).

## 4. Results to know

mini_val, 81 frames, preliminary:

| Measure | Apparent-size baseline | Showcase |
|---|---|---|
| Vehicle recall / precision | .21 / .31 | .902 / .806 |
| Person recall / precision | .29 / .46 | .831 / .734 |
| Track recall | .19 | .786 |
| Track velocity error | 2.16 m/s | .197 m/s |
| Mean NEES | 22.2 | 3.4 |
| Box cells marked free | .303 | .066 |
| Decisions, pass/modify/fallback | 17/9/55 | 73/5/3 |
| Human-path alarms of 68 | 52 | 1 |

Forecast position error at 1 s and 3 s: constant velocity .78 and 1.66 m, frozen .94 and 1.86 m.

Interventions against annotations: 3 nominal-plan conflicts, 8 interventions, 2 with a conflict, 6 without, 1 conflict passed.
Treating unknown as free gave 6 interventions and the same conflict counts.

Simulator only, 90 episodes per mode: filter collision rate 1.1%, IDM 23.3%; route completion .795 against .714. Unknown-as-free
gave the same collision rate, so the unknown policy shows no measured benefit there either.

## 5. Why these choices

| Choice | Reason |
|---|---|
| Space carving | Inspectable, needs no labels or GPU, and separates free from unknown explicitly |
| StreamPETR R50 | Published temporal camera detector that runs here on MPS. BEVFormer was also assessed |
| Kalman with Hungarian | Understandable, consistent and testable by NEES |
| Monte Carlo constant velocity | Strong short-term baseline with joint samples the risk layer needs |
| Filter around an IDM plan | Separates proposal from check, and gives traceable reasons |
| ROS 2 | Realistic process and message boundaries with a deterministic comparison |
| FastAPI and React | Typed contract and inspection, outside the decision path |

## 6. Established versus new

**Established:** IPM, space carving, SegFormer, StreamPETR, Kalman filtering, Hungarian assignment, IDM, swept-path checks,
isotonic calibration.

**Project work:** integrating these into one causal camera-centric chain; keeping observed-free, unknown and no-coverage
apart all the way to trajectory checks; carrying joint forecast samples into an explainable filter; an open test of
interventions against annotations. This is an engineering and evaluation contribution. It is not a proven scientific novelty.

## 7. Demonstrated, simulated, future

| Status | Items |
|---|---|
| Demonstrated on recorded real data | Cameras to decision in open-loop replay, ten nuScenes mini scenes, declared camera fault |
| Simulated only | Closed-loop driving, collision rates, risk calibration |
| Not done | CARLA, Autoware, live cameras, vehicle control, Occ3D evaluation, learned 4D occupancy, real risk calibration, OOD uncertainty, full nuScenes validation |
