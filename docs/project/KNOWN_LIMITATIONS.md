# Known limitations

## What this system is

A working prototype of the full architecture. It is not a validated autonomous-driving system, and
nothing here is safety evidence for real vehicles.

## Inputs

- **Real data is ten short scenes.** nuScenes mini supports verification, not generalisation claims or training.
- **Segmentation is cross-dataset.** SegFormer was trained on daytime Cityscapes. Night scenes are visibly worse.
- **The simulator uses four cameras with declared label noise.** The rig has near-field corner blind spots.
- **Flat ground is the largest single error on real roads.** Annotations show about ±0.5 m of ground spread, which moves a contact at 30 m by about 10 m. Lens distortion is rejected, not handled.
- **Keyframes only.** Replay runs at 2 Hz. The 12 Hz sweeps are unused, which hurts velocity estimates.

## Perception

- **Occupancy is observational.** Only near surfaces become OCCUPIED, so object interiors stay UNKNOWN.
  It is not semantic scene completion and is not comparable to Occ3D or SurroundOcc metrics.
- **36% of annotated box cells are marked FREE on held-out real scenes.** This is the most dangerous measured error. Static occupancy on real data is unscored because no labels exist.
- **Heuristic detectors have about 21–25% real recall.** StreamPETR reaches about 0.83–0.90 on two daytime mini_val scenes, which is preliminary. It needs RGB, so it cannot run on the simulator's semantic images or on the Cam2BEV sample. Its weights licence is unverified. It is an inference port with no bit-level comparison to the official code.
- **No instance separation.** Vehicles closer than the 1 m merge radius become one detection.
- **Box extents are class-size priors**, because only visible faces are measured.
- **Range error grows with the square of distance**, so far detections are weak.
- **`reliability` is a heuristic** usable-label fraction. There is no OOD score or epistemic uncertainty.

## Forecasting and risk

- **The risk chain uses constant velocity.** Cut-ins and braking appear only after they begin. With the baseline detector it is worse than assuming objects do not move; with StreamPETR it is slightly better on mini_val.
- **The safety filter is unvalidated on real footage.** With the baseline it alarms on 53% of held-out human-driving frames. With StreamPETR it alarms on 9% of 68 mini_val frames, which is too small a sample and says nothing about missed hazards.
- **Tracker covariance depends on the detector:** NEES 28 in simulation and 34 with the real-data baseline, but 3.3 with StreamPETR on mini_val. Risk scores inherit this.
- **The unknown-space policy shows no measured safety benefit** in these simulator scenarios.
- **The trained U-Net is not in the risk chain.** It outputs per-cell marginals, and using those for a
  trajectory event would need an independence assumption this project rejects.
- **Calibration is synthetic.** The isotonic map holds only for these scripted scenarios and this noise.
- **The unknown policy and the phantom speed are design choices** with hand-set values. The stationary
  test and exemption for shadows behind moving tracks are also heuristics.
- **Development results overlap tuning seeds.** Only `final` (seeds 400–414) is held out. Use `default` for nothing but development.

## Showcase build

- The final configuration was developed and reported on the same two previously seen mini_val scenes. No untouched real test set exists.
- Real logs hold only three nominal-plan conflicts, so miss and false-intervention rates cannot be estimated. Six of eight interventions had no annotated conflict.
- The conservative unknown policy shows no measured benefit on real or simulated data.
- Local ground fit is a coarse tilt and offset from detected objects. It is not a road surface.
- The occupancy figure is a proxy on dynamic-object footprints. Static scenery is unscored, and marked-free area is only one side of the trade-off.
- Model boxes fused into occupancy depend on detector quality, and that detector trained on the eight mini_train scenes.
- Stored runs replay cached segmentation and detections. Live StreamPETR takes about 290 ms a frame on MPS.
- ROS 2 replay of real frames takes about 3.4 s a frame, far from real time. The cause has not been profiled.
- The segmentation network was trained on Cityscapes daytime scenes. Night runs are weaker.
- Detection at night on scenes 1077, 1094 and 1100 is inflated: the detector trained on them.

## Simulation and integration

- **Scripted agents never react to the ego**, which makes counterfactual labels well defined and
  makes interaction scenarios unrealistic.
- **The ego follows plans perfectly.** There is no controller or vehicle dynamics model.
- **Lanes and routes are declared**, not perceived. The road is straight.
- **CARLA has not been executed.** The adapter is tested only against a fake module.
- **Autoware has not been executed.** Only a mapping note exists.
- **ROS 2 runs in lockstep.** Message loss is tested for a withheld camera, not under sustained load. On this Mac, large images can take seconds under heavy CPU load, so the watchdog limit matters.
- **The backend has no authentication.** It binds to localhost.

## Safety

No formal guarantee is offered or implied. The safety filter is a deterministic rule set, and a lower
collision count here is a measured property of these scenarios only.

## Stage 0

The prior-art comparison remains open: REA 05 (Cam4DOcc) has no report, and OccDriver, SURE-Plan and
Drive-OccWorld were not compared. No novelty claim is supported.
