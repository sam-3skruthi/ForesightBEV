# End-to-end data flow

Contracts and frame conventions: `src/foresightbev/contracts.py`. Ego frame is x forward, y left, z up.
`T_a_b` maps points from frame b into frame a. BEV arrays are `[ix, iy]`, voxels `[iz, ix, iy]`.

```text
Cam2BEV real frame ─ data/cam2bev_sample.py ┐
Built-in simulator ─ sim/backend.py         ├─► FrameBundle (labels, calibration, ego metadata)
CARLA (not run)    ─ integration/carla_backend.py ┘        │
                                                           ▼
 Stage 2-3  perception/occupancy.py     BEVFrame, OccupancyFrame (5 cell states, 3D voxels)
 Stage 4    perception/detection.py     Detection (with range-dependent covariance)
            tracking/tracker.py         TrackState (Kalman, world frame)
 Stage 5    forecast/forecaster.py      ForecastVolume (joint Monte Carlo samples, forward flow)
            forecast/learned.py         Trained U-Net, per-cell marginals, evaluated separately
 Stage 6    risk/trajectories.py        TrajectoryCandidate lattice, IDM nominal plan
            risk/evaluator.py           TrajectoryRisk (event score, unknown policy)
            risk/calibration.py         Isotonic calibrator, fitted on held-out simulator seeds
            risk/safety_filter.py       SafetyDecision with reason codes
 Stage 7    sim/closed_loop.py          Observe, decide, apply
            integration/ros_*.py        Same loop across two processes over ROS 2 DDS
 Stage 8    recording.py ─► experiments/end_to_end/runs/ ─► api/app.py ─► frontend/
 Stage 9    evaluation/ ─► experiments/end_to_end/results/<name>/REPORT.md
```

`pipeline.py` orchestrates Stages 2–6 for one `FrameBundle`. It receives only images, calibration, ego
metadata and a declared route. Simulator truth reaches it only through the explicitly named oracle
arguments used by the `filter_oracle_tracks` ablation. Future agent states exist only in offline labels.

## ROS 2 topics

Defined in `integration/ros_messages.py`. All under `/foresight`.

| Topic | Type | Direction |
|---|---|---|
| `cameras/<name>/labels` | `sensor_msgs/Image` mono8; height 0 means dropped | sim to pipeline |
| `ego` | `nav_msgs/Odometry` | sim to pipeline |
| `route` | `std_msgs/String` JSON | sim to pipeline |
| `occupancy` | `nav_msgs/OccupancyGrid` (lossy, -1 for all non-free, non-occupied) | pipeline out |
| `occupancy_state` | `sensor_msgs/Image` mono8, exact `CellState` | pipeline out |
| `decision` | `std_msgs/String` JSON with reason codes | pipeline out |
| `trajectory` | `nav_msgs/Path` | pipeline out |
| `trajectory_timed` | `std_msgs/String` JSON with times and speeds | pipeline to sim |

## Autoware compatibility (investigated, not integrated, not run)

Autoware targets Ubuntu with ROS 2 and was not available here. A future bridge would need these
conversions. They are design notes from documented message families, not tested interfaces, so
check names against the installed Autoware version.

| Tavyra | Likely Autoware counterpart | Gap |
|---|---|---|
| `occupancy` | Occupancy grid map input to planning | Keep `unknown` as -1; do not merge into free |
| `TrackState` | `autoware_perception_msgs/TrackedObjects` | Needs UUIDs and shape messages; our class set is smaller |
| `trajectory_timed` | `autoware_planning_msgs/Trajectory` | Needs per-point velocity and acceleration fields |
| Safety decision | Would act as a validator before control | Needs an adapter node and a defined override interface |
