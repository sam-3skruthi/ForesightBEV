# Data contracts

Source of truth: `src/foresightbev/contracts.py`. API forms: `src/foresightbev/api/schemas.py`.

## Conventions

| Item | Convention |
|---|---|
| Ego frame | x forward, y left, z up. Right-handed. Metres. Origin on the ground under the reference point. |
| World frame | Right-handed, z up. Metres. Yaw counter-clockwise about +z. |
| Camera frame | Optical: x right, y down, z forward. |
| Transforms | `T_a_b` maps points in frame b into frame a. |
| Time | Seconds. nuScenes microseconds are divided by 1e6. ROS stamps are rounded to nanoseconds. |
| BEV arrays | `[ix, iy]`, with index 0 at `x_min`, `y_min`. Cell centres are at +0.5 resolution. |
| Voxel arrays | `[iz, ix, iy]`. |
| BEV images | Row 0 is `x_max`, column 0 is `y_max`. Forward is up, left is left. |
| Flow | Forward flow: m/s, in the reference ego frame at `t_ref`. |
| Quaternions | nuScenes (w, x, y, z); ROS (x, y, z, w). Conversions are tested. |

## Cell states

`CellState` values are distinct on purpose. `FREE` needs positive evidence. `UNKNOWN` is inside camera coverage but
unobserved. `OUTSIDE_FOV` has no healthy camera. `INVALID` is excluded, such as the ego body.

## Pipeline types

| Type | Stage | Key semantics |
|---|---|---|
| `CameraCalibration`, `CalibrationBundle` | 1 | Validated on construction and by `validate_calibration`. Distortion is rejected. |
| `CameraFrame`, `FrameBundle` | 1 | Labels, optional image, per-camera timestamp and status. The only perception input. |
| `EgoState` | 1 | Declared metadata. |
| `BEVFrame` | 2 | Flat-ground labels and support. Not an occupancy claim. |
| `OccupancyFrame` | 3 | Voxel and BEV states, heuristic `reliability`. |
| `Detection` | 4 | Metric, ego frame. `support` is a heuristic, not a probability. |
| `TrackState` | 4 | World frame. Covariance is the filter's own. |
| `ForecastVolume` | 5 | Marginals plus joint `agent_samples`. |
| `TrajectoryCandidate` | 6 | Timed poses in the ego frame at `t_ref`. |
| `TrajectoryRisk` | 6 | `event_score` is a sample frequency. `calibrated_probability` exists only when a fitted calibrator was applied, and `calibration_status` names its data. |
| `SafetyDecision` | 6 | Status, selection, machine-readable `ReasonCode`s. |

## Stage interfaces

`perception/segmentation.py:Segmenter`, and `interfaces.py:Detector` and `Forecaster`, are the replacement points. A
forecaster must supply joint samples, because Stage 6 rejects independence between cells.

## Stored run

`experiments/end_to_end/runs/<id>/` holds `manifest.json` (provenance, environment, grid), `frames/` (records),
`arrays/` (occupancy, IPM), `forecast_arrays/` (bit-packed joint samples) and PNG previews.

## Label sequence

`experiments/end_to_end/sequences/<id>/` holds `sequence.json` and `labels/<frame>/<camera>.png`. See `data/sequence.py`.
Ground truth is never stored here; a `sample_token` lets scoring code find it afterwards.
