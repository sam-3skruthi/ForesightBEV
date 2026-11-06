# Backend architecture

```text
configs/*.yaml ─► config.py (strict, validated)
                     │
research/datasets ─► data/nuscenes.py ──► perception/segmentation.py ─┐
Cam2BEV sample ───► data/cam2bev_sample.py ──────────────────────────┼─► data/sequence.py (label sequence on disk)
Uploads (API) ───────────────────────────────────────────────────────┘           │
Simulator ──► sim/backend.py ───────────────────────────────┐                    │
                                                            ▼                    ▼
                                        pipeline.py  ◄───────────  replay.py, ros_replay_node
                      validate ─ occupancy ─ detection ─ tracking ─ forecast ─ candidates ─ risk ─ filter
                                                            │
             ┌──────────────────┬───────────────────────────┼──────────────────────┐
             ▼                  ▼                           ▼                      ▼
      recording.py        integration/ros_*         sim/closed_loop.py      evaluation/*
    (run directories)    (DDS topics, watchdog)                           (results, reports)
             │
             ▼
      api/app.py, api/v1.py ◄── api/store.py, api/jobs.py, api/logs.py
```

## Design rules

- **One pipeline path.** The simulator, Cam2BEV, nuScenes, uploads and ROS all reach the same `ForesightPipeline`. Nothing has a private code path.
- **One replay input.** A label sequence is the only recorded input format, so in-process and ROS runs see the same bytes.
- **Boundaries validate.** `validate_bundle` and `validate_calibration` run at the pipeline boundary. Failures raise a `ForesightError` with a stable code.
- **A lost camera is data, not an error.** Malformed input raises. A declared lost camera continues, and its coverage becomes unknown.
- **Silence is never a clear road.** No frame, a stale frame, or a rejected frame yields a logged fallback.
- **Ground truth stays out of inference.** Scorers read annotations and simulator truth after outputs exist. `FrameBundle` has no field for them.
- **Determinism.** Random draws are seeded. Cameras are always processed in calibration order, never arrival order.
- **No overwriting.** Result sets refuse to overwrite. `experiments.jsonl` is append-only. Each record stores a hash of the source tree and of the configuration.
- **The API is not in a control path.** Jobs run as subprocesses from a fixed allow-list, one at a time. A crash cannot take the server down.

## Modules

| Module | Responsibility | Kind |
|---|---|---|
| `contracts.py`, `errors.py`, `interfaces.py`, `config.py` | Types, error codes, replaceable stages, configuration | Infrastructure |
| `data/nuscenes.py`, `data/nuscenes_check.py` | Official-format loading and validation | Loader |
| `data/cam2bev_sample.py`, `data/calibration.py`, `data/taxonomy.py` | Cam2BEV input | Loader |
| `data/sequence.py`, `data/prepare.py`, `data/synchronization.py`, `data/validation.py` | Label sequences, conversion, sync, validation | Infrastructure |
| `geometry/projection.py` | Projection and precomputed camera-to-grid lookups | Classical |
| `perception/segmentation.py` | RGB to labels | Pretrained |
| `perception/occupancy.py` | IPM, semantic space carving, camera health | Classical, heuristic health |
| `perception/detection.py`, `perception/image_detection.py` | BEV-cluster and apparent-size detectors | Heuristic |
| `tracking/tracker.py` | Kalman filter, Hungarian association | Classical |
| `forecast/forecaster.py` | Monte Carlo constant velocity, joint samples | Classical |
| `forecast/learned.py` | U-Net on BEV history | Trained, synthetic only |
| `risk/*` | Candidates, event evaluation, calibrator, safety filter | Classical, heuristic filter |
| `sim/*` | Built-in simulator, renderer, scenarios, closed loop | Simulated |
| `integration/ros_*`, `integration/verify_ros.py` | ROS 2 nodes, replay, self-test, comparison | Middleware |
| `integration/carla_backend.py` | CARLA adapter | Never executed |
| `evaluation/*` | Metrics, experiments, nuScenes scoring, integrity guards, reports | Evaluation |
| `acceptance.py` | End-to-end acceptance with content checks | Verification |
| `api/*` | FastAPI service | Service |

Forecast selection: the risk chain uses the constant-velocity forecaster because Stage 6 requires joint samples and the
U-Net produces only per-cell marginals.
