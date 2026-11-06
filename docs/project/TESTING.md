# Testing

Everything, on an idle machine, with a recorded log in `experiments/end_to_end/results/verification/`:

```bash
scripts/verify_backend.sh
```

Automated tests only:

```bash
.venv/bin/python -m pytest tests -q
```

Tests that need local assets skip when those assets are absent. `tests/test_showcase.py` needs stored runs from `scripts/run_showcase.sh` for its API checks.

| Layer | Where |
|---|---|
| Geometry, transforms, Cam2BEV regression | `tests/test_geometry.py` |
| Occupancy semantics | `tests/test_occupancy.py` |
| Data validation, sync, sequences, pipeline boundary | `tests/test_pipeline_boundary.py`, `test_sequence.py`, `test_data_and_cli.py` |
| nuScenes loader, data boundaries, read-only guarantee | `tests/test_nuscenes.py` (synthetic fixture in the official schema), `test_nuscenes_eval.py` |
| Segmentation and detection | `tests/test_segmentation.py`, `test_image_detection.py` |
| Tracking and forecasting, causality | `tests/test_tracking_forecast.py`, `test_learned.py` |
| Trajectories, risk, safety invariants | `tests/test_risk.py` |
| Simulator and closed loop | `tests/test_simulation.py` |
| API contract, content, failure, jobs, ingestion | `tests/test_api.py`, `test_api_v1.py` |
| Evaluation integrity | `tests/test_evaluation.py`, `test_integrity.py` |
| ROS conversions | `tests/test_ros_messages.py` |
| CARLA wiring (fake module) | `tests/test_carla_adapter.py` |

Live checks, which need the ROS 2 environment:

| Command | Proves |
|---|---|
| `scripts/run_ros2_selftest.sh` | Real message round trips and conventions |
| `scripts/run_ros2_closed_loop.sh <scenario> <seed>` | Two processes, closed loop, decision by decision against in-process |
| `scripts/run_ros2_replay.sh <sequence> <config>` | Recorded frames over DDS, occupancy hashes and decisions against in-process |
| `WATCHDOG_S=5 scripts/run_ros2_replay.sh <sequence> "" 0` | A lost camera message becomes a fallback |
| `scripts/run_pipeline.sh [scene]` | Full acceptance |

The API content tests assert equality with pipeline outputs, not only status codes. Acceptance checks that the served
occupancy hash equals the computed one.
