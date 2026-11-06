# Backend API

Start with `scripts/run_app.sh`, or `scripts/run_backend.sh` for the API alone. Interactive docs are at `/docs`. The schema is in `docs/api/openapi.json`.
The server binds to 127.0.0.1 and has no authentication, so keep it local.

`/api/v1` is the typed API and the only API the application uses. Older `/api/*` routes remain for compatibility and are not extended.

## Conventions

- Errors always use `{"error": {"code", "message", "request_id"}}`, and every response has an `X-Request-ID` header.
- Requests reject unknown fields.
- `*_ego` values are metres in the ego frame at the frame timestamp. Speeds are m/s. Times are seconds.
- Arrays use `{shape, dtype, encoding: "zlib+base64", data}`. Decode with zlib, then reshape. See `api/store.py:decode_array`.

| Status | Meaning |
|---|---|
| 404 | `run_not_found`, `dataset_not_found`, `output_not_stored` |
| 409 | `job_conflict`: a job is running, or results already exist |
| 422 | `invalid_request`, `invalid_dataset`, `invalid_configuration`, `invalid_calibration`, `synchronization_failed` |
| 500 | `internal_error`: details are logged, not returned |
| 503 | `model_unavailable` |

## Endpoints

| Method and path | Purpose |
|---|---|
| `GET /health` | Liveness |
| `GET /status` | Readiness checks, assets, active jobs, source hash |
| `GET /config/pipeline` | Effective configuration |
| `POST /config/validate` | Validate overrides without applying them |
| `GET /datasets` | Sequences, the Cam2BEV sample, nuScenes |
| `GET /datasets/nuscenes/scenes` | Scenes in the configured root |
| `GET /datasets/{id}/calibration` | Validated calibration |
| `PUT /uploads/{id}/frames/{n}/{camera}` | Ingest one colour-coded semantic PNG for the Cam2BEV rig. Raw body. |
| `POST /uploads/{id}/finalize` | Add ego metadata and create a sequence |
| `GET /runs`, `GET /runs/{id}` | Runs with provenance and environment |
| `GET /runs/{id}/frames/{n}/bev` | IPM labels and camera support |
| `GET /runs/{id}/frames/{n}/occupancy` | State, class and height grids, reliability. `?include_voxels=true` adds the 3D grid. |
| `GET /runs/{id}/frames/{n}/perception` | Detections and tracks |
| `GET /runs/{id}/frames/{n}/forecast` | Probabilities, timing, uncertainty semantics |
| `GET /runs/{id}/frames/{n}/decision` | Decision record with evidence for every candidate |
| `POST /runs/{id}/frames/{n}/risk` | Evaluate your own trajectories against that frame's stored forecast |
| `POST /jobs` | Start work |
| `GET /jobs`, `GET /jobs/{id}`, `GET /jobs/{id}/log`, `DELETE /jobs/{id}` | Inspect, read output, cancel |
| `GET /experiments`, `GET /experiments/index`, `GET /experiments/{name}/files/{file}` | Stored results |
| `GET /logs`, `GET /diagnostics` | Structured logs and counters |

Job kinds: `sample`, `episode`, `prepare_cam2bev`, `prepare_nuscenes`, `pipeline`, `evaluation`, `evaluate_nuscenes`,
`train_forecaster`. Heavy jobs refuse to start while another evaluation or training process is active, including ones
started outside the API.

## Risk example

`candidates[0]` is the nominal plan. Comfort and emergency stops are added automatically as fallbacks.

```bash
curl -s localhost:8000/api/v1/runs/accept_scene-0553/frames/10/risk -H 'content-type: application/json' -d '{"candidates":[{"id":"mine","times":[0,1,2,3],"poses":[[0,0,0,5],[5,0,0,5],[10,0,0,5],[15,0,0,5]]}],"desired_speed":5,"thresholds":{"max_event_score":0.1}}'
```

`event_score` is a forecast sample frequency, not a collision probability. Each response says so.

## Showcase routes

| Route | Returns |
|---|---|
| `GET /api/v1/sequences` | Real sequences with split, status and stored runs |
| `GET /api/v1/runs/{run}/timeline` | One row per frame |
| `GET /api/v1/runs/{run}/frames/{f}/record` | Everything stored for one frame |
| `GET /api/v1/runs/{run}/frames/{f}/cameras` | Status, timing and calibration |
| `GET /api/v1/runs/{run}/frames/{f}/image/{camera}` | The real JPEG. 404 `output_not_stored` when that camera was lost |
| `GET /api/v1/runs/{run}/frames/{f}/layer/{layer}` | `bev`, `bev_original`, `coverage`, `ipm`, `forecast` |
| `GET /api/v1/runs/{run}/frames/{f}/layer/{layer}/{camera}` | `labels`, `projection` |
| `GET /api/v1/runs/{run}/tracks` | Track histories in the world frame |
| `POST /api/v1/runs/{run}/frames/{f}/plan` | Evaluate `{target_speed, lateral_offset}` with the run's configuration |
| `GET /api/v1/evaluation` | Real and synthetic results, kept apart |
| `GET /api/v1/system` | Models, integrations, dataset, ROS runs, configurations |
| `GET /api/v1/config/pipeline/{name}` | One effective configuration, read-only |

