# Configuration

| File | Purpose |
|---|---|
| `configs/backend.yaml` | Backend paths, log level, upload limit |
| `configs/pipeline_default.yaml` | Pipeline overrides for the simulator and Cam2BEV sample |
| `configs/pipeline_nuscenes.yaml` | Pipeline overrides for nuScenes |
| `configs/experiment_default.yaml` | Development evaluation |
| `configs/experiment_final.yaml` | Held-out simulator evaluation |
| `configs/nuscenes_splits.yaml` | nuScenes development and held-out scenes |
| `configs/rig_cam2bev_1frlr.yaml` | Cam2BEV camera rig |

Backend settings accept `FORESIGHTBEV_<KEY>` environment overrides, for example `FORESIGHTBEV_DATA_DIR`. Environment
wins over file, and file wins over defaults. `FORESIGHTBEV_CONFIG` selects a different file.

Pipeline YAML mirrors `PipelineConfig`: `occupancy`, `detection`, `tracker`, `forecast`, `planner`, `risk`, `safety`,
plus `nominal_mode`, `filter_enabled`, `seed` and `grid`. Unknown keys, wrong types and out-of-range values are
errors, so a typo cannot silently leave a default safety threshold in place. `foresightbev check` validates all of it
without running anything. `POST /api/v1/config/validate` does the same for overrides.

An experiment file may add a `pipeline:` section, applied to every mode.
