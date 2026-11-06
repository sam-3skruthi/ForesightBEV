# Run the system

Everything runs locally with no network access at runtime and no silent downloads.

## Setup

```bash
uv sync --group dev --group ml
```

```bash
cd frontend && npm ci && npm run build
```

ROS 2 only (about 1.3 GB, isolated in `ros2/.pixi`). Download `pixi` to `.tools/pixi` from the
prefix-dev/pixi GitHub releases, then:

```bash
cd ros2 && ../.tools/pixi install
```

## Start the application

One command builds the frontend if needed and serves everything at http://127.0.0.1:8000 (API docs at `/docs`):

```bash
scripts/run_app.sh
```

## End-to-end demo

One command processes a normal drive and a camera-loss drive from real frames, stores both, and opens the application:

```bash
scripts/run_demo.sh
```

Then, in the application:

1. Overview shows the selected sequence, its split status, the decision and any warnings.
2. Camera studio shows the six real cameras, their timing and calibration. Choose `scene-0103: front camera loss`, frame 19, to see a lost camera.
3. BEV and geometry shows the combined view, per-camera projections, coverage and the fitted ground plane.
4. Occupancy compares the previous and current maps against annotated boxes, with the false-free proxy.
5. Detection and tracking shows boxes, track trails, and the baseline comparison.
6. Future prediction shows predicted paths, model spread, horizons and which method produced them.
7. Risk and safety shows every candidate and lets you evaluate your own plan live.
8. Evaluation separates real results, simulator results and experiment history.
9. System shows health, integrations, models, read-only configuration and logs, and can reprocess a sequence.

Left and right arrow keys step frames. Space plays and pauses.

Regenerate all eleven stored runs:

```bash
scripts/run_showcase.sh
```

## Other commands

Run the pipeline with acceptance checks. With no argument it uses the Cam2BEV sample. With a scene name it uses real
nuScenes footage. Results go to `experiments/end_to_end/results/acceptance/`.

```bash
scripts/run_pipeline.sh scene-0553
```

Verify everything:

```bash
scripts/verify_backend.sh
```

## Individual commands

Use `.venv/bin/python -m foresightbev.cli`, or the commands below.

Stage 2-3 score and pipeline run on the real frame:

```bash
.venv/bin/python -m foresightbev.cli sample
```

One recorded closed-loop episode. Scenarios: `free_road`, `lead_brake`, `cut_in`, `occluded_pedestrian`,
`static_obstacle`, `dense_traffic`. Modes are listed in `configs/experiment_default.yaml`.

```bash
.venv/bin/python -m foresightbev.cli run --scenario occluded_pedestrian --seed 100 --mode filter
```

Full Stage 9 evaluation (calibration, ablations, report). About 20 minutes on 8 workers.

```bash
.venv/bin/python -m foresightbev.cli eval --name default --workers 8
```

Trained forecaster: dataset, training and scoring. Writes into the same results folder.

```bash
.venv/bin/python -m foresightbev.forecast.learned --name default
```

nuScenes (see `docs/datasets/NUSCENES.md`):

```bash
.venv/bin/python -m foresightbev.cli validate-nuscenes
```

```bash
.venv/bin/python -m foresightbev.cli prepare nuscenes
```

```bash
.venv/bin/python -m foresightbev.cli eval-nuscenes --split held_out --name nuscenes_rerun
```

Pretrained StreamPETR (needs the checkpoint, see `docs/project/STREAMPETR_COMPATIBILITY.md`). Generate detections, then evaluate:

```bash
.venv/bin/python -m foresightbev.cli detect-streampetr
```

```bash
.venv/bin/python -m foresightbev.cli eval-nuscenes --split mini_val --name streampetr_rerun --pipeline-config configs/pipeline_streampetr.yaml
```

Verify the port with its negative controls:

```bash
.venv/bin/python -m foresightbev.evaluation.streampetr_check
```

Choose the detector with `detector:` in a pipeline configuration: `bev_clusters`, `image_size` or `streampetr`.

Rebuild the report from stored results:

```bash
.venv/bin/python -m foresightbev.cli report --name default
```

ROS 2 message self-test, replay of a prepared sequence, and closed loop. Each compares decision by decision with an in-process run:

```bash
scripts/run_ros2_selftest.sh
```

```bash
scripts/run_ros2_replay.sh scene-0553 "$PWD/configs/pipeline_nuscenes.yaml"
```


```bash
scripts/run_ros2_closed_loop.sh occluded_pedestrian 101
```

Tests:

```bash
.venv/bin/python -m pytest tests -q
```

## Outputs

| Path | Contents | Tracked by Git |
|---|---|---|
| `experiments/end_to_end/runs/<id>/` | Per-frame JSON and images, `manifest.json` | No: images derive from Cam2BEV terms |
| `experiments/end_to_end/results/<name>/` | Experiment config, environment, metrics, `REPORT.md` | Yes |
| `experiments/end_to_end/results/ros2/` | ROS 2 run logs and comparison | Yes |
| `experiments/end_to_end/sequences/` | Prepared label sequences | No: derived from dataset images |
| `experiments/end_to_end/data/`, `checkpoints/`, `models/` | Training data, trained and pretrained weights | No |
| `experiments/end_to_end/logs/`, `jobs/` | Backend logs and job records | Logs no, jobs no |
| `research/datasets/nuscenes/` | The dataset | No |
