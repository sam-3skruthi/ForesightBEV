# Tavyra demo script (5–8 minutes)

Before the session, confirm the stored runs exist. The first command starts the application at http://127.0.0.1:8000:

```bash
scripts/run_app.sh
```

To reprocess both demo drives from real frames first (about a minute), then start it:

```bash
scripts/run_demo.sh
```

## 1. Problem and objective (30 s)
> "Tavyra studies a safety-critical question for camera-based driving: can we keep track of what the cameras actually know, and carry that
> through to a decision? Most importantly, unseen space must never be treated as free."

## 2. Architecture (45 s)
Show the rail on the left.
> "There are ten frozen stages. Six cameras go through geometry and occupancy, then detection and tracking, forecasting, and a safety layer. The
> backend and this dashboard only observe; they are not in the decision path."

## 3. Open a recorded sequence (30 s)
Open `http://127.0.0.1:8000/#/overview?run=showcase_scene-0916&frame=20`. Point to the amber line.
> "This is recorded nuScenes replay, open-loop evaluation. Decisions do not control subsequent frames. mini_val is two previously seen scenes,
> so these results are preliminary."

## 4. Cameras and frame navigation (45 s)
Open Camera studio. Use the left and right arrow keys, or press Play.
> "These are the real six images. Each shows its time offset. Calibration is read from the dataset and validated."

Choose `scene-0103: front camera loss`, frame 19.
> "Here I declared a fault by removing the front camera. The system reports it, and it does not invent the missing view."

## 5. BEV, occupancy, detections and tracks (90 s)
Return to `scene-0916`. Open BEV and geometry.
> "Each pixel is mapped to ground. Parked cars smear because only the road really lies on that plane. That is why we need more than IPM."

Open Occupancy.
> "Teal is observed free, red occupied, grey unknown. On the left is flat ground, and on the right the local plane with model boxes. Dashed boxes are
> annotations, display only. On mini_val the share of box cells wrongly free fell from .303 to .066. It is a proxy, because mini has no occupancy
> labels."

Open Detection and tracking.
> "StreamPETR gives 3D boxes and velocity. A Kalman filter tracks them. Against the apparent-size baseline, vehicle recall rises from .21 to .90."

## 6. Forecast and risk (60 s)
Open Future prediction and select +3 s.
> "This is constant-velocity extrapolation from tracks, not a trained 4D occupancy network. Circles are model spread, not a confidence interval."

Open Risk and safety at frame 21.

## 7. A modified and a fallback decision (60 s)
> "The nominal plan has event score .69 against a 0.10 limit and reaches zero clearance to a tracked vehicle within 2.5 s, so it is rejected and replaced by holding the lane at 70% of desired speed. Each row shows
> score, clearance and time to conflict. The score is a model frequency, not a collision probability."

Change the sequence to `scene-0103`, then go to `#/risk?run=showcase_scene-0103&frame=36`.
> "Here no candidate is admissible, so the filter declares fallback with an emergency stop, and records why."

## 8. Evaluation and limitations (60 s)
Open Evaluation, then scroll to the intervention table.
> "Real and simulated results are kept apart. On real logs only three conflicts occurred. Six of eight interventions had no annotated conflict, and one conflict
> passed. That is too little to validate the filter, so I do not claim it is safe. The simulator collision figures are not real-world evidence."

Open System.
> "ROS 2 is verified, with identical output to in-process runs. CARLA and Autoware show as not executed, because they were not."

## 9. Close (30 s)
> "The contribution is an inspectable camera-centric chain that keeps free, unknown and no-coverage apart through to an explained decision, and an honest
> test of it. Next come full nuScenes validation, Occ3D labels, CARLA, better depth and real-data risk calibration."

## If something goes wrong

If the page shows the backend is unavailable, restart `scripts/run_app.sh`. If no runs are listed, regenerate them:

```bash
scripts/run_showcase.sh
```
