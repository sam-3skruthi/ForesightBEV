# Tavyra final release report

**Tavyra — Camera-Centric 3D Occupancy, Spatiotemporal Forecasting and Risk-Constrained Decision Intelligence for Autonomous Driving.**

Released 2026-11-06. Formerly ForesightBEV; the repository and the Python package `foresightbev` keep that name. This is a research
prototype, not a driving system. On recorded nuScenes data it runs from six cameras to an explained decision in **open-loop replay**.

## 1. Final architecture

Ten frozen stages. One frame flows: six camera images and calibration → cached SegFormer-B2 labels and cached StreamPETR boxes →
local ground plane → semantic space-carving occupancy with fused model boxes → Kalman tracking → Monte Carlo constant-velocity
forecast → candidate plans scored for events, unknown exposure, clearance and TTC → safety filter (pass, modify, fallback) →
stored record → FastAPI → React application. The dashboard only observes; it is never in the decision path. Full detail is in
[TAVYRA_TECHNICAL_HANDOVER.md](TAVYRA_TECHNICAL_HANDOVER.md).

## 2. Implemented capabilities

| Stage | Status |
|---|---|
| 0 Literature, REA, gap | Prior audits kept. Candidate gap, not a proven novelty claim |
| 1 Data, cameras, calibration | Implemented, tested on nuScenes mini. Recorded ego pose is an input, not camera-derived |
| 2 Geometry, BEV | Implemented. Flat ground or coarse local plane. Exact match to Cam2BEV sample |
| 3 Occupancy, reliability | Classical space carving; free / occupied / unknown / no-coverage / invalid kept distinct. Heuristic camera health |
| 4 Detection, tracking | StreamPETR port plus baselines, Kalman tracker. Preliminary on mini_val |
| 5 Forecasting | Monte Carlo constant velocity in the risk path. Simulator U-Net excluded (does not transfer) |
| 6 Risk, safety | Candidates, swept events, unknown policy, calibrator, filter, fallback, reason codes, live plan evaluation |
| 7 CARLA, ROS 2, Autoware | ROS 2 verified (replay + simulator, bit-identical). CARLA adapter unexecuted. Autoware not integrated |
| 8 FastAPI, React | Typed `/api/v1`, nine-workspace app, API-backed, browser-inspected |
| 9 Evaluation | Ablations, calibration, tracking, forecast, occupancy, nuScenes scoring, leak guards, reports |

## 3. Verified test results

Full `scripts/verify_backend.sh` run on 2026-11-05/06: **29 of 29 steps**, including 352 automated tests, showcase acceptance on
scene-0916, StreamPETR port verification, ROS 2 replay of the showcase configuration, a stalled-camera replay, five watchdog
trials, the real-log safety study, a release audit over all 11 stored runs (444 frames, 0 invariant failures), and the frontend
production build.

**Honest note on the ROS 2 closed-loop steps.** In the full background run, the three closed-loop simulator steps
(`occluded_pedestrian`, `static_obstacle`, `lead_brake`) timed out because the application server and browser automation were
running at the same time; their wall times reached hours, which is the signature of CPU contention, not a code fault. Re-run in
isolation on 2026-11-06 they each passed: 58, 68 and 70 frames, `identical=True` against the in-process run with matching
occupancy hashes. The determinism guarantee holds; the lockstep is sensitive to a loaded machine.

Checks actually rerun now (2026-11-06): the three ROS 2 closed-loop scenarios, `scripts/run_demo.sh`, `scripts/run_app.sh` start
and health, and the full browser inspection (54 page views, no console errors, no broken images, replay banner present, playback
and pause working, mobile layout with no horizontal scroll). Earlier recorded results (simulator seeds 400–414, calibration,
learned forecaster, held-out classical decisions) were reproduced by the verification steps, not recomputed from scratch.

## 4. Measured metrics and conditions

**Real footage — mini_val, 81 frames, scene-0103 and scene-0916, both daytime, both previously seen by this project, preliminary.**
Matching 2 m within 40 m, score cutoff 0.3 fixed before evaluation.

| Measure | Apparent-size baseline | Showcase (StreamPETR) |
|---|---|---|
| Vehicle recall / precision | .21 / .31 | **.902 / .806** |
| Person recall / precision | .29 / .46 | **.831 / .734** |
| Track recall | .19 | **.786** |
| Track velocity error | 2.16 m/s | **.197 m/s** |
| ID switches per 100 matched | — | 7.7 (was 9.8) |
| Mean NEES (ideal ≈ 4) | 22.2 | **3.4** |
| Box cells marked free (proxy) | .303 | **.066** |
| Decisions pass / modify / fallback | 17 / 9 / 55 | 73 / 5 / 3 |
| Human-path alarms of 68 frames | 52 | 1 (0 true contacts) |
| Pipeline time (cached perception) | — | ~449 ms |

Forecast position error at 1 s / 3 s: constant velocity .78 / 1.66 m, frozen tracks .94 / 1.86 m. Forecast occupancy IoU on real
data (63 frames, 1 m cells) at 1 s / 3 s: constant velocity .362 / .297, frozen .378 / .317, simulator U-Net .154 / .116 — the
U-Net does not transfer. Live StreamPETR inference is ~290 ms/frame on MPS.

The occupancy figure is an **annotation-derived proxy** over dynamic-object footprints, not voxel accuracy; nuScenes mini has no
occupancy labels. Corrections: flat .550, previous tolerance .303, pose tilt .312 (no help), local plane .199, local plane +
model boxes .066. Free area fell from 1240 to 1113 m²/frame, so the gain has a cost.

**Risk on real logs (intervention study).** Of 81 frames the nominal plan conflicted with annotated objects 3 times; there were 8
interventions (2 with a conflict, 6 without), 1 conflict passed. Treating unknown as free gave 6 interventions, same conflict
counts. Three conflicts cannot support a rate, so the filter is **not validated** on real driving, and the conservative unknown
policy shows no measured benefit.

**Held-out classical pipeline (no pretrained detector), 8 scenes, 325 frames:** vehicle recall .214, person recall .251, human-path
alarm rate .529. This is the weak classical baseline the detector replaces.

**Simulator only (built-in, scripted non-reactive agents), 90 episodes/mode, held-out seeds 400–414:** filter collision rate 1.1%
vs IDM 23.3% vs blind 77.8%; route completion .795 vs .714. Unknown-as-free: 1.1% collision, .819 route — no measured benefit.
Calibration (simulator rows): ECE .078 → .004 with no change in decisions. **These figures are not evidence about real roads.**

## 5. Dataset and model provenance

- **Dataset:** official nuScenes v1.0-mini (10 scenes, 404 keyframes). Non-commercial (CC BY-NC-SA 4.0 with additional terms).
  Not modified, not redistributed, kept outside Git. Annotations used only for scoring and display, never as inference inputs.
- **Segmentation:** SegFormer-B2 Cityscapes, NVIDIA research licence, non-commercial. Trained on daytime data; night is weaker.
- **Detector:** StreamPETR R50, `stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth`,
  SHA-256 `72900a1c…dd2c5f`. **Weights licence unverified**; treated as non-commercial. Plain-PyTorch inference port, 573 needed
  tensors loaded strictly with `weights_only=True`. Port verified by controls: recall .877 with memory, .803 without, .045 with
  shuffled cameras. Metrics are measured here, not copied from the paper; published nuScenes val figures appear only in the
  assessment doc, clearly labelled.
- **Leakage:** mini_val scenes belong to the official val split; both were previously seen in this project, so results are
  preliminary. mini_train detector figures are labelled inflated. An architecture-guard test fails if inference modules import the
  ground-truth classes.

## 6. Known limitations

No untouched real test set. Safety filter unvalidated on real driving; event score is not a calibrated collision probability;
calibration is simulator-only. Occupancy figure is a dynamic-object proxy; static scenery unscored. Local ground plane is a coarse
tilt/offset, not a surface. Forecasting is tracking extrapolation, not a trained 4D occupancy network. Night quality weak. ROS 2
replay ~3.4 s/frame, not real-time, not profiled. Recorded ego pose (LiDAR-and-map localisation) is an input, so this is
camera-centric with recorded ego pose, not camera-only end to end. See [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md).

## 7. Unimplemented or partial

CARLA and Autoware never executed. No live camera input or vehicle control. No Occ3D evaluation. No learned 4D occupancy on real
data. No real-data risk calibration. No OOD/uncertainty output. No full nuScenes validation. See [PHASE2_BACKLOG.md](PHASE2_BACKLOG.md).

## 8. Run instructions

Start the application at http://127.0.0.1:8000:
```bash
scripts/run_app.sh
```
End-to-end demo (processes a normal drive and a camera-loss drive from real frames, then serves):
```bash
scripts/run_demo.sh
```
Regenerate all eleven stored runs:
```bash
scripts/run_showcase.sh
```
Full verification (idle machine; distinguishes freshly rerun checks from recorded results):
```bash
scripts/verify_backend.sh
```
Demo walkthrough: [TAVYRA_DEMO_SCRIPT.md](TAVYRA_DEMO_SCRIPT.md). Mentor preparation: [TAVYRA_MENTOR_QA.md](TAVYRA_MENTOR_QA.md).

## 9. Final release status

Working research prototype. Backend and frontend start reproducibly and show genuine stored results. Real sequences process end to
end; BEV, occupancy with the free/unknown/no-coverage distinction, detections, tracks, forecasts and traceable decisions are all
visible and API-backed. Important tests pass. Limitations are documented and surfaced in the UI. Not a safe, validated or
deployable autonomous-driving system, and must not be presented as one.
