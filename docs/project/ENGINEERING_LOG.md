# Engineering log

Concise record of work on 2026-11-05 (IST). Nothing is committed or published.

## 15:30 Audit and first slice

- Found only a three-line package. No Stage 1–9 code, data, checkpoints, ROS, CARLA or Docker. One Cam2BEV frame.
- Corrected `source.md`: Stages 7–9 are in scope.
- Built contracts, calibration, projection (exact match to the official IPM output), semantic space carving, detection, tracking, forecasting, trajectories, risk, safety filter, simulator, closed loop.
- Repeated tracing fixed range noise, fragmented detections, double-counted forecast noise, the emergency stop at standstill, lateral motion at standstill, replanning that never completed a lane change, yawed route coordinates and over-strict stopping.

## 16:15 Stages 7–9, dashboard, forecaster

- Approved installs: FastAPI, React, ROS 2 Jazzy through RoboStack, PyTorch.
- First ROS 2 two-process closed loop. Dashboard built and verified against stored runs.
- Development evaluation on seeds 100–109. These overlap tuning seeds, so a held-out set was added: seeds 400–414.

## 17:20 Backend-first phase (frontend frozen)

- Kept the held-out run clean: edits were limited to new files until its workers had loaded, and it was later shown to reproduce on the changed code.
- Fixed pipeline staleness and missing-perception paths, input validation, per-run provenance.
- Added strict configuration, typed API v1, job manager, structured logs, stage-output retrieval, a risk endpoint that recomputes against stored forecasts, leak guards, overwrite protection, experiment records.
- Added the label sequence format, replay, ROS replay, stage topics, watchdog, recording from ROS, message self-test, acceptance runner.

## 17:45 nuScenes

- Archive verified, extracted without overwriting, and validated with zero problems.
- SegFormer-B2 approved, pinned to a revision and loaded locally.
- First real run: filter in constant fallback. Traced to the near-field blind zone and to flat-ground range. Annotations showed about ±0.5 m ground spread.
- Added the apparent-size detector and ground-height tolerance. Tuned on scenes 0061 and 0103 only, then froze and scored eight held-out scenes.

## 18:00 Verification

- Real-footage ROS mismatch was traced to arrival-order dependence. Initial timestamp hypothesis was tested and rejected.
- Fixed an indefinite ROS hang, then a DDS discovery race that failed the watchdog in 2 of 5 trials. After the fix, 12 of 12 pass.
- Fixed a misleading Brier comparison and a check that could read stale output.

## 18:45 Pretrained detector

- Assessed BEVFormer and StreamPETR. Selected StreamPETR R50 428q. Corrected the split analysis: mini_val is scene-0103 and scene-0916.
- Gate 1: checkpoint downloaded, hashed, pickle globals inspected, loaded with `weights_only=True`.
- Gate 2: compatibility report written from the pinned source.
- Gate 3: PyTorch inference port. 573 of 573 tensors load strictly. Verified with negative controls.
- Gate 4: compared on mini_val with identical inputs and thresholds. Scorer fixed first for range and low-visibility objects, applied to both detectors.
- Fixed provenance, which still reported detection as heuristic.
- A held-out reproducibility failure showed the scorer change was removing predictions before matching. Decisions were identical. Scorer corrected and mini_val rescored.

## Final showcase build

- Occupancy false-free studied. Rejected pose tilt. Selected local plane plus fused model boxes, kept in a labelled layer.
- Tracker: model velocity selected on mini_train, scored once on mini_val. Two-stage association rejected.
- Risk: added clearance, time to conflict, user plan evaluation and a real-log intervention study.
- Rebuilt the frontend as nine API-backed workspaces with one shared sequence and frame.
- Browser review found run directories vanishing during regeneration (now swapped in atomically), a frame reset on run change, and images requested for a stale frame (a data load now only returns for its own path).
- The fault review found the filter passing an accelerating plan with the front camera lost. Added `hold_speed_on_coverage_loss` to the showcase configuration only, leaving frozen simulator behaviour unchanged.
- ROS 2 revalidated with the showcase configuration, including a stalled camera.

## Open

See `IMPLEMENTATION_STATUS.md`, `KNOWN_LIMITATIONS.md` and `PHASE2_BACKLOG.md`.

## Tavyra final release (2026-11-06)

- Renamed the project to Tavyra in user-facing text, branding, docs and `source.md`. Repository URL and Python package `foresightbev` unchanged, so paths, venv and scripts keep working.
- Added a replay banner to every workspace: "Recorded nuScenes replay — open-loop evaluation. Decisions do not control subsequent frames." Relabelled speed as logged speed (unavailable on frame 0), decision as filter decision, and the plan tool as a per-frame test.
- Fixed the Overview "last recorded experiment" label to read from stored evaluations and the empty-warnings panel to hide when there are none.
- Added `scripts/audit_release.py`: invariants over all 11 stored runs through the API (cameras, sync, state partition, detection classes/scores, unique track IDs, causal forecast, decision/candidate consistency, fault coverage). 444 frames, 0 failures.
- Wrote the final release report, technical handover, mentor Q&A and demo script.
- Full verification: 29/29 steps, 352 tests. The three ROS 2 closed-loop steps failed under concurrent app+browser load (multi-hour wall times) and passed on isolated re-run (identical, occupancy hashes match).
- GitHub release: single commit to `main`. Dataset, checkpoints, runs, venv, `.claude/` and logs stay ignored.
