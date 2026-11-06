# End-to-end workspace audit

Inspected 2026-11-05, 15:30 IST, before any implementation. This is the starting state.

| Item | Finding |
|---|---|
| Code | `src/foresightbev/__init__.py` only (3 lines). No code for any stage. |
| Stages 7–9 | Nothing existed. The teammate-ownership statement in `source.md` was wrong and has been corrected. |
| Git | `main`, one commit `aaef1cb`. Audits, `research/stage_0/` and `source.md` untracked. |
| Hardware | Apple M5, 24 GB, 686 GB free. No NVIDIA GPU, no Docker. No SSH hosts configured. |
| Python | `.venv` 3.12.15 with numpy, scipy, opencv, pyyaml, pillow, pytest. `.venvs/cam2bev-ipm` preserved and unused. |
| Node | Node 26, npm 11. |
| Datasets | No nuScenes, Occ3D, Cam4DOcc, CARLA maps or Lyft data anywhere under the home folder. |
| Checkpoints | None for any audited model. Only an unrelated Qwen model in the Hugging Face cache. |
| Real multi-camera input | One Cam2BEV `1_FRLR` frame: four 964x604 semantic images plus top-down labels, in `experiments/cam2bev_ipm/inputs/`. It comes from a simulated dataset, has no timestamps or ego pose, and is gitignored pending licence review. |
| Upstream clones | Cam2BEV, Birds_Eye_View, SurroundOcc, OPUS, OCCUQ, Cam4DOcc. No weights. All need legacy CUDA or TF stacks. No Cam4DOcc audit report exists. |
| CARLA | Not installed. `pip` offers only `carla==0.9.5`, and there is no macOS arm64 server. |
| Autoware | Not installed. Targets Ubuntu with ROS 2. |

Consequence: no learned perception could run from local assets. The route taken is a classical pipeline
validated on the one real frame, plus a built-in simulator that renders through the same calibration
to supply sequences and ground truth.
