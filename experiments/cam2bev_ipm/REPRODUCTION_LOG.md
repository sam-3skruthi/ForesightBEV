# Reproduction Log: Cam2BEV IPM, Stitching, and Occlusion

**Project:** ForesightBEV Research Reproduction
**Target Repository:** `Cam2BEV` (RWTH Aachen University, Institute for Automotive Engineering - ika)
**Execution Environment:** Isolated Python 3.12.15 venv (`~/Developer/ForesightBEV/.venvs/cam2bev-ipm`) on Apple Silicon M5 (macOS 26.6)
**Main Tooling:** Native OpenCV (`opencv-python 4.10.x`), NumPy (`2.1.x`), PyYAML (`6.0.x`), scikit-image (`0.24.x`), tqdm

---

## Chronological Execution Log

### Session 1: Experiment 1A — Analytical IPM Homography Calculation
- **Timestamp:** 2026-11-03 13:30 UTC
- **Action:** Executed upstream `preprocessing/ipm/ipm.py` with matrix-only output flag (`-v`).
- **Command:**
  ```bash
  /Users/samskruthi/Developer/ForesightBEV/.venvs/cam2bev-ipm/bin/python preprocessing/ipm/ipm.py -v \
    --drone preprocessing/camera_configs/1_FRLR/drone.yaml \
    preprocessing/camera_configs/1_FRLR/front.yaml front \
    preprocessing/camera_configs/1_FRLR/rear.yaml rear \
    preprocessing/camera_configs/1_FRLR/left.yaml left \
    preprocessing/camera_configs/1_FRLR/right.yaml right
  ```
- **Output:** Printed the four $3 \times 3$ OpenCV homography matrices for the `1_FRLR` surround-view rig.
- **Verification:** Computed matrices matched the upstream documentation reference in `preprocessing/homography_converter/README.md:40-48` down to machine floating-point precision ($< 10^{-14}$ delta).
- **Artifact:** Saved to `experiments/cam2bev_ipm/homographies_1_FRLR.txt`.
- **Status:** **PASS**

---

### Session 2: Image Asset Inventory & Provenance Verification
- **Timestamp:** 2026-11-03 16:35 UTC
- **Action:** Inspected `Cam2BEV` repository and discovered only 6 documentation assets. Verified presence of official RWTH Aachen GitLab example images in `experiments/cam2bev_ipm/inputs/1_FRLR_examples/`.
- **Artifact:** Created `experiments/cam2bev_ipm/input_inventory.csv`.
- **Key Finding:** Documentation thumbnails in `preprocessing/ipm/assets/` and `preprocessing/occlusion/assets/` are low-resolution previews ($200 \times 320$), whereas `1_FRLR_examples/` contains genuine full-resolution ($604 \times 964$) synchronized frames matching `1_FRLR` camera calibrations.
- **Status:** **VERIFIED**

---

### Session 3: Experiment 1B — Individual Camera IPM Warping
- **Timestamp:** 2026-11-03 16:42 UTC
- **Action:** Executed original `ipm.py` without `-v` individually for Front, Rear, Left, and Right cameras using nearest-neighbor color-coded warping (`--cc`).
- **Commands Executed:**
  ```bash
  # Front camera
  /Users/samskruthi/Developer/ForesightBEV/.venvs/cam2bev-ipm/bin/python preprocessing/ipm/ipm.py --cc \
    --drone preprocessing/camera_configs/1_FRLR/drone.yaml \
    preprocessing/camera_configs/1_FRLR/front.yaml /Users/samskruthi/Developer/ForesightBEV/experiments/cam2bev_ipm/inputs/1_FRLR_examples/front.png \
    --output /Users/samskruthi/Developer/ForesightBEV/experiments/cam2bev_ipm/ipm_outputs

  # Rear camera
  /Users/samskruthi/Developer/ForesightBEV/.venvs/cam2bev-ipm/bin/python preprocessing/ipm/ipm.py --cc \
    --drone preprocessing/camera_configs/1_FRLR/drone.yaml \
    preprocessing/camera_configs/1_FRLR/rear.yaml /Users/samskruthi/Developer/ForesightBEV/experiments/cam2bev_ipm/inputs/1_FRLR_examples/rear.png \
    --output /Users/samskruthi/Developer/ForesightBEV/experiments/cam2bev_ipm/ipm_outputs

  # Left camera
  /Users/samskruthi/Developer/ForesightBEV/.venvs/cam2bev-ipm/bin/python preprocessing/ipm/ipm.py --cc \
    --drone preprocessing/camera_configs/1_FRLR/drone.yaml \
    preprocessing/camera_configs/1_FRLR/left.yaml /Users/samskruthi/Developer/ForesightBEV/experiments/cam2bev_ipm/inputs/1_FRLR_examples/left.png \
    --output /Users/samskruthi/Developer/ForesightBEV/experiments/cam2bev_ipm/ipm_outputs

  # Right camera
  /Users/samskruthi/Developer/ForesightBEV/.venvs/cam2bev-ipm/bin/python preprocessing/ipm/ipm.py --cc \
    --drone preprocessing/camera_configs/1_FRLR/drone.yaml \
    preprocessing/camera_configs/1_FRLR/right.yaml /Users/samskruthi/Developer/ForesightBEV/experiments/cam2bev_ipm/inputs/1_FRLR_examples/right.png \
    --output /Users/samskruthi/Developer/ForesightBEV/experiments/cam2bev_ipm/ipm_outputs
  ```
- **Outcomes:**
  - `front.png`: $604 \times 964 \times 3$, 222,955 valid pixels ($38.29\%$ canvas coverage). Runtime: 0.035s.
  - `rear.png`: $604 \times 964 \times 3$, 232,627 valid pixels ($39.95\%$ canvas coverage). Runtime: 0.036s.
  - `left.png`: $604 \times 964 \times 3$, 149,545 valid pixels ($25.68\%$ canvas coverage). Runtime: 0.026s.
  - `right.png`: $604 \times 964 \times 3$, 148,572 valid pixels ($25.52\%$ canvas coverage). Runtime: 0.028s.
- **Verification:** All four perspective images were rectilinearly warped into the road plane without runtime errors or memory warnings.
- **Status:** **PASS**

---

### Session 4: Experiment 2 — Multi-Camera Stitching & Seam Verification
- **Timestamp:** 2026-11-03 16:42 UTC
- **Action:** Executed original `ipm.py` with all 4 cameras supplied simultaneously.
- **Command:**
  ```bash
  /Users/samskruthi/Developer/ForesightBEV/.venvs/cam2bev-ipm/bin/python preprocessing/ipm/ipm.py --cc \
    --drone preprocessing/camera_configs/1_FRLR/drone.yaml \
    preprocessing/camera_configs/1_FRLR/front.yaml /Users/samskruthi/Developer/ForesightBEV/experiments/cam2bev_ipm/inputs/1_FRLR_examples/front.png \
    preprocessing/camera_configs/1_FRLR/rear.yaml /Users/samskruthi/Developer/ForesightBEV/experiments/cam2bev_ipm/inputs/1_FRLR_examples/rear.png \
    preprocessing/camera_configs/1_FRLR/left.yaml /Users/samskruthi/Developer/ForesightBEV/experiments/cam2bev_ipm/inputs/1_FRLR_examples/left.png \
    preprocessing/camera_configs/1_FRLR/right.yaml /Users/samskruthi/Developer/ForesightBEV/experiments/cam2bev_ipm/inputs/1_FRLR_examples/right.png \
    --output /Users/samskruthi/Developer/ForesightBEV/experiments/cam2bev_ipm/stitching
  ```
- **Outcome:** Saved composite image to `stitching/front.png` (renamed to `stitching/stitched_bev.png`).
- **Bit-for-Bit Validation:** Compared generated `stitched_bev.png` against official RWTH Aachen GitLab reference `inputs/1_FRLR_examples/homography.png`:
  - **Max pixel difference:** `0.0`
  - **Mean pixel difference:** `0.0`
  - **Exact pixel match:** `100.0000%`
- **Seam & Overlap Metrics:**
  - Total canvas: 582,256 pixels
  - Stitched coverage: 576,460 pixels ($99.00\%$)
  - Ego vehicle blind spot: 5,796 pixels ($1.00\%$), bounded within $Y \in [268, 336], X \in [448, 531]$
  - Overlap Front-Left: 44,106 px ($7.58\%$)
  - Overlap Front-Right: 43,902 px ($7.54\%$)
  - Overlap Rear-Left: 44,810 px ($7.70\%$)
  - Overlap Rear-Right: 44,421 px ($7.63\%$)
- **Status:** **PASS**

---

### Session 5: Experiment 3 — Ground-Truth Occlusion Ray-Casting
- **Timestamp:** 2026-11-03 16:44 UTC
- **Action:** Executed original `preprocessing/occlusion/occlusion.py` on the top-down drone segmentation image.
- **Command:**
  ```bash
  /Users/samskruthi/Developer/ForesightBEV/.venvs/cam2bev-ipm/bin/python preprocessing/occlusion/occlusion.py \
    /Users/samskruthi/Developer/ForesightBEV/experiments/cam2bev_ipm/inputs/1_FRLR_examples/bev.png \
    preprocessing/camera_configs/1_FRLR/drone.yaml \
    preprocessing/camera_configs/1_FRLR/front.yaml \
    preprocessing/camera_configs/1_FRLR/rear.yaml \
    preprocessing/camera_configs/1_FRLR/left.yaml \
    preprocessing/camera_configs/1_FRLR/right.yaml \
    --output /Users/samskruthi/Developer/ForesightBEV/experiments/cam2bev_ipm/occlusion
  ```
- **Outcome:** Generated `occlusion/bev.png` ($604 \times 964 \times 3$, 24,272 bytes). Total execution runtime: ~15.2 seconds on M5 CPU.
- **Bit-for-Bit Validation:** Compared generated `occlusion/bev.png` against official RWTH Aachen GitLab ground-truth reference `inputs/1_FRLR_examples/bev+occlusion.png`:
  - **Max pixel difference:** `0.0`
  - **Mean pixel difference:** `0.0`
  - **Exact pixel match:** `100.0000%`
- **Status:** **PASS**

---

### Summary of Statuses

| Experiment | Target Functionality | Upstream Script | Status | Verification Evidence |
|---|---|---|---|---|
| **Experiment 1A** | Analytical Homography Matrix Calculation | `ipm.py -v` | **PASS** | Matrices match documentation down to $10^{-14}$ |
| **Experiment 1B** | Actual Perspective Warping (4 cameras) | `ipm.py --cc` | **PASS** | 4 individual warps generated in `ipm_outputs/` |
| **Experiment 2** | 360° Multi-Camera Stitching & Seams | `ipm.py --cc` | **PASS** | 100.0000% bit-for-bit match to official `homography.png` |
| **Experiment 3** | Ground-Truth Occlusion Ray Casting | `occlusion.py` | **PASS** | 100.0000% bit-for-bit match to official `bev+occlusion.png` |
