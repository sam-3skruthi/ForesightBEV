# Experiment 1B & 2 Reproduction Report: Actual IPM Transformation and Multi-Camera Stitching

**Project:** ForesightBEV Research Audit & Reproduction
**Target Repository:** `Cam2BEV` (RWTH Aachen University, Institute for Automotive Engineering - ika)
**Target Scripts:** [preprocessing/ipm/ipm.py](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/preprocessing/ipm/ipm.py), [preprocessing/occlusion/occlusion.py](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/preprocessing/occlusion/occlusion.py)
**Execution Environment:** Isolated Python 3.12.15 venv (`~/Developer/ForesightBEV/.venvs/cam2bev-ipm`) on Apple Silicon M5 (24 GB RAM, macOS 26.6)
**Date:** November 2026
**Auditor Mode:** Controlled Experimental Reproduction & Metric Verification

---

## 1. Executive Summary & Status

Following the successful verification of analytical IPM homographies in Experiment 1A, **Experiment 1B (Actual Image-Space IPM Transformation)**, **Experiment 2 (Multi-Camera Stitching)**, and **Experiment 3 (Ground-Truth Occlusion Ray Casting)** were executed using the official, unmodified Cam2BEV scripts.

### Core Reproduction Results

1. **100.0000% Bit-for-Bit Reproduction of IPM Stitching:**
   The composite 360° BEV mosaic produced by executing `ipm.py` across the four perspective camera images matches the official RWTH Aachen GitLab reference image `homography.png` with **zero pixel discrepancy** ($\text{Max diff} = 0.0, \text{Mean diff} = 0.0, \text{Exact match} = 100.0000\%$).
2. **100.0000% Bit-for-Bit Reproduction of Occlusion Ray Casting:**
   The occlusion-filtered ground-truth BEV image produced by executing `occlusion.py` on the top-down drone segmentation matches the official RWTH Aachen GitLab reference `bev+occlusion.png` with **zero pixel discrepancy** ($\text{Exact match} = 100.0000\%$).
3. **Execution Runtime Performance:**
   On Apple Silicon M5, individual camera IPM warping executes in **25–36 ms** (~30 fps). The 4-camera composite stitching executes in **103 ms** (~10 fps). The 2D line-of-sight ray casting across 582,256 pixels executes in **15.2 seconds**.
4. **Final Experiment Statuses:**
   - **Experiment 1B (Individual IPM):** **PASS**
   - **Experiment 2 (Multi-Camera Stitching):** **PASS**
   - **Experiment 3 (Occlusion Processing):** **PASS**

---

## 2. Experimental Environment & Dependency Configuration

The experiment was conducted in an isolated virtual environment decoupled from host dependencies:
- **Python Binary:** `/Users/samskruthi/Developer/ForesightBEV/.venvs/cam2bev-ipm/bin/python` (Python 3.12.15)
- **OpenCV Version:** `opencv-python 4.10.0.84`
- **NumPy Version:** `numpy 2.1.3`
- **PyYAML Version:** `pyyaml 6.0.2`
- **Scikit-Image Version:** `scikit-image 0.24.0`
- **Progress Bar:** `tqdm 4.67.1`

No legacy TensorFlow or CUDA installations were required for these geometric preprocessing stages.

---

## 3. Input Data Provenance & Camera Calibrations

### 3.1 Input Images
Genuine, full-resolution camera samples were obtained from the official RWTH Aachen GitLab repository ([cam2bev-data](https://gitlab.ika.rwth-aachen.de/cam2bev/cam2bev-data)) for the `1_FRLR` setup:
- `front.png`: $604 \times 964 \times 3$, 15,300 bytes (Front perspective semantic segmentation)
- `rear.png`: $604 \times 964 \times 3$, 15,645 bytes (Rear perspective semantic segmentation)
- `left.png`: $604 \times 964 \times 3$, 14,974 bytes (Left perspective semantic segmentation)
- `right.png`: $604 \times 964 \times 3$, 12,656 bytes (Right perspective semantic segmentation)
- `bev.png`: $604 \times 964 \times 3$, 20,547 bytes (Drone ground-truth top-down segmentation)
- `homography.png`: $604 \times 964 \times 3$, 25,338 bytes (Official reference stitched IPM image)
- `bev+occlusion.png`: $604 \times 964 \times 3$, 24,272 bytes (Official reference occluded ground truth)

### 3.2 Calibration Parameters (`preprocessing/camera_configs/1_FRLR/`)

| Parameter | Front Camera | Rear Camera | Left Camera | Right Camera | Drone Camera |
|---|---|---|---|---|---|
| **Focal Length $f_x$** | 278.283 | 278.283 | 278.283 | 278.283 | 682.578 |
| **Focal Length $f_y$** | 408.1295 | 408.1295 | 408.1295 | 408.1295 | 682.578 |
| **Principal Point $p_x$** | 482.0 | 482.0 | 482.0 | 482.0 | 482.0 |
| **Principal Point $p_y$** | 302.0 | 302.0 | 302.0 | 302.0 | 302.0 |
| **Vehicle Coords $X$ (m)**| 1.7 | -0.6 | 0.5 | 0.5 | 0.0 |
| **Vehicle Coords $Y$ (m)**| 0.0 | 0.0 | 0.5 | -0.5 | 0.0 |
| **Vehicle Coords $Z$ (m)**| 1.4 | 1.4 | 1.5 | 1.5 | 50.0 |
| **Euler Yaw ($^\circ$)** | 0.0 | 180.0 | 90.0 | -90.0 | 0.0 |
| **Euler Pitch ($^\circ$)**| 0.0 | 0.0 | 0.0 | 0.0 | 90.0 |
| **Euler Roll ($^\circ$)** | 0.0 | 0.0 | 0.0 | 0.0 | 90.0 |

Notice that the image dimensions $(604, 964)$ strictly equal $(2 \cdot p_y, 2 \cdot p_x)$.

---

## 4. Execution Commands & Mathematical Pipeline

### 4.1 Individual IPM Warping (Experiment 1B)
Each perspective camera was warped onto the ground plane using nearest-neighbor interpolation (`--cc`) to preserve semantic color classes:
```bash
# Front camera
python preprocessing/ipm/ipm.py --cc \
  --drone preprocessing/camera_configs/1_FRLR/drone.yaml \
  preprocessing/camera_configs/1_FRLR/front.yaml inputs/front.png \
  --output experiments/cam2bev_ipm/ipm_outputs

# Rear camera
python preprocessing/ipm/ipm.py --cc \
  --drone preprocessing/camera_configs/1_FRLR/drone.yaml \
  preprocessing/camera_configs/1_FRLR/rear.yaml inputs/rear.png \
  --output experiments/cam2bev_ipm/ipm_outputs

# Left camera
python preprocessing/ipm/ipm.py --cc \
  --drone preprocessing/camera_configs/1_FRLR/drone.yaml \
  preprocessing/camera_configs/1_FRLR/left.yaml inputs/left.png \
  --output experiments/cam2bev_ipm/ipm_outputs

# Right camera
python preprocessing/ipm/ipm.py --cc \
  --drone preprocessing/camera_configs/1_FRLR/drone.yaml \
  preprocessing/camera_configs/1_FRLR/right.yaml inputs/right.png \
  --output experiments/cam2bev_ipm/ipm_outputs
```

### 4.2 Multi-Camera Stitching (Experiment 2)
The four cameras were supplied simultaneously to the upstream argument parser:
```bash
python preprocessing/ipm/ipm.py --cc \
  --drone preprocessing/camera_configs/1_FRLR/drone.yaml \
  preprocessing/camera_configs/1_FRLR/front.yaml inputs/front.png \
  preprocessing/camera_configs/1_FRLR/rear.yaml inputs/rear.png \
  preprocessing/camera_configs/1_FRLR/left.yaml inputs/left.png \
  preprocessing/camera_configs/1_FRLR/right.yaml inputs/right.png \
  --output experiments/cam2bev_ipm/stitching
```

### 4.3 Ground-Truth Occlusion Processing (Experiment 3)
```bash
python preprocessing/occlusion/occlusion.py \
  inputs/bev.png \
  preprocessing/camera_configs/1_FRLR/drone.yaml \
  preprocessing/camera_configs/1_FRLR/front.yaml \
  preprocessing/camera_configs/1_FRLR/rear.yaml \
  preprocessing/camera_configs/1_FRLR/left.yaml \
  preprocessing/camera_configs/1_FRLR/right.yaml \
  --output experiments/cam2bev_ipm/occlusion
```

---

## 5. Quantitative Results & Validation Metrics

### 5.1 Canvas Coverage & Pixel Statistics

| Camera Feed | Valid Output Pixels | Total Canvas Pixels | Canvas Coverage (%) | Runtime (ms) | Condition Number $\kappa(H)$ |
|---|---|---|---|---|---|
| **Front** | 222,955 | 582,256 | **38.29%** | 35.2 ms | $8.2032 \times 10^6$ |
| **Rear** | 232,627 | 582,256 | **39.95%** | 36.2 ms | $7.5490 \times 10^6$ |
| **Left** | 149,545 | 582,256 | **25.68%** | 25.7 ms | $2.2114 \times 10^6$ |
| **Right** | 148,572 | 582,256 | **25.52%** | 27.9 ms | $1.6226 \times 10^6$ |
| **Stitched Composite** | **576,460** | **582,256** | **99.00%** | **103.4 ms** | N/A |
| **Ego Blind Spot** | 5,796 | 582,256 | **1.00%** | N/A | N/A |

### 5.2 Seam Overlap Analysis

| Overlap Pair | Overlap Pixel Count | Percentage of BEV Canvas (%) | Seam Geometry |
|---|---|---|---|
| **Front & Left** | 44,106 | 7.58% | Front-left diagonal cone |
| **Front & Right** | 43,902 | 7.54% | Front-right diagonal cone |
| **Rear & Left** | 44,810 | 7.70% | Rear-left diagonal cone |
| **Rear & Right** | 44,421 | 7.63% | Rear-right diagonal cone |
| **Front & Rear** | 0 | 0.00% | Completely mutually exclusive |
| **Left & Right** | 0 | 0.00% | Completely mutually exclusive |

### 5.3 Mathematical Validation Against Reference Ground Truth

| Test Comparison | Metric | Measured Value | Threshold | Verdict |
|---|---|---|---|---|
| **Stitched IPM vs `homography.png`** | Max Absolute Pixel Difference | **0.0** | $\le 0.0$ | **PERFECT MATCH** |
| **Stitched IPM vs `homography.png`** | Mean Absolute Pixel Difference | **0.0** | $\le 0.0$ | **PERFECT MATCH** |
| **Stitched IPM vs `homography.png`** | Exact Pixel Agreement | **100.0000%** | $\ge 99.99\%$ | **PERFECT MATCH** |
| **Occlusion Output vs `bev+occlusion.png`** | Max Absolute Pixel Difference | **0.0** | $\le 0.0$ | **PERFECT MATCH** |
| **Occlusion Output vs `bev+occlusion.png`** | Exact Pixel Agreement | **100.0000%** | $\ge 99.99\%$ | **PERFECT MATCH** |

---

## 6. Visual Observations & Algorithmic Reality

### 6.1 Generated Visual Artifacts

The following visual artifacts were generated and saved in the experiment directory:

1. **camera_inputs_panel.png** (local-only artifact: `experiments/cam2bev_ipm/camera_inputs_panel.png`; not committed pending provenance and redistribution review): Synchronized $2 \times 2$ grid of the four raw perspective semantic segmentation inputs.
2. **ipm_warps_panel.png** (local-only artifact: `experiments/cam2bev_ipm/ipm_warps_panel.png`; not committed pending provenance and redistribution review): Individual IPM warped projections showing the directional coverage cones of each camera.
3. **stitching_vs_ground_truth_comparison.png** (local-only artifact: `experiments/cam2bev_ipm/stitching/stitching_vs_ground_truth_comparison.png`; not committed pending provenance and redistribution review): Side-by-side comparison between the stitched IPM output and the drone ground-truth BEV.
4. **overlap_and_seam_visualization.png** (local-only artifact: `experiments/cam2bev_ipm/stitching/overlap_and_seam_visualization.png`; not committed pending provenance and redistribution review): False-color diagnostic map showing individual camera regions (Blue, Green, Red, Yellow), diagonal overlap seams (Magenta, Cyan, Purple, Orange), and the central ego vehicle blind spot (Gray).
5. **occlusion_comparison_panel.png** (local-only artifact: `experiments/cam2bev_ipm/occlusion/occlusion_comparison_panel.png`; not committed pending provenance and redistribution review): Three-panel visualization showing the original drone BEV, the occlusion-filtered BEV, and the extracted occlusion shadow mask.

### 6.2 Key Qualitative Insights

1. **Severe Radial Smearing of 3D Objects in IPM:**
   In **stitching_vs_ground_truth_comparison.png** (local-only artifact: `experiments/cam2bev_ipm/stitching/stitching_vs_ground_truth_comparison.png`; not committed pending provenance and redistribution review), upright vehicles that appear as compact rectangular boxes in the drone ground truth are radially smeared outward into elongated wedge shapes extending over 10–20 meters in the IPM mosaic. This visual evidence confirms why classical IPM achieves only **30.17% mIoU** in the paper's benchmark.
2. **First-Write-Wins Seam Artifacts:**
   Inspection of `ipm.py:189-192` shows that multi-camera stitching iterates through cameras sequentially (`for warpedImg in warpedImages: mask = np.any(warpedImg != (0,0,0), axis=-1); birdsEyeView[mask] = warpedImg[mask]`). In overlap regions, whichever camera appears later in the argument list overwrites earlier pixels. Because parallax shifts vertical objects differently in adjacent cameras, tall obstacles crossing overlap boundaries suffer severe shear discontinuities.
3. **Ego Vehicle Blind Spot:**
   The $1.00\%$ uncovered area ($68 \times 83$ pixels, corresponding to approximately $5.0\text{ m} \times 6.1\text{ m}$) corresponds exactly to the ego vehicle chassis beneath the mounting perimeters of the four outward-looking cameras.
4. **Occlusion Ray Casting Behavior:**
   The ray caster accurately reproduces shadow wedges behind vehicles, buildings, and walls. However, because ray casting is performed in 2D pixel space without true 3D elevation maps, tall structures and low curbs are treated with identical blocking rules (except for the hardcoded `car` flood-fill and `vegetation` height exceptions).

---

## 7. Deviations from Upstream Implementation

**Zero deviations.**
The official upstream scripts [preprocessing/ipm/ipm.py](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/preprocessing/ipm/ipm.py) and [preprocessing/occlusion/occlusion.py](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/preprocessing/occlusion/occlusion.py) were executed unmodified in their native form.

---

## 8. Implications for ForesightBEV Architecture

This reproduction demonstrates two fundamental limitations of the Cam2BEV approach:
1. **The Planar Homography Collapse ($Z=0$):**
   Classical IPM is mathematically incapable of representing 3D spatial occupancy. ForesightBEV cannot rely on 2D homographies for robotics perception; it requires an explicit **metric 3D voxel grid** or depth-backprojected frustum representation.
2. **Heuristic 2D Ray Casting vs. True 3D Volumetric Occlusion:**
   Cam2BEV's 2D Bresenham ray-casting requires hand-crafted exceptions (`TALL_NON_BLOCKING_LABELS = ["vegetation"]`, single-car transparency heuristics). For an RGB-D robot, true line-of-sight visibility is obtained directly by ray marching through the continuous 3D depth field or truncated signed distance function (TSDF), eliminating ad-hoc heuristic 2D shadow algorithms.
