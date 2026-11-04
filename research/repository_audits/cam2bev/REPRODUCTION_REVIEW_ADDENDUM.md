# Reproduction Review Addendum: Cam2BEV

**Project:** ForesightBEV Research Audit & Verification
**Target Repository:** `Cam2BEV` (RWTH Aachen University, Institute for Automotive Engineering - ika)
**Document Purpose:** Critical review, terminology correction, geometric revalidation, and closure of the Cam2BEV reproduction audit (Experiments 1A, 1B, 2, and 3).
**Date:** November 2026
**Auditor Mode:** Read-Only Verification, Methodological Review, & Metric Auditing

---

## 1. Verification of "Pixel-Identical" Reproduction Claims (Item A1)

In the preceding audit reports, the term "100% bit-for-bit match" was used colloquially. This review formally audits the verification procedure and refines the claim to standard computer vision precision.

### 1.1 Input Provenance & Properties
- **Source:** The input images used for Experiments 1B, 2, and 3 were sourced from the official RWTH Aachen GitLab repository ([cam2bev-data](https://gitlab.ika.rwth-aachen.de/cam2bev/cam2bev-data)) under the `1_FRLR` surround-view camera setup.
- **Image Properties:**
  - `front.png`, `rear.png`, `left.png`, `right.png`: Dimensions $604 \times 964$, 3-channel 8-bit RGB PNGs, representing semantic segmentation classes using the Cityscapes color palette.
  - `bev.png`: Dimensions $604 \times 964$, 3-channel 8-bit RGB top-down drone segmentation ground truth.
  - `homography.png`: Dimensions $604 \times 964$, 3-channel 8-bit RGB official preprocessed IPM mosaic reference.
  - `bev+occlusion.png`: Dimensions $604 \times 964$, 3-channel 8-bit RGB official ground-truth reference with occluded regions labeled.
- **Synchronization:** The four camera perspective views and the two drone BEV views represent a single, mutually synchronized temporal frame from the CARLA synthetic recording environment.

### 1.2 Comparison Methodology
The comparison between generated outputs and reference images was performed on **in-memory decoded pixel arrays**, not raw compressed file byte streams:
- Images were decoded using OpenCV:
  ```python
  stitched = cv2.imread("experiments/cam2bev_ipm/stitching/stitched_bev.png")
  ref = cv2.imread("experiments/cam2bev_ipm/inputs/1_FRLR_examples/homography.png")
  ```
- Comparison metrics were computed across all $H \times W \times C = 604 \times 964 \times 3 = 1,746,768$ values:
  $$\Delta_{\max} = \max_{y, x, c} |I_{\text{gen}}(y, x, c) - I_{\text{ref}}(y, x, c)| = 0$$
  $$\Delta_{\text{mean}} = \frac{1}{HWC} \sum_{y, x, c} |I_{\text{gen}}(y, x, c) - I_{\text{ref}}(y, x, c)| = 0.0$$
  $$\text{Pixel Agreement} = \frac{1}{HW} \sum_{y, x} \mathbb{I}\left(I_{\text{gen}}(y, x, :) = I_{\text{ref}}(y, x, :)\right) \times 100\% = 100.0000\%$$

### 1.3 Technical Qualification
- **Standardized Terminology:** The accurate technical designation is **pixel-identical decoded output**.
- **Scope Limitation:** This exact match was verified on the **single synchronized reference sample** provided in the official repository. It proves conclusively that the local execution environment, numerical libraries (OpenCV 4.10, NumPy 2.1), camera calibration parsing, homography inversion, and nearest-neighbor perspective interpolation reproduce the upstream author's environment for this sample. It does **not** assert broad numerical invariance across all 36,930 dataset sequences, where floating-point differences on different architectures could conceivably induce boundary rounding variations.

---

## 2. Stitching Implementation & Overwrite Terminology (Item A2)

### 2.1 Code Inspection of Multi-Camera Composition
Inspection of the stitching loop in [preprocessing/ipm/ipm.py:189-192](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/preprocessing/ipm/ipm.py#L189-L192) reveals the exact composition logic:

```python
# stitch separate images to total bird's-eye-view
birdsEyeView = np.zeros(warpedImages[0].shape, dtype=np.uint8)
for warpedImg in warpedImages:
  mask = np.any(warpedImg != (0,0,0), axis=-1)
  birdsEyeView[mask] = warpedImg[mask]
```

`warpedImages` is populated in the order cameras are passed via CLI arguments:
`[front, rear, left, right]`.

### 2.2 Terminology Correction: "Last-Write-Wins Sequential Camera Overwrite"
The earlier report inadvertently used the phrase "First-Write-Wins". This is mathematically and programmatically incorrect:
1. `birdsEyeView` is initialized to all zeros.
2. In iteration 1 (`front`), front warped pixels are written to `birdsEyeView`.
3. In iteration 2 (`rear`), rear warped pixels are written. (Front and rear do not overlap).
4. In iteration 3 (`left`), any non-zero pixel from the left camera is assigned to `birdsEyeView[mask]`, **overwriting** any overlapping pixels previously written by `front` or `rear`.
5. In iteration 4 (`right`), any non-zero pixel from the right camera is assigned to `birdsEyeView[mask]`, **overwriting** any overlapping pixels from `front` or `rear`.

The correct, implementation-specific description is:
**Last-Write-Wins Sequential Camera Overwrite** (or *Priority-Ordered Occlusion by Argument Sequence*).

### 2.3 Consequences for 3D Objects & Seam Discontinuities
- **No Blending:** There is no feathering, alpha blending, distance-weighted fusion, or graph-cut seam optimization.
- **Parallax Step Discontinuities:** Because 3D objects violate the planar road assumption ($Z \ne 0$), an upright object (e.g. a vehicle in the front-left overlap cone) appears at different image coordinates in the front camera versus the left camera. In the overlap region, the left camera abruptly overwrites the front camera's projection along a sharp radial boundary, causing severed or duplicated object geometry.

---

## 3. Revalidation of Geometric Coverage & Overlap Statistics (Item A3)

### 3.1 Semantic Support vs. True Geometric Support
In `ipm.py:191`, the presence of valid projection is determined by testing:
`mask = np.any(warpedImg != (0, 0, 0), axis=-1)`
In the Cityscapes palette used by Cam2BEV, RGB `(0, 0, 0)` designates `obstacle` / `static`. Consequently, if a camera observed a static obstacle colored black, `ipm.py` would treat that pixel as non-existent and fail to write it, or allow an earlier camera to remain un-overwritten.

To separate semantic coloration from camera geometry, an **independent geometric validity mask** was computed:
1. A binary canvas of ones ($604 \times 964$, dtype=uint8) was defined on each camera sensor plane.
2. Each mask was transformed via `cv2.warpPerspective(ones, IPM, (964, 604), flags=cv2.INTER_NEAREST)`.
3. The analytical FOV directional mask was applied (`warped_geom[fov_mask] = 0`).

### 3.2 Revalidated Geometric Statistics

| Region / Metric | Geometric Support | Semantic Support (Sample) | Discrepancy | Canvas Percentage |
|---|---|---|---|---|
| **Front Camera Footprint** | 222,955 px | 222,955 px | 0 px | 38.29% |
| **Rear Camera Footprint** | 232,627 px | 232,627 px | 0 px | 39.95% |
| **Left Camera Footprint** | 149,545 px | 149,545 px | 0 px | 25.68% |
| **Right Camera Footprint** | 148,572 px | 148,572 px | 0 px | 25.52% |
| **Total Canvas Area** | 582,256 px | 582,256 px | 0 px | 100.00% |
| **Combined 4-Camera Union** | **576,460 px** | **576,460 px** | **0 px** | **99.0046% (~99.00%)** |
| **Uncovered Area** | **5,796 px** | **5,796 px** | **0 px** | **0.9954% (~1.00%)** |

*(Note: For this specific sample, no valid black obstacle pixels occurred within the warped camera frustums, which explains why the semantic and geometric support matched identically down to 0 pixels).*

### 3.3 Revalidated Pairwise Overlaps

| Overlap Pair | Intersection Pixels | Percentage of Total Canvas |
|---|---|---|
| **Front $\cap$ Left** | 44,106 px | 7.575% (~7.58%) |
| **Front $\cap$ Right** | 43,902 px | 7.540% (~7.54%) |
| **Rear $\cap$ Left** | 44,810 px | 7.696% (~7.70%) |
| **Rear $\cap$ Right** | 44,421 px | 7.629% (~7.63%) |
| **Front $\cap$ Rear** | 0 px | 0.000% |
| **Left $\cap$ Right** | 0 px | 0.000% |

### 3.4 Uncovered Region: Bounding Box & Off-by-One Correction
The coordinates of the uncovered region were audited:
- Bounding Box Row Indices ($Y$): $\min = 268$, $\max = 336$
- Bounding Box Column Indices ($X$): $\min = 448$, $\max = 531$

**Off-by-One Correction:**
Using the standard **inclusive indexing** convention:
$$\text{Height} = Y_{\max} - Y_{\min} + 1 = 336 - 268 + 1 = 69\text{ pixels}$$
$$\text{Width} = X_{\max} - X_{\min} + 1 = 531 - 448 + 1 = 84\text{ pixels}$$
$$\text{Bounding Box Area} = 69 \times 84 = 5,796\text{ pixels}$$

The uncovered region is a **solid, contiguous rectangle of exactly $69 \times 84 = 5,796$ pixels**. The previous report incorrectly stated $68 \times 83$ due to omitting $+1$ for inclusive bounds.

**Geometric Attribution:**
In metric coordinates ($13.65156\text{ px/m}$), this unobserved rectangle measures $5.05\text{ m}$ (lateral) $\times 6.15\text{ m}$ (longitudinal). Rather than asserting without proof that this equals the exact CAD vehicle chassis boundary, it is properly defined as:
> **The unobserved central blind region created by the intersection of the outward-directed FOV clipping planes of the four perimeter-mounted cameras.**

---

## 4. Methodological Review of Homography Conditioning (Item A4)

In the previous report, condition numbers on the order of $\kappa(H) \sim 10^6$ were cited as proof that coordinate normalization is required in neural networks. This claim requires important mathematical qualification.

### 4.1 Dependency on Coordinate System & Scaling
The matrix 2-norm condition number $\kappa(H) = \sigma_{\max}(H) / \sigma_{\min}(H)$ is **not an invariant property** of a projective transformation. It is fundamentally coupled to the coordinate representation:
1. **Unnormalized Pixel Coordinates:** In `ipm.py`, $H_{cv}$ maps image pixel indices $(u, v) \in [0, 964] \times [0, 604]$ with homogeneous projective elements $w \sim 10^{-3}$. The matrix entries span six orders of magnitude ($10^{-3}$ to $10^2$), yielding singular values $\sigma \in [3.8 \times 10^{-5}, 3.1 \times 10^2]$ and $\kappa(H) \approx 8.2 \times 10^6$.
2. **Normalized Grid Coordinates:** When Hartley preconditioning is applied to map image coordinates to $[-1, 1]$ (as performed in `homography_converter.py` for the Spatial Transformer), the condition numbers drop dramatically:
   - Front Camera $H_{stn}$: $\kappa(H_{stn}) = \mathbf{14.93}$
   - Rear Camera $H_{stn}$: $\kappa(H_{stn}) = \mathbf{14.90}$
   - Left Camera $H_{stn}$: $\kappa(H_{stn}) = \mathbf{8.03}$
   - Right Camera $H_{stn}$: $\kappa(H_{stn}) = \mathbf{8.03}$

### 4.2 Clarification
The high condition number in pixel space is a standard artifact of uncentered, unscaled pixel coordinate systems in projective geometry. It does not indicate an ill-posed physical projection or a defective camera rig. Coordinate normalization in `homography_converter.py` restores standard conditioning ($\kappa \approx 8 - 15$).

---

## 5. Experiment 3 Re-Classification: Label Generation vs. Perception (Item A5)

### 5.1 Updated Formal Title
**Experiment 3: Ground-Truth Occlusion Label Generation Using 2D Camera-Ray Casting**

### 5.2 Functional Role Distinction
- **What Experiment 3 Actually Is:** An **offline data curation procedure** that takes an omniscient, unoccluded synthetic ground-truth top-down image (from a virtual drone camera in CARLA) and calculates which road-plane pixels *would be invisible* from ground-level vehicle cameras using 2D line-of-sight ray tracing. It injects a 10th semantic label (`occluded`, RGB `150, 150, 150`) into the training targets.
- **What Experiment 3 Is NOT:** It is not an online perception algorithm, real-time sensor processing pipeline, or live RGB-D visibility reasoning engine. It requires prior knowledge of the complete unoccluded environment to function.

---

## 6. Metric Attribution & Benchmark Separation (Item A6)

The mean Intersection-over-Union (mIoU) figures frequently referenced throughout the audit are **published benchmark metrics from the original IEEE ITSC 2020 paper (Table I)** evaluated across 3,731 validation scenes:
- Classical Homography (IPM Baseline): **30.17% mIoU** (Published claim)
- DeepLab MobileNetV2: **66.60% mIoU** (Published claim)
- DeepLab Xception: **71.35% mIoU** (Published claim)
- uNetXST (Multi-Input Proposed): **71.92% mIoU** (Published claim)

**Independent Reproduction Status:**
These mIoU benchmark values have **NOT** been independently evaluated or reproduced by our single-example test harness. Full reproduction of Table I is blocked until batch neural network inference is executed.

---

## 7. Finalized Audit Status Matrix (Item A7)

```text
+------------------------------------------------------------------------------------------------+
| Cam2BEV Upstream Audit & Classical Reproduction Final Classification Matrix                    |
+-------------------+---------------------------------------------+------------------------------+
| Subsystem         | Evaluation Classification                   | Justification / Reference    |
+-------------------+---------------------------------------------+------------------------------+
| Repository Audit  | Source Code Reviewed                        | Clean MIT license; 19 commits|
| Camera Geometry   | Numerical Result Independently Verified     | Homographies match to <10^-14|
| Classical IPM     | Experiment Executed & Verified (1 sample)   | Pixel-identical decoded match|
| Multi-Cam Stitch  | Experiment Executed & Verified (1 sample)   | Pixel-identical decoded match|
| Occlusion Labels  | Experiment Executed & Verified (1 sample)   | Pixel-identical decoded match|
| Paper Benchmark   | Published Result Referenced, Not Reproduced | Table I (71.92% mIoU)        |
| Neural Inference  | Work Blocked by Environment                 | Legacy TF 2.5 / Python 3.7   |
| Full Validation   | Work Not Started                            | Awaits neural batch runner   |
+------------------------------------------------------------------------------------------------+
```

With this addendum, the Cam2BEV classical reproduction audit is formally closed. No files within `research/repositories/Cam2BEV` were modified.
