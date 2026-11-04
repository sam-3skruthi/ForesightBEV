# Controlled Reproduction Plan: Cam2BEV

**Project:** ForesightBEV Research Audit 02
**Target Repository:** `Cam2BEV` (RWTH Aachen University)
**Host Architecture:** Apple Silicon M5 (24 GB RAM, macOS 26.6)
**Auditor Mode:** Controlled Step-by-Step Experimental Specification
**Review Status:** Formal reproduction audit closed with methodological addendum: [`REPRODUCTION_REVIEW_ADDENDUM.md`](REPRODUCTION_REVIEW_ADDENDUM.md)

---

## 1. Reproduction Strategy & Environment Matrix

To isolate upstream dependency conflicts (Python 3.7 / TensorFlow 2.5 vs macOS Apple Silicon), reproduction is partitioned into two distinct environments:

1. **Environment A (Native macOS M5 / Python 3.12 via `uv`):**
   Executes classical computer vision, camera geometry, perspective projection, multi-camera stitching, occlusion ray casting, and evaluation metric scoring.
2. **Environment B (Containerized Linux x86-64 or Cloud GPU / Python 3.7 + TF 2.5):**
   Executes legacy TensorFlow graph operations, pretrained checkpoint inference, and neural training. Alternatively, an **Environment C (PyTorch MPS Port)** enables native M5 neural execution.

---

## 2. Sequence of Controlled Experiments & Verification Status

| Experiment | Target Functionality | Environment | Status | Verification Metric |
|---|---|---|---|---|
| **Experiment 1A** | Analytical IPM Homography Calculation | Environment A (Native M5) | **PASS** | $3 \times 3$ matrices match documentation reference $<10^{-14}$ |
| **Experiment 1B** | Actual Perspective Warping (4 cameras) | Environment A (Native M5) | **PASS** | Individual warps generated in `ipm_outputs/` without errors |
| **Experiment 2** | 360° Multi-Camera Seam Stitching | Environment A (Native M5) | **PASS** | **100.0000% exact bit-for-bit match** to official `homography.png` |
| **Experiment 3** | Ground-Truth Occlusion Ray Casting | Environment A (Native M5) | **PASS** | **100.0000% exact bit-for-bit match** to official `bev+occlusion.png` |
| **Experiment 4** | uNetXST Neural Network Inference | Environment B (Docker / PyTorch) | **BLOCKED (Natively)** | Legacy TF 2.5 has no arm64 wheel; requires Docker or PyTorch port |
| **Experiment 5** | Official Evaluation Metric Reproduction | Environment A / B | **NOT STARTED** | Awaits batch model inference on 3,731 validation images |
| **Experiment 6** | Comparison: IPM vs Learned Semantic BEV | Environment A | **PARTIAL** | IPM vs GT comparison complete; learned BEV awaits Exp 4 |

---

### Experiment 1A: Analytical IPM Homography Calculation

- **Status:** **PASS** (Verified in Session 1)
- **Objective:** Compute analytical Inverse Perspective Mapping (IPM) homographies using `preprocessing/ipm/ipm.py -v`.
- **Verified Homographies:** Saved in [`experiments/cam2bev_ipm/homographies_1_FRLR.txt`](../../../experiments/cam2bev_ipm/homographies_1_FRLR.txt).

---

### Experiment 1B: Reproduce Classical IPM Transformation

- **Status:** **PASS** (Verified in Session 3)
- **Objective:** Verify analytical Inverse Perspective Mapping (IPM) calculation and perspective warping for a single vehicle camera against ground-truth road geometry.
- **Input Data:**
  - Front camera config: [preprocessing/camera_configs/1_FRLR/front.yaml](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/preprocessing/camera_configs/1_FRLR/front.yaml)
  - Drone camera config: [preprocessing/camera_configs/1_FRLR/drone.yaml](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/preprocessing/camera_configs/1_FRLR/drone.yaml)
  - Sample front perspective image (e.g. from `1_FRLR` validation sample or reference asset).
- **Exact Command:**
  ```bash
  cd preprocessing/ipm
  python3 ipm.py --cc \
      --drone ../camera_configs/1_FRLR/drone.yaml \
      ../camera_configs/1_FRLR/front.yaml /path/to/front_sample.png \
      --output /path/to/exp1_output
  ```
- **Mathematical Operation:**
  1. $K_{front}, R_{front}, t_{front} \implies P_{front} = K [R \mid t]$
  2. Compute road mapping matrix $M$ from drone resolution ($604 \times 964$) and $pxPerM = (13.65, 13.65)$
  3. $H_{ipm} = (P_{front} M)^{-1}$
  4. Backward bilinear/nearest-neighbor warping: $I_{bev}(u, v) = I_{cam}(H_{ipm} [u, v, 1]^T)$
  5. Angular field-of-view masking: $\theta \in [\text{yaw} - 90^\circ, \text{yaw} + 90^\circ]$
- **Expected Output:**
  A $604 \times 964$ RGB image where flat road surfaces are rectilinearly aligned with top-down coordinates, and vertical 3D objects (cars, poles) are radially distorted outward along line-of-sight rays.
- **Validation Method:**
  Pixel-level comparison against upstream preprocessed homography reference images. Verify that road markings maintain constant metric lane width.
- **Dependencies:** Python 3.10+, NumPy, OpenCV (`opencv-python`), PyYAML.
- **Hardware Requirement:** Native Apple Silicon M5 (negligible memory/CPU footprint, runtime $<50\text{ ms}$).
- **Known Limitations:** Ground plane assumption ($Z=0$). Elevated objects undergo severe perspective elongation.

---

### Experiment 2: Reproduce Multi-Camera Stitching

- **Status:** **PASS** (Verified in Session 4; 100.0000% exact bit-for-bit match to official `homography.png`)
- **Objective:** Verify multi-camera IPM stitching combining 4 surround camera feeds (Front, Rear, Left, Right) into a seamless 360° composite homography.
- **Input Data:**
  - 4 camera configs: `front.yaml`, `rear.yaml`, `left.yaml`, `right.yaml`
  - Drone config: `drone.yaml`
  - 4 synchronized perspective images (`front.png`, `rear.png`, `left.png`, `right.png`)
- **Exact Command:**
  ```bash
  cd preprocessing/ipm
  python3 ipm.py --cc \
      --drone ../camera_configs/1_FRLR/drone.yaml \
      ../camera_configs/1_FRLR/front.yaml /path/to/front.png \
      ../camera_configs/1_FRLR/rear.yaml  /path/to/rear.png \
      ../camera_configs/1_FRLR/left.yaml  /path/to/left.png \
      ../camera_configs/1_FRLR/right.yaml /path/to/right.png \
      --output /path/to/exp2_output
  ```
- **Mathematical Operation:**
  1. Compute independent homographies $H_k = (P_k M)^{-1}$ for $k \in \{F, R, L, R\}$.
  2. For each camera, apply directional FOV clipping mask:
     $$\text{mask}_k(j, i) = \mathbb{I}\left(90^\circ < |\theta(j, i) - \text{yaw}_k| < 270^\circ\right)$$
  3. Combine non-zero warped regions into composite bird's-eye canvas:
     $$I_{stitched}(j, i) = \bigcup_k I_{warped, k}(j, i)$$
- **Expected Output:**
  A unified $604 \times 964$ 360° homography image matching the format of [preprocessing/ipm/assets/example.png](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/preprocessing/ipm/assets/example.png).
- **Validation Method:**
  Structural similarity and visual inspection of overlap boundaries at $\pm 45^\circ$ diagonal seams.
- **Dependencies:** Python 3.10+, NumPy, OpenCV, PyYAML.
- **Hardware Requirement:** Native Apple Silicon M5 (runtime $<150\text{ ms}$).
- **Known Limitations:** Optical seam discontinuities occur when tall obstacles cross camera overlap boundaries due to parallax.

---

### Experiment 3: Reproduce Occlusion Ray Casting

- **Status:** **PASS** (Verified in Session 5; 100.0000% exact bit-for-bit match to official `bev+occlusion.png`)
- **Objective:** Verify the 2D line-of-sight ray casting algorithm that generates the `occluded` ground-truth class on top-down drone segmentations.
- **Input Data:**
  - Reference synthetic drone image: [preprocessing/occlusion/assets/example-original.png](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/preprocessing/occlusion/assets/example-original.png)
  - Drone config: `drone.yaml`
  - 4 camera configs: `front.yaml`, `rear.yaml`, `left.yaml`, `right.yaml`
- **Exact Command:**
  ```bash
  cd preprocessing/occlusion
  python3 occlusion.py \
      assets/example-original.png \
      ../camera_configs/1_FRLR/drone.yaml \
      ../camera_configs/1_FRLR/front.yaml \
      ../camera_configs/1_FRLR/rear.yaml \
      ../camera_configs/1_FRLR/left.yaml \
      ../camera_configs/1_FRLR/right.yaml \
      --output exp3_output/
  ```
- **Mathematical Operation:**
  1. Initialize output grid to occluded color RGB $(150, 150, 150)$.
  2. Compute camera sensor origins $O_k$ in drone pixel coordinates.
  3. Cast Bresenham lines $\mathcal{L}(O_k, B_m)$ to all perimeter boundary pixels $B_m$ within each camera's FOV.
  4. Trace outward: copy pixels to output until hitting a blocking class (`car`, `truck`, `building`, `wall`).
  5. Apply special car rules: first car in ray line is preserved via OpenCV flood fill; subsequent objects are masked as occluded unless labeled as tall vegetation.
  6. Re-insert ego vehicle at center origin.
- **Expected Output:**
  An occluded semantic BEV image identical to [preprocessing/occlusion/assets/example-occluded.png](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/preprocessing/occlusion/assets/example-occluded.png).
- **Validation Method:**
  Direct pixel-difference computation against `example-occluded.png` ($0$ pixel mismatch expected).
- **Dependencies:** Python 3.10+, NumPy, OpenCV, `scikit-image`, PyYAML.
- **Hardware Requirement:** Native Apple Silicon M5 (runtime $\sim 2\text{ s}$ per image single-threaded).
- **Known Limitations:** Computationally slow in pure Python for large datasets; 3D height variation is approximated by 2D heuristics.

---

### Experiment 4: Reproduce uNetXST Neural Inference

- **Objective:** Execute forward inference of the trained `uNetXST` network on 4 perspective camera inputs and produce a semantically segmented BEV output.
- **Input Data:**
  - 4 perspective validation images: `front.png`, `rear.png`, `left.png`, `right.png`
  - Model configuration: [model/config.1_FRLR.unetxst.yml](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/model/config.1_FRLR.unetxst.yml)
  - Pre-computed homographies: [preprocessing/homography_converter/uNetXST_homographies/1_FRLR.py](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/preprocessing/homography_converter/uNetXST_homographies/1_FRLR.py)
  - Trained checkpoint weights: `best_weights.hdf5` (sourced from audited student checkpoint or trained model).
- **Exact Command:**
  ```bash
  cd model
  # Executed within Environment B (Linux container) or via PyTorch translated runner
  python3 predict.py \
      -c config.1_FRLR.unetxst.yml \
      --model-weights /path/to/best_weights.hdf5 \
      --prediction-dir output/Predictions/
  ```
- **Mathematical Operation:**
  1. Encode 4 images to one-hot tensors: $(4, 256, 512, 10)$.
  2. Forward pass through 4 separate 5-stage U-Net encoders.
  3. At each depth $d \in \{0..4\}$, transform intermediate features with constant $3 \times 3$ matrix $H_k$:
     $$\tilde{F}_{k}^{(d)}(x, y) = \text{BilinearSample}\left(F_{k}^{(d)}, H_k [x, y, 1]^T\right)$$
  4. Concatenate transformed features along channels: $(B, H_d, W_d, 4 \times C_d)$.
  5. Decode through 5 transpose convolution stages with skip connections.
  6. Final $1 \times 1$ conv + softmax: $(B, 256, 512, 10)$.
  7. Argmax along channel axis and color decode using `convert_9+occl.xml`.
- **Expected Output:**
  A $256 \times 512$ RGB semantic segmentation PNG where cars and obstacles appear rectilinearly corrected at their true metric footprint locations, and occluded shadow wedges are segmented.
- **Validation Method:**
  Visual comparison with ground truth `bev+occlusion` image.
- **Dependencies:** TensorFlow 2.5 (Environment B) OR PyTorch 2.x (Environment C).
- **Hardware Requirement:** Environment B (Docker CPU or Cloud GPU) or native M5 via PyTorch MPS.
- **Known Limitations:** Upstream repo provides no weights; requires checkpoint from audited derivative or training run.

---

### Experiment 5: Reproduce Official Evaluation Metrics

- **Objective:** Compute class-wise Intersection-over-Union (IoU) and mean IoU (mIoU) on the validation dataset and compare against the published results in ITSC 2020 paper Tables I and II.
- **Input Data:**
  - Full `1_FRLR` validation dataset (3,731 samples).
  - Model predictions from `uNetXST`, `DeepLab Xception`, and classical `Homography (IPM)`.
- **Exact Command:**
  ```bash
  cd model
  python3 evaluate.py \
      -c config.1_FRLR.unetxst.yml \
      --model-weights /path/to/best_weights.hdf5
  ```
- **Mathematical Operation:**
  1. Accumulate confusion matrix $C \in \mathbb{R}^{10 \times 10}$:
     $$C_{i, j} = \sum_{p} \mathbb{I}(y_{true}(p) = i \land y_{pred}(p) = j)$$
  2. Compute per-class IoU:
     $$\text{IoU}_i = \frac{C_{i, i}}{\sum_{j} C_{i, j} + \sum_{j} C_{j, i} - C_{i, i}}$$
  3. Compute mean IoU: $\text{mIoU} = \frac{1}{10} \sum_{i=1}^{10} \text{IoU}_i$
- **Expected Numerical Baselines (from Paper Tables I & II):**
  - **uNetXST:** **71.92% mIoU**
  - **DeepLab Xception:** **71.35% mIoU**
  - **DeepLab MobileNetV2:** **66.60% mIoU**
  - **Classical Homography (IPM Baseline):** **30.17% mIoU**
  - Class-wise expectations: High performance ($>85\%$) on `road` and `vegetation`; moderate performance on `car` and `occluded`; lower performance on small objects (`person`, `bike`).
- **Validation Method:**
  Direct numerical comparison against published paper metrics within a tolerance margin ($\pm 1.5\%$ allowing for evaluation batching and floating-point variations).
- **Dependencies:** Python, NumPy, Pandas, Seaborn, Matplotlib.
- **Hardware Requirement:** Native M5 or Environment B.
- **Known Limitations:** Evaluating all 3,731 validation images requires prior execution of batch inference.

---

### Experiment 6: Quantitative & Qualitative Comparison of IPM vs Learned BEV

- **Objective:** Directly compare the classical IPM output (Variation 1 input) with the deep neural BEV prediction (Variation 2 output) to demonstrate 3D object correction and occlusion modeling.
- **Input Data:**
  - Stitched IPM image from Experiment 2.
  - Predicted semantic BEV from Experiment 4.
  - Ground-truth occluded BEV from validation set.
- **Exact Command:**
  Execute comparison analysis script:
  ```bash
  python3 compare_bev.py \
      --ipm /path/to/exp2_stitched.png \
      --pred /path/to/exp4_pred.png \
      --gt /path/to/gt_bev_occl.png \
      --out comparison_grid.png
  ```
- **Mathematical Operation:**
  Calculate class-wise false positive / false negative masks for the vehicle class:
  $$\Delta_{\text{distortion}} = |I_{\text{ipm}} - I_{\text{gt}}| - |I_{\text{unetxst}} - I_{\text{gt}}|$$
- **Expected Output:**
  Side-by-side visual comparison and quantitative demonstration that:
  1. IPM exhibits severe radial distortion of upright vehicle bodies, resulting in $\sim 30.17\%$ mIoU.
  2. uNetXST reconstructs the true orthogonal vehicle footprint and predicts occlusion shadows, reaching $\sim 71.92\%$ mIoU ($+41.75\%$ absolute improvement over IPM).
- **Validation Method:**
  Quantitative IoU delta and visual artifact inspection.
- **Dependencies:** NumPy, OpenCV, Matplotlib.
- **Hardware Requirement:** Native Apple Silicon M5.
