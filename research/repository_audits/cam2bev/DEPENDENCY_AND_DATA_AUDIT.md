# Dependency and Data Audit: Cam2BEV

**Target Repository:** `Cam2BEV` (RWTH Aachen University, Institute for Automotive Engineering - ika)
**Associated Publication:** *"A Sim2Real Deep Learning Approach for the Transformation of Images from Multiple Vehicle-Mounted Cameras to a Semantically Segmented Image in Bird’s Eye View"* (IEEE ITSC 2020)
**Audit Date:** November 2026
**Auditor Mode:** Read-Only Static Reverse-Engineering & Source Verification

---

## 1. Upstream Dependency Analysis

### 1.1 Dependency Evolution & The "Lambda Layer" Defect

The upstream repository contains two distinct dependency manifests reflecting an attempted framework upgrade and subsequent emergency rollback:

```text
Cam2BEV
├── requirements.original.txt      # Frozen dependencies from ITSC 2020 publication
└── requirements.txt               # Current downgraded dependencies (Jan 2022)
```

#### Commit History & Technical Root Cause

1. **Original Research State (May 2020, Commit `07919b1`):**
   The paper's experiments were conducted on **Python 3.7** with **TensorFlow 2.1.0** and **CUDA 10.1** on Linux x86_64.
2. **Upgrade Attempt (Nov 2021, Commit `dbd5a4fb`):**
   An upgrade was made to **Python 3.8** and **TensorFlow 2.7.0**.
3. **Rollback & Downgrade (Jan 2022, Commit `bfcaa289`):**
   Author Lennart Reiher reverted the upgrade, downgrading to **Python 3.7** and **TensorFlow 2.5.0** with the commit message:
   > *"downgrade to TensorFlow 2.5 and Python 3.7
   > The DeepLab implementations were broken with TensorFlow 2.7 due to non-trainable Keras lambda layers."*

This failure stems from the third-party implementation in [model/architecture/third_party/deeplabv3plus.py](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/model/architecture/third_party/deeplabv3plus.py). DeepLabv3+ relies on internal `tensorflow.python.keras` imports and `tf.keras.layers.Lambda` operations for image resizing and bilinear upsampling. In TensorFlow $\ge 2.6$, the Keras codebase was separated from the core TensorFlow repository, introducing changes to functional graph tracing that prevented lambda layers from properly handling backpropagation and model serialization.

### 1.2 Comparison of Dependency Manifests

| Package | `requirements.original.txt` (TF 2.1) | `requirements.txt` (TF 2.5) | Role in Codebase |
|---|---|---|---|
| `tensorflow` | `2.1.0` | `2.5.0` | Core deep learning framework |
| `opencv-python` | `4.2.0.32` | `4.2.0.32` | Perspective warping, image I/O, flood fill |
| `scikit-image` | `0.16.2` | `0.16.2` | Bresenham line drawing (`skimage.draw.line`) in `occlusion.py` |
| `PyYAML` | `5.3` | `5.3` | Camera YAML configuration parser |
| `ConfigArgParse` | `1.1` | `1.1` | CLI and YAML configuration parsing in training/eval |
| `pandas` | `1.0.2` | `1.0.2` | Confusion matrix DataFrame formatting in `evaluate.py` |
| `seaborn` | `0.10.0` | `0.10.0` | Confusion matrix heatmap plotting in `evaluate.py` |
| `matplotlib` | `3.2.0` | `3.2.0` | Visualization and PDF figure export in `evaluate.py` |
| `tqdm` | `4.43.0` | `4.43.0` | Progress bars across batch preprocessing and training |
| `numpy` | `1.18.1` | *(implied by TF)* | Array operations, matrix transformations, homography math |
| `Keras-Applications` | `1.0.8` | *(omitted)* | Legacy Keras model support |
| `Keras-Preprocessing` | `1.1.0` | *(omitted)* | Legacy Keras image utilities |

---

## 2. Hardware Compatibility Audit: Apple Silicon (MacBook Air M5)

### 2.1 Hardware Profile
- **SoC:** Apple Silicon M5
- **Unified Memory:** 24 GB
- **Operating System:** macOS 26.6 (Darwin 25.x)
- **Host Tools:** Native Python 3.12, system Python 3.9.6, `uv 0.13.4` (aarch64)

### 2.2 Compatibility Assessment by Pipeline Stage

| Pipeline Component | Upstream Requirements | Native macOS M5 Viability | Barriers & Constraints | Recommended Workaround |
|---|---|---|---|---|
| **Classical IPM Preprocessing** (`preprocessing/ipm/ipm.py`) | Python, NumPy, OpenCV, PyYAML | **100% FEASIBLE** | None. Pure mathematical and OpenCV operations run natively on Python 3.12. | Run directly with native virtual environment (`uv venv`). |
| **Occlusion Ray Casting** (`preprocessing/occlusion/occlusion.py`) | Python, NumPy, OpenCV, skimage, PyYAML | **100% FEASIBLE** | Multiprocessing on macOS requires `spawn` (default in Python 3.8+). | Runs natively on M5 CPU cores. |
| **Homography Conversion** (`preprocessing/homography_converter/*`) | Python, NumPy, argparse | **100% FEASIBLE** | None. Simple matrix arithmetic. | Runs natively. |
| **Dataset Inspection & Conversion** (`model/utils.py`) | Python, NumPy, OpenCV, ElementTree | **100% FEASIBLE** | None. | Runs natively. |
| **uNetXST Inference** (`model/predict.py`) | TensorFlow 2.5 / Keras | **BLOCKED NATIVELY** | No native macOS arm64 wheel exists for legacy TensorFlow 2.5. Python 3.7 is end-of-life and unavailable natively on Apple Silicon. | Run CPU inference in Linux x86-64 container via Docker/UTM, or port to PyTorch / Keras 3. |
| **DeepLab Inference** (`model/predict.py`) | TensorFlow 2.5 / Keras | **BLOCKED NATIVELY** | Incompatible with modern TensorFlow due to lambda layer breaking changes. | Run in Linux x86-64 container or port to modern PyTorch. |
| **Model Training** (`model/train.py`) | TensorFlow 2.5, CUDA 10.1 | **BLOCKED NATIVELY** | Apple Silicon does not support NVIDIA CUDA. Running training on emulated x86_64 CPU is computationally intractable for 33k samples. | Train on dedicated Linux x86-64 GPU machine (NVIDIA / CUDA), or rewrite training loop in PyTorch with Metal (`mps`) backend. |
| **Evaluation Metrics** (`model/evaluate.py`) | TensorFlow, NumPy, Seaborn, Pandas | **PARTIALLY BLOCKED** | Metric calculations require model inference. Once inference predictions are saved, metric script runs natively. | Run inference off-target or via container; run analysis script natively. |

---

## 3. Data Audit & GitLab Repository Inspection

### 3.1 Official Data Repository Overview

The official Cam2BEV datasets are hosted on RWTH Aachen University's institutional GitLab instance:
- **Repository URL:** [https://gitlab.ika.rwth-aachen.de/cam2bev/cam2bev-data](https://gitlab.ika.rwth-aachen.de/cam2bev/cam2bev-data)
- **Source Environment:** Generated synthetically using the open-source CARLA autonomous driving simulator under various urban weather and traffic scenarios.
- **Palette Convention:** Ground-truth semantic segmentation images follow Cityscapes RGB color encoding.

### 3.2 Ingestion Script Inspection (`data/download.sh`)

Source inspection of [data/download.sh](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/data/download.sh) reveals the archive hierarchy:

```bash
URL="https://gitlab.ika.rwth-aachen.de/cam2bev/cam2bev-data/-/archive/master/cam2bev-data-master.tar.gz"
FILE="cam2bev-data.tar.gz"
wget -q --show-progress -c -O $FILE $URL
tar -xzf $FILE
mv cam2bev-data-master/* .
for f in $(find . -name "*.tar.gz"); do
  tar -xzf $f -C "$(dirname $f)"
  rm $f
done
```

- **Download Footprint:** Compressed master archive is **~3.7 GB**.
- **Extracted Footprint:** Uncompressed datasets require **~7.7 GB** of disk space.
- **Archive Topology:** The master archive contains nested `.tar.gz` archives for individual view streams and dataset splits.

### 3.3 Dataset Topology & Split Breakdown

#### Dataset 1: `1_FRLR` (360° Surround View Rig)
- **Configuration:** 4 vehicle-mounted cameras (Front, Rear, Left, Right) positioned around the ego vehicle, plus 1 virtual top-down drone camera centered directly above the vehicle ($Z=50\text{ m}$).
- **Training Samples:** 33,199 samples
- **Validation Samples:** 3,731 samples
- **Directory Layout:**
  ```text
  data/1_FRLR/
  ├── train/
  │   ├── front/                   # Perspective RGB semantic segmentation (604 x 964)
  │   ├── rear/                    # Perspective RGB semantic segmentation (604 x 964)
  │   ├── left/                    # Perspective RGB semantic segmentation (604 x 964)
  │   ├── right/                   # Perspective RGB semantic segmentation (604 x 964)
  │   ├── bev/                     # Raw top-down drone segmentation (centered above ego)
  │   ├── bev+occlusion/           # Preprocessed ground truth with occluded areas (Class 10)
  │   └── homography/              # Preprocessed stitched IPM image (256 x 512)
  └── val/
      ├── front/, rear/, left/, right/
      ├── bev/
      ├── bev+occlusion/
      └── homography/
  ```

#### Dataset 2: `2_F` (Single Front Camera Rig)
- **Configuration:** 1 vehicle-mounted frontal camera, plus 1 virtual drone camera offset ahead of the vehicle ($X=28.84\text{ m}, Z=100\text{ m}$) producing a forward-looking BEV plane.
- **Input Resolution:** Perspective images are $1216 \times 1936$; drone images are $968 \times 1936$.
- **Target Classes:** Coarse 4-class segmentation (`road`, `vehicle`, `obstacle`, `occluded`).
- **Directory Layout:**
  ```text
  data/2_F/
  ├── train/
  │   ├── front/                   # Front perspective semantic segmentation
  │   ├── bev/                     # Raw drone segmentation
  │   ├── bev+occlusion/           # Preprocessed occluded ground truth
  │   └── homography/              # Single-camera IPM image
  └── val/
      ├── front/, bev/, bev+occlusion/, homography/
  ```

### 3.4 Preprocessed Outputs Provided by Upstream

To save users dozens of hours of preprocessing compute, the GitLab repository provides pre-computed artifacts:
1. `bev+occlusion`: Pre-computed 2D ray casting labels eliminating the need to run `preprocessing/occlusion/occlusion.py` across 36,930 samples.
2. `homography`: Pre-computed classical IPM transformed and stitched images eliminating the need to run `preprocessing/ipm/ipm.py`.

---

## 4. Legal & Licensing Audit

### 4.1 Upstream Source Code License: MIT License

The source code hosted on GitHub (`ika-rwth-aachen/Cam2BEV`) is licensed under the **MIT License** ([LICENSE:1-21](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/LICENSE#L1-L21)):
- **Permissions:** Commercial use, modification, distribution, private use.
- **Conditions:** Inclusion of original copyright notice (`Copyright 2020 Institute for Automotive Engineering of RWTH Aachen University`) and license text in all substantial copies.
- **Third-Party Code:**
  - `model/architecture/third_party/spatial_transformer.py`: MIT License (Octavio Arriaga).
  - `model/architecture/third_party/deeplabv3plus.py`: MIT License (Emil Zakirov).

### 4.2 Dataset License: Restrictive Non-Commercial Research License

The dataset hosted on RWTH Aachen GitLab (`cam2bev-data`) is **NOT** licensed under the MIT License. It is governed by a custom institutional non-commercial research agreement:

| Licensing Dimension | Dataset Term / Condition | Impact on ForesightBEV |
|---|---|---|
| **Commercial Use** | **STRICTLY PROHIBITED.** The dataset is provided solely for academic research, education, and scientific publication. | Cam2BEV synthetic raw data cannot be bundled into a commercial product. |
| **Redistribution** | **STRICTLY PROHIBITED.** Users may not mirror, redistribute, or publicly host the raw or modified dataset archives. | ForesightBEV must not re-host or publish Cam2BEV data archives in public repos. |
| **Derivative Models** | **PERMITTED.** Machine learning model weights trained on the data may be distributed, provided the raw data cannot be reconstructed. | Weights trained on Cam2BEV can be utilized for research benchmarking. |
| **Mandatory Citation** | **REQUIRED.** Any publication or derived work using the data must cite the ITSC 2020 paper. | Compliance verified; CITATION metadata documented. |
| **Warranty** | Data provided "AS IS" without liability or operational guarantee. | Expected standard for academic releases. |
