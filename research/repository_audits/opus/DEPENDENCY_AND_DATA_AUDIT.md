# Dependency, Hardware, Dataset, Checkpoint & License Audit: OPUS / OPUS-V2

**Project:** ForesightBEV Research Audit 02 — Stage 0 Baseline Reverse Engineering  
**Target Repository:** `OPUS`  
**Repository URL:** [https://github.com/jbwang1997/OPUS](https://github.com/jbwang1997/OPUS)  
**Target Git Commit Audited:** `fdb17339f40168ea0ceda9615863e3f5d0c3f5ce` (`fdb1733`, HEAD -> main, origin/main)  
**Date:** November 2026  
**Auditor Mode:** Read-Only Static Verification & Dependency Compatibility Analysis

---

## 1. Executive Dependency Matrix

The OPUS repository represents an evolutionary fork of the **SparseBEV** and **SparseOcc** codebases (MCG-NJU, Nanjing University). Its environment dependencies are significantly more modern than SurroundOcc's (anchored in late 2022 / early 2023), but still carry specific framework version locks.

| Dependency | Upstream Specified Version | Verified Compatible Range | ForesightBEV Host / Virtual Env Status | Critical Issues / Breaking Factors |
| :--- | :--- | :--- | :--- | :--- |
| **Python** | `3.8` | `3.8` – `3.10` | Workspace: `Python 3.12+` | Modern build tools (`uv`) work best with Python 3.10–3.12. Python 3.8 is near end-of-life. |
| **PyTorch** | `1.13.1` | `1.12.0` – `2.1.x` | Modern PyTorch 2.x | PyTorch 1.13.1 is specified; however, the author recently stripped `mmdet3d` dependencies (`commit fdb1733`), improving compatibility with PyTorch 2.x. |
| **CUDA Toolkit** | `11.6` | `11.3` – `12.1` | N/A on Apple Silicon Host | Required to compile custom CUDA extensions (`msmv_sampling`, `lib/dvr`). |
| **MMCV** | `mmcv-full==1.6.0` | `1.5.0` – `1.7.2` | Incompatible with PyTorch 2.x | Requires precompiled CUDA wheels matching specific PyTorch and CUDA versions. |
| **MMDetection** | `mmdet==2.28.2` | `2.25.0` – `2.28.2` | Incompatible with MMDetection 3.x | Uses MMDetection 2.x API (`build_detector`, `build_loss`). |
| **MMDetection3D** | **Removed** (`commit fdb1733`) | Not required | Removed | **Crucial finding:** The author explicitly removed the `mmdet3d` dependency on May 3, 2026, eliminating a major historical friction point! |
| **spconv** | `spconv-cu116` | `2.1.x` – `2.3.x` | Fails on macOS / non-CUDA | Essential for OPUS-V2's Continuous Prediction Discretization (CPD) module (`SparseConv3d`, `SubMConv3d`). |
| **torch-scatter**| PyG binary wheels | `2.0.9` – `2.1.2` | Requires compilation on macOS | Required by `PFNLayer` (`scatter_max`). |
| **Custom CUDA 1** | `models/csrc` (`MSMV`) | Internal C++/CUDA | Fails on macOS / Apple Silicon | Multi-scale multi-view sampling kernel. **A pure PyTorch fallback exists** (`wrapper.py:17-37`). |
| **Custom CUDA 2** | `lib/dvr` (`DVR`) | Internal C++/CUDA | Fails on macOS / Apple Silicon | Differentiable Voxel Raycasting kernel for RayIoU evaluation. No PyTorch fallback provided. |
| **Optional IO** | `pyturbojpeg`, `pillow-simd` | Any recent | Optional | Used to accelerate JPEG image decoding. |

---

## 2. Hardware Requirements & Resource Footprint

### 2.1 Upstream Reference Training Environment
- **GPU Hardware:**
  - Single-GPU execution supported via `train.py` (scaled batch size).
  - Multi-GPU distributed training: 8 $\times$ NVIDIA RTX 3090 / A100 GPUs (configured in `dist_train.sh`).
- **Batch Size:** Config defaults specify `batch_size = 8` (scaled automatically across available GPUs, e.g., 1 sample per GPU on an 8-GPU node).
- **Epoch Count:**
  - OPUS-V1: 100 epochs (~3 to 4 days on 8 $\times$ RTX 3090).
  - OPUS-V2: 50 epochs (~1.5 to 2 days on 8 $\times$ RTX 3090) due to faster convergence from the CPD module.
- **Memory Footprint:**
  - Tiny variant (`r50`, image size $256 \times 704$, 8 sweeps): Requires approximately **12 GB to 16 GB VRAM per GPU** during training with FP16 enabled (`Fp16OptimizerHook`).
  - Large variant (`r50`, image size $512 \times 1408$): Requires **24 GB to 32 GB VRAM per GPU**.
- **Host RAM & Storage:**
  - Host RAM: $\ge 64\text{ GB}$ system RAM recommended for multi-sweep caching.
  - Storage: nuScenes full dataset ($400\text{ GB}$) + Occ3D-nuScenes GT labels ($50\text{ GB}$).

### 2.2 Feasibility on Apple Silicon (MacBook Air M5, 24 GB Unified Memory)
1. **Model Forward Pass & Inference Feasibility:**
   - **Feasible with PyTorch Fallback:** Unlike SurroundOcc, OPUS includes an explicit pure PyTorch fallback for its multi-view sampling operator (`msmv_sampling_pytorch` in `models/csrc/wrapper.py:17-37`), using standard `torch.nn.functional.grid_sample`.
   - **Blocker 1 (`spconv`):** OPUS-V2 requires `spconv` for its CPD module. `spconv` does not support Apple Silicon MPS out-of-the-box. OPUS-V1 does not use `spconv` (it uses `mmcv.ops.Voxelization`), making OPUS-V1 easier to run on CPU/MPS than OPUS-V2.
   - **Blocker 2 (`torch-scatter`):** Requires manual compilation for macOS ARM64.
   - **Blocker 3 (`lib/dvr`):** Differentiable Voxel Raycasting metric contains CUDA `.cu` code with no CPU fallback. Standard mIoU (`old_metrics.py`) can run on CPU, but RayIoU cannot.
2. **Training Feasibility on Mac:**
   - **Not Feasible:** Training an 8-frame temporal model across 50–100 epochs on unified memory without CUDA will result in unviable training runtimes (>2 weeks).

---

## 3. Dataset Audit: nuScenes, Occ3D-nuScenes & OpenOccupancy

OPUS supports two primary occupancy ground-truth benchmarks on top of the nuScenes base dataset:

### 3.1 Supported Dataset Variants
1. **Occ3D-nuScenes (Primary Benchmark):**
   - Implemented in `loaders/nuscenes_occ3d_dataset.py`.
   - Ingests `labels.npz` containing:
     - `semantics`: Dense grid of shape $(200, 200, 16)$.
     - `mask_camera`: Boolean frustum visibility mask $(200, 200, 16)$.
     - `mask_lidar`: Boolean LiDAR visibility mask $(200, 200, 16)$.
   - **Spatial Extents:** $[-40.0\text{ m}, -40.0\text{ m}, -1.0\text{ m}, +40.0\text{ m}, +40.0\text{ m}, +5.4\text{ m}]$.
   - **Voxel Size:** $\Delta x = 0.4\text{ m}, \Delta y = 0.4\text{ m}, \Delta z = 0.4\text{ m}$.
   - **Class Count:** 17 semantic classes (indices $0 \dots 16$), with class `17` designated as empty/free space (`empty_label = 17`).
2. **OpenOccupancy (Occ3D-OpenOccupancy):**
   - Implemented in `loaders/nuscenes_occupancy_dataset.py` and `configs/opusv2_nusc-occupancy/`.
   - Ingests `.npy` sparse coordinates and semantic labels.
   - **Spatial Extents:** $[-51.2\text{ m}, -51.2\text{ m}, -5.0\text{ m}, +51.2\text{ m}, +51.2\text{ m}, +3.0\text{ m}]$.
   - **Voxel Size:** $\Delta x = 0.2\text{ m}, \Delta y = 0.2\text{ m}, \Delta z = 0.2\text{ m}$ ($512 \times 512 \times 40$ voxels).

### 3.2 Preprocessing Requirement (`scripts/preprocess_nuscenes.py`)
- Standard MMDetection3D `.pkl` files (e.g. `nuscenes_infos_train.pkl`) are **incompatible** with OPUS.
- OPUS requires generating sweep-aware pickle files (`nuscenes_infos_train_sweep.pkl`) using `scripts/preprocess_nuscenes.py`. This script extracts inter-sweep relative ego poses (`ego2global`), timestamps, and calibration parameters across multi-frame sequences.

---

## 4. Checkpoints & Pretrained Weights Audit

| Checkpoint Identifier | Upstream Hosting Location | Target Configuration | Reported Benchmark Performance |
| :--- | :--- | :--- | :--- |
| `cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_...pth` | OpenMMLab Official CDN | ResNet-50 2D backbone weight | Pretrained 2D initialization on NuImages |
| `opusv1-t_r50_704x256_8f_nusc-occ3d_100e.pth` | Hugging Face (`jbwang1997/OPUS`) | OPUS-V1 Tiny (8 frames, 100e) | $36.2\text{ mIoU}, \, 41.2\text{ RayIoU}$ |
| `opusv1-l_r50_704x256_8f_nusc-occ3d_100e.pth` | Hugging Face (`jbwang1997/OPUS`) | OPUS-V1 Large (8 frames, 100e) | $38.2\text{ mIoU}, \, 42.8\text{ RayIoU}$ |
| `opusv2-t_r50_704x256_8f_nusc-occ3d_50e.pth` | Hugging Face (`jbwang1997/OPUS`) | OPUS-V2 Tiny (8 frames, 50e) | $38.6\text{ mIoU}, \, 44.0\text{ RayIoU}$ |
| `opusv1-fusion-t_r50_704x256_8f_...pth` | Hugging Face (`jbwang1997/OPUS`) | OPUS-V1 Fusion (Camera + LiDAR) | $51.4\text{ mIoU}, \, 51.8\text{ RayIoU}$ |

> [!NOTE] Checkpoint Hosting Advantage
> Unlike SurroundOcc, which hosts checkpoints on Baidu Netdisk (requiring Chinese SMS verification), OPUS hosts all official model checkpoints on **Hugging Face** (`https://huggingface.co/jbwang1997/OPUS`), providing direct, unauthenticated programmatic access via HTTPS.

---

## 5. Intellectual Property & License Audit

- **Repository License:** **MIT License** (Copyright 2023 Multimedia Computing Group, Nanjing University).
  - Grants permissive rights to use, copy, modify, merge, publish, distribute, sublicense, and sell copies without commercial restriction.
  - Fully compatible with ForesightBEV's licensing and internal development rules.
- **Third-Party Code Lineage:**
  - Inherits multi-view camera sampling modules from **SparseBEV** (`github.com/MCG-NJU/SparseBEV`) and **SparseOcc** (`github.com/MCG-NJU/SparseOcc`), both developed by the same research laboratory at Nanjing University under MIT licenses.
  - Differentiable Voxel Raycasting kernel (`lib/dvr`) is adapted from Khurana et al. (*4D Occupancy Forecasting*, ECCV 2022).

