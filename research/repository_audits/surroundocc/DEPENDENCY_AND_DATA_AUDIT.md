# Dependency, Hardware, Dataset, Checkpoint & License Audit: SurroundOcc

**Project:** ForesightBEV Research Audit 01 — Stage 0 Baseline Reverse Engineering  
**Target Repository:** `SurroundOcc`  
**Repository URL:** [https://github.com/weiyithu/SurroundOcc](https://github.com/weiyithu/SurroundOcc)  
**Target Git Commit Audited:** `419bf5b47ee07ef0610950cce5ba99168c506753` (`419bf5b`, HEAD -> main, origin/main)  
**Date:** November 2026  
**Auditor Mode:** Read-Only Static Verification & Dependency Compatibility Analysis

---

## 1. Executive Dependency Matrix

SurroundOcc was developed during the late 2022 to mid-2023 timeframe and is anchored to the OpenMMLab 1.x ecosystem (`mmdet==2.14.0`, `mmcv-full==1.4.0`, `mmdetection3d==v0.17.1`). This ecosystem exhibits strict pin requirements and binary compatibility constraints.

| Component | Upstream Specified Version | Verified Compatible Range | ForesightBEV Host / Virtual Env Status | Critical Issues / Breaking Factors |
| :--- | :--- | :--- | :--- | :--- |
| **Python** | `3.7` | `3.7` – `3.8` | Current workspace: `Python 3.12+` | Python 3.7 is End-Of-Life; modern tools (`uv`, newer PyTorch) do not support 3.7. |
| **PyTorch** | `1.10.1` | `1.9.0` – `1.10.2` | Current workspace: PyTorch 2.x | PyTorch 2.x broke internal C++ extension APIs used by MMCV 1.4.0. |
| **CUDA Toolkit** | `11.1` | `11.1` – `11.3` | N/A (macOS Host) | Hard requirement for compiling custom CUDA kernels (`chamfer`, `ms_deform_attn`). |
| **Compiler (GCC)**| `gcc-6` / `gcc-7` | `gcc <= 7.5` | Apple Clang (macOS) | GCC 6/7 required by PyTorch 1.10 + CUDA 11.1 build scripts. Clang cannot compile NVIDIA CUDA kernels. |
| **MMCV** | `mmcv-full==1.4.0` | `1.4.0` – `1.5.0` | Incompatible with modern PyTorch | Requires prebuilt CUDA wheels or manual compilation with CUDA 11.1. |
| **MMDetection** | `mmdet==2.14.0` | `2.14.0` – `2.19.0` | Incompatible with modern MMDetection 3.x | Pre-OpenMMLab 2.0 architecture and API structure. |
| **MMSegmentation**| `mmsegmentation==0.14.1`| `0.14.1` – `0.16.0` | Incompatible | Legacy OpenMMLab segmentation API. |
| **MMDetection3D** | `v0.17.1` (from source)| Commit `v0.17.1` | Incompatible | Strict requirement on `v0.17.1` checkout (`setup.py install`). |
| **Open3D** | `open3d-python` / `open3d` | `0.13.0` – `0.16.0` | Open3D installed on modern macOS | Used for Poisson Surface Reconstruction (`create_from_point_cloud_poisson`). |
| **Chamfer Dist** | `extensions/chamfer_dist` | Internal commit | Unbuilt | Custom C++/CUDA extension; fails to compile on non-NVIDIA / macOS platforms. |

---

## 2. Hardware Requirements & Resource Footprint

### 2.1 Upstream Reference Training Environment
- **GPU Hardware:** 8 $\times$ NVIDIA GeForce RTX 3090 (24 GB VRAM each)
- **Batch Size:** `samples_per_gpu = 1` across 8 GPUs (effective global batch size = 8)
- **Host RAM:** $\ge 64\text{ GB}$ system RAM recommended (nuScenes dataset loading and caching)
- **Training Wallclock:** Approximately **2.5 days (60 hours)** for 24 epochs on 8 $\times$ RTX 3090 GPUs
- **Storage Requirement:**
  - nuScenes raw v1.0 data: $\sim 400\text{ GB}$ (samples, sweeps, maps)
  - Pre-generated dense occupancy labels (200 $\times$ 200 $\times$ 16): $\sim 3.8\text{ GB}$
  - Optional full-resolution Poisson mesh vertices: $\sim 204\text{ GB}$ ($170\text{ GB}$ train + $34\text{ GB}$ val)
  - Work dirs and checkpoints: $\sim 25\text{ GB}$

### 2.2 Apple Silicon (macOS) Incompatibility Audit
SurroundOcc **cannot execute training or model inference directly on Apple Silicon / macOS** without substantial architectural reimplementation:
1. **CUDA Extensions:**
   - `projects/mmdet3d_plugin/surroundocc/modules/multi_scale_deformable_attn_function.py` directly binds to MMCV's `ms_deform_attn_forward` CUDA kernel. While a pure PyTorch fallback `multi_scale_deformable_attn_pytorch` exists in the codebase (`spatial_cross_attention.py:395`), it is extremely slow for 80,000 queries.
   - `extensions/chamfer_dist` consists of pure NVIDIA CUDA `.cu` source code (`chamfer.cu`, `chamfer_cuda.cpp`). It has no MPS (Metal Performance Shaders) or CPU implementation. Non-semantic evaluation (`evaluation_reconstruction`) and ground-truth semantic transfer cannot run without it.
2. **Caffe-style ResNet101 Backbone:**
   - Relies on Deformable Convolutions v2 (`DCNv2`), which require compiled CUDA kernels in MMCV.
3. **MMDetection3D v0.17.1 Compatibility:**
   - Pre-dates PyTorch Apple Silicon MPS backend support; builds expect NVIDIA NVCC compiler.

---

## 3. Dataset Audit: nuScenes & Dense Occupancy Ground Truth

### 3.1 Raw Data Ingestion
- **Dataset:** nuScenes full dataset (`v1.0-trainval` and `v1.0-test`).
- **Required Sensors:** 6 surround cameras (`CAM_FRONT`, `CAM_FRONT_RIGHT`, `CAM_FRONT_LEFT`, `CAM_BACK`, `CAM_BACK_LEFT`, `CAM_BACK_RIGHT`) operating at 12 Hz, sampled at 2 Hz key-frames.
- **Image Resolution:** $1600 \times 900$ pixels.
- **Coordinate Systems:** Sensor frame $\rightarrow$ Ego vehicle frame $\rightarrow$ Global frame. SurroundOcc converts directly from Ego/Sensor frame to the primary `LIDAR_TOP` coordinate system.

### 3.2 Pre-Generated Occupancy Ground Truth (`data/nuscenes_occ/`)
Upstream generates and distributes pre-computed occupancy ground truth as `.npy` files:
- **Spatial Bounds (`pc_range`):** $[-50.0\text{ m}, -50.0\text{ m}, -5.0\text{ m}, +50.0\text{ m}, +50.0\text{ m}, +3.0\text{ m}]$
- **Voxel Grid Dimensions (`occ_size`):** $[200, 200, 16]$
- **Voxel Resolution:** $\Delta x = 0.5\text{ m}, \Delta y = 0.5\text{ m}, \Delta z = 0.5\text{ m}$
- **Storage Format:** NumPy arrays of shape $(N, 4)$ where $N$ is the number of occupied voxels ($\approx 20,000 - 45,000$ non-empty voxels per keyframe).
  - Columns $[0, 1, 2]$: Integer voxel coordinates $(x \in [0, 199], y \in [0, 199], z \in [0, 15])$.
  - Column $[3]$: Semantic class index ($1 \dots 16$).
  - In `pipelines/loading.py:32`, label $0$ is remapped to $255$ (`ignore_index`).

### 3.3 Semantic Class Ontology (16 Driving Classes + Empty)
The 16 foreground semantic classes mapped in `nuscenes.yaml` are:

| Index | Class Name | Description | Evaluation Mapping |
| :---: | :--- | :--- | :---: |
| 0 | `empty / unobserved` | Air or empty space | Excluded from mIoU; evaluated in geometry IoU |
| 1 | `barrier` | Road barriers, guardrails | Evaluated in mIoU |
| 2 | `bicycle` | Bicycles (moving and parked) | Evaluated in mIoU |
| 3 | `bus` | Transit and school buses | Evaluated in mIoU |
| 4 | `car` | Passenger cars and sedans | Evaluated in mIoU |
| 5 | `construction_vehicle` | Heavy equipment, cranes, bulldozers | Evaluated in mIoU |
| 6 | `motorcycle` | Motorcycles and mopeds | Evaluated in mIoU |
| 7 | `pedestrian` | Walking persons | Evaluated in mIoU |
| 8 | `traffic_cone` | Traffic cones and warning pylons | Evaluated in mIoU |
| 9 | `trailer` | Cargo trailers | Evaluated in mIoU |
| 10 | `truck` | Cargo trucks, semi-cabs | Evaluated in mIoU |
| 11 | `driveable_surface` | Asphalt, road surface | Evaluated in mIoU |
| 12 | `other_flat` | Paved non-road surfaces, curbs | Evaluated in mIoU |
| 13 | `sidewalk` | Pedestrian footpaths | Evaluated in mIoU |
| 14 | `terrain` | Grass, dirt, hillsides | Evaluated in mIoU |
| 15 | `manmade` | Buildings, walls, poles, bridges | Evaluated in mIoU |
| 16 | `vegetation` | Trees, bushes, shrubs | Evaluated in mIoU |
| 255 | `ignore / unknown` | Unannotated, occluded, or unobserved | Ignored in loss & evaluation |

---

## 4. Checkpoints & Pretrained Weights Audit

| Checkpoint Name | Hosting Source | Architecture | Training Target | Downstream Utility |
| :--- | :--- | :--- | :--- | :--- |
| `r101_dcn_fcos3d_pretrain.pth` | Zhiqi Li GitHub Storage | ResNet-101 + DCNv2 | FCOS3D monocular 3D detection on nuScenes | Essential initialization weight for backbone feature extraction |
| `surroundocc_semantic.pth` | Baidu Netdisk (`dmcq`) | Full SurroundOcc | 3D Semantic Occupancy (17 classes, 24 epochs) | Published baseline evaluation weight |
| `surroundocc_reconstruction.pth` | Baidu Netdisk (`nvm9`) | SurroundOcc (no-semantic) | 3D Geometric Scene Reconstruction (binary occupancy) | Baseline geometric evaluation weight |

> [!WARNING] Checkpoint Availability Constraint
> The official model weights and full datasets are hosted primarily on **Baidu Netdisk** (`pan.baidu.com`) and Tencent Weiyun (`share.weiyun.com`), requiring Chinese SMS-authenticated accounts or VPN access for high-speed downloads. The backbone weight (`r101_dcn_fcos3d_pretrain.pth`) is hosted on GitHub Releases and remains directly accessible via HTTPS.

---

## 5. Intellectual Property & License Audit

- **Repository License:** **Apache License, Version 2.0** (January 2004)
  - Full license text verified in repository root `LICENSE`.
  - Grants perpetual, worldwide, non-exclusive, no-charge royalty-free rights to reproduce, prepare derivative works of, publicly display, sublicense, and distribute.
  - Requires clear attribution notices and disclaimer of warranty.
- **Patent Rights:** Section 3 includes an express grant of patent rights by each contributor for contributions incorporated into the work.
- **Third-Party Code Provenance:**
  - Core transformer and cross-attention architectures (`spatial_cross_attention.py`, `encoder.py`) are derived directly from **BEVFormer** (licensed under Apache 2.0).
  - OpenMMLab / MMDetection3D code components carry Apache 2.0 licenses.
  - Custom Chamfer distance CUDA extension (`extensions/chamfer_dist`) carries copyright by Haozhe Xie (MIT License style).
- **ForesightBEV Compliance:** Completely compatible with ForesightBEV's proprietary and open research governance, provided attribution to the original authors is maintained.

