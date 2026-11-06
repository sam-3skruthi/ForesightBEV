# REA 03 — OCCUQ Dependency, Hardware, and Data Audit

**Repository:** [https://github.com/ika-rwth-aachen/OCCUQ](https://github.com/ika-rwth-aachen/OCCUQ)  
**Audited Commit:** `2aa54297151b695929524ad8a4f4863ba5ae6387`  
**Parent Review Date:** 2026-11-04  
**Status:** Complete Environment, Hardware, and Dataset Specification  

---

## 1. Runtime Environment & Framework Stack

OCCUQ relies on the classic OpenMMLab 1.x ecosystem coupled with PyTorch 1.10 and CUDA 11.3/11.4. The authoritative environment definition is verified across `docker/Dockerfile`, `docs/install.md`, and `requirements.txt`:

| Dependency Layer | Package / Tool | Exact Required Version | Purpose / Failure Mode if Mismatched |
| :--- | :--- | :--- | :--- |
| **Operating System** | Linux (Ubuntu 20.04 LTS tested) | Ubuntu 18.04 / 20.04 / 22.04 | Required for custom CUDA kernel compilation and OpenMMLab wheel compatibility. |
| **Python** | CPython | `3.8.x` | OpenMMLab 1.x (`mmcv-full==1.4.0`) wheels are precompiled for Python 3.7 / 3.8. Python $\geq 3.10$ breaks. |
| **CUDA Toolkit** | NVIDIA CUDA | `11.3` (or `11.4` in NGC 21.10) | Strictly required to compile `extensions/chamfer_dist` and link with `mmcv-full`. |
| **PyTorch** | `torch` | `1.10.1` (tested) / `1.10.0` | High-level APIs (`torch.distributions.MultivariateNormal`) used for GMM; TorchScript JIT compiler target. |
| **Torchvision** | `torchvision` | `0.11.2` | Image transforms and backbone tensor manipulation. |
| **OpenMMLab Core** | `mmcv-full` | `1.4.0` | Provides deformable attention CUDA operators and base runner machinery. |
| **Detection 2D** | `mmdet` | `2.14.0` | ResNet backbone, FPN neck, and base loss/transformer abstractions. |
| **Segmentation** | `mmsegmentation` | `0.14.1` | Semantic segmentation utilities and base pipelines. |
| **3D Detection** | `mmdetection3d` | `v0.17.1` (built from source) | Multi-view camera data loaders, 3D coordinate converters, and runner hooks. |
| **Uncertainty Tool** | `netcal` | Latest / standard | Provides Expected Calibration Error (`netcal.metrics.ECE(10)`) and reliability diagrams. |
| **Sparse Conv** | `spconv-cu114` | Optional / docker | Voxelization and sparse point processing. |
| **Point Cloud / Vis** | `open3d` / `mayavi` | `open3d-python`, `mayavi` | 3D mesh and voxel point cloud rendering in `tools/gmm_video.py`. |
| **Evaluation** | `scikit-learn` | `1.0.1` | ROC curves, AUROC, AUPR, and confusion matrix calculation. |
| **Compilation** | `gcc` / `g++` | `gcc-6` or `gcc-7` | Modern GCC ($\geq 11$) fails when compiling MMCV 1.4.0 and CUDA 11.3 kernels. |

---

## 2. Custom CUDA Extensions

OCCUQ inherits one primary compiled CUDA extension from SurroundOcc:

### `extensions/chamfer_dist/`
- **Source files:**
  - `extensions/chamfer_dist/chamfer.cu`: CUDA kernel evaluating two-way Chamfer distance between predicted voxel vertices and ground-truth LiDAR points.
  - `extensions/chamfer_dist/chamfer_cuda.cpp`: C++ PyTorch binding (`torch::Tensor chamfer_distance(...)`).
  - `extensions/chamfer_dist/setup.py`: `CUDAExtension` builder using `torch.utils.cpp_extension`.
- **Architectural Requirement:** Used during training for geometry reconstruction loss ($\mathcal{L}_{\text{geo}}$) and offline surface mesh evaluation.
- **Hardware Lock-in:** Strictly requires NVIDIA GPU with Compute Capability 6.0, 7.0, 7.5, 8.0, or 8.6.

---

## 3. Dataset Footprint & Directory Topology

To reproduce OCCUQ training, GMM fitting, and MultiCorrupt evaluation, three distinct data sources are required:

```
/workspace/data/
├── nuscenes/                         # Standard nuScenes Full Dataset (~350 GB)
│   ├── maps/                         # Expansion maps
│   ├── samples/                      # Keyframe camera images (6 views, 1600x900)
│   ├── sweeps/                       # Intermediate sweeps (optional for OCCUQ)
│   ├── v1.0-test/                    # Official test set
│   └── v1.0-trainval/                # Official train/val metadata JSONs
├── nuscenes_occ/                     # SurroundOcc Dense GT Labels (~3.8 GB)
│   ├── scene-xxxx/
│   │   └── xxxx.npy                  # (N, 4) array: [x, y, z, semantic_class]
│   └── ...
├── nuscenes_infos_train.pkl          # Pre-parsed training metadata (~150 MB)
├── nuscenes_infos_val.pkl            # Pre-parsed validation metadata (~35 MB)
└── multicorrupt/                     # MultiCorrupt Dataset (~50-100 GB)
    ├── snow/
    │   ├── 1/ (severity 1 images)
    │   ├── 2/ (severity 2 images)
    │   └── 3/ (severity 3 images)
    ├── fog/
    │   ├── 1/, 2/, 3/
    ├── motionblur/
    │   ├── 1/, 2/, 3/
    ├── brightness/
    │   ├── 1/, 2/, 3/
    └── missingcamera/
        ├── 1/ (1 camera dropped: front camera zeroed)
        ├── 2/ (2 cameras dropped)
        └── 3/ (3 cameras dropped)
```

### Detailed Dataset Storage Inventory

| Dataset Component | Source Link / Hosting | Uncompressed Size | Role in OCCUQ |
| :--- | :--- | :--- | :--- |
| **nuScenes v1.0 TrainVal** | `https://www.nuscenes.org/download` | $\approx 350\text{ GB}$ | Source multi-camera images (6 surround views, CAM_FRONT, CAM_FRONT_LEFT, etc.). |
| **SurroundOcc Dense Occupancy GT** | Tsinghua Cloud (`cloud.tsinghua.edu.cn/f/ef8357724574491d9ddb`) | $3.2\text{ GB}$ (train) + $627\text{ MB}$ (val) | $200 \times 200 \times 16$ voxel labels at $0.5\text{ m}$ resolution. Format: `.npy` files with $(N, 4)$ coordinates. |
| **MultiCorrupt OOD Dataset** | Hugging Face: `TillBeemelmanns/MultiCorrupt` | $\approx 60\text{ GB}$ | Corrupted validation images across 5 corruption families at 3 severity levels. |
| **Total Disk Space Needed** | — | **$\approx 420 - 450\text{ GB}$** | Complete training, GMM fitting, and evaluation workspace. |

---

## 4. Hardware Compute Requirements

### 4.1 Training OCCUQ (`scripts/train_occuq.sh`)
- **GPU Configuration:** $4 \times \text{NVIDIA A100 (80GB / 40GB)}$ or $4 \times \text{NVIDIA RTX 3090 / 4090 (24GB)}$.
- **Batch Size:** 1 sample per GPU (`samples_per_gpu=1`), total effective batch size 4.
- **Memory Footprint:** Peak VRAM consumption $\approx 18.5\text{ GB}$ per GPU during backward pass (due to multi-scale 3D deformable attention and transposed 3D convolution upsampling).
- **Duration:** 6 epochs fine-tuning initialized from `surroundocc.pth` takes $\approx 18\text{ hours}$ on $4 \times \text{A100}$.

### 4.2 GMM Fitting (`tools/gmm_fit.py`)
- **GPU Configuration:** Single GPU ($1 \times \text{NVIDIA RTX 3090}$ or $1 \times \text{A100}$).
- **System RAM:** High requirement. Collecting up to 200,000,000 features across 17 classes requires **$\geq 64\text{ GB}$ of host RAM** (reported $\approx 32\text{ GB}$ pure tensor allocation in `train_features`).
- **Duration:** $\approx 45 - 60\text{ minutes}$ for balanced feature extraction across the nuScenes training split, followed by $< 2\text{ minutes}$ for covariance matrix computation.

### 4.3 GPU-Accelerated GMM Inference (`tools/gmm_evaluate.py`)
- **VRAM Footprint:** $\approx 6.8\text{ GB}$ VRAM during test inference.
- **Latency on RTX 3090 / A100:**
  - Backbone + FPN + 3D Cross-Attention + Decoder: $\approx 180\text{ ms}$.
  - TorchScript JIT GMM density evaluation: **$\approx 9.5\text{ ms}$**.
  - Total latency per frame: $\approx 190\text{ ms}$ ($\approx 5.2\text{ FPS}$).

---

## 5. macOS / Apple Silicon (MPS) Audit

Attempting to run OCCUQ natively on macOS (Apple Silicon M-series via `torch.device("mps")`) faces the following hard blockers:

1. **Deformable Attention Operators:** `MSDeformableAttention3D` depends on OpenMMLab's `mmcv-full` compiled CUDA C++ extensions (`MultiScaleDeformableAttention`), which have no native MPS or Metal Performance Shaders backend.
2. **Chamfer Distance Extension:** `extensions/chamfer_dist/chamfer.cu` is written in raw NVIDIA CUDA assembly and cannot compile under Apple Clang.
3. **OpenMMLab 1.x Legacy Dependencies:** `mmcv-full==1.4.0` does not provide prebuilt ARM64 wheels for macOS with modern PyTorch ($\geq 2.0$). Compiling MMCV 1.4.0 on macOS requires outdated Clang/GCC compilers and fails due to deprecated PyTorch C++ header APIs.
4. **Feasibility for ForesightBEV:**
   - Full native end-to-end model training on macOS: **Impossible**.
   - Lightweight standalone GMM inference: **Possible**. The TorchScript JIT kernel (`jit_log_prob`) in `tools/gmm_utils.py` uses standard PyTorch tensor matrix operations (`matmul`, `square`, `sum`, `log`, `diagonal`) that run seamlessly on CPU or MPS without custom CUDA kernels.
   - ForesightBEV development strategy: Containerize upstream training and full 3D occupancy generation on remote CUDA servers (Linux + NVIDIA A100), while decoupling the downstream safety layer (Stage 6), FastAPI backend (Stage 8), and dashboard for local macOS development.

