# REA 04 — ReliOcc Dependency, Hardware, and Data Audit

**Audited Paper:** *Reliable and Calibrated Semantic Occupancy Prediction by Hybrid Uncertainty Learning* (IJCAI 2025)  
**Implementation Classification:** **Status D — No Verifiable Official Codebase Found**  
**Audit Purpose:** Comprehensive audit of computational requirements, framework dependencies, dataset protocols, and licensing constraints reconstructed from the peer-reviewed specification.

---

## 1. Official Code Status & Framework Lineage

As established in Section 2 of the REA Report, **no official source-code repository has been published by the authors**. However, because ReliOcc is implemented as an add-on module to established open-source baselines, its runtime dependencies and framework stack can be reconstructed with high fidelity from the underlying frameworks evaluated in the paper:

| Layer | Underlying Component | Authoritative License | Operational Role in ReliOcc |
| :--- | :--- | :--- | :--- |
| **Primary Base Model 1** | **VoxFormer** (`NVlabs/VoxFormer`) | NVIDIA Source Code License-NC | Primary 3D transformer baseline. ReliOcc replaces its Stage-II classification head. |
| **Primary Base Model 2** | **SGN** (`Jieqianyu/SGN`) | Open-Source / Academic | Secondary "dense-sparse-dense" baseline. |
| **Post-Hoc Baseline** | **Calib3D** (`ldkong1205/Calib3D`) | MIT License | Source code used by authors to implement DeptS baseline adaptation for occupancy. |
| **Evaluation Metrics** | **relis** (`naver/relis`) | CC BY-NC-SA 4.0 | Source for Prediction Rejection Ratio (PRR) rejection curve logic. |
| **Calibration Graphics** | **reliability-diagrams** (`hollance/reliability-diagrams`) | MIT License | Visualization scripts for 15-bin confidence-accuracy gap diagrams. |
| **Corruptions Suite** | **3D_Corruptions_AD** (`thu-ml/3D_Corruptions_AD`) | MIT License | Generates synthetic weather and sensor perturbations. |
| **Dataset API** | **SemanticKITTI-API** (`PRBonn/semantic-kitti-api`) | MIT License | Voxel loading, evaluation, and voxel-to-point transformations. |

---

## 2. Reconstructed Software Runtime Dependencies

Based on VoxFormer, SGN, and OpenMMLab 3D perception requirements, reproducing ReliOcc requires:

| Dependency | Required Version (Tested) | Rationale & Potential Blockers |
| :--- | :--- | :--- |
| **Operating System** | Linux (Ubuntu 18.04 / 20.04 LTS) | Required by VoxFormer CUDA deformable attention kernels and SGN sparse convolutions. |
| **Python** | `3.8.x` | OpenMMLab 1.x and VoxFormer environment compatibility. |
| **CUDA Toolkit** | `CUDA 11.1` or `11.3` | Required for compiling Deformable Attention CUDA operators on Tesla V100 / Ampere GPUs. |
| **PyTorch** | `torch==1.10.0` or `1.10.1` | Autograd reparameterization trick (`torch.randn_like`), standard linear algebra, and multi-GPU DDP. |
| **Torchvision** | `torchvision==0.11.1` | ResNet-50 vision encoder transforms. |
| **MMCV** | `mmcv-full==1.4.0` or `1.6.0` | Deformable cross-attention and spatial transformer layers. |
| **MMDetection3D** | `mmdet3d==0.17.1` or `1.0.0rcx` | 3D coordinate frame conventions and volumetric loss wrappers. |
| **SpConv** | `spconv==2.x` (or `spconv-cu113`) | Required by SGN's sparse voxel encoder. |
| **Compiler** | `GCC 7.5` or `8.4` | C++14 support for CUDA kernel compilation. |

---

## 3. Dataset and Input Modality Audit

### 3.1 Dataset Specification: SemanticKITTI
The empirical evaluation of ReliOcc relies exclusively on the **SemanticKITTI** benchmark, derived from the KITTI Odometry Benchmark:

- **Sequences 00–10 (excluding 08):** Training split (10 sequences, $\approx 19,130$ frames).
- **Sequence 08:** Validation split (1 sequence, $\approx 4,071$ frames).
- **Sequences 11–21:** Official test split (hidden test server; semantic occupancy ground truth is withheld, which is why ReliOcc evaluates all reliability metrics on Sequence 08).

### 3.2 Detailed Input Modality & Sensor Setup
A critical architectural finding of this audit is that **ReliOcc operates on a forward-facing monocular temporal camera setup, NOT a 360° multi-camera surround array**:

```
+---------------------------------------------------------------------------------+
|                                 VEHICLE HEADING                                 |
|                                        ^                                        |
|                                        |                                        |
|                                   [LEFT CAMERA]                                 |
|                       (CAM 2: Current Frame t + 4 Past Frames)                  |
|                             Azimuth Coverage: ~90 deg                           |
|                                                                                 |
|       NO SENSORS                      [EGO]                       NO SENSORS    |
|       Left Blind Zone                                             Right Blind   |
|                                                                                 |
|                                    NO SENSORS                                   |
|                                  Rear Blind Zone                                |
+---------------------------------------------------------------------------------+
```

- **Sensors Ingested:** Single perspective camera (`image_2` / Left RGB camera of the KITTI stereo rig).
- **Temporal Buffer:** $T = 5$ frames total:
  - Current image: frame $t$
  - Past context: frames $t-1, t-2, t-3, t-4$
- **Image Resolution:** Cropped to $1220 \times 370$ pixels.
- **Volumetric Bounding Box (SemanticKITTI Ego-Frame):**
  - Longitudinal ($Y$): $[0.0\text{ m}, 51.2\text{ m}]$ (Forward direction only!)
  - Lateral ($X$): $[-25.6\text{ m}, +25.6\text{ m}]$ (Symmetric left-to-right)
  - Vertical ($Z$): $[-2.0\text{ m}, +4.4\text{ m}]$ (Road surface to overhead height)
- **Grid Resolution:** $256 \times 256 \times 32$ voxels.
- **Voxel Size:** $\Delta x = 0.2\text{ m}, \Delta y = 0.2\text{ m}, \Delta z = 0.2\text{ m}$ (isotropic).
- **Total Voxel Count:** $2,097,152$ voxels per scene.
- **Classes:** 20 total classes (class 0 = free/unoccupied, classes 1–19 = semantic categories: car, bicycle, motorcycle, truck, other-vehicle, person, bicyclist, motorcyclist, road, parking, sidewalk, other-ground, building, fence, vegetation, trunk, terrain, pole, traffic-sign).

### 3.3 The Camera-Only Inference Boundary & LiDAR Role
- **Offline Supervision Role:** Ground-truth occupancy labels in SemanticKITTI were constructed by accumulating LiDAR sweeps using accurate ego-poses from GPS/IMU, followed by point-cloud voxelization and ray-casting.
- **Inference Pipeline:** During camera inference in ReliOcc, **no LiDAR data is accessed, processed, or leaked into the model**. The input is purely the 5-frame monocular camera sequence.
- **Contrast with DeptS:** The baseline calibration method DeptS (*Depth-Aware Scaling*, Ding et al., 2024) requires true point-cloud depth $d_i$ at inference. ReliOcc demonstrates that its uncertainty-aware calibrator ($f_\phi$) outperforms DeptS **without using any depth input at test time**.

---

## 4. Hardware Compute Requirements

### 4.1 Author-Reported Training Infrastructure
The official experiments were executed on an enterprise cluster with the following specifications:
- **Compute:** $8 \times \text{NVIDIA Tesla V100 GPUs}$ (32GB VRAM per GPU).
- **Batch Size:** 1 scene sample per GPU (total effective batch size 8).
- **Memory Consumption:** $\approx 24 - 28\text{ GB}$ VRAM per GPU during Stage-II 3D transformer cross-attention and dense voxel backpropagation.
- **Training Duration:**
  - VoxFormer + ReliOcc (Online): 20 epochs ($\approx 36\text{ hours}$ on 8x V100).
  - SGN + ReliOcc (Online): 40 epochs ($\approx 48\text{ hours}$ on 8x V100).
  - Offline Calibration ($f_\phi$): 20 epochs ($\approx 1.5 - 2.5\text{ hours}$ on 8x V100, since base model is frozen).

### 4.2 Storage Footprint
To replicate the SemanticKITTI experiment pipeline:
- KITTI Odometry Left Color Images (`data_odometry_color.zip`): $\approx 65\text{ GB}$.
- SemanticKITTI Voxel Ground Truth (`data_odometry_voxels.zip`): $\approx 42\text{ GB}$.
- Pretrained VoxFormer / SGN Checkpoints: $\approx 1.5\text{ GB}$.
- Total Disk Allocation: $\approx 110 - 120\text{ GB}$.

---

## 5. Licensing and Academic/Commercial Compliance

| Asset | Source / Host | License | Compliance Constraint for ForesightBEV |
| :--- | :--- | :--- | :--- |
| **SemanticKITTI Annotations** | `semantic-kitti.org` | **CC BY-NC-SA 4.0** | **Strictly Non-Commercial.** Any model trained directly on SemanticKITTI labels cannot be commercialized. |
| **KITTI Raw Images** | `cvlibs.net` | **CC BY-NC-SA 3.0** | Non-Commercial share-alike. |
| **VoxFormer Base Code** | NVIDIA Research | **NVIDIA Source Code License-NC** | Non-Commercial research use only. |
| **relis (PRR Metric)** | NAVER Corp | **CC BY-NC-SA 4.0** | Non-Commercial. |
| **ReliOcc Method** | IJCAI 2025 Paper | Intellectual Property (Zhejiang Univ / Huawei) | The mathematical principles ($L_{\text{au}}$, $L_{\text{ru}}$, $f_\phi$) can be reimplemented cleanly under Apache 2.0 within ForesightBEV without copying proprietary code. |

