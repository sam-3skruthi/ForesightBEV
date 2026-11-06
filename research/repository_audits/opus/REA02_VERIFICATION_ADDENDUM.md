# REA 02 — OPUS / OPUS-V2 Verification Addendum

**Audited Repository:** [https://github.com/jbwang1997/OPUS](https://github.com/jbwang1997/OPUS)  
**Audited Commit:** `fdb17339f40168ea0ceda9615863e3f5d0c3f5ce`  
**Parent Review Date:** 2026-11-04  
**Status:** Verification Addendum & Metric Reconciliation for ForesightBEV Stage 0 REA 02  

---

## 1. Executive Summary

This verification addendum resolves critical technical reconciliations identified following the completion of REA 02 (OPUS and OPUS-V2). It serves as an authoritative bridge between the primary audit reports (`REA_REPORT.md`, `SOURCE_MAP.md`, `DEPENDENCY_AND_DATA_AUDIT.md`, `REPRODUCTION_ASSESSMENT.md`) and our ongoing research baseline comparisons.

Specifically, this document:
1. Formally tabulates all OPUS-V2 architecture variants (T, S, M, L) and their exact query allocations ($Q \in \{600, 1200, 2400, 4800\}$).
2. Authoritatively attributes the benchmark metric of **$38.6\text{ mIoU}$ and $44.0\text{ RayIoU}$ to OPUS-V2-L at $8.6\text{ FPS}$**, correcting any earlier conflation with the real-time **OPUS-V2-T** configuration ($35.6 - 36.5\text{ mIoU}$ at $25.8\text{ FPS}$).
3. Clarifies occupancy class counts across V1, V2, and benchmarks (Occ3D vs. OpenOccupancy), detailing the exact semantic channels and `empty_label = 17` handling.
4. Analyzes the 8-frame temporal ingest and establishes that `fut-15f` configurations perform offline, non-causal bidirectional trajectory smoothing rather than future occupancy forecasting.
5. Clarifies the dependency lineage resulting from commit `fdb1733` (removal of `mmdet3d` and `mmsegmentation`), identifying persisting dependencies on legacy MMCV 1.6.0, `spconv`, `torch-scatter`, and custom CUDA kernels.
6. Establishes comparability constraints when evaluating OPUS against SurroundOcc and other Stage 3 perception baselines.

---

## 2. OPUS-V2 Variants & Query Count Disaggregation

In OPUS-V2, queries are split into:
- $Q$: Number of dynamic base 3D bounding box queries (`num_queries`).
- $P$: Number of points sampled per query box (`num_points`) across its $Z$, $Y$, $X$ sub-dimensions (`num_points_in_box=[z, y, x]`).

The four official OPUS-V2 variants are configured in `projects/configs/opus_v2/` as follows:

| Variant | Config Path | Queries ($Q$) | Points/Box ($P$) | Total Points ($Q \times P$) | Points Grid $[Z, Y, X]$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **OPUS-V2-T** (Tiny) | `opus_v2_t_8f_occ3d.py` | **600** | 128 | 76,800 | $[4, 8, 4]$ |
| **OPUS-V2-S** (Small) | `opus_v2_s_8f_occ3d.py` | **1,200** | 64 | 76,800 | $[4, 4, 4]$ |
| **OPUS-V2-M** (Medium) | `opus_v2_m_8f_occ3d.py` | **2,400** | 32 | 76,800 | $[2, 4, 4]$ |
| **OPUS-V2-L** (Large) | `opus_v2_l_8f_occ3d.py` | **4,800** | 16 | 76,800 | $[2, 2, 4]$ |

### Structural Observation
Across all four variants, the densifier receives the identical point budget ($Q \times P = 76,800$ points) into the Continuous Point Densifier (`Densifier` in `projects/models/dense_heads/densifier.py`). 
- **Tiny (T)** uses coarse object proposals ($Q=600$) and samples finely inside each proposal ($P=128$).
- **Large (L)** decomposes 3D space into fine object proposals ($Q=4800$) and samples sparsely inside each proposal ($P=16$).

Because the Transformer decoder operates on $Q$ queries with self-attention complexity $\mathcal{O}(Q^2)$ and deformable cross-attention scaling with $Q$, increasing $Q$ from 600 to 4800 dramatically reduces latency.

---

## 3. Authoritative Metric & Latency Attribution

The official upstream benchmark table in `README.md` details the Occ3D-nuScenes validation set results evaluated on an NVIDIA RTX 3090:

| Model Variant | Training Schedule | Loss Variant | mIoU (%) | RayIoU (%) | FPS (RTX 3090) | Latency / Sample |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **OPUS-V2-T** | 50 epochs | CD (Chamfer Distance) | 35.6 | 41.0 | **25.8** | ~38.8 ms |
| **OPUS-V2-T** | 50 epochs | DCD (Dual Chamfer Distance) | 36.1 | 41.7 | **25.8** | ~38.8 ms |
| **OPUS-V2-T** | 100 epochs | CD (Chamfer Distance) | 36.5 | 42.0 | **25.8** | ~38.8 ms |
| **OPUS-V2-S** | 100 epochs | CD | 37.3 | 42.7 | **23.7** | ~42.2 ms |
| **OPUS-V2-M** | 100 epochs | CD | 37.7 | 43.3 | **16.0** | ~62.5 ms |
| **OPUS-V2-L** | 100 epochs | CD | **38.6** | **44.0** | **8.6** | ~116.3 ms |

### Attribution Reconciliation
- **The maximum reported accuracy ($38.6\text{ mIoU}$ / $44.0\text{ RayIoU}$)** is achieved **exclusively by OPUS-V2-L** running at **$8.6\text{ FPS}$**.
- **The real-time benchmark ($25.8\text{ FPS}$)** belongs **exclusively to OPUS-V2-T**, which delivers **$35.6 - 36.5\text{ mIoU}$**.
- Citations must never conflate the $38.6\text{ mIoU}$ result with the $25.8\text{ FPS}$ throughput: OPUS-V2 cannot deliver $38.6\text{ mIoU}$ at real-time speeds on an RTX 3090.

---

## 4. Occupancy Class Definitions & Empty/Unknown Handling

Inspection of `projects/models/datasets/nuscenes_occ3d_dataset.py` and `projects/models/dense_heads/opus_head_v2.py` reveals the precise class structures:

### 4.1 Occ3D-nuScenes Benchmark
- **Class count:** 18 distinct categories defined in `occ_names`:
  1. `others` (0)
  2. `barrier` (1)
  3. `bicycle` (2)
  4. `bus` (3)
  5. `car` (4)
  6. `construction_vehicle` (5)
  7. `motorcycle` (6)
  8. `pedestrian` (7)
  9. `traffic_cone` (8)
  10. `trailer` (9)
  11. `truck` (10)
  12. `driveable_surface` (11)
  13. `other_flat` (12)
  14. `sidewalk` (13)
  15. `terrain` (14)
  16. `manmade` (15)
  17. `vegetation` (16)
  18. `free / empty` (17)
- **Empty label handling:** `empty_label = 17`.
- **Classification Head:** Predictions are made over classes $0 \dots 17$. The semantic occupancy head outputs 18 logits per voxel (or 17 semantic foreground classes + 1 empty class).
- **Unknown label:** There is **NO** explicit `unknown` class output channel. The camera visibility mask (`mask_camera`) filters unobserved or occluded voxels during loss calculation; voxels with label 255 (ignore) are masked out in evaluation metrics.

### 4.2 OpenOccupancy Benchmark
- Uses 16 semantic classes plus free space. Class `noise` is mapped to index 0 or ignored.
- The `empty_label` is passed directly to the focal loss or cross-entropy loss function to down-weight or penalize empty space reconstruction errors.

---

## 5. Temporal Modeling vs. Future Forecasting

Configurations in `projects/configs/opus_v2/` include:
- `opus_v2_t_8f_occ3d.py` (standard online camera inference).
- `opus_v2_t_fut_15f_occ3d.py` (extended temporal sequence processing).

### Source-Level Dissection of `fut-15f`
Inspection of `projects/models/datasets/nuscenes_occ3d_dataset.py` demonstrates:
```python
# In Dataset configuration for fut-15f:
prev_frames = 7
next_frames = 7
```
1. `opus_v2_t_8f_occ3d.py` ingests $T=8$ sweeps: 7 past frames ($t-7 \dots t-1$) plus the current frame ($t$). This constitutes a causal, temporal sliding window for current occupancy prediction at timestamp $t$.
2. `opus_v2_t_fut_15f_occ3d.py` ingests 7 past frames, the current frame, and **7 future frames** ($t+1 \dots t+7$), totaling 15 frames ($[-7, +7]$).
3. **Crucial Architectural Clarification:** `fut-15f` does **NOT** forecast future occupancy states at $t+1 \dots t+7$. It predicts the occupancy state of the current frame $t$ using non-causal bidirectional camera context. This is an offline auto-labeling or sequence-smoothing regime.
4. **ForesightBEV Architectural Alignment:** In ForesightBEV:
   - **Stage 3 (3D Occupancy Perception)** requires causal, past-to-present inference ($t-K \dots t$).
   - **Stage 5 (4D Occupancy and Flow Forecasting)** is specifically tasked with forecasting future occupancy states $\hat{\mathbf{O}}_{t+1:t+H}$ from causal history $\mathbf{O}_{t-K:t}$. OPUS does not implement Stage 5 forecasting.

---

## 6. Dependency Lineage and Removal of `mmdet3d`

Upstream commit `fdb17339f40168ea0ceda9615863e3f5d0c3f5ce` titled *"remove mmdet3d and mmseg"* altered repository dependencies:

### What was removed:
- Direct dependency on `mmdet3d==1.0.0rc6` (OpenMMLab 3D detection framework).
- Direct dependency on `mmsegmentation==0.30.0`.
- All `mmdet3d.models.builder` imports were refactored to native `mmdet.models.builder`.

### What remains (persisting lock-ins):
1. **MMCV-Full:** Still strictly depends on `mmcv-full==1.6.0` (OpenMMLab 1.x series, deprecated in OpenMMLab 2.x).
2. **MMDetection:** Still strictly requires `mmdet==2.28.2`.
3. **PyTorch / CUDA:** Requires `torch==1.10.0` or `1.12.0` with `cuda==11.3` or `11.6`.
4. **V1 Custom CUDA Extension:** `projects/models/csrc/msmv_sampling/` compiles a custom CUDA multi-scale multi-view sampling kernel (`msmv_sampling_cuda.cu`). While a pure PyTorch fallback is provided in `msmv_sampling_pytorch()`, performance degrades significantly without CUDA compilation.
5. **V2 Dependencies:** OPUS-V2 introduces `spconv-cu116` (sparse convolution) and `torch-scatter` for point-cloud voxelization in `Densifier`. These packages require compiled CUDA C++ binaries, making OPUS-V2 strictly CUDA Linux-dependent and incompatible with native macOS/MPS execution.

---

## 7. Comparative Assessment Constraints

When evaluating OPUS-V2 against SurroundOcc and other Stage 3 models:

1. **Resolution Equivalence:** 
   - SurroundOcc predicts dense $200 \times 200 \times 16$ voxel grids ($0.5\text{ m}$ voxel size).
   - OPUS predicts dense $200 \times 200 \times 16$ grids on Occ3D via continuous point densification, but query supervision operates in sparse Chamfer space.
2. **Hardware Uniformity:**
   - SurroundOcc reported FPS was measured on NVIDIA RTX 3090 / A100 under different image input dimensions.
   - OPUS-V2-T delivers $25.8\text{ FPS}$ on an RTX 3090, while OPUS-V2-L drops to $8.6\text{ FPS}$. Cross-paper comparisons must specify the exact variant and hardware setup.
3. **Temporal Ingest:**
   - Baseline SurroundOcc uses single-frame multi-camera inference (with offline multi-frame LiDAR GT generation).
   - OPUS-V2 ingests $T=8$ frames of camera history, which gives it a significant temporal context advantage over single-frame architectures.

---

## 8. Conclusion

This addendum formally completes the verification requirements for REA 02. The distinctions detailed above are frozen into the research record for ForesightBEV Stage 0. We now proceed to REA 03: OCCUQ.

