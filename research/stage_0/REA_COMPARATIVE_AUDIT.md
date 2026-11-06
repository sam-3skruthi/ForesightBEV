# ForesightBEV — Stage 0: Comparative Audit Register & Research Index

**Document Reference:** `research/stage_0/REA_COMPARATIVE_AUDIT.md`  
**Phase:** Stage 0 — Literature Review, REA, and Research Gap  
**Status:** Consolidated Audit Registry & Baseline Index  

---

## 1. Scope & Master Audit Directory

This document establishes the master technical index linking all completed reverse-engineering audits across ForesightBEV Stage 0. It summarizes the scope, audited commit/version, primary findings, open technical risks, and evidential status of each investigation:

```
research/
├── repository_audits/
│   ├── birds_eye_view/       # REA Baseline: Monocular IPM & Inverse Projection
│   ├── cam2bev/              # REA Baseline: 4-Camera IPM & Spatial Homography
│   ├── surroundocc/          # REA 01: Dense 3D Volumetric Cross-Attention
│   ├── opus/                 # REA 02: Sparse Query 3D Continuous Point Densification
│   ├── occuq/                # REA 03: Feature-Level GMM Epistemic Uncertainty & OOD
│   └── reliocc/              # REA 04: Hybrid Aleatoric Mix-Up & Dynamic Calibration
└── stage_0/
    ├── OCCUPANCY_AND_UNCERTAINTY_SYNTHESIS.md  # Unified cross-architecture comparison
    ├── RESEARCH_GAP_AND_HYPOTHESES.md         # 5 testable core research hypotheses
    ├── EVIDENCE_AND_OPEN_QUESTIONS.md          # 10 unresolved empirical risk items
    └── REA_COMPARATIVE_AUDIT.md                # Master audit registry (This document)
```

---

## 2. Master Audit Registry

### Baseline Audit 01: Birds_Eye_View
- **Repository:** `https://github.com/adept/birds_eye_view` (MonoScene lineage)
- **Audit Type:** Source-level static code audit
- **Audited Target:** Monocular inverse perspective mapping (IPM) & 2D-to-BEV baseline
- **Audit Directory:** [`research/repository_audits/birds_eye_view/`](file://research/repository_audits/birds_eye_view/)
- **Primary Deliverables:**
  - [`REA_REPORT.md`](file://research/repository_audits/birds_eye_view/REA_REPORT.md)
  - [`AUDIT_REVIEW_ADDENDUM.md`](file://research/repository_audits/birds_eye_view/AUDIT_REVIEW_ADDENDUM.md)
- **Primary Findings:** Monocular IPM suffers from catastrophic geometric distortion on non-flat ground planes; provides intuition for ray-casting limitations.
- **Open Issues:** Flat-ground homography assumptions fail on urban elevation changes; lacks 3D height reasoning.

---

### Baseline Audit 02: Cam2BEV
- **Repository:** `https://github.com/ika-rwth-aachen/Cam2BEV`
- **Audit Type:** Source-level static code audit & containerized environment trace
- **Audited Target:** Multi-camera synthetic IPM baseline for urban BEV segmentation
- **Audit Directory:** [`research/repository_audits/cam2bev/`](file://research/repository_audits/cam2bev/)
- **Primary Deliverables:**
  - [`REA_REPORT.md`](file://research/repository_audits/cam2bev/REA_REPORT.md)
  - [`SOURCE_MAP.md`](file://research/repository_audits/cam2bev/SOURCE_MAP.md)
  - [`DEPENDENCY_AND_DATA_AUDIT.md`](file://research/repository_audits/cam2bev/DEPENDENCY_AND_DATA_AUDIT.md)
  - [`REPRODUCTION_PLAN.md`](file://research/repository_audits/cam2bev/REPRODUCTION_PLAN.md)
  - [`REPRODUCTION_REVIEW_ADDENDUM.md`](file://research/repository_audits/cam2bev/REPRODUCTION_REVIEW_ADDENDUM.md)
- **Primary Findings:** Multi-camera IPM blending creates severe seam artifacts at camera overlap zones; 2D BEV projection destroys vertical height structure required for 3D navigation.
- **Open Issues:** Inability to handle dynamic vertical occluders (overpasses, bridges, multi-level parking).

---

### REA 01: SurroundOcc
- **Repository:** `https://github.com/weiyithu/SurroundOcc`
- **Audit Type:** Comprehensive source-code reverse engineering (Status A)
- **Audited Commit SHA:** `419bf5b47ee07ef0610950cce5ba99168c506753` (Branch `main`, Apache 2.0)
- **Publication:** ICCV 2023 (*"SurroundOcc: Multi-Camera 3D Occupancy Prediction for Autonomous Driving"*)
- **Audit Directory:** [`research/repository_audits/surroundocc/`](file://research/repository_audits/surroundocc/)
- **Primary Deliverables:**
  - [`REA_REPORT.md`](file://research/repository_audits/surroundocc/REA_REPORT.md)
  - [`SOURCE_MAP.md`](file://research/repository_audits/surroundocc/SOURCE_MAP.md)
  - [`DEPENDENCY_AND_DATA_AUDIT.md`](file://research/repository_audits/surroundocc/DEPENDENCY_AND_DATA_AUDIT.md)
  - [`REPRODUCTION_ASSESSMENT.md`](file://research/repository_audits/surroundocc/REPRODUCTION_ASSESSMENT.md)
  - [`AUDIT_REVIEW_ADDENDUM.md`](file://research/repository_audits/surroundocc/AUDIT_REVIEW_ADDENDUM.md)
- **Primary Findings:**
  - First complete multi-camera dense 3D spatial cross-attention lifting framework.
  - Multi-scale volumetric query pyramids ($25\times 25\times 2 \to 100\times 100\times 8$) decoded via 3D transposed convolutions into $200 \times 200 \times 16$ grid ($0.5\text{ m}$ resolution).
  - Multi-frame LiDAR fusion + Poisson reconstruction pipeline for ground-truth generation.
- **Open Issues & Limitations:**
  - Slow inference speed ($\approx 4.5\text{ FPS}$ on RTX 3090).
  - Zero uncertainty quantification; outputs uncalibrated deterministic logits.
  - No explicit unknown-space representation; static single-frame inference.

---

### REA 02: OPUS / OPUS-V2
- **Repository:** `https://github.com/jbwang1997/OPUS`
- **Audit Type:** Comprehensive source-code reverse engineering & metric reconciliation (Status A)
- **Audited Commit SHA:** `fdb17339f40168ea0ceda9615863e3f5d0c3f5ce` (Branch `main`, Apache 2.0)
- **Preprint / Reports:** arXiv:2311.16487 / arXiv:2407.08630
- **Audit Directory:** [`research/repository_audits/opus/`](file://research/repository_audits/opus/)
- **Primary Deliverables:**
  - [`REA_REPORT.md`](file://research/repository_audits/opus/REA_REPORT.md)
  - [`SOURCE_MAP.md`](file://research/repository_audits/opus/SOURCE_MAP.md)
  - [`DEPENDENCY_AND_DATA_AUDIT.md`](file://research/repository_audits/opus/DEPENDENCY_AND_DATA_AUDIT.md)
  - [`REPRODUCTION_ASSESSMENT.md`](file://research/repository_audits/opus/REPRODUCTION_ASSESSMENT.md)
  - [`REA02_VERIFICATION_ADDENDUM.md`](file://research/repository_audits/opus/REA02_VERIFICATION_ADDENDUM.md)
- **Primary Findings:**
  - Solves the density bottleneck using sparse 3D bounding box queries ($Q \in \{600, 1200, 2400, 4800\}$) densified via Continuous Point Densification (CPD).
  - Authoritative benchmark disaggregation: **$38.6\%\text{ mIoU}$ belongs to OPUS-V2-L at $8.6\text{ FPS}$**, while **$25.8\text{ FPS}$ belongs to OPUS-V2-T at $35.6 - 36.5\%\text{ mIoU}$**.
  - Ingests $T=8$ causal past frames; `fut-15f` is bidirectional offline auto-labeling, not forward forecasting.
- **Open Issues & Limitations:**
  - Sparse queries hallucinate bounding boxes with high confidence under sensor failure.
  - Strictly locked to CUDA Linux via `spconv-cu116` and `torch-scatter`.

---

### REA 03: OCCUQ
- **Repository:** `https://github.com/ika-rwth-aachen/OCCUQ`
- **Audit Type:** Comprehensive source-code reverse engineering & GPU JIT verification (Status A)
- **Audited Commit SHA:** `2aa54297151b695929524ad8a4f4863ba5ae6387` (Branch `main`, Apache 2.0)
- **Publication:** ICRA 2025 (*"OCCUQ: Exploring Efficient Uncertainty Quantification for 3D Occupancy Prediction"*)
- **Audit Directory:** [`research/repository_audits/occuq/`](file://research/repository_audits/occuq/)
- **Primary Deliverables:**
  - [`REA_REPORT.md`](file://research/repository_audits/occuq/REA_REPORT.md)
  - [`SOURCE_MAP.md`](file://research/repository_audits/occuq/SOURCE_MAP.md)
  - [`DEPENDENCY_AND_DATA_AUDIT.md`](file://research/repository_audits/occuq/DEPENDENCY_AND_DATA_AUDIT.md)
  - [`REPRODUCTION_ASSESSMENT.md`](file://research/repository_audits/occuq/REPRODUCTION_ASSESSMENT.md)
  - [`COMPARISON_WITH_PREVIOUS_BASELINES.md`](file://research/repository_audits/occuq/COMPARISON_WITH_PREVIOUS_BASELINES.md)
  - [`AUDIT_REVIEW_ADDENDUM.md`](file://research/repository_audits/occuq/AUDIT_REVIEW_ADDENDUM.md)
- **Primary Findings:**
  - Adapts Deep Deterministic Uncertainty (DDU) to 3D occupancy.
  - Applies spectral normalization to prediction head (`MLPHeadv5`) to enforce a bi-Lipschitz condition, preventing feature collapse.
  - Fits class-conditional GMM in latent space; evaluates data log-density $\log p(\mathbf{z})$ as an epistemic uncertainty/OOD score.
  - Precomputes precision Cholesky factor $\mathbf{P}_c = \mathbf{L}_c^{-1}$ and executes TorchScript JIT GPU kernel (`jit_log_prob`), reducing 640k voxel inference time from **$938.41\text{ ms}$ on CPU to $9.48\text{ ms}$ on GPU ($\approx 100\times$ speedup)**.
  - Validated on MultiCorrupt: achieves **$95.39\%\text{ AUROC}$** under 3-camera blackout (where standard softmax entropy achieves only $50.82\%$).
- **Open Issues & Limitations:**
  - Built on dense SurroundOcc architecture, inheriting its slow $\approx 5.2\text{ FPS}$ backbone latency.
  - Does not calibrate aleatoric softmax probabilities or evaluate future horizons.

---

### REA 04: ReliOcc
- **Publication Target:** Proceedings of the 34th International Joint Conference on Artificial Intelligence (**IJCAI 2025**), pp. 1973–1981 / arXiv:2409.18026
- **Audit Type:** Peer-reviewed paper & supplemental specification reverse engineering (**Status D: No official code repository found**)
- **Authors:** Song Wang, Zhongdao Wang, Jiawei Yu, Wentong Li, Bailan Feng, Junbo Chen, Jianke Zhu (Zhejiang Univ / Huawei Noah's Ark Lab)
- **Audit Directory:** [`research/repository_audits/reliocc/`](file://research/repository_audits/reliocc/)
- **Primary Deliverables:**
  - [`REA_REPORT.md`](file://research/repository_audits/reliocc/REA_REPORT.md)
  - [`PAPER_AND_METHOD_MAP.md`](file://research/repository_audits/reliocc/PAPER_AND_METHOD_MAP.md)
  - [`DEPENDENCY_AND_DATA_AUDIT.md`](file://research/repository_audits/reliocc/DEPENDENCY_AND_DATA_AUDIT.md)
  - [`REPRODUCTION_ASSESSMENT.md`](file://research/repository_audits/reliocc/REPRODUCTION_ASSESSMENT.md)
  - [`COMPARISON_WITH_PREVIOUS_BASELINES.md`](file://research/repository_audits/reliocc/COMPARISON_WITH_PREVIOUS_BASELINES.md)
  - [`AUDIT_REVIEW_ADDENDUM.md`](file://research/repository_audits/reliocc/AUDIT_REVIEW_ADDENDUM.md)
- **Primary Findings:**
  - Dissects the reliability gap between LiDAR (ECE $\approx 0.7 - 2.9\%$) and camera (ECE $\approx 5.7 - 7.2\%$) on SemanticKITTI.
  - Formulates Hybrid Uncertainty: Absolute uncertainty ($L_{\text{au}}$ via Gaussian logit reparameterization) + Relative uncertainty ($L_{\text{ru}}$ via uncertainty-weighted voxel feature MixUp $\lambda = \frac{\sigma_i}{\sigma_i + \sigma_j}$).
  - In offline mode, fits uncertainty-aware diagonal temperature scaling $f_\phi$, slashing semantic ECE from $5.90\%$ to $2.05\%$ without degrading mIoU.
- **Open Issues & Limitations:**
  - No official code released.
  - In online mode, the uncertainty branch is discarded at inference (acts strictly as a training regularizer).
  - Evaluated on monocular forward camera (SemanticKITTI); does not evaluate 360° surround camera dropouts.

---

## 3. Summary Cross-Baseline Comparison Matrix

| Property | SurroundOcc | OPUS-V2 | OCCUQ | ReliOcc |
| :--- | :--- | :--- | :--- | :--- |
| **Audit Status** | Status A (Git verified) | Status A (Git verified) | Status A (Git verified) | Status D (Paper verified) |
| **Volumetric Paradigm** | Dense Query Volume | Sparse 3D Boxes + CPD | Dense Query Volume | Dense Feature Grid |
| **Camera Configuration** | 6 Surround Cameras | 6 Surround Cameras | 6 Surround Cameras | 1 Monocular Front Camera |
| **Temporal Context** | Single-frame ($T=1$) | 8-frame Causal ($T=8$) | Single-frame ($T=1$) | 5-frame Causal ($T=5$) |
| **Primary Metric** | $28.5 - 30.2\%\text{ mIoU}$ | $38.6\%\text{ mIoU}$ (Large) / $25.8\text{ FPS}$ (Tiny) | $29.8 - 30.5\%\text{ mIoU}$ | $13.43 - 15.65\%\text{ mIoU}$ (SemanticKITTI) |
| **Epistemic OOD Score** | None | None | **GPU JIT GMM Density** | None |
| **Aleatoric Calibration** | None | None | None | **Diagonal TempS ($f_\phi$)** |
| **Missing-Camera Detection**| Blind failure | Blind failure | **$95.39\%\text{ AUROC}$** | Temporal frame drop only |
| **Future 4D Forecasting** | None | None | None | None |
| **Trajectory Risk Output** | None | None | None | None |

---

## 4. Recommended Next Audit Target

To complete Stage 0 and directly address the transition to **Stage 5 (4D Occupancy and Flow Forecasting)** and **Stage 6 (Risk-Constrained Safety Layer)**, the most urgent research gap is:

> **Recommended Next Audit Target: OccWorld (ECCV 2024) or Cam4DOcc (CVPR 2024)**  
> *Objective:* Reverse engineer the spatiotemporal world model and temporal token forecasting dynamics to establish how future 4D occupancy and flow are predicted, and determine where uncertainty propagation can be injected into the temporal transition function.

