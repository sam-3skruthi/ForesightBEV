# REA 04 — ReliOcc Audit Review and Formal Closure Addendum

**Paper Title:** Reliable and Calibrated Semantic Occupancy Prediction by Hybrid Uncertainty Learning  
**Publication:** Proceedings of the 34th International Joint Conference on Artificial Intelligence (IJCAI 2025), pp. 1973–1981  
**DOI:** [10.24963/ijcai.2025/220](https://doi.org/10.24963/ijcai.2025/220)  
**Preprint:** arXiv:2409.18026v1  
**Audited Repository Status:** **Status D — No Verifiable Official Codebase Found**  
**Audit Completion Date:** 2026-11-04  
**Audit Scope:** Stage 0 REA 04 Formal Closure, Architecture Governance, and Gap Analysis  

---

## 1. Authoritative ForesightBEV Architecture Compliance

All findings, mappings, and evaluations in this audit strictly comply with the frozen ten-stage ForesightBEV framework:

- **Stage 0 — Literature Review, REA and Research Gap** (Active stage; REA 01 SurroundOcc, REA 02 OPUS, REA 03 OCCUQ, and REA 04 ReliOcc completed)
- **Stage 1 — Automotive Datasets, Cameras and Calibration**
- **Stage 2 — Multi-view Geometry and BEV Projection**
- **Stage 3 — 3D Occupancy Perception and Sensor Reliability** (Primary recipient of hybrid uncertainty learning and calibration)
- **Stage 4 — Dynamic Agent Detection and Tracking**
- **Stage 5 — 4D Occupancy and Flow Forecasting**
- **Stage 6 — Risk-Constrained Decision and Safety Layer** (Downstream consumer of calibrated voxel probabilities and sensor failure warnings)
- **Stage 7 — CARLA + ROS 2 + Autoware**
- **Stage 8 — FastAPI Backend and React Dashboard**
- **Stage 9 — Evaluation, Ablations and Research Reports**

No stage names, stage numbering, or boundaries have been altered, added, or reordered.

---

## 2. Enforcement of the Camera-Only Inference Mandate

ForesightBEV strictly enforces the camera-only inference rule:
1. **Camera Inputs Solely:** During runtime inference, the perception model receives only camera video streams.
2. **LiDAR Boundary:** LiDAR point clouds are restricted exclusively to offline dataset ground-truth generation, supervision losses during training, or separately declared comparison baselines.
3. **Audit Verification:** ReliOcc adheres strictly to the camera-only inference mandate. Unlike DeptS (*Depth-Aware Scaling*), which improperly relies on true point-cloud depth $d_i$ during test-time calibration, ReliOcc calculates its calibration temperature $T_\sigma = k_1 \sigma_i + k_2$ solely from the predicted uncertainty $\sigma_i$ derived from vision features $v_i$. No LiDAR data enters the model during camera inference.

---

## 3. Code Availability & Status Verification

### Status D: No Verifiable Official Implementation Found
- Exhaustive searches across IJCAI 2025 proceedings, arXiv:2409.18026, authors' academic and personal GitHub accounts (Song Wang, Jianke Zhu), and community tracking registries confirm that **the authors have not released an official code repository for ReliOcc**.
- In strict adherence to our audit guidelines, **no unverified or third-party reimplementations have been cloned or cited as official**.
- All conclusions in REA 04 are grounded directly in the peer-reviewed IJCAI 2025 archival text, formulas, and experimental data.

---

## 4. Methodological Inconsistencies & Unresolved Questions

Our independent review identified several critical technical nuances and unresolved questions in the published paper:

### 4.1 The Online Inference Paradox
- In Section 5 (*Training and Inference*), the authors explicitly state:
  > *"During inference, the model operates consistently with the original network design as the predicted uncertainties are not utilized."*
- **Audit Critique:** While the paper is framed around "uncertainty learning", the online uncertainty branch ($\sigma_i$) is discarded at test time. The network outputs standard discrete predictions or softmax scores. Therefore, in online mode, hybrid uncertainty is **purely a regularizing loss function during backpropagation**, not an inference uncertainty estimator. Only when the offline calibrator is trained and attached does the model output calibrated confidences ($c_i$) and derived uncertainties ($\sigma_i = 1 - c_i$).

### 4.2 Empty Voxel Domination in Random Shuffling
- In the SemanticKITTI volume ($256 \times 256 \times 32$), $>90\%$ of voxels represent empty space ($y=0$).
- If voxel features $v \in \mathcal{V}$ are randomly shuffled without class-aware masking, over $81\%$ of generated pairs $(v_i, v_j)$ will be empty-empty pairs ($0 + 0 = 0$).
- The paper does not specify whether empty voxels are filtered out during shuffling. If left unmasked, the relative loss $\mathcal{L}_{\text{ru}}$ would be dominated by empty space, diluting its regularizing effect on foreground semantic boundaries.

### 4.3 Multi-Label Cross-Entropy Formulation
- The mixed label is defined as $y = y_i + y_j$, where $y_i, y_j$ are one-hot vectors.
- Footnote 2 clarifies that label weighting by $\lambda$ is omitted for stability. However, standard categorical cross-entropy requires targets to be probability distributions ($\sum y = 1$). The paper does not disclose whether they utilized binary cross-entropy with logits (`BCEWithLogitsLoss`), soft cross-entropy with $y/2$, or multi-target cross-entropy.

### 4.4 Monocular vs. Surround-View Transferability
- ReliOcc is evaluated strictly on **monocular temporal forward cameras** (current left camera + 4 past frames) on SemanticKITTI.
- Its "sensor failure" robustness evaluates **temporal frame dropping** in the historical buffer.
- This setup cannot be assumed to transfer directly to **multi-camera surround arrays** (such as nuScenes in SurroundOcc, OPUS, and OCCUQ), where sensor failure entails losing an entire spatial camera perspective ($60^\circ - 180^\circ$ blind zone).

---

## 5. Research Novelty & Gap Assessment for ForesightBEV

The audit affirms that while ReliOcc provides valuable regularizing techniques, its concepts build on established foundations:
- Heteroscedastic aleatoric uncertainty sampling originates from Kendall & Gal (NeurIPS 2017).
- Relative uncertainty mix-up originates from Zhang et al. (NeurIPS 2021, facial expression recognition).
- Temperature scaling originates from Guo et al. (ICML 2017).

### True Remaining Research Gaps for ForesightBEV:
1. **Unified Epistemic-Aleatoric Architecture for Surround-View 3D Occupancy:** Combining OCCUQ-style feature density monitoring (for spatial camera failure detection) with ReliOcc-style relative mix-up regularizers (for boundary calibration) in a unified multi-camera framework.
2. **4D Spatiotemporal Uncertainty Calibration:** Extending voxel reliability calibration from static 3D occupancy (Stage 3) to forward temporal occupancy and flow forecasting (Stage 5).
3. **Trajectory-Event Closed-Loop Risk Assessment (Stage 6):** Converting calibrated voxel probabilities and epistemic sensor degradation flags into formally constrained path planning safety margins.

---

## 6. Formal Closure of REA 04

All six required deliverables have been published in `research/repository_audits/reliocc/`:
1. `REA_REPORT.md`
2. `PAPER_AND_METHOD_MAP.md`
3. `DEPENDENCY_AND_DATA_AUDIT.md`
4. `REPRODUCTION_ASSESSMENT.md`
5. `COMPARISON_WITH_PREVIOUS_BASELINES.md`
6. `AUDIT_REVIEW_ADDENDUM.md`

All upstream repositories in `research/repositories/` remain pristine and unmodified.

**REA 04 (ReliOcc) is hereby formally closed.**

