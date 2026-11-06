# REA 03 — OCCUQ Audit Review and Formal Closure Addendum

**Audited Repository:** [https://github.com/ika-rwth-aachen/OCCUQ](https://github.com/ika-rwth-aachen/OCCUQ)  
**Audited Commit SHA:** `2aa54297151b695929524ad8a4f4863ba5ae6387`  
**Remote URL:** `https://github.com/ika-rwth-aachen/OCCUQ`  
**Branch:** `main` (verified clean working tree)  
**License:** Apache License 2.0 (`LICENSE`)  
**Audit Completion Date:** 2026-11-04  
**Audit Scope:** Stage 0 REA 03 Formal Closure and Architecture Governance Sign-Off  

---

## 1. Authoritative ForesightBEV Architecture Compliance

All findings, mappings, and technical conclusions in this audit adhere strictly to the frozen ten-stage ForesightBEV architecture:

- **Stage 0 — Literature Review, REA and Research Gap** (Active stage; REA 01, REA 02, and REA 03 completed)
- **Stage 1 — Automotive Datasets, Cameras and Calibration**
- **Stage 2 — Multi-view Geometry and BEV Projection**
- **Stage 3 — 3D Occupancy Perception and Sensor Reliability** (Primary home of OCCUQ uncertainty quantification)
- **Stage 4 — Dynamic Agent Detection and Tracking**
- **Stage 5 — 4D Occupancy and Flow Forecasting**
- **Stage 6 — Risk-Constrained Decision and Safety Layer** (Primary downstream consumer of Stage 3 reliability signals)
- **Stage 7 — CARLA + ROS 2 + Autoware**
- **Stage 8 — FastAPI Backend and React Dashboard**
- **Stage 9 — Evaluation, Ablations and Research Reports**

No stage names, stage numbering, or boundaries have been altered, added, or reordered.

---

## 2. Enforcement of the Camera-Only Inference Mandate

ForesightBEV operates under an uncompromised camera-centric inference mandate:
1. **Camera-Only Inference:** Camera images are the sole sensory input accepted by the perception network at runtime.
2. **LiDAR Boundary:** LiDAR point clouds are strictly restricted to:
   - Offline ground-truth generation and mesh surface reconstruction (e.g. Poisson surface reconstruction from SurroundOcc).
   - Supervised loss computation during training ($\mathcal{L}_{\text{geo}}$, Chamfer distance).
   - Separately declared multimodal comparison baselines.
3. **No Silent Leakage:** The OCCUQ codebase was audited line by line to verify that LiDAR is not silently accessed or leaked into the camera inference forward path. In `projects/configs/occuq/occuq_mlpv5_sn.py`, `input_modality.use_lidar = False` and `use_camera = True` are strictly maintained during `simple_test` and `gmm_evaluate.py`.

---

## 3. Authoritative Technical Reconciliation for OCCUQ

### 3.1 Mathematical Foundation: DDU Adapted to 3D Occupancy
- Standard deep networks exhibit feature collapse under distribution shifts, mapping out-of-distribution inputs into the same latent space as training data.
- OCCUQ prevents feature collapse by applying **Spectral Normalization** (`torch.nn.utils.spectral_norm`) across all linear layers of its residual prediction head (`MLPHeadv5`). This enforces a bi-Lipschitz mapping:
  $$L_1 \|\mathbf{x}_1 - \mathbf{x}_2\| \leq \|f(\mathbf{x}_1) - f(\mathbf{x}_2)\| \leq L_2 \|\mathbf{x}_1 - \mathbf{x}_2\|$$
  preserving metric distances between in-distribution and corrupted inputs in feature space $\mathbb{R}^{32}$.
- An offline class-conditional Gaussian Mixture Model (GMM) fits empirical mean $\boldsymbol{\mu}_c$ and covariance $\boldsymbol{\Sigma}_c$ for all 17 classes ($0 = \text{unoccupied}$, $1 \dots 16 = \text{semantics}$).
- The marginal data log-likelihood:
  $$\log p(\mathbf{z}) = \text{LogSumExp}_{c=0}^{16}\left( \log \pi_c + \log \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c) \right)$$
  quantifies epistemic uncertainty: corrupted or out-of-distribution features yield low density ($\log p(\mathbf{z}) \to -\infty$).

### 3.2 Computational Feasibility: 100x GPU Acceleration
- Evaluating multivariate normal log-likelihood on $640,000$ voxels ($200 \times 200 \times 16$) using standard PyTorch CPU methods takes **$938.41\text{ ms}$**, an intolerable delay for autonomous vehicle control.
- OCCUQ precomputes the Cholesky factor of the precision matrix:
  $$\mathbf{P}_c = \mathbf{L}_c^{-1} \quad \text{where} \quad \boldsymbol{\Sigma}_c = \mathbf{L}_c \mathbf{L}_c^\top$$
  transforming the Mahalanobis distance calculation into a matrix multiplication followed by an elementwise norm.
- Implemented as a compiled TorchScript JIT kernel (`jit_log_prob` in `tools/gmm_utils.py`), GPU execution time drops to **$9.48\text{ ms}$** ($\approx 100\times$ speedup) with numerical discrepancy $< 0.0026$.

### 3.3 Sensor Degradation & OOD Detection Performance
- Standard softmax entropy and maximum softmax probability fail completely under severe sensor dropouts (`missingcamera_3` AUROC: **$50.82\%$** and **$54.22\%$** — near coin-toss chance). The baseline network remains dangerously overconfident in blind spots.
- OCCUQ GMM achieves **$95.39\%\text{ AUROC}$** with an FPR95 of **$18.03\%$** under 3-camera dropouts, and **$80.15\%\text{ mAUROC}$** across the entire MultiCorrupt benchmark (snow, fog, motion blur, brightness, missing cameras).

---

## 4. Upstream Integrity Verification

- Audited upstream directory: `research/repositories/OCCUQ/`
- Upstream Git status: Clean (`nothing to commit, working tree clean`)
- Upstream Git commit: `2aa54297151b695929524ad8a4f4863ba5ae6387`
- No upstream files were modified, committed, or deleted during this audit.

---

## 5. Formal Closure of REA 03

With the publication of the five core audit documents and this review addendum in `research/repository_audits/occuq/`:
1. `REA_REPORT.md`
2. `SOURCE_MAP.md`
3. `DEPENDENCY_AND_DATA_AUDIT.md`
4. `REPRODUCTION_ASSESSMENT.md`
5. `COMPARISON_WITH_PREVIOUS_BASELINES.md`
6. `AUDIT_REVIEW_ADDENDUM.md`

and the preceding completion of `research/repository_audits/opus/REA02_VERIFICATION_ADDENDUM.md`, **REA 03 (OCCUQ) is hereby formally validated, approved, and closed**.

