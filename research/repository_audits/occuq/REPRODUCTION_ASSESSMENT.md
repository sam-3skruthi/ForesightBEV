# REA 03 — OCCUQ Reproduction Assessment

**Repository:** [https://github.com/ika-rwth-aachen/OCCUQ](https://github.com/ika-rwth-aachen/OCCUQ)  
**Audited Commit:** `2aa54297151b695929524ad8a4f4863ba5ae6387`  
**Parent Review Date:** 2026-11-04  
**Status:** Practical Feasibility, Risk Matrix, and Reproduction Blueprint  

---

## 1. Reproduction Feasibility Overview

OCCUQ achieves a **High to Moderate-High** practical reproducibility rating. Unlike many academic perception repositories where checkpoint links decay or custom CUDA kernels fail to build on newer compilers, the RWTH Aachen team provides:
1. Complete official Docker container definitions (`docker/Dockerfile` based on NVIDIA NGC PyTorch 21.10).
2. Public cloud storage hosting all trained weights and fitted GMM checkpoints on Sciebo (RWTH Academic Cloud: `https://rwth-aachen.sciebo.de/s/2o1LOb4PwFbPzSb`).
3. Complete OOD test set availability on Hugging Face (`TillBeemelmanns/MultiCorrupt`).
4. Self-contained TorchScript JIT benchmarking notebook (`notebooks/gmm_jit_log_prob.ipynb`) that runs independently on modern PyTorch ($\geq 2.0$).

---

## 2. Artifact Availability & Checkpoint Verification

| Checkpoint / Artifact | Upstream Hosting Location | Verified Size | Status / Notes |
| :--- | :--- | :--- | :--- |
| **OCCUQ Fine-Tuned Weights (`epoch_6.pth`)** | Sciebo RWTH Cloud (`s/2o1LOb4PwFbPzSb`) | $\approx 2.1\text{ GB}$ | ResNet-101 + FPN + 3D Cross-Attention + MLPHeadv5 with Spectral Normalization. |
| **SurroundOcc Pretrained Base (`surroundocc.pth`)** | Tsinghua Cloud / Sciebo | $\approx 2.1\text{ GB}$ | Checkpoint used to initialize OCCUQ fine-tuning. |
| **Fitted GMM Tensors (`train_gmm_scale_3.pt`)** | Sciebo RWTH Cloud / Local repo `notebooks/` | $\approx 78\text{ KB}$ | Contains `MultivariateNormal(loc, covariance_matrix)` for 17 classes in 32-dim feature space. |
| **Class Prior Tensors (`train_prior_log_prob_scale_3.pt`)** | Sciebo RWTH Cloud | $\approx 1\text{ KB}$ | Prior log probabilities $\log \pi_c$ for 17 classes. |
| **MultiCorrupt Benchmark Dataset** | Hugging Face: `TillBeemelmanns/MultiCorrupt` | $\approx 60\text{ GB}$ | Validated active repository on Hugging Face hub. |

---

## 3. Step-by-Step Reproduction Blueprint

### Step 1: Docker Environment Setup
To isolate the legacy OpenMMLab 1.x and MMCV 1.4.0 dependencies, use the provided Docker container:

```bash
cd research/repositories/OCCUQ/docker
./build.sh
./run.sh
```

Inside the container:
```bash
cd /workspace/extensions/chamfer_dist
python setup.py install --user
```

### Step 2: Download Weights and Datasets
```bash
mkdir -p /workspace/work_dirs/occuq_mlpv5_sn
# Download from Sciebo link into /workspace/work_dirs/occuq_mlpv5_sn:
# - epoch_6.pth
# - train_gmm_scale_3.pt
# - train_prior_log_prob_scale_3.pt
```

### Step 3: Run Fast Standalone GPU Verification
To verify the $\approx 9.5\text{ ms}$ GPU inference acceleration without downloading the entire 350 GB nuScenes dataset:
```bash
jupyter nbconvert --to notebook --execute notebooks/gmm_jit_log_prob.ipynb
```
This loads `notebooks/train_gmm_scale_3.pt`, synthesizes 640,000 32-dim features on GPU, verifies precision Cholesky factorization, and validates the $100\times$ speedup over CPU execution.

### Step 4: Out-of-Distribution Benchmark Reproduction
```bash
cd /workspace
./scripts/gmm_multicorrupt_evaluate.sh
```
This script executes clean evaluation, loops through all 15 corruption scenarios in `multicorrupt/`, saves per-sample uncertainty logs, and executes `scripts/ood_detection_evaluation.py` to output the macro AUROC, AUPR, and FPR95 summary table.

---

## 4. Technical Failure Modes & Mitigation Strategies

| Failure Mode | Root Cause | Impact | Verified Mitigation |
| :--- | :--- | :--- | :--- |
| **Host System RAM Exhaustion during GMM Fit** | `get_features_balanced` collects up to 200M features ($32\text{ GB}$ raw tensors) in host memory. | Kernel OOM killer terminates Python process. | In `tools/gmm_fit.py`, lower `max_features_per_class` from `200000000` to `2000000` (2M features per class yields identical covariance convergence with $< 4\text{ GB}$ RAM footprint). |
| **Cholesky Covariance Matrix Singularity** | Class features in rare categories (e.g. `traffic_cone` or `construction_vehicle`) have near-zero variance along specific dimensions. | `RuntimeError: cholesky_cpu: U(k,k) is zero`. | Handled automatically in `tools/gmm_utils.py` lines 193–207 via adaptive jitter loop ($0 \dots 10^{-1}$). If degenerate, synthetic regularization $\epsilon = 10^{-4}$ ensures numerical stability. |
| **Host Clang / GCC Incompatibility** | Compiling MMCV or Chamfer distance on modern host systems running GCC $\geq 11$. | Build failure during C++ extension setup. | Enforce GCC 6/7 in Conda environment or utilize the audited NVIDIA NGC 21.10 Docker container. |
| **Apple Silicon Execution Attempt** | Native `torch.device("mps")` lacks deformable attention CUDA operators. | Immediate failure upon model construction. | Decouple pipeline: run heavy 3D occupancy backbone on remote Linux CUDA host; run lightweight downstream safety policies and dashboard on macOS. |

---

## 5. Architectural Recommendation for ForesightBEV

1. **Adopt OCCUQ's Uncertainty Formulation in Stage 3:**  
   OCCUQ provides the exact mathematical and computational building block needed for ForesightBEV Stage 3 (3D Occupancy Perception and Sensor Reliability). Integrating `MLPHeadv5` with spectral normalization and the TorchScript JIT GMM density evaluator adds only **$\approx 9.5\text{ ms}$** of overhead on GPU, transforming raw occupancy predictions into safety-critical, uncertainty-calibrated perception.
2. **Reuse MultiCorrupt for Stage 9 Validation:**  
   The MultiCorrupt dataset protocol (snow, fog, motion blur, brightness, missing camera) should be adopted as our standard benchmark for evaluating sensor degradation and out-of-distribution robustness in Stage 9 reports.

