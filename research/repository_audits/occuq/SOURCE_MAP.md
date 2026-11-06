# REA 03 — OCCUQ Source-Code Implementation Map

**Repository:** [https://github.com/ika-rwth-aachen/OCCUQ](https://github.com/ika-rwth-aachen/OCCUQ)  
**Audited Commit:** `2aa54297151b695929524ad8a4f4863ba5ae6387`  
**Parent Review Date:** 2026-11-04  
**Status:** Complete Source-Code Inventory and Lineage Analysis  

---

## 1. Directory Structure Overview

The OCCUQ codebase extends the SurroundOcc / MMDetection3D framework with uncertainty estimation modules, GPU JIT acceleration, and OOD evaluation routines:

```
research/repositories/OCCUQ/
├── assets/                       # Paper figures, GIFs, and poster
├── docker/                       # Docker container setup (PyTorch 21.10, CUDA 11.4/11.3)
│   ├── Dockerfile                # Environment build instructions
│   ├── build.sh                  # Container build script
│   ├── run.sh                    # Container interactive runner
│   └── run_in_container.sh       # Execution script inside container
├── docs/                         # Documentation guides
│   ├── data.md                   # nuScenes & MultiCorrupt data preparation
│   ├── install.md                # Conda, CLAIX HPC cluster, and Docker setup
│   └── run.md                    # Training, GMM fitting, and evaluation walkthrough
├── extensions/                   # Custom C++/CUDA extensions
│   └── chamfer_dist/             # Chamfer distance GPU operator (from SurroundOcc)
│       ├── chamfer.cu            # CUDA kernel for Chamfer distance
│       ├── chamfer_cuda.cpp      # PyTorch C++ bindings
│       └── setup.py              # Compilation setup
├── notebooks/                    # Jupyter notebooks for verification
│   ├── gmm_jit_log_prob.ipynb    # GPU acceleration benchmark notebook
│   └── train_gmm_scale_3.pt      # Pretrained GMM tensor checkpoint
├── projects/                     # Core plugin directory
│   ├── configs/                  # Configuration files
│   │   ├── _base_/               # Base runtime configurations
│   │   ├── datasets/             # Base dataset loaders
│   │   └── occuq/                # OCCUQ model configurations
│   │       ├── occuq_mlpv5_sn.py # Authoritative OCCUQ config (ResNet101 + MLPHeadv5 + SN)
│   │       └── surroundocc.py    # Baseline SurroundOcc config
│   └── mmdet3d_plugin/           # Custom MMDetection3D plugin
│       ├── datasets/             # Occupancy dataset loaders and metrics
│       │   ├── evaluation_metrics.py # Semantic IoU, ECE (netcal), NLL evaluation
│       │   ├── nuscenes_occupancy_dataset.py # nuScenes loader for .npy ground truth
│       │   └── pipelines/        # Loading, augmentations, and formatting
│       └── surroundocc/          # Detector and head implementation
│           ├── dense_heads/
│           │   └── occ_head.py   # MLPHead variants (v1..v6), SN, OccHead
│           ├── detectors/
│           │   └── surroundocc.py # SurroundOcc detector with MCD and DE test modes
│           └── modules/          # Spatial cross-attention, transformer encoder
├── scripts/                      # Evaluation and training bash scripts
│   ├── gmm_fit_evaluate.sh       # Fit and evaluate GMM
│   ├── gmm_multicorrupt_evaluate.sh # Full MultiCorrupt OOD benchmark script
│   ├── gmm_multicorrupt_video.sh    # MultiCorrupt video generation
│   ├── ood_detection_evaluation.py  # AUROC, AUPR, FPR95 evaluator
│   └── train_occuq.sh            # Distributed 4-GPU training script
└── tools/                        # Command-line executables
    ├── gmm_analyze.py            # Feature, logit, and label extraction
    ├── gmm_evaluate.py           # Clean and corrupt inference with UQ metrics
    ├── gmm_fit.py                # Class-balanced feature collection and GMM fitting
    ├── gmm_utils.py              # TorchScript JIT log-prob, precision-Cholesky, helpers
    ├── gmm_video.py              # Open3D / Matplotlib point cloud video generator
    └── train.py / test.py        # Standard MMDetection training and testing runners
```

---

## 2. Granular Source-Code Component Inventory

### 2.1 Uncertainty-Quantified Occupancy Head
**File:** `projects/mmdet3d_plugin/surroundocc/dense_heads/occ_head.py`

| Symbol / Class | Line Range | Architectural Role & Implementation Details |
| :--- | :--- | :--- |
| `MLPHead` | Lines 37–58 | Three-layer fully-connected head (`Linear(in, hidden) -> ReLU -> Linear(hidden, hidden) -> ReLU -> Linear(hidden, out)`). Supports `use_sn` (spectral normalization via `torch.nn.utils.spectral_norm`). Returns logits and detached penultimate features. |
| `MLPHeadv2` | Lines 60–86 | Four-layer head with `LayerNorm` after each linear mapping and residual connection before output layer. |
| `MLPHeadv3` | Lines 88–142 | Five-layer head with 3D batch normalization (`BatchNorm3d`) and `LeakyReLU` activations. |
| `MLPHeadv4` | Lines 144–197 | Five-layer head with `BatchNorm3d` and standard `ReLU` activations. |
| **`MLPHeadv5`** | **Lines 199–234** | **Authoritative OCCUQ Head.** Five-layer residual architecture: each layer has an identity skip connection (`x = F.relu(self.fc_k(x)) + identity`). Uses `spectral_norm` on all weight matrices when `use_sn=True`. Binds Lipschitz constant to prevent feature collapse. Detaches penultimate layer feature `feature = x.clone().detach()` ($\mathbb{R}^{32}$) and projects to 17 output classes via `fc5`. |
| `MLPHeadv5NoSkip` | Lines 236–263 | Ablation variant of `MLPHeadv5` without residual skip connections. |
| `MLPHeadv6` | Lines 265–289 | Two-layer residual head with spectral normalization. |
| `OccHead` | Lines 293–596 | Multiscale 3D occupancy head. Manages volumetric queries across 4 decoder stages. Conditionally instantiates `mlp_heads` across output indices (`out_indices=[0, 2, 4, 6]`). Permutes volume tensors to `(B, H, W, Z, C)`, passes through MLP, caches penultimate features in `self.feature`, and permutes back to `(B, C, H, W, Z)`. |

---

### 2.2 Detector Architecture & Uncertainty Inference Modes
**File:** `projects/mmdet3d_plugin/surroundocc/detectors/surroundocc.py`

| Function / Method | Line Range | Implementation Details |
| :--- | :--- | :--- |
| `entropy_prob(probs)` | Lines 23–27 | Computes Shannon entropy: $\mathcal{H}(p) = -\sum_c p_c \log(p_c + 10^{-12})$. |
| `mutual_information_prob(probs, predictive_entropy)` | Lines 29–35 | Computes mutual information for ensemble / MCD: $\mathcal{I}(y, \mathbf{w}) = \mathcal{H}(\mathbb{E}[p]) - \mathbb{E}[\mathcal{H}(p)]$. Disentangles model uncertainty from data noise. |
| `SurroundOcc.__init__` | Lines 39–90 | Initializes ResNet-101 2D backbone, FPN neck, and `OccHead`. Sets up deep ensemble checkpoint containers (`self.state_dicts`). |
| `forward` | Lines 144–160 | Dispatches inference mode: `forward_train` if `return_loss=True`, `forward_mcd_test` if `mcdropout=True`, `forward_de_test` if `deepensembles=True`, else standard `forward_test`. |
| `forward_test` | Lines 179–199 | Standard single-pass forward test. Extracts multi-scale occupancy predictions `self.logits`, computes softmax confidences, and returns ECE / mIoU metrics via `evaluation_semantic`. |
| `forward_mcd_test` | Lines 201–230 | Monte Carlo Dropout inference. Executes $N=5$ forward passes with dropout active (`copy.deepcopy(img)`). Stacks prediction probabilities, calculates predictive mean, predictive entropy, and mutual information. |
| `forward_de_test` | Lines 233–265 | Deep Ensemble inference. Iterates over 5 pretrained model state dicts (`self.state_dicts`), runs forward evaluation, stacks softmax outputs, and computes ensemble predictive entropy and mutual information. |

---

### 2.3 Mathematical Core: GMM Utilities & GPU Acceleration
**File:** `tools/gmm_utils.py`

| Function | Line Range | Implementation Details |
| :--- | :--- | :--- |
| **`jit_log_prob`** | **Lines 17–33** | **TorchScript JIT Kernel for GPU Acceleration.** Inputs features $\mathbf{x} \in \mathbb{R}^{V \times D}$, means $\boldsymbol{\mu} \in \mathbb{R}^{K \times D}$, and Cholesky precision factors $\mathbf{P} \in \mathbb{R}^{K \times D \times D}$. Evaluates inner dot products: `inner = x.matmul(prec_chol) - mu.matmul(prec_chol)`. Computes quadratic norm `inner.square().sum(1)`. Adds log-determinant and Gaussian normalization constant. Achieves **$\approx 9.5\text{ ms}$ GPU latency** for $640,000$ voxels. |
| `means_precisions_cholesky` | Lines 35–44 | Precomputes Cholesky factor $\mathbf{L}_c$ and precision factor $\mathbf{P}_c = \mathbf{L}_c^{-1} = \text{inv}(\text{chol}(\boldsymbol{\Sigma}_c))$ using `torch.linalg.cholesky` and `torch.inverse`. |
| `multiscale_supervision` | Lines 60–68 | Downsamples dense ground-truth occupancy coordinates by ratio $r = 2^s$ to match multi-scale feature grids ($[200, 100, 50, 25]$). |
| `get_features_balanced` | Lines 71–160 | Balanced training feature collection. Loops over training dataloader, samples up to `max_features_per_class` (default 200,000,000) with a per-frame cap `max_collect_features_per_class_per_frame` (100,000), shuffling samples to avoid spatial correlation bias. |
| `gmm_fit` | Lines 163–215 | Fits class-conditional multivariate Gaussians. Computes empirical mean $\boldsymbol{\mu}_c$ and covariance $\boldsymbol{\Sigma}_c = \text{cov}(\mathbf{z}_c)$. Tests adaptive jitter diagonal increments $\epsilon \in \{0, 10^{-308}, \dots, 10^{-1}\}$ until `torch.distributions.MultivariateNormal` succeeds without Cholesky singularity errors. |
| `get_prior_log_prob` | Lines 217–220 | Computes empirical class prior $\log \pi_c = \log(N_c / N_{\text{total}})$. |
| `gmm_evaluate` | Lines 269–309 | Core evaluation loop. Ingests test samples, evaluates `jit_log_prob`, adds prior log-probabilities, evaluates marginal epistemic density $\log p(\mathbf{z}) = \text{logsumexp}(\log p(\mathbf{z}, y))$, calculates softmax entropy, and records maximum softmax probability. |

---

### 2.4 Command-Line Executables & Scripts

| File Path | Purpose & CLI Invocations |
| :--- | :--- |
| `tools/gmm_fit.py` | CLI entry point to extract features from a trained checkpoint and export `train_gmm_scale_X.pt` and `train_prior_log_prob_scale_X.pt`. |
| `tools/gmm_evaluate.py` | Evaluates clean or corrupted nuScenes sets. Saves CSV density logs (`gmm_uncertainty_per_sample.csv`, `softmax_entropy_per_sample.csv`, `max_softmax_per_sample.csv`) and generates matplotlib histogram distributions. |
| `tools/gmm_video.py` | 855-line visualization engine. Combines multi-camera 2D perspective views, predicted 3D semantic voxel meshes, and 3D epistemic uncertainty point-cloud heatmaps into animated video frames. |
| `scripts/ood_detection_evaluation.py` | Computes ROC curves, AUROC, AUPR, and FPR95 (false positive rate at 95% true positive rate) using scikit-learn metrics comparing in-distribution (clean) vs. OOD (corrupted) samples. |
| `scripts/gmm_multicorrupt_evaluate.sh` | Orchestration script that fits GMM at selected scale (default Scale 3), runs clean evaluation, loops through all 15 corruption scenarios (snow 1..3, fog 1..3, motionblur 1..3, brightness 1..3, missingcamera 1..3), and invokes `ood_detection_evaluation.py`. |

---

### 2.5 Evaluation Metrics & Expected Calibration Error
**File:** `projects/mmdet3d_plugin/datasets/evaluation_metrics.py`

| Metric Computation | Lines | Implementation Details |
| :--- | :--- | :--- |
| Geometry IoU | Lines 59–63 | Binary intersection-over-union of non-empty space: $\text{IoU}_{\text{geo}} = \frac{|\text{pred} \neq 0 \cap \text{gt} \neq 0|}{|\text{pred} \neq 0 \cup \text{gt} \neq 0|}$. |
| Semantic IoU | Lines 64–68 | Class-wise semantic intersection-over-union for classes $1 \dots 16$. |
| **Expected Calibration Error (ECE)** | **Lines 73–76** | Uses `netcal.metrics.ECE(10).measure(confidences[mask], gt[mask] == pred[mask])`. Partitions voxel predictions into 10 confidence bins and calculates weighted deviation between accuracy and confidence. |
| **Negative Log-Likelihood (NLL)** | **Lines 78–84** | Evaluates cross-entropy / negative log-likelihood on masked non-ignore voxels: `F.nll_loss(torch.log(softmax + 1e-8), gt)`. |

---

## 3. Configuration Architecture
**File:** `projects/configs/occuq/occuq_mlpv5_sn.py`

- **Point cloud range:** `[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]` ($100\text{ m} \times 100\text{ m} \times 8\text{ m}$ volumetric bounding box).
- **Occupancy grid resolution:** `[200, 200, 16]` ($0.5\text{ m}$ isotropic voxel resolution).
- **Backbone:** ResNet-101 (`depth=101`, stages 1..3, DCNv2 on stages 2 and 3).
- **Neck:** FPN with 3 output levels, `out_channels=512`.
- **Transformer:** Multi-scale deformable 3D spatial cross-attention across 3 query pyramid stages:
  - Stage 0: $25 \times 25 \times 2$, embed dim 512, 8 deformable points, 6 layers.
  - Stage 1: $50 \times 50 \times 4$, embed dim 256, 4 deformable points, 3 layers.
  - Stage 2: $100 \times 100 \times 8$, embed dim 128, 2 deformable points, 1 layer.
- **Volumetric Decoder:** 3D transposed convolution blocks upsampling from $25 \times 25 \times 2$ to $200 \times 200 \times 16$.
- **UQ Settings:**
  - `use_mlp = "MLPHeadv5"`
  - `use_sn = True` (spectral normalization active on all head layers)
- **Training schedule:** AdamW optimizer (`lr=5e-5`, weight decay `0.01`), cosine annealing, 6 epochs, initialized from pretrained SurroundOcc checkpoint (`surroundocc.pth`).

---

## 4. Key Architectural Insights for ForesightBEV

1. **Lightweight Head Adaptation:** OCCUQ does not alter the underlying SurroundOcc backbone, transformer encoder, or 3D decoder. The entire modification resides in substituting the linear 1x1x1 conv output layer with `MLPHeadv5` and enabling spectral normalization.
2. **Post-Hoc Epistemic Calibration:** GMM density fitting is performed offline on training representations without requiring retraining of the feature extractor.
3. **Dual-Channel Stage 3 Output:** ForesightBEV Stage 3 can output both semantic predictions and epistemic reliability heatmaps in a single forward pass with under $10\text{ ms}$ overhead on GPU.

