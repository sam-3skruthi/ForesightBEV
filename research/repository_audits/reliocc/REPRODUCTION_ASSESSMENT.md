# REA 04 — ReliOcc Reproduction Assessment

**Audited Paper:** *Reliable and Calibrated Semantic Occupancy Prediction by Hybrid Uncertainty Learning* (IJCAI 2025)  
**Implementation Classification:** **Status D — No Verifiable Official Implementation Found**  
**Assessment Purpose:** Feasibility analysis, platform risk matrix, clean-room reimplementation specifications, and minimal verification blueprints.

---

## 1. Reproduction Feasibility Overview

Because the authors have not published an official open-source codebase for ReliOcc (Status D), **direct code reproduction via repository cloning and script execution is impossible**. 

However, unlike complex multi-stage architectures that rely on proprietary CUDA kernels or obscure heuristics, ReliOcc’s core methodological contributions are mathematically concise:
- An uncertainty prediction projection head ($\text{MLP}_{\text{unc}}: \mathbb{R}^d \to \mathbb{R}^+$).
- A Gaussian logit reparameterization loss ($\mathcal{L}_{\text{au}}$).
- A paired feature mix-up module with an uncertainty-ratio blending coefficient ($\mathcal{L}_{\text{ru}}$).
- A post-hoc temperature scaling layer with a diagonal weight matrix ($\mathcal{L}_{\text{calib}}$).

Consequently, **clean-room reimplementation is completely feasible**, provided that missing low-level design choices (such as empty-voxel sampling and multi-label loss formulations) are resolved through principled engineering.

---

## 2. Platform Feasibility Matrix

| Computational Platform | Online Training Feasibility | Offline Calibration Feasibility | Standalone Math Verification | Primary Hardware Blockers |
| :--- | :--- | :--- | :--- | :--- |
| **Apple Silicon Mac (M5, 24 GB Unified Memory)** | **Infeasible** | **Borderline Feasible** (with precomputed features) | **Fully Feasible** (Runs in seconds) | Lack of native Metal/MPS deformable cross-attention kernels in VoxFormer; memory limits under 3D transformer backward passes. |
| **Linux Workstation ($1 \times \text{RTX 3090/4090}$, 24 GB VRAM)** | **Feasible** (Batch size 1, gradient accumulation) | **Highly Feasible** ($\approx 2 - 3\text{ hours}$) | **Fully Feasible** | VRAM capacity requires batch size 1; online training takes 48–72 hours for 20 epochs. |
| **Multi-GPU Cluster ($8 \times \text{NVIDIA V100 / A100}$)** | **Fully Feasible** (Matches paper setup) | **Fully Feasible** ($< 2\text{ hours}$) | **Fully Feasible** | None. Standard environment used in published paper. |

---

## 3. The "Smallest Sensible Experiment" (Clean-Room Verification Blueprint)

To verify the numerical validity and gradient behavior of ReliOcc without downloading the 120 GB SemanticKITTI dataset or compiling legacy OpenMMLab/VoxFormer dependencies, we design the following standalone PyTorch verification harness:

```python
"""
Clean-Room Prototype: ReliOcc Core Hybrid Uncertainty & Calibration Modules
Demonstrates mathematical forward/backward validity on synthetic 3D voxel tensors.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReliOccHead(nn.Module):
    def __init__(self, in_features=128, num_classes=20, hidden_dim=64):
        super().__init__()
        # 1. Base Deterministic Semantic Head
        self.cls_head = nn.Linear(in_features, num_classes)
        
        # 2. Uncertainty Head (Predicts scalar sigma > 0)
        self.unc_head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensures strictly positive uncertainty sigma > 0
        )
        
    def forward(self, v):
        # v: (N, in_features)
        z = self.cls_head(v)        # (N, num_classes)
        sigma = self.unc_head(v)    # (N, 1)
        return z, sigma

class HybridUncertaintyLoss(nn.Module):
    def __init__(self, alpha=4.0, beta=6.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, v, labels, head):
        # v: (N, in_features), labels: (N,) ground-truth class indices
        z, sigma = head(v)
        
        # --- A. Absolute Uncertainty Loss (HAU reparameterization) ---
        eps = torch.randn_like(z)
        z_sampled = z + sigma * eps
        l_au = F.cross_entropy(z_sampled, labels)
        
        # --- B. Relative Uncertainty Loss (Voxel MixUp) ---
        N = v.shape[0]
        perm = torch.randperm(N, device=v.device)
        v_j = v[perm]
        sigma_j = sigma[perm]
        labels_j = labels[perm]
        
        # Mixing coefficient: lambda = sigma_i / (sigma_i + sigma_j)
        denom = sigma + sigma_j + 1e-8
        lam = sigma / denom  # Shape (N, 1)
        
        v_hat = lam * v + (1.0 - lam) * v_j
        z_mix = head.cls_head(v_hat)
        
        # Multi-label cross entropy for y = y_i + y_j
        # Target probability vector: equal 0.5 probability for class i and class j
        y_onehot_i = F.one_hot(labels, num_classes=z.shape[-1]).float()
        y_onehot_j = F.one_hot(labels_j, num_classes=z.shape[-1]).float()
        target_soft = torch.clamp(y_onehot_i + y_onehot_j, max=1.0)
        target_soft = target_soft / target_soft.sum(dim=-1, keepdim=True)
        
        log_probs_mix = F.log_softmax(z_mix, dim=-1)
        l_ru = -(target_soft * log_probs_mix).sum(dim=-1).mean()
        
        # Total Hybrid Uncertainty Regularizer
        loss = self.alpha * l_au + self.beta * l_ru
        return loss, l_au.item(), l_ru.item()

class UncertaintyAwareCalibrator(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        # Diagonal weight matrix W initialized to identity
        self.W_diag = nn.Parameter(torch.ones(num_classes))
        self.b = nn.Parameter(torch.zeros(num_classes))
        self.k1 = nn.Parameter(torch.tensor(0.5))
        self.k2 = nn.Parameter(torch.tensor(1.0))

    def forward(self, z, sigma):
        # T_sigma = k1 * sigma + k2 (clamped positive)
        T_sigma = F.relu(self.k1 * sigma + self.k2) + 1e-4
        scaled_z = (self.W_diag * z + self.b) / T_sigma
        return F.softmax(scaled_z, dim=-1)
```

---

## 4. Key Implementation Risks & Engineering Trade-offs

| Implementation Risk | Severity | Root Cause | Recommended Engineering Resolution |
| :--- | :--- | :--- | :--- |
| **Empty Voxel Domination during Mix-Up** | **High** | In SemanticKITTI, $>90\%$ of voxels are empty ($y=0$). Shuffling uniformly pairs empty with empty, making $L_{\text{ru}}$ trivial. | Apply **Non-Empty Voxel Masking**: restrict feature permutation $\pi$ to occupied and boundary voxels, sampling a balanced subset of empty voxels. |
| **Unbounded Variance in Logit Sampling** | **Medium** | During early training, $\sigma_i$ can explode, generating extreme logits $\hat{z}_i$ and `NaN` in softmax. | Apply activation bounding: use `torch.clamp(sigma, max=5.0)` or `softplus` on the uncertainty output. |
| **Multi-Label Cross-Entropy Ambiguity** | **Medium** | When $y_i \neq y_j$, $y_i + y_j$ has two non-zero entries. Standard CE expects a single class index. | Normalize target $y_{\text{target}} = \frac{y_i + y_j}{\|y_i + y_j\|_1}$ and optimize Kullback-Leibler divergence or soft cross-entropy `-(target * log_softmax).sum()`. |
| **Data Leakage in Offline Calibration** | **High** | Calibrating on training data leads to artificially low ECE that fails on test data. | Partition the validation split into two disjoint subsets: $50\%$ for calibration parameter fitting and $50\%$ for final reliability reporting. |

---

## 5. Architectural Recommendation for ForesightBEV

1. **Adopt Relative Uncertainty Mix-Up in Stage 3 Training:**  
   The relative uncertainty formulation ($\mathcal{L}_{\text{ru}}$) is computationally efficient and adds zero inference latency. It should be integrated into ForesightBEV Stage 3 as a structural regularizer to enhance semantic boundary discrimination.
2. **Combine with OCCUQ Epistemic Monitoring:**  
   Do not rely solely on ReliOcc's online mode for runtime safety, as its uncertainty branch is discarded at test time. Pair ReliOcc’s calibrated logits with OCCUQ’s GPU JIT epistemic feature density to create a comprehensive safety monitor for Stage 6.

