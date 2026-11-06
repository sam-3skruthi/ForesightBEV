# REA 04 — ReliOcc Paper, Method, and Equation Implementation Map

**Paper Title:** Reliable and Calibrated Semantic Occupancy Prediction by Hybrid Uncertainty Learning  
**Publication:** Proceedings of the 34th International Joint Conference on Artificial Intelligence (IJCAI 2025), pp. 1973–1981  
**Preprint:** arXiv:2409.18026v1  
**Audited Repository Status:** **Status D (No verifiable official code release)**  
**Document Purpose:** Exhaustive mathematical and structural mapping of all equations, figures, tables, and implementation gaps.

---

## 1. Structural Comparison: IJCAI 2025 vs. arXiv:2409.18026

The paper underwent structural reorganization between the arXiv preprint and the final IJCAI 2025 archival publication:

| Structural Component | arXiv:2409.18026v1 Section | IJCAI 2025 Proceedings Section | Description & Scope |
| :--- | :--- | :--- | :--- |
| **Title** | *ReliOcc: Towards Reliable Semantic Occupancy Prediction via Uncertainty Learning* | *Reliable and Calibrated Semantic Occupancy Prediction by Hybrid Uncertainty Learning* | Title expanded in conference version to explicitly highlight "hybrid uncertainty" and "calibration". |
| **Related Work** | Section 6 (pp. 7–8) | Section 2 (pp. 1973–1974) | Shifted forward in IJCAI version to precede technical preliminaries. |
| **Problem Formulation & Metrics** | Section 2 (pp. 2–3) | Section 3 (pp. 1974–1975) | Defines occupancy task, PRR, and ECE metrics. |
| **Adaptation of Prior Arts** | Section 3 (pp. 3–4) | Section 4 (pp. 1975–1976) | Adapts HAU, DUL, MCD, TempS, DiriS, MetaC, and DeptS to 3D occupancy. |
| **Core Method (ReliOcc)** | Section 4 (pp. 4–5) | Section 5 (pp. 1976–1977) | Absolute uncertainty, relative uncertainty mix-up, and uncertainty-aware calibration. |
| **Experiments & Ablations** | Section 5 (pp. 5–7) | Section 6 (pp. 1977–1980) | Benchmark tables, online/offline results, robustness curves, and ablations. |
| **Conclusion** | Section 7 (p. 8) | Section 7 (p. 1980) | Summary and future data-loop perspectives. |
| **Supplemental Material** | Appendices A, B, C, D | Published as online appendix | Contains implementation hyperparameters, extra reliability diagrams, and third-party code licenses. |

---

## 2. Complete Mathematical Inventory & Equation Derivations

### 2.1 Problem Formulation
Let $\bm{x} \in \mathbb{R}^{T \times 3 \times H_{\text{img}} \times W_{\text{img}}}$ denote input multi-frame camera images. The 3D occupancy backbone $V_\theta(\bm{x})$ generates a dense volumetric feature grid:
$$\mathcal{V} \in \mathbb{R}^{d \times L \times W \times H}$$
where $L=256, W=256, H=32$ for SemanticKITTI ($V = 2,097,152$ total voxels), and $d$ is the feature dimension per voxel. Each voxel feature is denoted $v_i \in \mathbb{R}^d$.

The classification head estimates the conditional probability distribution:
$$p(y_i = y \mid v_i), \quad y \in \{0, 1, \dots, S\}$$
where $y=0$ denotes unoccupied/empty space, and $S=19$ denotes foreground semantic categories.

---

### 2.2 Reliability & Calibration Metrics

#### Prediction Rejection Ratio (PRR) — Eq. (1) in Paper
Used to evaluate misclassification detection via rejection curves. Predictions are sorted in ascending order of predicted confidence $c_i$. As the rejection threshold is varied, erroneous low-confidence predictions are progressively eliminated:
$$\text{PRR} = \frac{\text{AUC}_{\text{random}} - \text{AUC}_{\text{uncertainty}}}{\text{AUC}_{\text{random}} - \text{AUC}_{\text{oracle}}}$$
- $\text{AUC}_{\text{random}} = 0.5$ represents the baseline Area Under Curve for uninformative random rejection.
- $\text{AUC}_{\text{oracle}}$ represents the ideal curve where all incorrect predictions are rejected before any correct prediction.
- A perfectly reliable perception system achieves $\text{PRR} = 1.0$.
- Separately evaluated as $\text{PRR}_{\text{geo}}$ (binary occupancy correctness) and $\text{PRR}_{\text{sem}}$ (multi-class semantic correctness).

#### Expected Calibration Error (ECE) — Eqs. (2) & (3) in Paper
Measures the expectation of the difference between predicted confidence and observed accuracy:
$$e_{\text{ECE}} = \mathbb{E}_{\hat{c}_i} \left[ \left| p(\hat{y}_i = y_i \mid \hat{c}_i = c) - c \right| \right]$$
Approximated in practice by partitioning the interval $[0, 1]$ into $M = 15$ equally spaced bins $\{B_m\}_{m=1}^M$:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
where:
$$\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbf{1}(\hat{y}_i = y_i), \quad \text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{c}_i$$
Evaluated as $\text{ECE}_{\text{geo}}$ and $\text{ECE}_{\text{sem}}$. A perfectly calibrated model achieves $\text{ECE} = 0.0$.

---

### 2.3 Baseline Adaptations (Section 4 in IJCAI / Section 3 in arXiv)

| Baseline Method | Mathematical Formulation | Mechanism / Failure Mode |
| :--- | :--- | :--- |
| **Heteroscedastic Aleatoric Uncertainty (HAU)** | $\hat{z}_i = z_i + \sigma_i \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$ | Places Gaussian over logits. Reparameterized cross-entropy. Suffers from optimization plateau when $\sigma_i \to 0$. |
| **Data Uncertainty Learning (DUL)** | $\hat{v}_i = v_i + \boldsymbol{\sigma}_i \epsilon, \quad \mathcal{L}_{\text{DUL}} = \mathcal{L}_{\text{CE}} + D_{\text{KL}}(\mathcal{N} \parallel \mathcal{N}(0, \mathbf{I}))$ | Models feature-level Gaussian with KL regularizer. Causes training instability in sparse 3D volumes. |
| **Monte Carlo Dropout (MCD)** | $\hat{z}_i = \frac{1}{K} \sum_{k=1}^K z_{k, i}, \quad \mathcal{H} = -\sum_s \bar{p}_s \log \bar{p}_s$ | Requires $K=40$ forward stochastic passes during inference. Prohibitively slow for real-time driving. |
| **Temperature Scaling (TempS)** | $f_\phi(z_i) = \mathcal{S}\left(\frac{z_i}{T}\right)$ | Single scalar $T > 0$ shared across all classes. Improves ECE but degrades misclassification detection ($\text{PRR}_{\text{geo}}$ drops). |
| **Dirichlet Scaling (DiriS)** | $f_\phi(z_i) = \mathcal{S}\left(\mathbf{W} \log(\mathcal{S}(z_i)) + \mathbf{b}\right)$ | Full weight matrix $\mathbf{W} \in \mathbb{R}^{(S+1) \times (S+1)}$ and bias. High parameter count; overfits on small calibration sets. |
| **Meta-Calibration (MetaC)** | $f_\phi(z_i) = \mathcal{S}(z_i / T)$ if $\mathcal{H} < \eta$, else $\frac{1}{S+1}$ | Injects constant uniform probability for high-entropy voxels. Destroys geometric accuracy (IoU drops $43.96 \to 34.73\%$). |
| **Depth-Aware Scaling (DeptS)** | $\alpha_i = k_1 d_i + k_2, \quad f_\phi(z_i) = \mathcal{S}\left(\frac{z_i}{\alpha_i T}\right)$ | Scales temperature based on LiDAR depth $d_i$. Requires depth sensor at inference; violates camera-only mandate. |

---

### 2.4 ReliOcc Core Method Formulations (Section 5 in IJCAI / Section 4 in arXiv)

#### 1. Absolute Uncertainty Learning
- Voxel feature: $v_i \in \mathbb{R}^d$
- Predicted uncertainty: $\sigma_i = \text{MLP}_{\text{unc}}(v_i) \in \mathbb{R}^+$
- Deterministic logit: $z_i = \mathbf{W}_{\text{head}} v_i + \mathbf{b}_{\text{head}} \in \mathbb{R}^{S+1}$
- Reparameterized stochastic logit:
  $$\hat{z}_i = z_i + \sigma_i \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
- Absolute uncertainty loss:
  $$\mathcal{L}_{\text{au}} = -\log \left( \frac{\exp(\hat{z}_{i, y_i})}{\sum_{s=0}^S \exp(\hat{z}_{i, s})} \right)$$

#### 2. Relative Uncertainty Learning (The Mix-Up Formulation)
- Random permutation of voxel indices: $\pi \in \text{Perm}(V)$ generates pairs $(v_i, v_{\pi(i)})$, denoted $(v_i, v_j)$.
- Uncertainty-dependent blend coefficient:
  $$\lambda = \frac{\sigma_i}{\sigma_i + \sigma_j} \in [0, 1]$$
- Feature blend:
  $$\hat{v} = \lambda v_i + (1 - \lambda) v_j$$
- Label blend (Unweighted multi-label target):
  $$y_{\text{blended}} = y_i + y_j \in \{0, 1, 2\}^{S+1}$$
  *(As noted in Footnote 2, label weighting by $\lambda$ is omitted for training stability).*
- Shared completion head processing:
  $$\hat{z}_{\text{mix}} = \mathbf{W}_{\text{head}} \hat{v} + \mathbf{b}_{\text{head}}$$
- Relative uncertainty loss:
  $$\mathcal{L}_{\text{ru}} = \mathcal{L}_{\text{CE}}(\hat{z}_{\text{mix}}, y_i + y_j)$$

#### 3. Uncertainty-Aware Post-Hoc Calibration — Eq. (5) in Paper
- Uncertainty-dependent dynamic temperature:
  $$T_\sigma = k_1 \cdot \sigma_i + k_2$$
  where $k_1, k_2$ are scalar learnable parameters initialized to maintain $T_\sigma > 0$.
- Calibrated probability mapping:
  $$f_\phi(z_i) = \mathcal{S}\left( \frac{\mathbf{W} \cdot z_i + \mathbf{b}}{T_\sigma} \right)$$
  where:
  - $\mathbf{W} = \text{diag}(w_0, w_1, \dots, w_S) \in \mathbb{R}^{(S+1) \times (S+1)}$ is constrained strictly to the diagonal, initialized with $w_s = 1.0$.
  - $\mathbf{b} \in \mathbb{R}^{S+1}$ is a learnable bias vector initialized to $\mathbf{0}$.
- Calibration loss:
  $$\mathcal{L}_{\text{calib}} = -\sum_{i} \log f_\phi(z_i)^{(y_i)}$$

---

## 3. Figures and Tables Map

| Item | Title / Caption | Technical Role |
| :--- | :--- | :--- |
| **Figure 1** | (a) Overview of proposed ReliOcc. (b) Absolute uncertainty learning. (c) Relative uncertainty learning. | Architecture schematic showing the dual supervision pathways ($L_{\text{au}}$ and $L_{\text{ru}}$) and shared completion head. |
| **Figure 2** | Visual results of error map and uncertainty map from prediction by SGN and ReliOcc. | Demonstrates that vanilla SGN remains highly confident in misclassified voxels, whereas ReliOcc assigns high uncertainty to error zones. |
| **Figure 3** | Reliability diagrams of semantic calibration gaps from SGN and ReliOcc. | Visualizes confidence vs. accuracy across 15 bins; demonstrates drastic reduction in the under-confidence / over-confidence gap. |
| **Figure 4** | Accuracy and reliability performance between SGN and ReliOcc under four out-of-domain conditions. | Radar / line plots showing performance under dropped frames, strong sunlight, fog, and rain at weak ($w$) and strong ($s$) severities. |
| **Figure A1** | Geometric comparison in accuracy and reliability under four OOD conditions. | Supplemental geometric robustness curves showing $\text{PRR}_{\text{geo}}$ and $\text{ECE}_{\text{geo}}$ stability. |
| **Figure A2** | Additional reliability diagrams. | Supplemental calibration diagrams for VoxFormer baseline. |
| **Table 1** | Benchmark accuracy and reliability of 10 SOTA models on SemanticKITTI validation set. | Groundwork table demonstrating the reliability gap between LiDAR (ECE $\approx 0.7 - 2.9\%$) and Camera (ECE $\approx 5.7 - 7.2\%$). |
| **Table 2** | Quantitative results of online uncertainty learning on SemanticKITTI. | Compares ReliOcc against HAU, DUL, and MCD on VoxFormer and SGN frameworks. |
| **Table 3** | Quantitative results of offline model calibration on SemanticKITTI. | Compares ReliOcc against TempS, DiriS, MetaC, and DeptS. |
| **Table 4** | Ablation of online uncertainty learning. | Evaluates combinations of None, Absolute Only, Relative Only, and Hybrid. |
| **Table 5** | Ablation of offline model calibration. | Evaluates combinations of Scaling Only, Scaling + Relative, and Scaling + Relative + Absolute. |

---

## 4. Implementation Gap Inventory (Unreleased Code Details)

Because no source-code repository has been released (Status D), the following specific low-level parameters and implementation choices cannot be verified from code and must be engineered if adapting into ForesightBEV:

1. **Uncertainty MLP Architecture:**
   - The paper specifies that $\sigma_i$ is predicted from $v_i$ using an "MLP".
   - *Unavailable details:* Exact number of layers (e.g. 2-layer vs. 3-layer), hidden dimension (e.g. 64, 128, or 256), activation function (ReLU vs. LeakyReLU), and output activation (Softplus vs. Sigmoid vs. exponential) used to guarantee $\sigma_i > 0$.
2. **Loss Formulation for Blended Multi-Label Cross-Entropy:**
   - The paper states that $\hat{v}$ is trained with blended label $y = y_i + y_j$ using "cross-entropy loss".
   - *Unavailable details:* When $y_i \neq y_j$, $y$ contains two positive entries. Standard multi-class softmax cross-entropy expects a single ground-truth class index. Did the authors implement multi-label binary cross-entropy (BCEWithLogitsLoss), normalized soft-target cross-entropy ($y/2$), or two separate cross-entropy loss terms $\mathcal{L}_{\text{CE}}(\hat{z}_{\text{mix}}, y_i) + \mathcal{L}_{\text{CE}}(\hat{z}_{\text{mix}}, y_j)$?
3. **Empty Voxel Filtering during Shuffling:**
   - In a $256 \times 256 \times 32$ SemanticKITTI grid, over $90\%$ of voxels represent empty space ($y=0$).
   - *Unavailable details:* If voxels are shuffled completely at random across all $2.1\times 10^6$ voxels, the vast majority of pairs $(v_i, v_j)$ will consist of two empty voxels ($0 + 0 = 0$). Did the authors shuffle all voxels uniformly, or did they perform class-balanced or non-empty voxel sampling to prevent empty-space domination?
4. **Offline Calibration Data Splitting:**
   - Offline calibration requires fitting $k_1, k_2, \mathbf{W}, \mathbf{b}$ on data not seen during model training.
   - *Unavailable details:* The authors evaluate on SemanticKITTI validation set (seq 08). Did they split sequence 08 into calibration and evaluation halves, or calibrate on a held-out slice of sequences 00–10?
5. **Numerical Clipping of Variance in Logit Sampling:**
   - In $\hat{z}_i = z_i + \sigma_i \epsilon$, if $\sigma_i$ becomes large, $\hat{z}_i$ causes numerical instability / NaN in the softmax exponential.
   - *Unavailable details:* Whether a clamp operator (e.g. `torch.clamp(sigma, min=1e-6, max=10.0)`) or log-variance formulation ($\log \sigma^2$) was used.

