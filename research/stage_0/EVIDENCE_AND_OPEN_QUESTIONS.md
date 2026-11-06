# ForesightBEV — Stage 0: Evidence and Unresolved Claims Register

**Document Reference:** `research/stage_0/EVIDENCE_AND_OPEN_QUESTIONS.md`  
**Phase:** Stage 0 — Literature Review, REA, and Research Gap  
**Status:** Living Technical Risk Register & Open Questions Tracker  

---

## 1. Register Overview

This register documents the critical empirical uncertainties, unverified claims, and methodological ambiguities identified during our source-level and paper-level audits of SurroundOcc, OPUS-V2, OCCUQ, and ReliOcc. 

Every entry is structured using four mandatory analytical categories:
1. **What Is Known:** The empirically verified or author-documented facts.
2. **What Remains Unknown:** The specific unverified parameters, omitted mechanisms, or unreleased implementation choices.
3. **Why the Distinction Matters:** The direct architectural, performance, or safety risk to ForesightBEV.
4. **What Investigation Could Resolve It:** The targeted empirical experiment, unit test, or baseline audit required to close the gap.

---

## 2. Technical Item Register

### Item 1: ReliOcc's Unavailable Official Source Code Implementation
- **What Is Known:**  
  The paper was published in IJCAI 2025 and on arXiv (2409.18026). A rigorous search across conference proceedings, arXiv LaTeX source, author profiles (Zhejiang University, Huawei Noah's Ark Lab), and public GitHub repositories confirmed that no official repository has been published (Status D).
- **What Remains Unknown:**  
  The exact codebase structure, exact hyperparameter values (e.g. MLP hidden dimensions, optimizer weight decay schedules), random seed dependencies, and specific engineering hacks used during training.
- **Why the Distinction Matters:**  
  Direct reproduction via code checkout is impossible. Any deployment within ForesightBEV must rely on a clean-room reimplementation, which carries the risk of divergence from published metrics if hidden heuristics were omitted from the paper text.
- **What Investigation Could Resolve It:**  
  Execute our designed standalone clean-room prototype (`research/repository_audits/reliocc/REPRODUCTION_ASSESSMENT.md`) and validate gradient flow and loss convergence on synthetic voxel features.

---

### Item 2: Exact Cross-Entropy Interpretation of ReliOcc's Unweighted Label Sum ($y = y_i + y_j$)
- **What Is Known:**  
  Footnote 2 of the ReliOcc paper states: *"Different from the original mix-up paper, we omit weighting the labels with $\lambda$ for stable training."* The blended label is defined as $y = y_i + y_j$, where $y_i, y_j$ are one-hot encoded ground-truth vectors.
- **What Remains Unknown:**  
  Standard multi-class categorical cross-entropy requires targets to be normalized probability distributions ($\sum y = 1$). If $y_i \neq y_j$, $y$ has two non-zero entries (summing to 2). The paper does not specify whether the authors implemented:
  1. Multi-label Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`).
  2. Soft-target Cross-Entropy normalized by 2 (`F.kl_div` or `-(0.5*y * log_softmax).sum()`).
  3. Two separate cross-entropy loss terms evaluated against the same prediction: $\mathcal{L}_{\text{CE}}(\hat{z}, y_i) + \mathcal{L}_{\text{CE}}(\hat{z}, y_j)$.
- **Why the Distinction Matters:**  
  Different multi-label implementations produce vastly different gradient magnitudes across the shared completion head, directly impacting whether relative uncertainty optimization stabilizes or causes representation drift.
- **What Investigation Could Resolve It:**  
  Run a unit ablation in the clean-room harness comparing all three loss variants and measuring gradient variance on the mix-up head.

---

### Item 3: Data Leakage Protection in ReliOcc Offline Calibration Protocol
- **What Is Known:**  
  ReliOcc fits its offline uncertainty-aware temperature calibrator ($f_\phi$) on the SemanticKITTI validation set (Sequence 08) and reports ECE and PRR on the same sequence.
- **What Remains Unknown:**  
  Whether Sequence 08 was partitioned into two disjoint subsets (e.g. $50\%$ calibration fitting, $50\%$ final evaluation), or whether the calibrator was fit and evaluated on the exact same data frames.
- **Why the Distinction Matters:**  
  If the calibrator was fit and evaluated on the same validation sequence, the reported reduction in $\text{ECE}_{\text{sem}}$ from $5.90\%$ to $2.05\%$ suffers from **in-sample data leakage** and will overstate the true out-of-sample calibration performance on unseen road sequences.
- **What Investigation Could Resolve It:**  
  Evaluate our clean-room calibrator on nuScenes validation scenes using a strict 2-fold cross-validation split (fitting on split A, evaluating on split B, and vice-versa).

---

### Item 4: Transfer of Monocular Temporal Uncertainty to 360° Surround-View Arrays
- **What Is Known:**  
  ReliOcc was evaluated strictly on a single forward-facing camera (`image_2` on KITTI) with a 5-frame temporal buffer. Its "sensor failure" robustness tests evaluated **temporal frame dropping** in that forward camera buffer.
- **What Remains Unknown:**  
  Whether the relative mix-up regularizer and temperature calibrator maintain their reliability gains when scaled to a 6-camera surround array (nuScenes) featuring wide-angle lenses, multi-camera overlap seams, and asynchronous shutter timings.
- **Why the Distinction Matters:**  
  Temporal frame dropping in a single forward camera maintains a continuous field of view with degraded motion cues. In contrast, losing a camera in a 6-camera surround rig creates a complete $60^\circ - 90^\circ$ spatial blind sector. A method robust to temporal stutter may fail completely under spatial blindness.
- **What Investigation Could Resolve It:**  
  Evaluate the relative uncertainty loss formulation on a 6-camera surround model under simulated camera dropouts.

---

### Item 5: OCCUQ's Architectural Dependence on Dense Volumetric Features
- **What Is Known:**  
  OCCUQ’s GMM is fit over dense penultimate-layer features $v_i \in \mathbb{R}^{32}$ extracted from all $640,000$ voxels of a dense $200 \times 200 \times 16$ 3D deconvolutional volume.
- **What Remains Unknown:**  
  Can OCCUQ's feature-level GMM density formulation be successfully applied to **sparse-query architectures** like OPUS-V2, where features exist only at $Q=600$ sparse 3D bounding box queries rather than in a dense voxel tensor?
- **Why the Distinction Matters:**  
  Dense 3D deconvolution limits SurroundOcc/OCCUQ to $\approx 4.5 - 5.2\text{ FPS}$ on an RTX 3090. If OCCUQ's GMM density monitor cannot be transferred to OPUS-V2's sparse query space, ForesightBEV Stage 3 would be forced to choose between real-time inference ($25.8\text{ FPS}$) and epistemic sensor reliability.
- **What Investigation Could Resolve It:**  
  Formulate and benchmark a Query-GMM layer fit on OPUS-V2 query embeddings ($\mathbb{R}^{Q \times 256}$) rather than dense voxel features.

---

### Item 6: OCCUQ GMM Spatial Resolution: Local Camera Failure vs. Global Distributional Abnormality
- **What Is Known:**  
  OCCUQ evaluates GMM log-density voxel by voxel on GPU using `jit_log_prob`. On MultiCorrupt, it achieves $95.39\%\text{ AUROC}$ on `missingcamera_3`.
- **What Remains Unknown:**  
  Does the GMM log-density field cleanly localize the specific angular wedge corresponding to the blacked-out camera (e.g. only marking voxels in the front camera frustum as low-density), or does the entire global feature volume exhibit depressed log-density due to global backbone/FPN cross-attention leakage?
- **Why the Distinction Matters:**  
  For Stage 6 trajectory planning, knowing that *only the rear-right quadrant* is blind allows safe forward driving. If camera failure causes global density collapse across all voxels, the planner will unnecessarily freeze.
- **What Investigation Could Resolve It:**  
  Analyze the spatial distribution of the epistemic uncertainty map $\mathbf{U} \in \mathbb{R}^{200 \times 200 \times 16}$ under single-camera blackout (`missingcamera_1`) to measure the ratio of intra-frustum vs. extra-frustum density drop.

---

### Item 7: OPUS-V2 Query-Count / Latency Disaggregation (Tiny vs. Large)
- **What Is Known:**  
  Source verification in REA 02 established that OPUS-V2 provides four variants: Tiny ($Q=600$), Small ($Q=1200$), Medium ($Q=2400$), and Large ($Q=4800$).
  The headline benchmark of **$38.6\%\text{ mIoU}$ and $44.0\%\text{ RayIoU}$ belongs exclusively to OPUS-V2-L at $8.6\text{ FPS}$**, while the real-time throughput of **$25.8\text{ FPS}$ belongs exclusively to OPUS-V2-T at $35.6 - 36.5\%\text{ mIoU}$**.
- **What Remains Unknown:**  
  Whether an intermediate configuration (e.g. $Q=1200$ with $P=64$) can achieve $>37.0\%\text{ mIoU}$ while sustaining $\geq 20\text{ FPS}$ on consumer hardware (RTX 3090 / 4090).
- **Why the Distinction Matters:**  
  Autonomous vehicle planners running at $10 - 20\text{ Hz}$ cannot consume a perception model running at $8.6\text{ FPS}$ without introducing severe perception latency ($> 116\text{ ms}$).
- **What Investigation Could Resolve It:**  
  Profile OPUS-V2-S ($Q=1200$) latency and memory under TensorRT / FP16 optimization.

---

### Item 8: OPUS-V2 Unknown-Space and Empty-Label Handling
- **What Is Known:**  
  In Occ3D-nuScenes, classes $0 \dots 16$ represent semantic categories, and `empty_label = 17` represents free space. OPUS-V2 supervises bounding box queries using Chamfer Distance against surface points and densifies query boxes into dense voxels via CPD.
- **What Remains Unknown:**  
  How does OPUS-V2 classify space outside its $Q$ bounding box proposals? If a region is unqueried, does CPD default to class 17 (empty space), or does it leave unqueried regions unclassified?
- **Why the Distinction Matters:**  
  If unqueried or occluded space automatically defaults to "free space", the model will dangerously predict that occluded road behind buildings or blind corners is safe to drive on.
- **What Investigation Could Resolve It:**  
  Inspect the voxelization scatter operation in `projects/models/dense_heads/densifier.py` to verify whether default voxel values represent "empty" or "unobserved".

---

### Item 9: The Absence of Spatiotemporal Uncertainty in Current 4D Forecasting Baselines
- **What Is Known:**  
  All four primary audited architectures (SurroundOcc, OPUS-V2, OCCUQ, ReliOcc) predict occupancy strictly for the **current observation timestamp $t$**.
  Existing 4D occupancy forecasting world models (OccWorld, Cam4DOcc) predict deterministic future tokens or continuous volumes over $t+1 \dots t+H$ without uncertainty estimation.
- **What Remains Unknown:**  
  How does perception uncertainty at time $t$ propagate through nonlinear world-model dynamics over a 3.0-second horizon? Does dynamic velocity ambiguity compound exponentially, and can an autoregressive model produce calibrated future confidence intervals?
- **Why the Distinction Matters:**  
  Stage 5 (4D Forecasting) cannot provide reliable inputs to Stage 6 (Safety Layer) if it treats future occupancy predictions as absolute certainties.
- **What Investigation Could Resolve It:**  
  Conduct an REA audit of Cam4DOcc / OccWorld to inspect the latent temporal transition function and design an uncertainty propagation mechanism.

---

### Item 10: The Absence of Trajectory-Event Risk Calibration in Prior Art
- **What Is Known:**  
  Current autonomous driving planning literature evaluates either binary geometric collision checking along swept paths or differentiable cost-map penalties.
- **What Remains Unknown:**  
  Whether any existing framework has demonstrated a statistically calibrated mapping from **4D uncertain voxel distributions** to **trajectory collision probabilities** satisfying empirical reliability criteria ($\text{T-ECE} \leq 5\%$).
- **Why the Distinction Matters:**  
  Without calibrated collision probabilities, safety filters either become dangerously optimistic (risking real collisions) or excessively conservative (freezing the autonomous vehicle in place).
- **What Investigation Could Resolve It:**  
  Formulate the spatial correlation integral along swept trajectory volumes in Stage 6 and validate calibration error using closed-loop log replay.

---

## 3. Register Summary & Priority Next Actions

```
+---------------------------------------------------------------------------------------------------+
|                                 EVIDENCE & UNRESOLVED CLAIMS SUMMARY                              |
+---------------------------------------------------------------------------------------------------+
| ITEM   DESCRIPTION                      IMPACT ON FORESIGHTBEV       PRIORITY ACTION REQUIRED     |
+---------------------------------------------------------------------------------------------------+
| 1      ReliOcc unreleased source code   Reproduction risk            Execute clean-room prototype |
| 2      Multi-label mixup loss math      Optimization stability       Benchmark 3 BCE/KL variants  |
| 3      ReliOcc validation data leakage  Reported ECE validity        Validate 2-fold split on nuSc|
| 4      Monocular -> Surround transfer   Sensor failure scope         Test mixup on 6-camera rig   |
| 5      OCCUQ dense-feature dependence   Inference latency (FPS)      Formulate Query-GMM for OPUS |
| 6      OCCUQ local vs global density    Spatial sector masking       Profile spatial drop map     |
| 7      OPUS-V2 metric attribution       FPS / mIoU trade-off         Profile OPUS-V2-S (Q=1200)   |
| 8      OPUS-V2 unqueried voxel state    Occlusion safety hazard      Audit CPD scatter default val|
| 9      Zero 4D forecasting uncertainty  Stage 5 research gap         Audit OccWorld / Cam4DOcc    |
| 10     Zero trajectory risk calibration Stage 6 research gap         Formulate Trajectory-ECE     |
+---------------------------------------------------------------------------------------------------+
```

This register will remain active and be updated as empirical experiments in Stages 1–6 resolve each open question.

