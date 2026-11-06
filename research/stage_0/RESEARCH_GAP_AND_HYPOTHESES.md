# ForesightBEV — Stage 0: Research Gap and Hypotheses Framework

**Document Reference:** `research/stage_0/RESEARCH_GAP_AND_HYPOTHESES.md`  
**Phase:** Stage 0 — Literature Review, REA, and Research Gap  
**Status:** Living Research Framework & Testable Hypotheses Registry  

---

## 1. Core Research Problem & Candidate Overarching Direction

Following our rigorous source-code audits of SurroundOcc, OPUS/OPUS-V2, OCCUQ, and the research audit of ReliOcc, we articulate the central candidate research problem for ForesightBEV:

> **Candidate Core Research Question:**  
> *Can uncertainty in future camera-derived 4D occupancy be translated into calibrated probabilities of safety-critical trajectory events under incomplete, degraded, and temporally inconsistent observations?*

### The Scientific Dilemma
Current autonomous driving perception systems separate 3D geometric occupancy prediction from downstream planning. Existing state-of-the-art models (SurroundOcc, OPUS-V2, OccWorld, Cam2BEV) produce dense or sparse 3D/4D grids of logits, assuming that every voxel is either deterministically occupied or deterministically free. When sensor degradation occurs (e.g. camera occlusion, lens contamination, fog, or blinding glare), these networks fail silently: they output high-confidence predictions in unobserved areas, causing downstream trajectory planners to make catastrophic assumptions about free space.

While recent works like OCCUQ (ICRA 2025) and ReliOcc (IJCAI 2025) have introduced static uncertainty estimation and confidence calibration for current-frame 3D occupancy, **no existing baseline propagates uncertainty forward into 4D spatiotemporal forecasting or translates voxel-level uncertainty into formally calibrated trajectory collision risks**.

---

## 2. Decomposed Testable Hypotheses Registry

To transform this overarching question into an empirical, falsifiable research program, we formulate five independent hypotheses ($H_1$ through $H_5$):

---

### Hypothesis 1 (H1) — Spatial Reliability: Explicit Unknown-Space Representation

- **Research Question:**  
  Can a camera-only 3D occupancy network explicitly distinguish between **occupied space**, **observed-free space**, and **unknown/unobserved space** (occluded or outside camera lines-of-sight) without relying on LiDAR ray-tracing at inference?
- **Existing Relevant Prior Work:**  
  - *Occ3D* (Tian et al., NeurIPS 2023): Constructs `mask_camera` by ray-tracing from camera optical centers to mesh surfaces; used only as an evaluation filter.
  - *OpenOccupancy* (Wang et al., ICCV 2023): Evaluates occupancy with visibility masks, treating unseen voxels as ignore labels (255).
  - *SurroundOcc, OPUS-V2, OCCUQ, ReliOcc:* All treat free space and occluded space as a single homogeneous "unoccupied" class ($y=0$ or $17$) or mask unobserved voxels out of the loss entirely. None output an explicit unobserved-space prediction.
- **What Has Already Been Demonstrated:**  
  Offline ray-tracing from camera centers through LiDAR depth can reliably label whether a voxel is visible to camera rays.
- **What Is NOT Yet Demonstrated:**  
  Whether a camera-only model can infer its own line-of-sight frustum visibility and learn an explicit ternary voxel state:
  $$\mathbf{V}(x, y, z) \in \{\text{Occupied}, \quad \text{Observed-Free}, \quad \text{Unknown/Occluded}\}$$
  directly from multi-camera visual parallax without ground-truth LiDAR ray-casting at test time.
- **Experimental Protocol:**  
  1. Train a dual-head occupancy model with: (a) a 17-class semantic occupancy head, and (b) a binary visibility line-of-sight classification head supervised by ray-traced camera frustum masks.
  2. Test under synthetic camera occlusion (masking out angular camera wedges).
- **Required Datasets:** Occ3D-nuScenes (with official `mask_camera` ground truth).
- **Baselines:** Vanilla SurroundOcc, OPUS-V2-T, Baseline Binary Mask Softmax.
- **Evaluation Metrics:**  
  - Visibility IoU ($\text{IoU}_{\text{vis}}$).
  - False Free Rate ($\text{FFR}$): Percentage of occluded voxels erroneously classified as observed-free space.
- **Likely Failure Modes:**  
  The network may memorize training-set vehicle geometry priors rather than inferring geometric lines-of-sight, failing in novel road topographies.
- **Acceptance / Rejection Criteria:**  
  - *Accept if:* The camera-only visibility head achieves $\text{IoU}_{\text{vis}} \geq 75\%$ and reduces $\text{FFR}$ behind large dynamic occluders (e.g. buses/trucks) by $\geq 40\%$ compared to baseline models.
  - *Reject if:* Visibility prediction accuracy degrades below random guessing on non-flat terrain.

---

### Hypothesis 2 (H2) — Uncertainty Calibration under Sensor Degradation

- **Research Question:**  
  Can voxel-level epistemic and aleatoric uncertainties be simultaneously calibrated under severe camera degradation, missing views, and distribution shifts without degrading underlying semantic occupancy accuracy (mIoU)?
- **Existing Relevant Prior Work:**  
  - *OCCUQ* (ICRA 2025): Demonstrates GMM log-density for epistemic OOD detection on MultiCorrupt.
  - *ReliOcc* (IJCAI 2025): Demonstrates diagonal temperature scaling for in-distribution ECE calibration on SemanticKITTI.
  - *DDU* (Mukhoti et al., NeurIPS 2021): Establishes spectral normalization for preserving feature distances.
- **What Has Already Been Demonstrated:**  
  OCCUQ detects camera dropouts with $95.39\%\text{ AUROC}$, and ReliOcc reduces semantic ECE from $5.90\%$ to $2.05\%$ on monocular inputs.
- **What Is NOT Yet Demonstrated:**  
  A unified surround-view framework that simultaneously delivers:
  1. High OOD detection ($\text{AUROC} \geq 90\%$) on multi-camera dropouts.
  2. Low calibration error ($\text{ECE} \leq 3.0\%$) across both in-distribution and corrupted conditions.
  3. Zero degradation in baseline occupancy mIoU.
- **Experimental Protocol:**  
  Evaluate a unified model equipped with spectrally normalized MLP heads, relative uncertainty mix-up regularizers, and post-hoc diagonal calibration on clean nuScenes val vs. MultiCorrupt benchmark (snow, fog, motion blur, brightness, 1–3 missing cameras).
- **Required Datasets:** Occ3D-nuScenes validation set + MultiCorrupt dataset.
- **Baselines:** Vanilla OPUS-V2, OCCUQ (GMM only), ReliOcc (TempS only), Deep Ensembles ($M=5$), MC-Dropout ($K=10$).
- **Evaluation Metrics:**  
  - Macro AUROC ($\text{mAUROC}$), Macro AUPR ($\text{mAUPR}$), False Positive Rate at 95% TPR ($\text{mFPR95}$).
  - Expected Calibration Error ($\text{ECE}_{\text{sem}}, \text{ECE}_{\text{geo}}$) using 15 equal-frequency bins.
  - Semantic mIoU and Geometric IoU.
- **Likely Failure Modes:**  
  Spectral normalization may overly constrain feature capacity, causing a drop in mIoU for fine categories (e.g. `pedestrian` or `traffic_cone`).
- **Acceptance / Rejection Criteria:**  
  - *Accept if:* The model achieves $\text{mAUROC} \geq 85\%$ on MultiCorrupt, maintains $\text{ECE} \leq 3.0\%$, and incurs $< 1.0\%\text{ mIoU}$ loss relative to the unconstrained baseline.
  - *Reject if:* Calibration collapses under heavy fog/snow ($\text{ECE} > 8.0\%$) or mIoU drops by $> 2.5\%$.

---

### Hypothesis 3 (H3) — Spatiotemporal Propagation into Future 4D Forecasting

- **Research Question:**  
  Can spatial uncertainty and visibility masks estimated at time $t$ be propagated into future time horizons ($t+1 \dots t+H$) through a 4D occupancy and flow world model, such that uncertainty grows monotonically with forecast horizon and dynamic velocity ambiguity?
- **Existing Relevant Prior Work:**  
  - *OccWorld* (Zheng et al., ECCV 2024): Predicts future 3D occupancy tokens using spatial-temporal discrete diffusion / autoregressive GPT-style transformers, but outputs deterministic tokens without uncertainty.
  - *Cam4DOcc* (Ma et al., CVPR 2024): Benchmarks 4D occupancy forecasting and point flow from multi-view cameras, evaluating Chamfer distance and IoU over a 3-second horizon without uncertainty quantification.
  - *MotionSC* (Wilson et al., IEEE RA-L 2022): Estimates occupancy flow using LiDAR, but lacks confidence calibration.
- **What Has Already Been Demonstrated:**  
  Autoregressive transformers can forecast geometric occupancy states and velocity flow up to 3.0 seconds into the future.
- **What Is NOT Yet Demonstrated:**  
  Whether a world model can forecast a **spatiotemporal distribution** over future occupancy states, quantifying how observation noise at time $t$ propagates into expanded spatial variance at $t+H$.
- **Experimental Protocol:**  
  1. Train a 4D occupancy and flow forecaster (Stage 5) taking historical states $\mathbf{O}_{t-K:t}$ and latent epistemic uncertainties $\mathbf{U}_{t-K:t}$.
  2. Evaluate predicted mean occupancy $\hat{\mathbf{O}}_{t+h}$ and predicted uncertainty field $\hat{\mathbf{\Sigma}}_{t+h}$ across horizons $h \in \{0.5\text{s}, 1.0\text{s}, 2.0\text{s}, 3.0\text{s}\}$.
- **Required Datasets:** nuScenes-Occupancy (with flow ground truth) or Cam4DOcc benchmark.
- **Baselines:** OccWorld, Cam4DOcc deterministic baseline, Autoregressive MC-Dropout Forecaster.
- **Evaluation Metrics:**  
  - Forecasting mIoU at $t+1\text{s}, t+2\text{s}, t+3\text{s}$.
  - Spatiotemporal Calibration Error: $\text{ECE}(t+h)$.
  - Negative Log-Likelihood (NLL) of future occupancy states.
  - Uncertainty Growth Monotonicity ($\rho_{\text{spearman}}(\text{Uncertainty}, h)$).
- **Likely Failure Modes:**  
  Autoregressive error accumulation: future token predictions diverge into unconstrained blur, causing uncertainty values to saturate at maximum entropy after 1.5 seconds.
- **Acceptance / Rejection Criteria:**  
  - *Accept if:* Forecasted uncertainty correlates strongly with empirical future error ($\rho \geq 0.70$), and dynamic object flow uncertainty expands along motion vectors without immediate global saturation.
  - *Reject if:* Future ECE exceeds $15\%$ or uncertainty values become uniform across all future spatial cells.

---

### Hypothesis 4 (H4) — Translation to Trajectory-Event Collision Probabilities

- **Research Question:**  
  Can voxel-level 4D occupancy probabilities and spatiotemporal uncertainties along a candidate ego-vehicle trajectory corridor be integrated into **statistically calibrated probabilities of safety-critical trajectory events** (e.g. collision, boundary breach, unsafe clearance)?
- **Existing Relevant Prior Work:**  
  - *DIPP* (Huang et al., CVPR 2024): Differentiable integrated prediction and planning using cost maps, but assumes deterministic obstacle occupancy.
  - *Safe Horizon Planning* (Albrecht et al., IEEE T-RO 2021): Computes collision bounds under Gaussian state estimation error.
  - *Frenet-Serret Trajectory Optimization:* Standard automotive trajectory generation evaluating obstacle distance penalties.
- **What Has Already Been Demonstrated:**  
  Voxel grids can be queried along a swept trajectory volume to detect binary intersection collisions.
- **What Is NOT Yet Demonstrated:**  
  A mathematically rigorous mapping function that converts a swept spatio-temporal volume of uncertain voxel distributions:
  $$\mathcal{T}_{\text{ego}}(t+1:t+H) \cap \hat{\mathbf{O}}(t+1:t+H, \mathbf{x})$$
  into an exact, calibrated trajectory collision probability $P_{\text{collision}}(\tau) \in [0, 1]$ satisfying:
  $$\mathbb{E}\big[\mathbf{1}(\text{Collision}) \mid P_{\text{collision}}(\tau) = p\big] = p$$
- **Experimental Protocol:**  
  1. Generate $N=1,000$ candidate trajectory rollouts per scene in closed-loop or log-replay evaluation.
  2. Compute trajectory collision probability using stochastic integration over voxel occupancy distributions.
  3. Measure empirical collision rates vs. predicted collision probabilities across 10 probability bins.
- **Required Datasets:** nuScenes planning benchmark (CAN bus ego-motion + 4D occupancy ground truth).
- **Baselines:** Deterministic occupancy collision checker (binary thresholding at $p=0.5$), Euclidean signed distance field (ESDF) heuristics.
- **Evaluation Metrics:**  
  - Trajectory-Level Expected Calibration Error ($\text{T-ECE}$).
  - Brier Score for trajectory collision prediction.
  - Area Under Precision-Recall Curve ($\text{AUPR}$) for near-miss events.
- **Likely Failure Modes:**  
  Voxel independence assumption: naive multiplication of voxel free-space probabilities along a trajectory ($\prod_i (1 - p_i)$) assumes spatial independence, leading to severe under-estimation of collision risk across contiguous physical objects.
- **Acceptance / Rejection Criteria:**  
  - *Accept if:* Trajectory-level collision calibration error $\text{T-ECE} \leq 5.0\%$, and collision prediction Brier score improves by $\geq 25\%$ over deterministic thresholding.
  - *Reject if:* Spatial correlation collapse causes the model to predict near-zero collision probability for trajectories passing through dense obstacle edges.

---

### Hypothesis 5 (H5) — Risk-Constrained Filtering without Excessive Conservatism

- **Research Question:**  
  Does filtering candidate trajectories using calibrated collision probabilities and epistemic failure flags improve closed-loop driving safety under sensor corruptions **without inducing excessive conservatism** (e.g. vehicle freezing or unwarranted emergency braking)?
- **Existing Relevant Prior Work:**  
  - *Control Barrier Functions (CBFs):* Formally guarantee safety bounds under known disturbance models, but suffer from vehicle freezing in high-uncertainty environments.
  - *Reachability Analysis:* Guarantees worst-case safety, but frequently yields empty feasible trajectory sets in cluttered urban environments.
- **What Has Already Been Demonstrated:**  
  Risk-constrained filters can stop an autonomous vehicle when an obstacle is detected with certainty.
- **What Is NOT Yet Demonstrated:**  
  Whether an uncertainty-aware filter can distinguish between:
  1. *High-risk occlusions that mandate immediate deceleration* (e.g. a pedestrian step-out from behind an occluding truck).
  2. *Benign sensor noise that permits safe continuation* (e.g. light rain noise or high-altitude tree canopy uncertainty).
- **Experimental Protocol:**  
  1. Integrate the Stage 6 Risk-Constrained Safety Layer with Autoware / ROS 2 in CARLA closed-loop simulator (Stage 7).
  2. Subject the vehicle to Town01–Town10 urban scenarios with randomized sensor faults (camera blackout, lens drops) and unexpected pedestrian intrusions.
  3. Compare planning outcomes against deterministic cost-map planners.
- **Required Simulation Platform:** CARLA Simulator v0.9.15 + ROS 2 Humble + Autoware Universe.
- **Baselines:** Vanilla Autoware Obstacle Stop Planner, Worst-Case Robust Reachability Planner, Deterministic OccWorld Planner.
- **Evaluation Metrics:**  
  - Driving Score (DS) & Infraction Score (IS) from CARLA Autonomous Driving Leaderboard.
  - Collision Rate per 100 km.
  - Minimum Time to Collision (TTC) in critical events.
  - Unwarranted Deceleration Rate (stops per hour caused by false-positive uncertainty).
  - Progress Rate / Route Completion Percentage.
- **Likely Failure Modes:**  
  The "freezing robot problem": whenever front cameras experience partial glare or fog, the planner aborts all valid paths and remains stationary indefinitely.
- **Acceptance / Rejection Criteria:**  
  - *Accept if:* Closed-loop collision rate decreases by $\geq 50\%$ under sensor corruptions while route completion rate remains $\geq 85\%$ of the clean-sensor baseline.
  - *Reject if:* The vehicle exhibits $> 5$ unwarranted emergency stops per hour or fails to proceed through normal shadow/sunlight transitions.

---

## 3. Prior-Art Grounding & Novelty Boundary Analysis

To prevent premature claims of novelty, we explicitly demarcate what prior literature has established versus what remains open:

```
+---------------------------------------------------------------------------------------------------+
|                                  PRIOR ART VS. OPEN RESEARCH GAP                                  |
+---------------------------------------------------------------------------------------------------+
| RESEARCH DOMAIN           ESTABLISHED IN PRIOR ART              OPEN RESEARCH GAP FOR FORESIGHTBEV|
+---------------------------------------------------------------------------------------------------+
| 3D Occupancy Lifting      Dense deformable attention (SurroundOcc);  Explicit unknown/occluded    |
|                           Sparse CPD bounding boxes (OPUS-V2).       representation learned from  |
|                                                                      camera parallax without LiDAR|
| Static Uncertainty        GMM feature density for OOD (OCCUQ);       Unified epistemic + aleatoric|
|                           Relative mix-up & TempS (ReliOcc).         surround-view multi-camera   |
|                                                                      calibration architecture     |
| Spatiotemporal 4D         Autoregressive discrete token occupancy    Uncertainty-propagating world|
| Forecasting               generation up to 3.0s (OccWorld, Cam4DOcc) model; variance growth over  |
|                                                                      dynamic velocity vectors     |
| Trajectory Planning       Binary collision checking along swept paths;Rigorous integration of 4D  |
| & Risk Evaluation         Differentiable cost maps (DIPP).           voxel distributions into     |
|                                                                      calibrated collision chances |
| Safety Filtering          Control barrier functions; Worst-case      Uncertainty-gated filtering  |
|                           reachability envelopes.                    preventing vehicle freezing  |
|                                                                      under benign sensor noise    |
+---------------------------------------------------------------------------------------------------+
```

---

## 4. Summary & Verification Sign-Off

The five hypotheses ($H_1 \dots H_5$) form an unbroken logical chain connecting our Stage 0 reverse-engineering audits to the ten-stage ForesightBEV roadmap:
- **$H_1$ and $H_2$** govern **Stage 3** (3D Occupancy Perception and Sensor Reliability).
- **$H_3$** governs **Stage 5** (4D Occupancy and Flow Forecasting).
- **$H_4$ and $H_5$** govern **Stage 6** (Risk-Constrained Decision and Safety Layer) and validate in **Stage 7** (CARLA/Autoware).

This document serves as the formal research hypothesis charter for ForesightBEV Stage 0.

