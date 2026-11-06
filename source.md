# TAVYRA — PROJECT SOURCE OF TRUTH

**Tavyra — Camera-Centric 3D Occupancy, Spatiotemporal Forecasting and Risk-Constrained Decision Intelligence for Autonomous Driving.**
Formerly ForesightBEV. That name stays in earlier history below, in the repository URL, and in the Python package `foresightbev`.

## Final release status (2026-11-06)

The ten-stage numbering below stays frozen. Tavyra is a research prototype. On recorded nuScenes data it runs from six cameras to an
explained decision in **open-loop replay**. Closed-loop operation exists only in the built-in simulator and over ROS 2 with that
simulator. CARLA and Autoware have not been executed. The safety filter is not validated on real driving, and the event score is
not a calibrated collision probability.

| Stage | Final status |
|---|---|
| 0 Literature, REA, gap | Audits kept. Candidate gap not proven; direct prior-art comparison incomplete |
| 1 Data, cameras, calibration | Implemented and tested on nuScenes mini. Recorded ego pose is an input and is not camera-derived |
| 2 Geometry, BEV | Implemented. Flat ground or a coarse local plane |
| 3 Occupancy, reliability | Classical space carving plus model boxes. Heuristic camera health, no OOD model |
| 4 Detection, tracking | StreamPETR port and baselines, Kalman tracker. Preliminary on mini_val |
| 5 Forecasting | Constant-velocity Monte Carlo in the risk path. Simulator U-Net excluded. No trained 4D occupancy network on real data |
| 6 Risk, safety | Implemented. Not validated on real driving. Calibration is simulator-only |
| 7 CARLA, ROS 2, Autoware | ROS 2 verified with replay and simulator. CARLA adapter unexecuted. Autoware not integrated |
| 8 FastAPI, React | Implemented and inspected. Observability only |
| 9 Evaluation | Implemented on ten mini scenes and the simulator. Not generalisable |

Final documents: [release report](docs/project/TAVYRA_FINAL_RELEASE_REPORT.md), [technical handover](docs/project/TAVYRA_TECHNICAL_HANDOVER.md), [mentor questions](docs/project/TAVYRA_MENTOR_QA.md), [demo script](docs/project/TAVYRA_DEMO_SCRIPT.md).

Verified decisions: showcase configuration `configs/pipeline_showcase.yaml`; mini_val reported as preliminary; thresholds unchanged; `hold_speed_on_coverage_loss` set from a fault review for real-data replay only; U-Net excluded after failing to transfer; two-stage association rejected.


**Document:** `source.md`  
**Version:** 1.2  
**Prepared:** 5 November 2026  
**Project scope owned by this brief:** Stages 0–9 (all ten stages)  
**Scope correction (2026-11-05):** Stages 7–9 have NOT been implemented by anyone and are not teammate-owned. The user's correction is authoritative; see §16.

> This is the canonical project-context brief for Claude Code. Read it fully before inspecting or changing the workspace. It consolidates the project definition, existing research, known audit findings, provisional architecture, implementation principles, acceptance gates, and the current end-to-end engineering execution priorities. It is not a claim that every research conclusion or subsystem has already been independently reproduced.

---

## 1. Executive mandate

**ForesightBEV** is a camera-centric autonomous-driving research framework that aims to connect:

1. Multi-camera perception and geometrically consistent 3D occupancy.
2. Explicit representation of occupied, observed-free, and unknown/unobserved space.
3. Dynamic-agent detection and tracking.
4. Future 4D occupancy and motion/flow forecasting.
5. Uncertainty-aware, calibrated assessment of safety-critical trajectory events.
6. A risk-constrained safety filter that can reject or modify unsafe candidate trajectories.

The intended research chain is:

**camera observations → 3D scene representation → reliability/uncertainty → dynamic scene state → future occupancy/flow → trajectory-event risk → safety decision.**

The project must be technically defensible, modular, reproducible, and evaluated against relevant prior work. A visually convincing dashboard or a model that produces a plausible occupancy grid is not sufficient evidence of a research contribution.

### 1.1 Working research question (provisional)

> Can uncertainty in future camera-derived 4D occupancy be translated into calibrated probabilities of safety-critical trajectory events under incomplete, degraded, and temporally inconsistent observations, and can those estimates improve safety decisions without causing excessive conservatism?

This is a **candidate research question**, not a proven novelty claim. It must be stress-tested against recent work in occupancy forecasting, world models, occupancy-guided planning, trajectory uncertainty, reachability, and risk calibration.

### 1.2 What success means

The final research system should be able to take camera-only observations, represent the current 3D environment and its observation limits, estimate the future evolution of relevant scene occupancy, and assess candidate ego trajectories against that uncertain future scene. It must expose enough evidence and uncertainty for a downstream safety decision to be explainable and testable.

The work should distinguish:
- **Perception accuracy:** whether the current scene is represented correctly.
- **Observability:** whether a region was actually observed or is occluded/unobserved.
- **Predictive uncertainty:** how uncertain a future scene prediction is.
- **OOD evidence:** whether the current feature representation differs from the training distribution.
- **Calibration:** whether stated probabilities match empirical event frequencies.
- **Trajectory risk:** the probability of a defined safety event along a candidate trajectory.
- **Safety guarantee:** a formal property only if supported by explicit assumptions and a valid proof; empirical performance alone is not a formal guarantee.

---

## 2. Project boundaries and non-goals

### 2.1 Frozen ten-stage architecture

Do not renumber or redefine these stages.

| Stage | Frozen name | Scope in this work |
|---|---|---|
| 0 | Literature Review, REA and Research Gap | Prior art, source audits, research question, hypotheses, baselines, dataset/reproduction decisions. |
| 1 | Automotive Datasets, Cameras and Calibration | Dataset selection, camera metadata, calibration, synchronization, preprocessing, splits and ground-truth provenance. |
| 2 | Multi-view Geometry and BEV Projection | Camera feature extraction, calibration-aware projection/lifting, multi-view fusion, ego-motion alignment and BEV/3D feature construction. |
| 3 | 3D Occupancy Perception and Sensor Reliability | Semantic/geometric 3D occupancy, free/occupied/unknown state, visibility, uncertainty and OOD/reliability outputs. |
| 4 | Dynamic Agent Detection and Tracking | Dynamic road-user detection, temporal association, state estimation, motion/track uncertainty and interaction context. |
| 5 | 4D Occupancy and Flow Forecasting | Causal future occupancy prediction, voxel/agent flow and uncertainty propagation over explicit horizons. |
| 6 | Risk-Constrained Decision and Safety Layer | Trajectory-event risk estimation, calibrated risk, candidate assessment, safety filtering and fallback behavior. |
| 7 | CARLA + ROS 2 + Autoware | In scope. Strongest feasible CARLA + ROS 2 integration; Autoware compatibility investigated, never claimed unless it actually runs. |
| 8 | FastAPI Backend and React Dashboard | In scope. Must display actual pipeline outputs, never fabricated data. |
| 9 | Evaluation, Ablations and Research Reports | In scope. Executable evaluation, quantitative metrics, experiment configs, stored results and reproducible reports. |

### 2.2 Ownership boundary

Claude Code owns research, design, implementation and verification for **all ten stages (0–9)**. No teammate deliverable for Stages 7–9 exists. Earlier wording that assigned them to a teammate was an error and is superseded (see §16).

### 2.3 Explicit non-goals

- Do not present ForesightBEV as a replacement for the complete production Autoware stack.
- Do not put the dashboard in the safety-critical control path.
- Do not claim that LiDAR is an inference sensor for the camera-only model.
- Do not equate 2D BEV projection/IPM with full 3D occupancy understanding.
- Do not equate historical temporal input with future forecasting.
- Do not equate occupancy softmax confidence with calibrated collision probability.
- Do not claim novelty for established components such as BEV lifting, dense occupancy, sparse queries, GMM-based OOD scoring, MixUp, temperature scaling, or autoregressive occupancy forecasting.
- Do not claim formal safety guarantees unless the system assumptions, reachable-set construction, collision model and proof obligations are explicitly met.

---

## 3. Mandatory engineering and research principles

1. **Evidence before assertion.** Separate facts verified in code, statements in peer-reviewed papers, author-reported measurements, our own measurements, analytical interpretations, and unknowns.
2. **Camera-only inference.** Camera images are the only sensor inputs to a camera-only model at inference. Permitted metadata (calibration, timestamps, ego pose where available) must be declared. LiDAR is restricted to offline ground-truth generation, training supervision, or separately declared LiDAR baselines.
3. **Causal deployment protocol.** At prediction time, use only current and past information. Future frames may be used only to create labels or for explicitly offline analyses, never silently in the model input path.
4. **Separate uncertainty concepts.** Do not use aleatoric uncertainty, epistemic uncertainty, predictive confidence, OOD score, unknown space, and calibration error interchangeably.
5. **Unknown is not empty.** An invalid/ignored voxel (`255`, a visibility mask, or a missing label) is not automatically a model prediction of unknown. Explicitly define observed-free, occupied, and unknown/unobserved semantics and how they are supervised/evaluated.
6. **No metric laundering.** Do not compare metrics across different datasets, label taxonomies, grid sizes, camera setups, forecast horizons, masks, or protocols as if they were directly comparable.
7. **Reproducibility.** Pin commits, configs, dependencies, data versions, random seeds and evaluation commands. Keep experiment metadata and exact source references.
8. **Modularity.** Keep backbone, occupancy head, uncertainty/reliability, tracking, forecasting, risk and integration interfaces separable.
9. **Fail safely, but quantify conservatism.** A safety filter must be evaluated for collision reduction and for failure modes such as freezing, route abandonment, unnecessary braking and excessive detours.
10. **No overnight quality collapse.** Deadline pressure changes sequencing and implementation scope, not the truth standard, test standard or scientific claims.
11. **No destructive automation.** Inspect before editing. Preserve existing user work. Never reset, force-push, rewrite history, delete branches, alter remotes, or mass-delete files. The current execution assignment explicitly authorizes Claude Code to commit its own verified changes on a dedicated feature branch and push that branch to the existing `origin` remote after tests and diff review. Never push directly to `main`.
12. **Small, verifiable increments.** One stage/gate at a time; run focused tests and inspect diffs before advancing.

---

## 4. User environment and working constraints

Known development environment (verify current state before relying on it):

- Main machine: MacBook Air M5, Apple Silicon (ARM64), 24 GB unified memory, 1 TB SSD.
- macOS: reported 26.6, build 25G72.
- Homebrew, Git, GitHub CLI and `uv` are installed.
- Python 3.12.15 is available through `uv`.
- Main workspace: `~/Developer/ForesightBEV`.
- Existing main project `.venv` uses Python 3.12; `uv` is the preferred Python environment/dependency manager.
- Existing isolated legacy environment `.venvs/cam2bev-ipm/` must be preserved.
- Existing `ai-ml-workspace` has PyTorch with MPS support and basic scientific packages; do not assume the ForesightBEV `.venv` has the same packages.
- A separate Ubuntu/Linux NVIDIA machine with an RTX 4060 has been mentioned as a possible test target. Verify continued access, GPU model and VRAM before planning training.

Working preferences:
- Explain commands with meaningful side effects before executing them.
- Use `uv` for Python dependencies; never install packages globally with `pip`.
- Prefer official documentation and actively maintained repositories.
- Keep dependencies isolated per model/repository when legacy stacks conflict.
- Verify every installation with a small smoke test.
- Do not force CUDA-only training onto Apple Silicon/MPS.
- Do not install legacy OpenMMLab/CUDA dependencies on the Mac merely to satisfy an audit.

### 4.1 Git and workspace history (verify; do not assume current state)

Known initial repository setup:
- Root repository: `ForesightBEV`, branch `main`.
- GitHub remote: `https://github.com/sam-3skruthi/ForesightBEV.git`.
- Initial root commit: `aaef1cb chore: initialize ForesightBEV research repository`.
- `.gitignore` excludes `.venv`, `.venvs`, datasets/data, model weights, environment files, upstream clones under `research/repositories/*`, macOS `.DS_Store`, and certain experiment images.
- Upstream clones are intentionally local-only/untracked; research reports and metadata belong in the project repository.

Always inspect `git status --short --branch` before changes. Do not assume the workspace remains clean or at the initial commit.

### 4.2 GitHub collaboration and push policy

The existing GitHub repository is already created and connected. Expected remote (verify live; do not blindly overwrite): `https://github.com/sam-3skruthi/ForesightBEV.git`.

For the current Claude Code Desktop execution: use a **Local** Code session opened on the existing repository folder. The user authorizes a dedicated implementation branch (recommended: `claude/foresightbev-end-to-end-20261105`), reviewable commits containing only Claude's own task-related changes, and a push of that branch to `origin` after tests, diff review, and secret/large-file checks pass. Do not push to `main`, force-push, rewrite history, change remotes, or merge the branch. If the local checkout contains unrelated user changes, preserve them and exclude them from commits. Stage explicit file paths, not an indiscriminate `git add -A`. Report the pushed branch and commit hashes.

---

## 5. Existing project structure and research record

Expected/previously observed layout (inspect actual filesystem; paths may have grown):

```text
ForesightBEV/
├── README.md
├── SOURCE.md or source.md                  # canonical context; create/use root source.md
├── pyproject.toml
├── .python-version
├── .gitignore
├── src/foresightbev/
│   └── __init__.py
├── docs/
│   ├── architecture/
│   └── project/
│       └── RESEARCH_STATUS.md
├── research/
│   ├── datasets/
│   ├── experiments/
│   ├── repositories/                       # ignored upstream clones
│   │   ├── Birds_Eye_View/
│   │   ├── Cam2BEV/
│   │   ├── SurroundOcc/
│   │   ├── OPUS/
│   │   ├── OCCUQ/
│   │   └── Cam4DOcc/                       # pending/verify
│   ├── repository_audits/
│   │   ├── birds_eye_view/
│   │   ├── cam2bev/
│   │   ├── surroundocc/
│   │   ├── opus/
│   │   ├── occuq/
│   │   └── reliocc/
│   └── stage_0/
└── experiments/
    └── cam2bev_ipm/
```

### 5.1 Previously completed baseline audits

Treat these as existing research assets. Inspect their full reports before making changes.

**Birds_Eye_View**
- Official repository: `https://github.com/TejasKalsait/Birds_Eye_View`
- Audited commit: `e66817791ea897cadbf3a9093409e24ce8bc9dbc`.
- Findings: derivative of Cam2BEV; analytical homographies rather than learned geometry (`theta_const=True`); purported blind-spot detection is a small three-class CNN steering classifier; missing `part_2_model.pth`; syntax problems in scripts; no license at root; underlying Cam2BEV files have MIT license, dataset has separate non-commercial academic terms.
- Reports: `research/repository_audits/birds_eye_view/REA_REPORT.md`, `AUDIT_REVIEW_ADDENDUM.md`.

**Cam2BEV**
- Official repository: `https://github.com/ika-rwth-aachen/Cam2BEV`
- Audited commit: `5691703b2979d6af88ce8dd44abd6a49130b79e4`.
- Findings: fixed analytical IPM homography, not learned; official source had no pretrained weights; source MIT but dataset has separate non-commercial conditions; legacy TensorFlow 2.5 / uNetXST is not a native Apple Silicon path.
- Small reproductions already completed in isolated `.venvs/cam2bev-ipm`: analytical matrices agree with docs to <1e-14; four-camera warp; decoded uint8 multi-camera stitch and 2D occlusion map exactly matched official outputs on one sample each. These are one-sample checks, not generalization claims.
- Reports: `research/repository_audits/cam2bev/` including `REA_REPORT.md`, `SOURCE_MAP.md`, dependency/data audit, reproduction plan and review addendum.

**SurroundOcc — REA 01**
- Official repository: `https://github.com/weiyithu/SurroundOcc`
- Audited commit: `419bf5b47ee07ef0610950cce5ba99168c506753`.
- Apache 2.0.
- Dense camera-only 3D semantic occupancy baseline, six concurrent cameras, ResNet101 + DCNv2/FPN and spatial cross-attention / 3D deformable attention. The implementation constructs volumetric queries at multiple scales and uses a 3D decoder.
- Training supervision uses multi-frame LiDAR-derived occupancy labels and offline geometry; inference consumes six camera images at a current timestamp (the audited setup has no temporal image queue).
- Output has semantic occupancy classes plus empty/free handling; there is no dedicated predicted unknown class. Invalid labels are ignored/masked. Do not treat ignored labels as explicit unknown predictions.
- Legacy stack: Python 3.7, PyTorch 1.10.1, MMCV-full 1.4.0, MMDetection 2.14, MMDetection3D v0.17.1, custom CUDA kernels; authors report 8× RTX 3090 and around 60 hours of training.
- Stage mapping: GT/data/calibration Stage 1; multi-view geometry and camera feature interaction Stage 2; occupancy head Stage 3; evaluation Stage 9. It does not implement Stages 4–8.
- Five reports at `research/repository_audits/surroundocc/`: `REA_REPORT.md`, `SOURCE_MAP.md`, `DEPENDENCY_AND_DATA_AUDIT.md`, `REPRODUCTION_ASSESSMENT.md`, `AUDIT_REVIEW_ADDENDUM.md`.

**OPUS / OPUS-V2 — REA 02**
- Official repository: `https://github.com/jbwang1997/OPUS`
- Audited commit: `fdb17339f40168ea0ceda9615863e3f5d0c3f5ce`.
- MIT license.
- OPUS and OPUS-V2 are modules in one repository (`models/opusv1/`, `models/opusv2/`); OPUS-Fusion is a separate camera+LiDAR baseline and must not be represented as camera-only.
- OPUS-V2 variants: T 600 queries × 128 points; S 1,200 × 64; M 2,400 × 32; L 4,800 × 16; each yields 76,800 sampled points before densification.
- Reported 38.6 mIoU / 44.0 RayIoU / 8.6 FPS is OPUS-V2-L; 25.8 FPS belongs to OPUS-V2-T with lower mIoU (reported 35.6–36.5 depending on config/table). Never conflate these configurations or compare FPS without the same hardware/protocol.
- Standard camera inference aggregates eight historical/current surround sweeps and predicts current occupancy; this is not future occupancy forecasting. `fut-15f` is an offline bidirectional future-context/autolabeling setup, not causal future occupancy prediction.
- V2 uses 16 semantic occupancy classes plus empty label index 17 (18 label channels under the reported config); no explicit unknown class. Verify exact config if using.
- The commit removed MMDetection3D/MMsegmentation, but legacy dependencies remain (reported MMCV-full 1.6.0, MMDetection 2.28.2, spconv-cu116, torch-scatter, custom CUDA kernel). It is not a drop-in Mac/MPS setup.
- Reports at `research/repository_audits/opus/`: four core reports plus `REA02_VERIFICATION_ADDENDUM.md`.

**OCCUQ — REA 03**
- Official repository: `https://github.com/ika-rwth-aachen/OCCUQ`
- Audited commit: `2aa54297151b695929524ad8a4f4863ba5ae6387`.
- Apache 2.0. ICRA 2025 paper: *OCCUQ: Exploring Efficient Uncertainty Quantification for 3D Occupancy Prediction*.
- Builds on a dense SurroundOcc-like camera occupancy architecture. Its reported method includes a feature projection head (including an MLPHeadv5 variant with spectral normalization/residual structure) and a class-conditional GMM feature-density score. Low feature likelihood is used as OOD/epistemic-indicator evidence.
- GMM scores are not automatically calibrated collision probabilities or calibrated class probabilities.
- Reported missing-camera experiment: severe three-camera dropouts; GMM AUROC reported 95.39%, compared with much weaker softmax entropy/max-probability scores. Full-scale macro MultiCorrupt values reported in the audit should be verified against exact paper table, variant, severity and metric aggregation before reuse.
- GMM GPU timing around 9.48 ms for 640k voxels is author-reported. Comparisons with 938.41 ms CPU naive distribution evaluation mix hardware and algorithmic changes; do not call it an isolated 100× algorithmic speedup.
- Spectral normalization alone provides an upper Lipschitz bound, not a lower bound or automatic non-collapse. The Stage 0 review claims a residual path helps; the strict lower-bound argument needs verification of the actual branch Lipschitz constant and activation details. Do not repeat a bi-Lipschitz guarantee without proof.
- Legacy CUDA/OpenMMLab environment. Not a native MPS assumption.
- Reports: `research/repository_audits/occuq/` including REA report, source map, dependency/data audit, reproduction assessment, baseline comparison and review addendum.

**ReliOcc — REA 04 (paper audit; no official implementation found)**
- Final publication: *Reliable and Calibrated Semantic Occupancy Prediction by Hybrid Uncertainty Learning*, IJCAI 2025, pp. 1973–1981, DOI `10.24963/ijcai.2025/220`.
- Preprint: arXiv:2409.18026v1, earlier title *ReliOcc: Towards Reliable Semantic Occupancy Prediction via Uncertainty Learning*.
- No verifiable official source repository was found in the audit. Treat all details as paper-derived, not code-verified.
- Method concepts: absolute uncertainty via Gaussian logit sampling; relative voxel uncertainty learning via uncertainty-weighted feature mixing; offline uncertainty-aware diagonal calibration.
- Critical behavior: online training mode's sigma branch is discarded at inference; it acts as a training regularizer. Offline mode freezes the base model and trains a calibrator; calibrated class confidence is not the same as a feature-density OOD score or epistemic uncertainty.
- Uses SemanticKITTI monocular forward camera temporal input (current + four past frames in the audited protocol), not six-camera surround view. Frame-drop robustness does not establish surround-camera dropout robustness.
- Important unresolved items: exact loss implementation for unweighted sum of one-hot labels; empty voxel imbalance in pair shuffling; calibration split/leakage; adaptation from monocular to surround-view.
- Reports: `research/repository_audits/reliocc/` including `REA_REPORT.md`, `PAPER_AND_METHOD_MAP.md`, dependency/data audit, reproduction assessment, comparison and review addendum.

### 5.2 Stage 0 synthesis status

Existing synthesis documents were reported complete under `research/stage_0/`:
- `OCCUPANCY_AND_UNCERTAINTY_SYNTHESIS.md`
- `RESEARCH_GAP_AND_HYPOTHESES.md`
- `EVIDENCE_AND_OPEN_QUESTIONS.md`
- `REA_COMPARATIVE_AUDIT.md`
- `STAGE0_SYNTHESIS_REVIEW_ADDENDUM.md`

The first synthesis and review addendum are not permission to declare all of Stage 0 scientifically closed. The Cam4DOcc audit was next in sequence and 4D forecasting / occupancy-guided planning / trajectory-risk prior art still requires direct investigation. Read these documents and correct only with sourced addenda; preserve history.

### 5.3 Corrections already required in the Stage 0 record

- Dense grid **storage** is O(V), where V is voxel count. Multi-scale deformable cross-attention is not full O(N_query²) pairwise self-attention; document actual sampling complexity and activation memory separately.
- Distinguish memory for stored voxel features from training activation memory and peak GPU memory.
- Spectral normalization alone does not establish bi-Lipschitzness; any residual lower-bound argument needs a strict Lipschitz bound on the residual branch and exact architectural proof.
- OCCUQ CPU/GPU timings are not a controlled algorithm-only speedup benchmark.
- ReliOcc training regularization, inference outputs and post-hoc calibrated confidence are separate.
- OPUS-V2 variant / query / FPS / accuracy values must stay tied to exact variants.
- Historical inputs, offline future-context label generation and causal future forecasting must stay distinct.

---

## 6. Research landscape to extend and verify

The following are priority papers/repos for Stage 0 continuation. This list is a starting point, not a complete literature review. Verify official paper, repository, version, release and code status independently.

### 6.1 Present occupancy / geometry / efficiency

- BEVFormer: `https://github.com/fundamentalvision/BEVFormer`
- BEVDepth: `https://github.com/Megvii-BaseDetection/BEVDepth`
- SurroundOcc: `https://github.com/weiyithu/SurroundOcc`
- OPUS: `https://github.com/jbwang1997/OPUS`
- StreamOcc project: `https://moonseokha.github.io/StreamOcc/` (ECCV 2026; investigate whether streaming temporal aggregation estimates current occupancy or forecasts future occupancy, and do not conflate the two).
- OccAny: `https://github.com/valeoai/OccAny`
- TT-Occ: `https://github.com/Xian-Bei/TT-Occ`

### 6.2 Uncertainty / robustness / OOD

- OCCUQ: `https://github.com/ika-rwth-aachen/OCCUQ`
- ReliOcc: `https://ijcai.org/proceedings/2025/220` (paper; no official code found in REA 04).
- UniOcc: `https://github.com/tasl-lab/UniOcc`
- OccStress: `https://github.com/InSAI-Lab/OccStress`
- Investigate related probabilistic occupancy / reliability work, including PORA and other papers found in a fresh citation search; verify title, publication and official artifacts before citing.

### 6.3 4D occupancy / world models / forecasting

- Cam4DOcc: `https://github.com/haomo-ai/Cam4DOcc` (CVPR 2024; REA 05 pending at handoff).
- OccWorld: `https://github.com/wzzheng/OccWorld` (ECCV 2024; occupancy world model, ego and scene token evolution; verify modality and action conditioning).
- Drive-OccWorld: `https://drive-occworld.github.io/` (AAAI 2025 oral; action-conditioned 4D occupancy world model and planning connection).
- UniOcc and other occupancy forecasting benchmarks should be assessed for data and protocol compatibility.

### 6.4 Planning / trajectory risk / uncertainty-aware safety

These recent works mean broad novelty claims about “occupancy + planning” or “uncertainty-aware safe planning” are not defensible without direct comparison:

- OccDriver, ICLR 2026: *Future Occupancy Guided Dual-branch Trajectory Planner in Autonomous Driving*. Official proceedings: `https://proceedings.iclr.cc/paper_files/paper/2026/hash/e59bf9a9077c361ed57180fc79c5b8d8-Abstract-Conference.html`. It combines trajectory and future occupancy/flow prediction; inspect its contingency objective and reactive/non-reactive closed-loop evaluation.
- SURE-Plan, Transportation Research Part C, September 2026, DOI `10.1016/j.trc.2026.105800`: *A spatio-temporal uncertainty-aware reachable set embedding framework for safe and efficient motion planning of autonomous vehicles*. It uses evidential trajectory prediction, reachable sets, hard/soft risk constraints and optimization. It is not automatically equivalent to camera-derived voxel uncertainty, but overlaps trajectory uncertainty and safety planning; direct gap comparison is mandatory.
- Drive-OccWorld (above) connects action-conditioned world models and planning.
- PORA and other probabilistic occupancy-to-risk work must be located, verified and compared.

### 6.5 Critical novelty warning

The earlier Stage 0 synthesis proposed an intersection around future occupancy uncertainty and calibrated trajectory collision probability. Treat this only as a **candidate gap**. OccDriver (2026), SURE-Plan (September 2026), Drive-OccWorld (2025), and other recent work may cover pieces of the intersection. A defensible gap may need to narrow to a precise combination such as camera-only surround-view uncertainty propagation into calibrated trajectory-event risks under spatially localized camera degradation, with an explicit observability state and a measured safety/conservatism tradeoff. Even that narrower idea requires a direct novelty search and fair evaluation plan.

---

## 7. Candidate end-to-end system architecture (NOT FROZEN)

This is a candidate logical architecture. Each block must be validated through REA, data checks and small experiments before implementation is committed.

```text
Six-view surround camera images + timestamps + calibration + ego pose
                              │
                              ▼
                 Stage 1: Data / calibration
                              │
                              ▼
              Stage 2: Multi-view feature encoder
          image backbone + FPN + camera geometry/lifting
            temporal alignment / BEV or 3D feature fusion
                              │
                              ▼
               Stage 3: Present 3D occupancy
       semantic occupancy + observed/free/unknown semantics
             visibility evidence + reliability outputs
                  ┌───────────┴───────────┐
                  ▼                       ▼
       OOD / feature-density       predictive confidence /
          diagnostic score        calibration / uncertainty
                  └───────────┬───────────┘
                              │
                              ▼
                Stage 4: Dynamic object state
           detection + association + tracks + velocity
              state covariance / identity / class
                              │
                              ▼
              Stage 5: Future 4D scene forecast
        future semantic occupancy at multiple time horizons
               voxel/agent flow + temporal dynamics
            uncertainty and observability propagated over time
                              │
                              ▼
                Stage 6: Trajectory risk evaluation
          candidate ego paths + swept-volume/time alignment
           event definition + occupancy interaction + risk
                              │
                              ▼
                  Stage 6: Safety filter
       safe/unsafe/unknown decision + constraints + fallback
                              │
                              ▼
                Stage 7 integration
          CARLA / ROS 2 / Autoware interface and execution
```

### 7.1 Architecture decisions still open

Do not yet choose among dense voxel, sparse query, hybrid or streaming representations. Evaluate against:
- Occupancy quality and coverage.
- Explicit unknown-space support.
- Camera-dropout and weather robustness.
- Temporal consistency.
- Future forecasting performance.
- Memory/latency on realistic hardware.
- Compatibility with future risk evaluation.
- Reproduction cost and license.

Do not stack OCCUQ and ReliOcc simply because they appear complementary. First determine whether the signals are distinct, stable, calibratable and spatially actionable.

### 7.2 Important interfaces (provisional contracts)

The implemented contracts live in `src/foresightbev/contracts.py`. These are semantic contracts, not a demand to rewrite their modules.

**OccupancyFrame (Stage 3 output)**
- timestamp and coordinate frame;
- grid bounds, voxel resolution, dimensions and transform;
- class taxonomy and occupancy probabilities/logits;
- observed/visible mask;
- explicit unknown/unobserved state or mask;
- reliability/calibration metadata;
- optional separate OOD feature score;
- model/config/checkpoint version.

**TrackState (Stage 4 output)**
- timestamp and frame;
- track ID and class;
- 3D position/extent and velocity;
- state covariance or uncertainty representation;
- track confidence, age and validity.

**ForecastVolume (Stage 5 output)**
- reference timestamp and coordinate frame;
- forecast horizons and timestamps;
- occupancy probabilities/logits per horizon;
- semantic taxonomy;
- flow vectors with precise coordinate/sign convention;
- unknown/validity mask;
- uncertainty representation and calibration metadata;
- model version.

**TrajectoryRisk (Stage 6 output)**
- candidate trajectory ID and time parameterization;
- event definition (e.g. collision with specified obstacle/region and time window);
- risk probability / score and calibration status;
- per-horizon risk contributions;
- unknown-space / OOD exposure;
- assumptions, model versions and diagnostics.

**SafetyDecision (Stage 6 output)**
- pass/reject/modify/indeterminate status;
- approved or modified trajectory;
- active constraints;
- fallback/minimum-risk action recommendation;
- reason codes and diagnostic record.

### 7.3 Risk calculation caution

Do not calculate a trajectory collision probability by naïvely multiplying independent voxel probabilities unless independence assumptions are justified. Occupancy across neighboring voxels and across time is correlated. A trajectory event needs an explicit definition, a dependence treatment or a conservative bound, and empirical calibration on held-out scenarios. Unknown space must be handled explicitly rather than silently treated as free.

---

## 8. Stage-by-stage plan, expected outputs and acceptance gates

### Stage 0 — Literature Review, REA and Research Gap

**Purpose:** establish what is known, what is available, and what is genuinely unresolved.

Work:
- Complete REA 05 Cam4DOcc; then examine OccWorld, Drive-OccWorld, OccDriver, SURE-Plan, StreamOcc and relevant probabilistic/risk work.
- Verify official repos, exact commit SHA, paper versions, licenses, code/checkpoints, data and environment.
- Trace architectures, modality, causal temporal inputs, prediction horizons, losses, metrics and failure cases.
- Compare all prior work against ForesightBEV, not just in isolation.
- Update research question and hypotheses only after the direct comparison.

Outputs:
- literature matrix with citations;
- audit reports with source paths/line references;
- prior-art comparison;
- candidate research gap with counter-evidence;
- testable hypotheses, baselines, metrics and falsification criteria;
- go/no-go decision for datasets and architecture candidates.

Gate 0: no unsupported novelty claim; key competitors audited; metrics and datasets specified; open uncertainties documented.

### Stage 1 — Automotive Datasets, Cameras and Calibration

**Purpose:** create an auditable data pipeline and calibration foundation.

Candidate route:
- nuScenes six-camera surround images and metadata.
- Occ3D-nuScenes occupancy annotations for 3D semantic occupancy supervision/evaluation, subject to exact license and protocol verification.
- Cam4DOcc data/labels for future occupancy/flow experiments if licensing, availability and protocol are suitable.
- Other datasets such as Lyft-Level5 only if they add a clear benchmark or temporal task and can be supported reproducibly.

Work:
- inspect data rights, storage, download mechanism and redistribution conditions before acquisition;
- define canonical dataset manifest and splits;
- parse camera intrinsics/extrinsics, ego poses, timestamps and calibration transforms;
- validate synchronization and temporal alignment;
- define coordinate convention and transformations;
- document LiDAR role in label generation only;
- validate annotations and missing/corrupt sample handling.

Outputs:
- dataset decision record;
- calibration/data schema;
- small-sample data integrity report;
- sequence/split manifests;
- provenance and license register.

Gate 1: a small data subset can be loaded and calibrated correctly, transforms pass numerical/visual checks, and license/usage is recorded. Do not download full datasets until required storage and rights are confirmed.

### Stage 2 — Multi-view Geometry and BEV Projection

**Purpose:** create geometry-consistent multi-camera features.

Candidate components:
- per-camera image backbone and feature pyramid;
- camera calibration-aware frustum/depth projection or learned lifting;
- multi-view feature fusion into BEV/3D latent representation;
- ego-motion alignment for past observations;
- masking for camera visibility and out-of-view projections.

Work:
- compare dense voxel queries, BEV queries, sparse point queries and streaming aggregation;
- reproduce a small projection geometry example before adopting a large network;
- validate coordinate frames and camera overlap;
- test calibration perturbations and missing camera handling;
- profile memory and latency separately from full training.

Outputs:
- geometry module and unit tests;
- projection visualizations;
- coordinate convention document;
- baseline feature encoder configuration.

Gate 2: projection/transform tests pass and multi-camera alignment is demonstrated on a small verified sample.

### Stage 3 — 3D Occupancy Perception and Sensor Reliability

**Purpose:** estimate present 3D scene occupancy and make observation/reliability limitations explicit.

Required conceptual outputs:
- semantic occupied voxels;
- observed-free space where there is evidence supporting free space;
- unknown/unobserved/occluded space;
- visibility/observation support;
- separate prediction confidence/calibration;
- optional OOD/feature-density indicator;
- calibration and model provenance.

Work:
- establish a strong reproducible occupancy baseline before custom novelty;
- define class taxonomy and empty/free/ignore/unknown semantics;
- define how unknown labels are generated and evaluated;
- compare dense SurroundOcc-like and sparse OPUS-like representation choices;
- evaluate OCCUQ-style OOD score and ReliOcc-like calibration separately;
- assess camera failures, localized dropout, weather, motion blur, brightness, occlusion and out-of-distribution scenes;
- keep LiDAR only in offline training/GT paths.

Outputs:
- occupancy model and inference schema;
- label semantics and visibility protocol;
- uncertainty/reliability head only if justified;
- calibration and corruption results;
- model card and reproducibility report.

Gate 3: baseline occupancy reproducible; no LiDAR leakage; unknown/free semantics are testable; uncertainty metrics add value over simple confidence baselines.

### Stage 4 — Dynamic Agent Detection and Tracking

**Purpose:** provide dynamic scene state and continuity across time.

Work:
- camera-only 3D detection or occupancy-derived object/instance extraction;
- multi-object temporal association;
- ego-motion compensation;
- position, extent, velocity and class estimation;
- track birth/death and occlusion handling;
- uncertainty/covariance estimation and consistency evaluation.

Outputs:
- `TrackState` interface;
- tracking module and evaluation report;
- dynamic/static scene decomposition with explicit limitations.

Gate 4: track identities and velocities are sufficiently stable on held-out sequences; occlusion/re-identification failure modes documented.

### Stage 5 — 4D Occupancy and Flow Forecasting

**Purpose:** predict future scene occupancy and motion from causal observations.

Work:
- audit and benchmark OCFNet/Cam4DOcc, OccWorld and Drive-OccWorld; distinguish forecasting from reconstruction and action-conditioned simulation;
- compare direct future occupancy prediction, autoregressive forecasting, token/world-model prediction, and flow-conditioned representations;
- forecast at several horizons chosen to match dataset timestamps and the downstream planner;
- define flow convention and evaluate it independently;
- propagate observation uncertainty and track uncertainty into future occupancy;
- measure horizon-wise degradation and calibration;
- use only past/current input at forecast time.

Outputs:
- forecast model and `ForecastVolume` contract;
- horizon-wise occupancy and flow metrics;
- uncertainty propagation method (only if tested);
- temporal consistency and corruption ablations.

Gate 5: future outputs beat persistence/static and relevant published baselines under a matched protocol; causal inference is proven; uncertainty quality is measured by horizon.

### Stage 6 — Risk-Constrained Decision and Safety Layer

**Purpose:** convert uncertain future scene predictions into interpretable trajectory-event risk and a bounded safety intervention.

Work:
- define candidate ego trajectories and time alignment;
- define safety events (collision, minimum separation violation, crossing of occupied/unknown region, etc.) precisely;
- evaluate swept vehicle volume/corridor against future occupancy and dynamic agents;
- address spatial/temporal correlation and unknown-space exposure;
- obtain risk probabilities using a justified probabilistic model;
- calibrate event probabilities on disjoint held-out sequences/scenarios;
- design a risk threshold policy and fallback/minimum-risk behavior;
- benchmark safety versus conservatism, not collision count alone;
- examine reachability/barrier/optimization methods only with clearly stated assumptions.

Outputs:
- trajectory-event risk API;
- risk calibration report and reliability plots;
- safety filter with reason codes;
- closed-loop/simulation evaluation inputs for Stages 7–9.

Gate 6: event probabilities are calibrated on held-out scenarios; filter reduces risk under declared conditions; route completion and excessive intervention are measured; fallback behavior is deterministic and tested. Do not call it a formal safety guarantee unless formally established.

---

## 9. Data and benchmark strategy (provisional)

### 9.1 Primary candidate

**nuScenes + Occ3D-nuScenes** is the leading candidate for camera-only six-view present-time 3D occupancy because it offers surround-camera images, calibration/ego metadata and occupancy supervision. Occupancy labels may be LiDAR-derived; this is allowed only as offline supervision/ground truth under the camera-only inference rule.

Before locking it, verify:
- exact nuScenes release and Occ3D annotation release;
- license and academic/non-commercial restrictions;
- occupancy class taxonomy, empty and ignore semantics;
- train/validation/test split and scene leakage;
- camera sync and pose alignment;
- data volume and storage availability;
- whether temporal forecasting labels are available or must be generated.

### 9.2 Forecasting candidates

**Cam4DOcc** is a priority benchmark for camera-only 4D occupancy forecasting. Its paper describes sequential occupancy for general movable/static objects and 3D backward centripetal flow, with multiple baseline families and OCFNet. Audit exact per-dataset task, modality and protocol before adopting.

Lyft-Level5 may appear in Cam4DOcc; do not assume its data availability, labels or license match nuScenes. SemanticKITTI is useful for monocular prior-art analysis but is not equivalent to a six-camera surround-view benchmark.

### 9.3 No apples-to-oranges comparisons

Never compare:
- SemanticKITTI monocular results to nuScenes six-camera results as equivalent;
- Occ3D present-frame mIoU to future forecasting metrics;
- different class taxonomies/grid resolutions/masks as one leaderboard;
- paper-reported FPS from different GPUs or resolutions;
- a point-cloud inference baseline to a camera-only model result.

---

## 10. Metrics and evaluation plan (provisional; finalize after dataset audit)

### Stage 3 — Occupancy

- per-class IoU and mIoU;
- geometric occupancy IoU / precision / recall;
- RayIoU only where benchmark-defined;
- free-space false-free rate and occupied false-negative rate;
- explicit unknown/observability precision, recall and coverage;
- risk-weighted or visibility-aware error analysis;
- corruption severity curves.

### Reliability / uncertainty

- Expected Calibration Error (ECE) with binning details;
- Brier score and negative log-likelihood;
- reliability diagrams;
- AUROC/AUPR/FPR95 for OOD/error detection, with positive class clearly defined;
- calibration under clean and corrupted distributions;
- selective risk/coverage curves;
- localization quality of uncertainty around failed cameras and occluded regions.

An OOD AUROC does not prove calibrated class probabilities. ECE does not prove OOD detection. Confidence is not an unknown-space label.

### Stage 4 — Tracking

Choose dataset-appropriate metrics after inspecting label availability: HOTA, IDF1, AMOTA/AMOTP, MOTA and/or tracking-specific 3D metrics. Do not promise a metric unsupported by the chosen dataset.

### Stage 5 — Forecasting

- horizon-wise occupancy IoU/mIoU;
- future semantic and geometric metrics defined by the benchmark;
- flow endpoint error / angular error where ground truth supports it;
- temporal consistency;
- horizon-wise calibration / Brier / NLL;
- robustness versus forecast horizon and corruption severity.

### Stage 6 — Risk and safety

- event-level Brier score, NLL and reliability calibration;
- trajectory-event calibration error (if a custom T-ECE is defined, specify bins, event and aggregation; do not call it a standard metric without citation);
- AUROC/AUPR for collision-event ranking if relevant;
- collision rate / near-collision rate;
- minimum separation / TTC where valid;
- route completion;
- safety intervention rate, emergency braking and fallback rate;
- progress, jerk/comfort and efficiency;
- conservative-freeze rate / false rejection rate.

A credible safety result must show the tradeoff curve, not only a lower collision count.

---

## 11. Provisional hypotheses (not yet accepted or numerically frozen)

The Stage 0 synthesis identified five candidate hypotheses:

**H1 — Spatial observability/reliability:** Can a camera-only model distinguish occupied, observed-free and unknown/unobserved space under partial camera visibility and occlusion?

**H2 — Uncertainty calibration under degradation:** Can voxel-level uncertainty and class confidence remain useful/calibrated under camera degradation, missing views and distribution shift without unacceptable occupancy accuracy loss?

**H3 — Temporal uncertainty propagation:** Can uncertainty about present occupancy and dynamic-agent motion be propagated into future occupancy/flow, with quality evaluated at each horizon?

**H4 — Trajectory-event risk calibration:** Can predicted future occupancy and its uncertainty produce calibrated probabilities of defined safety-critical events along continuous candidate vehicle trajectories?

**H5 — Safety filtering:** Can those calibrated risks improve closed-loop safety without unacceptable loss of route completion, progress or comfort?

Earlier synthesis included numeric targets such as 40% false-free reduction, mAUROC ≥85%, ECE ≤3%, T-ECE ≤5%, 50% fewer collisions and ≥85% route completion. These are **aspirational placeholders**, not validated acceptance criteria. Do not freeze them until a baseline, dataset size, confidence interval, metric definition and literature comparison justify them. Claude must propose statistically defensible thresholds and label them as provisional.

---

## 12. End-to-end engineering delivery strategy

**Current execution target:** complete the strongest runnable and integrated ForesightBEV path supported by the actual repository, data, checkpoints, software environment and hardware by the user's stated deadline. Do not create presentation slides or spend time on presentation preparation unless the user separately requests it.

### 12.1 Definition of end-to-end completion

The target data path is:

`timestamped multi-camera inputs + calibration/ego metadata`

→ input validation and preprocessing

→ multi-view geometry / BEV representation

→ occupancy and observation-state representation

→ dynamic-agent detection and tracking

→ future occupancy or motion forecasting

→ candidate trajectory generation/evaluation

→ risk assessment and safety-filter decision

→ Stage 7 simulation/middleware, Stage 8 backend/dashboard and Stage 9 evaluation

→ recorded outputs, tests and a reproducible run command.

A component is complete only when code is present, executed with appropriate inputs, tested, and its result inspected. Clearly label learned, pretrained, classical, heuristic, synthetic-only, partial and deferred components.

### 12.2 Implementation sequencing

1. Inspect current Git state, all existing source modules, audit reports, data, calibration, checkpoints and Stages 7–9.
2. Reuse verified code and experiments, especially the Cam2BEV geometry work, without modifying upstream clones.
3. Establish typed data contracts and coordinate/time conventions.
4. Build the smallest real-data end-to-end path and verify each boundary.
5. Add occupancy/observability, tracking, forecasting and trajectory-risk modules in dependency order.
6. Build Stages 7–9 against the same typed contracts.
7. Run tests and actual end-to-end execution; inspect outputs and repair failures.
8. Stabilize, document and verify all changes locally first. Publish to the existing GitHub repository only after the full local implementation has passed review and end-to-end verification.

### 12.3 Scope and truthfulness

The deadline does not make GPU training, multi-dataset label generation, calibrated uncertainty, statistically validated future forecasting or closed-loop safety validation instantaneous. If an existing compatible checkpoint, dataset and environment are unavailable, use a scientifically labelled baseline when useful and put the learned/research-grade component into Phase 2 with explicit acceptance criteria. Never replace a missing learned model with a heuristic and claim equivalence.

### 12.4 Stages 7–9 ownership

Stages 7–9 are in scope and owned by this implementation. There is no teammate code to preserve. Do not claim CARLA, ROS 2 or Autoware integration unless the actual launch and message path was executed.

---

## 13. Expected final research outputs (after Phase 2 and validated implementation)

By the end of Stages 0–9, the expected outputs are:

- documented prior-art and reproducibility decision;
- reproducible multi-camera dataset/calibration pipeline;
- geometry-consistent camera-to-BEV/3D representation;
- present-time semantic occupancy with an explicit treatment of observed-free and unknown space;
- reliability/uncertainty outputs whose meanings are independently validated;
- camera-derived dynamic tracks and motion state;
- future 4D occupancy and flow predictions with explicit horizons;
- horizon-dependent uncertainty estimates and calibration analysis;
- candidate trajectory-event risk estimates with held-out calibration;
- a risk-constrained filter with measurable safety/efficiency tradeoffs;
- simulation/middleware integration, a dashboard showing real pipeline outputs, and reproducible evaluation reports.

These are desired deliverables, not claims that they currently exist.

---

## 14. Risk register

| Risk | Why it matters | Required mitigation |
|---|---|---|
| Novelty overlap with 2025–2026 planning/world-model papers | The broad research question may already be substantially addressed. | Direct prior-art comparison; narrow or reject the gap if necessary. |
| Missing explicit unknown labels | Ignored voxels can be mistaken for free or unknown. | Define and validate observability/unknown supervision. |
| LiDAR leakage | Invalidates camera-only claims. | Audit every inference dataloader and model input. |
| Temporal leakage | Future frames can contaminate purported causal forecasting. | Trace data indices and timestamp alignment end to end. |
| Uncalibrated uncertainty | Cannot be interpreted as event probability. | Held-out calibration, Brier/NLL/ECE and OOD tests. |
| Spatial/temporal correlation | Naive voxel independence can misestimate collision probability. | Dependence-aware event evaluation or conservative validated bounds. |
| Empty-space dominance | Losses and random sampling may be dominated by empty voxels. | Class/visibility-aware sampling and ablations. |
| Legacy dependencies | Reproduction may be infeasible on Apple Silicon. | Isolated Linux/CUDA environments, pinned commits; adapt only after baseline audit. |
| Data licensing / storage | Datasets may be restricted or very large. | Review license, data terms and disk requirements before downloading. |
| Deadline pressure | Encourages untested shortcuts and exaggerated claims. | Prioritize the smallest real-data end-to-end path, enforce tests, and defer unsupported research-grade claims to Phase 2. |
| Contract drift | Stage 7–9 layers may break if contracts drift. | Single typed contract module; adapters tested against it. |

---

## 15. Source and evidence hierarchy

When making a claim, prefer evidence in this order:

1. Exact audited source code at a pinned commit (with file path and line range).
2. Final peer-reviewed paper and official supplementary material.
3. Official project documentation / model card / dataset documentation.
4. Author-released checkpoint/configuration and official benchmark results.
5. Reputable secondary research sources, clearly marked secondary.
6. Existing ForesightBEV audit summaries, used as a guide but rechecked for material claims.

Do not use a GitHub README as the sole source for mathematical implementation details if code/paper can be inspected. Do not treat benchmark leaderboards as comparable without protocol matching.

### Core references

- BEVFormer: https://github.com/fundamentalvision/BEVFormer
- BEVDepth: https://github.com/Megvii-BaseDetection/BEVDepth
- SurroundOcc: https://github.com/weiyithu/SurroundOcc
- OPUS / OPUS-V2: https://github.com/jbwang1997/OPUS
- OCCUQ: https://github.com/ika-rwth-aachen/OCCUQ
- ReliOcc (IJCAI 2025): https://ijcai.org/proceedings/2025/220
- Cam4DOcc (CVPR 2024): https://github.com/haomo-ai/Cam4DOcc
- OccWorld (ECCV 2024): https://github.com/wzzheng/OccWorld
- Drive-OccWorld (AAAI 2025): https://drive-occworld.github.io/
- OccDriver (ICLR 2026): https://proceedings.iclr.cc/paper_files/paper/2026/hash/e59bf9a9077c361ed57180fc79c5b8d8-Abstract-Conference.html
- SURE-Plan (2026): https://doi.org/10.1016/j.trc.2026.105800
- StreamOcc (ECCV 2026): https://moonseokha.github.io/StreamOcc/
- OccAny: https://github.com/valeoai/OccAny
- TT-Occ: https://github.com/Xian-Bei/TT-Occ
- UniOcc: https://github.com/tasl-lab/UniOcc
- OccStress: https://github.com/InSAI-Lab/OccStress

---

## 16. Change and decision log

Claude Code must maintain this section rather than silently changing the project definition.

| Date | Decision/change | Evidence / rationale | Status |
|---|---|---|---|
| 2026-11-05 | Updated project context for Claude Code Desktop execution and branch-based GitHub delivery through Stages 0–6. | Prior conversations and existing audit summaries; verify against workspace files. | Baseline brief; decisions remain provisional where noted. |
| 2026-11-05 | **Correction:** Stages 7, 8 and 9 were never implemented and are not teammate-owned; all ten stages are in scope. Removed the teammate-ownership wording from §§ header, 2.1, 2.2, 7, 7.2, 8, 12, 13, 14, 17.3. | User's authoritative correction; workspace inspection confirmed no Stage 7–9 code exists. | Correction |
| 2026-11-05 | **Correction:** no GitHub push, pull or branch creation during implementation; publication happens only after local completion and user review. Supersedes the §3.11 and §4.2 branch-and-push authorization until the user says otherwise. | User instruction. | Correction |
| 2026-11-05 | **Decision:** nuScenes v1.0-mini is the real-data input, loaded without the devkit from `research/datasets/nuscenes`. Annotations are scoring-only; LiDAR is unused; no Occ3D labels exist in it. Pretrained SegFormer-B2 supplies labels. | `docs/datasets/NUSCENES.md`, `experiments/end_to_end/results/datasets/`. | Decision |
| 2026-11-05 | **Finding:** on eight held-out real scenes, recall is 21–25%, 36% of box cells are false-free, and the filter alarms on 53% of normal driving frames. Simulator results (1.1% collisions) do not transfer. Unknown-space policy shows no measured benefit in simulation. | `experiments/end_to_end/results/nuscenes_heldout/`, `results/final/`, `docs/project/IMPLEMENTATION_STATUS.md`. | Hypothesis update |
| 2026-11-05 | **Correction:** frontend is frozen pending backend review; backend priority per user instruction. | User instruction. | Correction |
| 2026-11-05 | **Correction:** frontend unfrozen for the final showcase build; redesigned and completed against the API. | User instruction. | Correction |
| 2026-11-06 | **Rename:** project identity is now Tavyra. Repository URL and package name unchanged. | User instruction. | Decision |
| 2026-11-05 | **Decision:** StreamPETR R50 428q added as a pretrained detector through an inference-only PyTorch port, behind `detector:`. Baselines kept. Weights licence unverified, treated as non-commercial. | `docs/project/STREAMPETR_COMPATIBILITY.md`. | Decision |
| 2026-11-05 | **Correction:** headline pretrained-model evaluation uses official mini_val only (scene-0103, scene-0916), both previously seen. An earlier four-scene proposal is withdrawn. | Devkit `splits.py`; user correction. | Correction |
| 2026-11-05 | **Finding:** on mini_val, recall rose from 0.21 to 0.90 for vehicles and from 0.29 to 0.83 for people, and NEES fell to 3.3. Preliminary, two daytime scenes. Occupancy false-free is unchanged. The safety filter remains unvalidated. | `experiments/end_to_end/results/streampetr_mini_val/`. | Hypothesis update |
| TBD | Dataset decision. | Complete Stage 1 data/license/protocol audit. | Open |
| TBD | Stage 2 backbone/representation decision. | Matched geometry, accuracy, memory and runtime experiments. | Open |
| TBD | Stage 3 occupancy/unknown/uncertainty architecture decision. | Stage 0/3 evidence and controlled ablations. | Open |
| TBD | Stage 5 forecasting architecture decision. | Cam4DOcc/OccWorld/Drive-OccWorld and additional prior-art/reproduction review. | Open |
| TBD | Stage 6 event-risk and safety-filter decision. | OccDriver/SURE-Plan and calibrated event-level evaluation. | Open |

**Instruction:** When new evidence changes any item in this document, add a dated entry, cite the evidence, and explain whether the change is a correction, a decision, or a hypothesis update. Never silently rewrite historical results.


---

## 17. Current Claude Code Desktop execution handoff

The user is working in **Claude Code Desktop**, not the terminal CLI. The execution instructions must therefore begin with the Desktop workflow:

1. Open Claude Desktop and select the **Code** tab.
2. Start a new session using the **Local** environment.
3. Select the existing local repository folder: `~/Developer/ForesightBEV`.
4. If a Worktree option is shown, turn it **off** for this task so the session uses the selected local checkout and can see its local datasets, ignored files and existing Python environments. If Desktop requires a worktree, verify what local ignored assets are unavailable and do not assume `.venv`, datasets or checkpoints have been copied.
5. Ensure the user can review and approve file edits and shell commands. Never approve destructive Git/file operations automatically.
6. Read `source.md` and the execution prompt before editing.

### 17.1 Local-first collaboration and GitHub publication

The user clarified the workflow: **Claude Code must finish the work in the selected local ForesightBEV folder first. GitHub is the final publication destination, not the live working environment.** Use Claude Code Desktop's Local session and the selected local folder. Do not edit through GitHub's website or use a cloud workspace that cannot see local assets.

During implementation, do not fetch, pull, push, invoke GitHub API actions, or otherwise contact/modify the remote. Inspect local Git metadata if available. Local commits on a dedicated local branch are permitted, but no remote publication is allowed before local implementation and end-to-end verification are complete. If the selected folder has no `.git`, do not initialize Git or move the work into another clone without explaining the options first.

At the final publication stage only: freeze implementation; run tests and a real end-to-end execution; inspect the full diff; scan for secrets, credentials, `.env`, datasets, checkpoints, archives, virtual environments and large generated binaries; run `git diff --check`; stage explicit task-related paths; verify that the existing `origin` is `https://github.com/sam-3skruthi/ForesightBEV.git`; fetch and inspect remote history; then publish the completed changes to a dedicated feature branch such as `claude/foresightbev-end-to-end-20261105`.

Never push directly to `main`; never force-push, reset user changes, rewrite history, delete branches or change remotes. Keep unrelated local changes out of commits. If a non-fast-forward, divergence, permissions or authentication issue occurs, preserve local work and report the blocker. Verify remote branch and exact commit SHA after a successful final push.

### 17.2 Implementation completion evidence

The user wants a runnable system, not a slide deck. The required evidence is actual execution on available real inputs, output files, inspected visual/numeric results, focused tests, integration results and a single reproducible run command. Distinguish a working prototype from a research-grade autonomous-driving system.

### 17.3 Scope integrity

The ten-stage architecture remains frozen. All ten stages are Claude's responsibility. Do not invent assets or performance results. If unavailable datasets, model weights, CUDA hardware, or evaluation services prevent a complete learned pipeline, implement the strongest valid baseline and document exactly what remains for Phase 2.

### 17.4 End-of-session Git report

Report the repository URL, current branch, commit SHAs, files committed, push status, and link to the pushed branch. Clearly list any pre-existing user changes that were intentionally left uncommitted.
