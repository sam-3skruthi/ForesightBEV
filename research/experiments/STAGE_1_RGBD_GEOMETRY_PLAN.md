# ForesightBEV Stage 1: RGB-D Metric Geometry Engineering Plan

**Project:** ForesightBEV Research & Implementation
**Document:** `research/experiments/STAGE_1_RGBD_GEOMETRY_PLAN.md`
**Status:** Approved Experimental & Implementation Specification
**Date:** November 2026
**Target Environment:** Isolated Python 3.12 (`uv`) on Apple Silicon M5

---

## 1. Subsystem Scope & Engineering Objectives

Stage 1 establishes the core geometric perception foundation of ForesightBEV. It implements a standalone, highly modular Python package that ingests raw synchronized RGB-D frames, un-projects them into metric 3D point clouds, applies extrinsic vehicle mounting transformations, and rasterizes them into a multi-layer robot-centered Bird's-Eye-View (BEV) grid.

### Critical Engineering Boundaries
1. **Zero Homography Dependency:** Completely independent of Cam2BEV planar homography assumptions ($Z=0$).
2. **Explicit Epistemic Uncertainty:** Free space, obstacle endpoints, and unobserved/occluded space must remain strictly decoupled.
3. **Hardware Independence:** Validatable on synthetic analytical benchmarks and public datasets (TUM RGB-D) prior to physical RealSense D435i deployment.
4. **Modular Package Design:** Packaged under `src/foresightbev/geometry/` without dependencies on ROS 2, neural network weights, or GUI dashboards.

---

## 2. Proposed Implementation Architecture (`src/foresightbev/geometry/`)

```text
src/foresightbev/
├── __init__.py
└── geometry/
    ├── __init__.py
    ├── camera_model.py          # Pinhole intrinsics, optical center, focal lengths, de-quantization
    ├── transforms.py            # SE(3) rigid body transforms, coordinate frame conventions (REP-103 vs OpenCV)
    ├── backprojection.py        # Vectorized depth de-projection to metric 3D point cloud
    ├── ground_filter.py         # Ground plane detection, height-banding, obstacle segmentation
    ├── raycasting.py            # 3D-aware DDA ray marching for free-space and visibility updates
    └── bev_grid.py              # Robot-centered multi-layer metric grid data structure
```

---

## 3. Detailed Experimental Protocol (Experiments 1.1 – 1.5)

### Experiment 1.1 — Pinhole Back-Projection Verification

- **Objective:** Mathematically verify that 2D image coordinates $(u, v)$ with known depth $d$ back-project to exact analytical 3D coordinates $p_{\text{cam}} = [X_c, Y_c, Z_c]^T$ in the camera optical frame.
- **Mathematical Formulation:**
  $$Z_c = d / s_{\text{depth}}, \quad X_c = \frac{u - c_x}{f_x} Z_c, \quad Y_c = \frac{v - c_y}{f_y} Z_c$$
- **Input Data:**
  - Analytical synthetic grid points:
    1. Optical center $(c_x, c_y)$ with depth $d = 2000\text{ mm}$ ($s_{\text{depth}} = 1000$).
    2. Focal corner $(c_x + f_x, c_y + f_y)$ with depth $d = 1000\text{ mm}$.
    3. Opposing corner $(c_x - f_x, c_y - f_y)$ with depth $d = 3000\text{ mm}$.
  - Camera Intrinsics: $f_x = 500.0, f_y = 500.0, c_x = 320.0, c_y = 240.0$.
- **Expected Analytical Output:**
  1. Optical center: $p_1 = [0.0\text{ m}, 0.0\text{ m}, 2.0\text{ m}]^T$
  2. Focal corner: $p_2 = [1.0\text{ m}, 1.0\text{ m}, 1.0\text{ m}]^T$
  3. Opposing corner: $p_3 = [-3.0\text{ m}, -3.0\text{ m}, 3.0\text{ m}]^T$
- **Numerical Validation Tests:**
  $$\|p_{\text{computed}} - p_{\text{expected}}\|_{\infty} < 10^{-7}\text{ m}$$
- **Failure Conditions:** Any non-zero lateral component at the principal point; inverted $Y$ direction; scaling discrepancy.
- **Scope Limitation:** Establishes mathematical correctness of un-projection; does not test lens distortion or real sensor noise.

---

### Experiment 1.2 — Rigid Coordinate Frame Transformation ($T_{\text{base\_cam}}$)

- **Objective:** Verify coordinate transformation from OpenCV camera frame ($\mathcal{F}_{\text{cam}}$: right, down, forward) to ROS REP-103 robot base frame ($\mathcal{F}_{\text{base}}$: forward, left, up).
- **Mathematical Formulation:**
  $$p_{\text{base}} = R_{\text{base\_cam}} p_{\text{cam}} + t_{\text{base\_cam}}$$
  where $R_{\text{base\_cam}} = R_{\text{mount}}(\psi, \theta, \phi) \cdot R_{\text{base\_cam0}}$.
- **Input Data:**
  - Test Point in Camera Frame: $p_{\text{cam}} = [0.5\text{ m}, 0.2\text{ m}, 2.0\text{ m}]^T$ (Right, Down, Forward).
  - Mounting Extrinsics: Mount position $t = [0.2\text{ m}, 0.0\text{ m}, 0.4\text{ m}]^T$ (front of robot, centered, $0.4\text{ m}$ height), zero tilt.
- **Expected Analytical Output:**
  $$X_{\text{base}} = Z_{\text{cam}} + t_x = 2.0 + 0.2 = 2.2\text{ m}$$
  $$Y_{\text{base}} = -X_{\text{cam}} + t_y = -0.5 + 0.0 = -0.5\text{ m}$$
  $$Z_{\text{base}} = -Y_{\text{cam}} + t_z = -0.2 + 0.4 = 0.2\text{ m}$$
  $$p_{\text{base}} = [2.2\text{ m}, -0.5\text{ m}, 0.2\text{ m}]^T$$
- **Numerical Validation Tests:**
  $$\|p_{\text{base, computed}} - [2.2, -0.5, 0.2]^T\|_2 < 10^{-7}\text{ m}$$
  $$\det(R_{\text{base\_cam}}) = 1.0 \pm 10^{-7}, \quad \|R^T R - I\|_F < 10^{-7}$$
- **Failure Conditions:** Inverted handedness ($\det(R) = -1$); axes misalignment; non-orthogonal rotation.
- **Scope Limitation:** Establishes spatial mapping between frames; does not validate physical CAD mounting calibration.

---

### Experiment 1.3 — Metric BEV Rasterization & Grid Metric Scaling

- **Objective:** Rasterize a 3D point cloud into robot-centered BEV grid cells and verify that metric distances and cell indices map bijectively without metric distortion.
- **Mathematical Formulation:**
  For grid resolution $\Delta_{\text{res}} = 0.05\text{ m}$, origin $(X_{\min}=0, X_{\max}=6.0, Y_{\min}=-3.0, Y_{\max}=3.0)$:
  $$r = \left\lfloor \frac{X_{\max} - X_{\text{base}}}{\Delta_{\text{res}}} \right\rfloor, \quad c = \left\lfloor \frac{Y_{\text{base}} - Y_{\min}}{\Delta_{\text{res}}} \right\rfloor$$
- **Input Data:**
  Synthetic vertical cylinder obstacle of radius $R = 0.20\text{ m}$ ($40\text{ cm}$ diameter) centered at $(X=3.0\text{ m}, Y=0.0\text{ m})$.
- **Expected Analytical Output:**
  - Expected Center Cell:
    $$r_{\text{center}} = \lfloor (6.0 - 3.0) / 0.05 \rfloor = 60$$
    $$c_{\text{center}} = \lfloor (0.0 - (-3.0)) / 0.05 \rfloor = 60$$
  - Obstacle Cell Diameter:
    $$\text{Diameter} = 2 R / \Delta_{\text{res}} = 0.40 / 0.05 = 8\text{ cells}$$
- **Numerical Validation Tests:**
  - Centroid error: $\|\bar{r} - 60\| \le 0.5\text{ cells}, \|\bar{c} - 60\| \le 0.5\text{ cells}$.
  - Cell area: Count of occupied cells inside cylinder $\approx \pi (4)^2 \approx 50 \pm 2\text{ cells}$.
- **Failure Conditions:** Anisotropic scaling (elliptical rasterization of circular obstacle); off-by-one cell indexing; bounding box clamping failure.
- **Scope Limitation:** Validates 2D planar quantization; does not address elevation aggregation.

---

### Experiment 1.4 — Visibility, Unknown Space, & Free-Space Ray-Casting

- **Objective:** Verify that an obstacle generates an observed free-space cone in front of it and leaves an explicit, unobserved shadow wedge behind it without collapsing unknown space into free space.
- **Mathematical Formulation:**
  For optical ray $L(s) = O_{\text{base}} + s (p_{\text{base}} - O_{\text{base}})$:
  - Cells intersected for $s \in [0, 1 - \epsilon]$ increment $N_{\text{free}}$ and decrement $L_{\text{occ}}$.
  - Cell containing $s = 1$ increments $N_{\text{points}}$ and increments $L_{\text{occ}}$.
  - Cells along ray for $s > 1$ receive zero updates and retain $\mathcal{M}_{\text{unknown}} = 1$.
- **Input Data:**
  Camera at origin $O = [0, 0, 0.4]^T$. Solid wall segment placed at $X = 3.0\text{ m}$, spanning $Y \in [-1.0\text{ m}, 1.0\text{ m}]$. Space behind wall is empty.
- **Expected Output:**
  1. $X \in [0.0, 2.95\text{ m}]$: High free-space count ($N_{\text{free}} \ge 1, \mathcal{M}_{\text{unknown}} = 0, P_{\text{occ}} < 0.1$).
  2. $X \in [2.95, 3.05\text{ m}]$: Occupied barrier ($N_{\text{points}} \ge 1, P_{\text{occ}} > 0.9$).
  3. $X \in [3.05, 6.0\text{ m}]$ behind wall: Explicit Unknown ($\mathcal{M}_{\text{unknown}} = 1, N_{\text{free}} = 0, N_{\text{points}} = 0, P_{\text{occ}} = 0.5$).
- **Numerical Validation Tests:**
  - $0$ cells behind the wall classified as free.
  - $0$ cells in front of the wall classified as unknown.
- **Failure Conditions:** Rays passing through the wall; shadow wedge erroneously cleared as free space.
- **Scope Limitation:** Evaluates single-frame geometry; does not evaluate multi-view temporal consensus.

---

### Experiment 1.5 — Ground Plane vs. Obstacle Vertical Banding

- **Objective:** Verify that traversable floor returns are recognized as ground surfaces and do not inject spurious obstacle occupancy into the BEV path.
- **Mathematical Formulation:**
  For base height $Z_{\text{base}}$:
  $$\text{Classification}(p) = \begin{cases} \text{GROUND} & \text{if } |Z_{\text{base}}| \le 0.05\text{ m} \\ \text{OBSTACLE} & \text{if } 0.05\text{ m} < Z_{\text{base}} \le 1.80\text{ m} \\ \text{OVERHEAD} & \text{if } Z_{\text{base}} > 1.80\text{ m} \end{cases}$$
- **Input Data:**
  Simulated synthetic floor patch ($Z_{\text{base}} = 0.0\text{ m}$) with Gaussian height noise ($\sigma = 0.015\text{ m}$) and a raised step obstacle ($Z_{\text{base}} = 0.25\text{ m}$).
- **Expected Output:**
  - Floor patch: $100\%$ classified as `GROUND` (occupancy log-odds unaffected, traversability layer marked true).
  - Step obstacle: $100\%$ classified as `OBSTACLE` (occupancy log-odds incremented).
- **Numerical Validation Tests:**
  - False positive obstacle rate on floor: $\text{FPR} < 0.1\%$.
  - Obstacle detection recall: $100\%$.
- **Failure Conditions:** Floor returns rasterized as obstacles; low obstacles misclassified as floor.
- **Scope Limitation:** Assumes flat floor plane; evaluated without extreme floor pitching or ramps.

---

## 4. Acceptance Criteria & Implementation Readiness Checklist

Stage 1 implementation will be authorized only after this experimental plan and the companion architecture specifications are reviewed.

- [x] Cam2BEV reproduction review addendum finalized.
- [x] RGB-D candidate datasets evaluated and TUM RGB-D selected for primary unit tests.
- [x] Coordinate systems and transformation conventions explicitly documented.
- [x] Multi-layer metric BEV grid designed with explicit unknown-space preservation.
- [x] 3D-aware ray-casting visibility policy defined.
- [x] Five concrete engineering experiments specified with numerical validation thresholds.
- [ ] Technical design review and user authorization to begin Stage 1 module coding.
