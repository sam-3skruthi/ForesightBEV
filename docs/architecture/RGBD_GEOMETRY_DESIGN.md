# Metric RGB-D Bird's-Eye-View Geometry Engine Design

**Project:** ForesightBEV Stage 1 Architecture
**Document:** `docs/architecture/RGBD_GEOMETRY_DESIGN.md`
**Status:** Approved Technical Design Specification
**Date:** November 2026

---

## 1. System Motivation & Architectural Distinction

The preceding audit of `Cam2BEV` proved that 2D Inverse Perspective Mapping (IPM) fundamentally collapses under non-flat geometry ($Z \ne 0$). By assuming that all visual rays terminate on a flat planar road, IPM radially smears upright 3D objects (pedestrians, vehicles, walls) into severe streaks over 10–20 meters of ground plane.

ForesightBEV eliminates planar homographies entirely. Instead, ForesightBEV constructs a **robot-centered metric Bird's-Eye-View (BEV) representation** via explicit 3D back-projection of synchronized depth measurements:

$$\text{Raw Depth Map } D(u, v) \xrightarrow[\text{Intrinsics } K]{\text{Back-Projection}} \text{Metric 3D Cloud } p_{\text{cam}} \xrightarrow[\text{Extrinsics } T]{\text{Rigid Transform}} p_{\text{base}} \xrightarrow[\text{Height Banding}]{\text{3D-Aware Raster}} \text{Multi-Layer Metric BEV}$$

---

## 2. Multi-Layer Metric BEV Representation

Rather than compressing environment evidence into an ambiguous single-channel binary image, ForesightBEV defines a **structured multi-layer tensor** $\mathcal{M} \in \mathbb{R}^{H \times W \times C}$ centered on the mobile robot base.

### 2.1 Grid Spatial Parameters
The BEV grid is defined in the robot base frame $\mathcal{F}_{\text{base}}$ ($X$ forward, $Y$ left, $Z$ up):

| Parameter | Symbol | Default Value | Metric Meaning |
|---|---|---|---|
| **Forward Extent** | $[X_{\min}, X_{\max}]$ | $[0.0\text{ m}, 6.0\text{ m}]$ | Covers robot immediate navigation path up to $6\text{ m}$ ahead |
| **Lateral Extent** | $[Y_{\min}, Y_{\max}]$ | $[-3.0\text{ m}, +3.0\text{ m}]$ | Covers $3\text{ m}$ left and $3\text{ m}$ right ($6\text{ m}$ corridor width) |
| **Grid Resolution** | $\Delta_{\text{res}}$ | $0.05\text{ m/cell}$ ($5\text{ cm}$) | Sufficient to resolve fine obstacles (chair legs, curbs) |
| **Grid Width ($W$)** | $W_{\text{grid}}$ | $120\text{ cells}$ | $(Y_{\max} - Y_{\min}) / \Delta_{\text{res}} = 6.0 / 0.05$ |
| **Grid Height ($H$)** | $H_{\text{grid}}$ | $120\text{ cells}$ | $(X_{\max} - X_{\min}) / \Delta_{\text{res}} = 6.0 / 0.05$ |
| **Total Cell Count** | $N_{\text{cells}}$ | $14,400\text{ cells}$ | Ultra-lightweight footprint ($< 1\text{ MB}$ RAM) |

### 2.2 Layer Architecture Specification
The metric BEV tensor incorporates 6 dedicated information channels:

$$\mathcal{M}(r, c) = \begin{bmatrix} L_{\text{occ}}(r, c) & N_{\text{free}}(r, c) & \mathcal{M}_{\text{unknown}}(r, c) & Z_{\max}(r, c) & N_{\text{points}}(r, c) & \sigma_{\text{depth}}^2(r, c) \end{bmatrix}^T$$

1. **Layer 0 — Occupancy Log-Odds ($L_{\text{occ}}$):**
   Continuous occupancy probability represented in log-odds form:
   $$L(r, c) = \ln\left(\frac{P(\text{occupied})}{1 - P(\text{occupied})}\right)$$
   Initialized to $L_0 = 0.0$ ($P = 0.5$, unobserved state).
2. **Layer 1 — Observed Free-Space Clearance ($N_{\text{free}}$):**
   Integer counter recording the number of independent optical rays that have passed through this cell without terminating.
3. **Layer 2 — Explicit Unknown Mask ($\mathcal{M}_{\text{unknown}}$):**
   Binary flag ($\in \{0, 1\}$):
   $$\mathcal{M}_{\text{unknown}}(r, c) = \begin{cases} 1 & \text{if } N_{\text{free}}(r, c) = 0 \land N_{\text{points}}(r, c) = 0 \\ 0 & \text{otherwise (observed)} \end{cases}$$
   **Crucial Rule:** Unknown space is **never** collapsed into free space. Unobserved areas retain maximum epistemic uncertainty.
4. **Layer 3 — Obstacle Elevation / Height ($Z_{\max}$):**
   Maximum base-frame height of physical surface points observed within this cell:
   $$Z_{\max}(r, c) = \max_{p \in \mathcal{P}(r, c)} Z_{\text{base}}(p)$$
   Enables distinguishing low step-over curbs ($Z < 0.15\text{ m}$) from full-height barriers.
5. **Layer 4 — Point Density / Hit Count ($N_{\text{points}}$):**
   Accumulated count of valid depth endpoints rasterized into this cell. Filters single-point sensor noise.
6. **Layer 5 — Measurement Variance / Uncertainty ($\sigma_{\text{depth}}^2$):**
   Quadratic depth uncertainty based on physical triangulation geometry ($\sigma_Z \propto Z^2 / (f \cdot B)$).

---

## 3. Ground Plane vs. Obstacle Separation

A central failure of naive depth projection is marking the traversable floor as an impenetrable obstacle. ForesightBEV enforces rigorous vertical spatial segmentation.

```
       +Z_base
          ^
          |      [Ceiling / Overhead Clutter]          (Z > Z_clearance)  -> DISCARD
          |  ------------------------------------  Z_clearance = 1.80 m
          |
          |      [Navigation-Relevant Obstacles]       (Z_ground_max < Z <= Z_clearance) -> OBSTACLE
          |
          |  ------------------------------------  Z_ground_max = +0.05 m
          |      [Traversable Floor Band]              (Z_ground_min <= Z <= Z_ground_max) -> GROUND
  GROUND  |  ====================================  Z_base = 0.00 m (Ground surface)
          |  ------------------------------------  Z_ground_min = -0.05 m
          |      [Negative Obstacles / Drop-offs]      (Z < Z_ground_min) -> HAZARD
          v
```

### 3.1 Height-Banding Parameters
In the robot base frame where $Z_{\text{base}} = 0$ is calibrated to the wheel contact plane:
1. **Traversable Floor Band:**
   $Z_{\text{base}} \in [-0.05\text{ m}, +0.05\text{ m}]$. Points falling within this band represent the flat floor surface. They are logged as traversable evidence and **do not** trigger occupancy log-odds penalties.
2. **Navigation-Critical Obstacle Band:**
   $Z_{\text{base}} \in (+0.05\text{ m}, +1.80\text{ m}]$. Points in this range correspond to physical objects that obstruct robot travel (pedestrian legs, furniture, walls, carts). These points trigger **Occupancy Endpoint Updates**.
3. **Overhead Clutter Band:**
   $Z_{\text{base}} > +1.80\text{ m}$. Ceiling lights, beams, and overhead hanging signs that clear the robot's physical envelope. These points are filtered out to prevent false path blocking.
4. **Negative Obstacle Band:**
   $Z_{\text{base}} < -0.05\text{ m}$. Missing floor points or downward steps (stairs, pits). Flagged as traversability hazards.

### 3.2 Dynamic Ground Normal Estimation (RANSAC Fallback)
For uneven terrain or robot chassis pitching during acceleration, the ground plane $n^T p + d = 0$ is dynamically estimated from near-field floor points using a lightweight 3-point RANSAC fit ($< 50\text{ iterations}$, runtime $< 1\text{ ms}$). Points are then banded relative to the estimated plane distance:
$$h(p) = \frac{|n^T p + d|}{\|n\|_2}$$

---

## 4. Visibility & Occupancy Update Policy

### 4.1 Epistemic Ray-Casting Principles
For every valid depth measurement $p_{\text{base}}$:
- **Free-Space Ray Segment ($O_{\text{base}} \to p_{\text{base}}$):** The line of sight connecting the sensor center $O_{\text{base}} = t_{\text{base\_cam}}$ to the surface point $p_{\text{base}}$ constitutes confirmed free space.
- **Surface Endpoint ($p_{\text{base}}$):** The discrete location of reflective surface interaction.
- **Occluded Region (Behind $p_{\text{base}}$):** All space along the ray beyond $p_{\text{base}} + \delta_{\text{surf}}$ is completely unobserved. It receives **zero updates** and remains **Unknown**.
- **Missing / Drop-out Depth ($D = 0$):** Rays with zero depth represent non-reflective surfaces or range exceedance. They must **never** be assumed free; they provide zero evidence.

### 4.2 3D-Aware Ray Traversal
A critical flaw in standard 2D grid ray-casting occurs when an elevated camera looks over a low obstacle. If a ray is cast purely in 2D $(x, y)$, a ray passing at height $Z = 1.2\text{ m}$ through the air will clear a low $0.1\text{ m}$ box on the floor beneath it, erasing the obstacle!

ForesightBEV implements **3D-Aware Ray Traversal**:
1. The 3D line segment $L(s) = O_{\text{base}} + s (p_{\text{base}} - O_{\text{base}})$ for $s \in [0, 1)$ is sampled via a 3D Digital Differential Analyzer (3D DDA) or parametric step algorithm.
2. For each intermediate point along the ray, the base height $Z(s)$ is evaluated.
3. A BEV grid cell $(r, c)$ only receives a free-space clearing update if the ray height at that cell falls **within the robot collision volume**:
   $$Z_{\text{clear\_min}} \le Z(s) \le Z_{\text{clear\_max}}$$
   If the optical ray passes above $Z_{\text{clear\_max}}$, it traverses the air without clearing the ground-level cell below.

### 4.3 Log-Odds Bayesian Update Formulation
For each raster cell $(r, c)$:
- **Free-space update:**
  $$L_{t}(r, c) = \max\left(L_{\min}, L_{t-1}(r, c) - l_{\text{free}}\right)$$
  *(Default: $l_{\text{free}} = 0.4$, $L_{\min} = -4.0 \implies P_{\min} \approx 0.018$)*.
- **Obstacle endpoint update:**
  $$L_{t}(r, c) = \min\left(L_{\max}, L_{t-1}(r, c) + l_{\text{occ}}\right)$$
  *(Default: $l_{\text{occ}} = 0.85$, $L_{\max} = +5.0 \implies P_{\max} \approx 0.993$)*.

---

## 5. Architectural Trade-Off Analysis

| Architectural Metric | Height-Banded Multi-Layer BEV (Selected) | Full 3D Voxel Grid (TSDF / OctoMap) |
|---|---|---|
| **Memory Footprint** | **$\sim 350\text{ KB}$** ($120 \times 120 \times 6$ floats) | $\sim 50–200\text{ MB}$ ($120 \times 120 \times 80$ voxels) |
| **Compute Overhead** | **$< 2\text{ ms}$ per frame** (Single M5 CPU thread) | $20–60\text{ ms}$ (Requires GPU acceleration) |
| **Downstream Usability** | Directly ingestion-ready for 2D Nav2 / CNNs | Requires 2D projection or 3D convolutions |
| **Height Preservation** | Preserved via $Z_{\max}$ and floor-band layers | Fully discrete volumetric cells |
| **Overhang Modeling** | Captures under-table / overhang clearances | Fully volumetric |
| **Recommendation** | **PRIMARY SELECTION FOR STAGE 1** | Future extension if multi-deck overhangs required |

**Design Decision:** The **Height-Banded Multi-Layer BEV** is officially selected. It provides the optimal balance of metric fidelity, explicit unknown/free separation, sub-millisecond execution on Apple Silicon, and zero-copy integration with standard navigation stacks.
