# Stage 1 Design Review Addendum: Geometry Specifications & Methodological Corrections

**Project:** ForesightBEV Stage 1 Engineering
**Document:** `research/experiments/STAGE_1_DESIGN_REVIEW_ADDENDUM.md`
**Status:** Approved Design Review Specification
**Date:** November 2026
**Auditor / Engineer Mode:** Implementation Design Review & Metric Specifications

---

## 1. Coordinate Frame Origins & Ground-Aligned Reference (Item 2.1)

### 1.1 `base_link` vs. Ground-Aligned Frame (`base_footprint` / `base_ground`)
In ROS REP-103, body coordinate axes are defined as:
- $+X$: Forward
- $+Y$: Left
- $+Z$: Up

However, REP-103 defines axis orientation and right-handedness, but **does not specify the vertical location of the `base_link` origin**. In robotics platforms:
- `base_link` origin is typically located at the vehicle kinematic center of rotation, often elevated at a chassis reference height $h_{\text{chassis}} > 0$ above the supporting ground plane.
- Setting $Z_{\text{base\_link}} = 0$ does **not** physically correspond to the ground surface unless explicitly configured.

**ForesightBEV Ground-Aligned Convention:**
ForesightBEV formally designates the **`base_footprint`** frame (or project-level synonym **`base_ground`**) as its canonical ground-level reference frame:
- **Origin:** The vertical projection of the robot center of rotation onto the horizontal ground contact plane ($Z_{\text{base\_footprint}} = 0.00\text{ m}$ is the floor surface).
- **Relationship to `base_link`:**
  Assuming zero roll and pitch of the chassis on flat ground:
  $$T_{\text{base\_footprint\_base\_link}} = \begin{bmatrix} I_{3 \times 3} & \begin{bmatrix} 0 \\ 0 \\ -h_{\text{chassis}} \end{bmatrix} \\ 0_{1 \times 3} & 1 \end{bmatrix}$$
  $$p_{\text{base\_footprint}} = p_{\text{base\_link}} + \begin{bmatrix} 0 \\ 0 \\ h_{\text{chassis}} \end{bmatrix}$$
- **Camera Extrinsics:** Camera mounting translations must be expressed relative to this ground-aligned origin $t_{\text{base\_footprint\_cam}} = [x_{\text{mount}}, y_{\text{mount}}, z_{\text{mount}}]^T$, where $z_{\text{mount}}$ is the physical optical center height above the floor plane.

### 1.2 Coordinate System Summary

```text
CAMERA OPTICAL FRAME (OpenCV)           GROUND-ALIGNED BASE FRAME (base_footprint / REP-103)
      +Z_cam (Forward/Optical)                         +Z_ground (Up / Normal to floor)
            /                                                 ^
           /                                                  |
          /                                                   |
         +--------> +X_cam (Right)                            +--------> +X_ground (Forward)
         |                                                   /
         |                                                  /
         v                                                 v
    +Y_cam (Down)                                     +Y_ground (Left)
```

### 1.3 Canonical Optical-to-Ground Basis Conversion ($R_{\text{base\_cam0}}$)
When the camera has zero mounting tilt (optical axis points parallel to floor along $+X_{\text{ground}}$):
- $+X_{\text{ground}}$ (Forward) $\leftarrow +Z_{\text{cam}}$
- $+Y_{\text{ground}}$ (Left) $\leftarrow -X_{\text{cam}}$
- $+Z_{\text{ground}}$ (Up) $\leftarrow -Y_{\text{cam}}$

$$R_{\text{base\_cam0}} = \begin{bmatrix} 0 & 0 & 1 \\ -1 & 0 & 0 \\ 0 & -1 & 0 \end{bmatrix}$$

**Determinant & Orthogonality Verification:**
$$\det(R_{\text{base\_cam0}}) = 0(0) - 0(0) + 1((-1)(-1) - 0) = +1.0000000$$
$$R_{\text{base\_cam0}}^T R_{\text{base\_cam0}} = \begin{bmatrix} 0 & -1 & 0 \\ 0 & 0 & -1 \\ 1 & 0 & 0 \end{bmatrix} \begin{bmatrix} 0 & 0 & 1 \\ -1 & 0 & 0 \\ 0 & -1 & 0 \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} = I_{3 \times 3}$$
This confirms that $R_{\text{base\_cam0}}$ is a proper orthogonal rotation in $\text{SO}(3)$.

### 1.4 Strict Transform Direction Notation: $T_{\text{target\_source}}$
To avoid ambiguity, all rigid transformations follow the notation:
$$p_{\text{target}} = T_{\text{target\_source}} \cdot p_{\text{source}}$$
where:
$$T_{\text{target\_source}} = \begin{bmatrix} R_{\text{target\_source}} & t_{\text{target\_source}} \\ 0_{1 \times 3} & 1 \end{bmatrix}$$
- $p_{\text{source}} = [X_s, Y_s, Z_s, 1]^T$ is a point in the source frame.
- $p_{\text{target}} = [X_t, Y_t, Z_t, 1]^T$ is the identical physical point expressed in target frame coordinates.
- Composition rule: $T_{\text{c\_a}} = T_{\text{c\_b}} \cdot T_{\text{b\_a}}$.
- Inverse rule: $T_{\text{source\_target}} = T_{\text{target\_source}}^{-1} = \begin{bmatrix} R^T & -R^T t \\ 0 & 1 \end{bmatrix}$.

---

## 2. Metric BEV Grid Boundary Conventions (Item 2.2)

### 2.1 Forward-Facing Local Grid Dimensions
The proposed $120 \times 120$ grid is explicitly a **forward-facing local sensor grid**, covering:
- Forward ($X_{\text{base}}$): $[0.0, 6.0)\text{ meters}$
- Lateral ($Y_{\text{base}}$): $[-3.0, 3.0)\text{ meters}$
- Resolution ($\Delta$): $0.05\text{ meters/cell}$ ($5\text{ cm}$)

### 2.2 Half-Open Interval Definition & Discretization
To prevent off-by-one errors and array indexing violations:
- The spatial domain is strictly **half-open**: $[X_{\min}, X_{\max}) \times [Y_{\min}, Y_{\max})$.
- A physical point $(X, Y)$ is valid for rasterization if and only if:
  $$X_{\min} \le X < X_{\max} \quad \text{and} \quad Y_{\min} \le Y < Y_{\max}$$
- Points with $X = X_{\max} = 6.0\text{ m}$ or $Y = Y_{\max} = 3.0\text{ m}$ fall **strictly outside** the grid and must be rejected or masked as out-of-bounds. They must **never** evaluate to index $120$.

### 2.3 Row and Column Indexing Convention
In the raster matrix:
- **Row Index ($r \in [0, 119]$):** Represents forward distance $X$. We define row $0$ at $X_{\max}$ (top of image / furthest ahead) and row $119$ near the robot origin:
  $$r = \left\lfloor \frac{X_{\max} - X}{\Delta} \right\rfloor = \left\lfloor \frac{6.0 - X}{0.05} \right\rfloor$$
  *(For $X \in [5.95, 6.0)$, $r = 0$; for $X \in [0.0, 0.05)$, $r = 119$)*.
- **Column Index ($c \in [0, 119]$):** Represents lateral distance $Y$, from left to right:
  $$c = \left\lfloor \frac{Y - Y_{\min}}{\Delta} \right\rfloor = \left\lfloor \frac{Y - (-3.0)}{0.05} \right\rfloor$$
  *(For $Y \in [-3.0, -2.95)$, $c = 0$; for $Y \in [2.95, 3.0)$, $c = 119$)*.

### 2.4 Metric Cell-Center Coordinates
For any cell $(r, c)$, its geometric center in base metric coordinates $(X_c, Y_c)$ is:
$$X_c(r) = X_{\max} - \left(r + 0.5\right) \Delta = 6.0 - (r + 0.5) \times 0.05$$
$$Y_c(c) = Y_{\min} + \left(c + 0.5\right) \Delta = -3.0 + (c + 0.5) \times 0.05$$

---

## 3. Visibility, Ray-Casting, & Volumetric Shadow Model (Item 2.3)

### 3.1 3D Volumetric Wall Model for Synthetic Verification
In Experiment 1.4, a synthetic obstacle cannot be represented as an infinitely tall 2D line. It must possess a well-defined 3D physical volume:
- Forward position: $X \in [3.00\text{ m}, 3.10\text{ m}]$ (wall thickness $\Delta X = 10\text{ cm}$).
- Lateral span: $Y \in [-1.00\text{ m}, +1.00\text{ m}]$ (wall width $\Delta Y = 2.0\text{ m}$).
- Vertical extent: $Z_{\text{base}} \in [0.00\text{ m}, 2.00\text{ m}]$ (covers full robot collision height).

### 3.2 Angular Extent of the Shadow Wedge
The unobserved shadow wedge behind the wall must be restricted strictly to optical rays intersecting the wall's physical lateral angle:
$$\theta_{\text{wall\_min}} = \arctan2(-1.0, 3.0) \approx -18.43^\circ$$
$$\theta_{\text{wall\_max}} = \arctan2(+1.0, 3.0) \approx +18.43^\circ$$
Cells at $X > 3.10\text{ m}$ whose bearing $|\theta| > 18.43^\circ$ (to the left and right of the wall) receive direct line of sight from the camera origin and must **not** be marked unknown if an obstacle or ground ray reaches them.

### 3.3 Log-Odds Calibration & Probability Mapping
- Prior initial log-odds: $L_0 = 0.0 \implies P_0 = \frac{1}{1 + e^{-0}} = 0.50$ (Maximum uncertainty).
- Free-space decrement: $l_{\text{free}} = 0.40$.
- Obstacle hit increment: $l_{\text{occ}} = 0.85$.
- Clamping limits: $L_{\min} = -4.0 \implies P_{\min} \approx 0.018$; $L_{\max} = +5.0 \implies P_{\max} \approx 0.993$.
- Mapping function:
  $$P(\text{occupied}) = \frac{1}{1 + \exp(-L)}$$

---

## 4. Ground-Filter Statistical Noise Thresholds (Item 2.4)

### 4.1 Statistical Distribution Under Gaussian Sensor Noise
In Experiment 1.5, floor points on $Z_{\text{base}} = 0.00\text{ m}$ are perturbed with zero-mean Gaussian noise $\mathcal{N}(0, \sigma^2)$ where $\sigma = 0.015\text{ m}$ ($15\text{ mm}$ depth noise, realistic for RealSense at $2\text{ m}$).

The ground classification acceptance band is $|Z_{\text{base}}| \le 0.05\text{ m}$ ($\pm 50\text{ mm}$).
The normalized z-score threshold is:
$$z = \frac{0.05}{0.015} = \frac{10}{3} \approx 3.3333$$

The theoretical probability of a true ground point falling within this band is:
$$P(|Z| \le 0.05) = \text{erf}\left(\frac{3.3333}{\sqrt{2}}\right) \approx 0.999142 \implies 99.914\%$$
The theoretical False Positive Rate (FPR) of floor points classified as obstacles is:
$$\text{FPR}_{\text{theoretical}} = 2 \times \left(1 - \Phi(3.3333)\right) \approx 0.000858 \implies \mathbf{0.086\%}$$

### 4.2 Acceptance Criteria for Experiment 1.5
To avoid fragile unit tests:
- A fixed pseudorandom seed (`seed = 42`) is mandated for synthetic point generation ($N = 10,000$ points).
- **Statistically Defensible Threshold:**
  $$\text{Measured FPR} \le 0.20\% \quad (\text{Recall} \ge 99.80\%)$$
  A strict requirement of $100.00\%$ is mathematically rejected as incompatible with continuous Gaussian noise.

---

## 5. Scope of TUM RGB-D Benchmark (Item 2.5)

### 5.1 Smoke-Test Boundary
The TUM RGB-D dataset (`rgbd_dataset_freiburg1_xyz`) is officially bounded as:
> **An analytical geometry, un-projection, and coordinate transformation smoke-test benchmark.**
It evaluates the numerical accuracy of 3D back-projection against calibrated optical motion-capture ground truth. It does **not** evaluate indoor autonomous navigation, dynamic pedestrian avoidance, or human-robot interaction.

### 5.2 Depth Encoding & Calibration Semantics
- **Depth Scale:** Raw 16-bit integer $D(u, v) \in [0, 65535]$. Metric depth in meters is:
  $$Z = \frac{D(u, v)}{5000.0} \quad (\text{Depth scale: } 5000\text{ units/m}, \text{ or } 0.0002\text{ m/unit})$$
- **RGB-Depth Optical Alignment:** In TUM `freiburg1`, the depth camera and color camera have slightly different intrinsic matrices and an optical baseline offset ($\sim 2.5\text{ cm}$). When evaluating depth back-projection, depth image pixels must be un-projected using the **depth camera intrinsics** ($f_x = 591.1, f_y = 590.1, c_x = 331.0, c_y = 234.0$ for raw depth), not the RGB camera intrinsics, unless working with pre-registered/aligned depth pairs.
