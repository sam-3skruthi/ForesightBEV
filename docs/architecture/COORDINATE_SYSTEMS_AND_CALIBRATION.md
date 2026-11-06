# Coordinate Systems, Calibration, and Metric Geometry

**Project:** Tavyra Stage 1 Architecture
**Document:** `docs/architecture/COORDINATE_SYSTEMS_AND_CALIBRATION.md`
**Status:** Approved Technical Specification
**Date:** November 2026

---

## 1. Mathematical Coordinate Frame Definitions

Tavyra strictly adheres to standard robotics and computer vision spatial conventions. To prevent silent coordinate contamination, all transformations must explicitly designate source and destination frames.

```
       [ROBOT BASE FRAME: REP-103]                [CAMERA OPTICAL FRAME: OpenCV]
                 +Z_base (Up)                              +Z_cam (Forward/Optical)
                    ^                                             /
                    |                                            /
                    |                                           /
                    +--------> +X_base (Forward)               +--------> +X_cam (Right)
                   /                                           |
                  /                                            |
                 v                                             v
            +Y_base (Left)                                +Y_cam (Down)
```

### 1.1 Camera Optical Frame ($\mathcal{F}_{\text{cam}}$) — OpenCV Convention
- **Origin:** Optical focal center of the camera pinhole.
- **$+X_{\text{cam}}$ Axis:** Points horizontally to the **right** in the image plane.
- **$+Y_{\text{cam}}$ Axis:** Points vertically **downwards** in the image plane.
- **$+Z_{\text{cam}}$ Axis:** Points **forward** along the optical line of sight (depth axis).
- **Handedness:** Right-handed: $\hat{X}_{\text{cam}} \times \hat{Y}_{\text{cam}} = \hat{Z}_{\text{cam}}$.

### 1.2 Robot Base Frame ($\mathcal{F}_{\text{base}}$) — ROS REP-103 Standard
- **Origin:** Geometric footprint projection on the ground surface directly beneath the vehicle/robot center of rotation.
- **$+X_{\text{base}}$ Axis:** Points horizontally **forward** along the nominal driving direction.
- **$+Y_{\text{base}}$ Axis:** Points horizontally to the **left** of the robot.
- **$+Z_{\text{base}}$ Axis:** Points vertically **upward** opposing gravity.
- **Handedness:** Right-handed: $\hat{X}_{\text{base}} \times \hat{Y}_{\text{base}} = \hat{Z}_{\text{base}}$.

### 1.3 Metric Bird's-Eye-View Grid Frame ($\mathcal{F}_{\text{bev}}$)
- **Origin:** Grid pixel $(r_0, c_0)$ corresponding to the projected robot base origin $(0, 0)$ on the horizontal plane.
- **Row Axis ($r$):** Corresponds to $+X_{\text{base}}$ (forward travel) mapped downwards or upwards by convention. In matrix row index convention:
  $$r = \left\lfloor \frac{X_{\max} - X_{\text{base}}}{\Delta_{\text{res}}} \right\rfloor$$
- **Column Axis ($c$):** Corresponds to $-Y_{\text{base}}$ (lateral span):
  $$c = \left\lfloor \frac{Y_{\text{base}} - Y_{\min}}{\Delta_{\text{res}}} \right\rfloor$$
- **Resolution ($\Delta_{\text{res}}$):** Cell grid spacing in meters per pixel (default: $0.05\text{ m/cell}$).

---

## 2. Rigid Body Spatial Transformation ($T_{\text{base\_cam}}$)

### 2.1 Transformation Formulation
Any 3D point measured in camera coordinates $p_{\text{cam}} = [X_{\text{cam}}, Y_{\text{cam}}, Z_{\text{cam}}]^T$ is mapped into the robot base frame $p_{\text{base}} = [X_{\text{base}}, Y_{\text{base}}, Z_{\text{base}}]^T$ via the homogeneous affine transform:

$$\begin{bmatrix} X_{\text{base}} \\ Y_{\text{base}} \\ Z_{\text{base}} \\ 1 \end{bmatrix} = T_{\text{base\_cam}} \begin{bmatrix} X_{\text{cam}} \\ Y_{\text{cam}} \\ Z_{\text{cam}} \\ 1 \end{bmatrix} = \begin{bmatrix} R_{\text{base\_cam}} & t_{\text{base\_cam}} \\ 0_{1 \times 3} & 1 \end{bmatrix} \begin{bmatrix} X_{\text{cam}} \\ Y_{\text{cam}} \\ Z_{\text{cam}} \\ 1 \end{bmatrix}$$

where:
- $R_{\text{base\_cam}} \in \text{SO}(3)$ is a $3 \times 3$ orthonormal rotation matrix ($\det(R) = +1, R^T R = I$).
- $t_{\text{base\_cam}} = [t_x, t_y, t_z]^T \in \mathbb{R}^3$ is the translation vector specifying the position of the camera optical center in meters relative to the robot base origin.

### 2.2 Canonical Axis Permutation Matrix ($R_{\text{base\_cam0}}$)
Consider an ideal front-facing camera with zero mounting tilt mounted at the robot origin. The mapping between axes is:
- $+X_{\text{base}}$ (Forward) $\leftarrow +Z_{\text{cam}}$ (Optical depth)
- $+Y_{\text{base}}$ (Left) $\leftarrow -X_{\text{cam}}$ (Opposite of camera right)
- $+Z_{\text{base}}$ (Up) $\leftarrow -Y_{\text{cam}}$ (Opposite of camera down)

This defines the canonical base rotation matrix $R_{\text{base\_cam0}}$:

$$R_{\text{base\_cam0}} = \begin{bmatrix} 0 & 0 & 1 \\ -1 & 0 & 0 \\ 0 & -1 & 0 \end{bmatrix}$$

**Mathematical Verification:**
$$\det(R_{\text{base\_cam0}}) = 0(0) - 0(0) + 1((-1)(-1) - 0) = +1 \quad (\text{Proper rotation in SO(3)})$$
$$R_{\text{base\_cam0}} \begin{bmatrix} X_{\text{cam}} \\ Y_{\text{cam}} \\ Z_{\text{cam}} \end{bmatrix} = \begin{bmatrix} Z_{\text{cam}} \\ -X_{\text{cam}} \\ -Y_{\text{cam}} \end{bmatrix} = \begin{bmatrix} X_{\text{base}} \\ Y_{\text{base}} \\ Z_{\text{base}} \end{bmatrix}$$

### 2.3 General Mounting Orientation (Extrinsics)
If the sensor mount has extrinsic Euler rotation (yaw $\psi$, pitch $\theta$, roll $\phi$) or quaternion $q_{\text{mount}}$ relative to the base frame, the compound rotation is:

$$R_{\text{base\_cam}} = R_{\text{mount}}(\psi, \theta, \phi) \cdot R_{\text{base\_cam0}}$$

For pitch down by angle $\theta_{\text{pitch}}$ (common for mobile robots to observe near-ground obstacles):
$$R_{\text{pitch}}(\theta) = \begin{bmatrix} \cos\theta & 0 & \sin\theta \\ 0 & 1 & 0 \\ -\sin\theta & 0 & \cos\theta \end{bmatrix}$$

---

## 3. Pinhole Back-Projection Formulation

### 3.1 Pinhole Camera Model
Given the intrinsic camera calibration matrix $K$:

$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

where:
- $f_x, f_y$ are horizontal and vertical focal lengths in pixel units.
- $c_x, c_y$ are principal point coordinates (optical center) in image pixel coordinates.

### 3.2 Depth Unpacking
Raw depth sensors (Intel RealSense, Kinect) record depth as an unsigned 16-bit integer $D(u, v) \in [0, 65535]$. The continuous metric depth $Z_{\text{cam}}(u, v)$ in meters is computed using the sensor-specific depth scale factor $s_{\text{depth}}$ (in units of counts per meter):

$$Z_{\text{cam}}(u, v) = \frac{D(u, v)}{s_{\text{depth}}}$$

- **Intel RealSense D435i Default:** $s_{\text{depth}} = 1000.0\text{ counts/m}$ ($1\text{ count} = 1\text{ mm}$).
- **TUM RGB-D Benchmark:** $s_{\text{depth}} = 5000.0\text{ counts/m}$ ($1\text{ count} = 0.2\text{ mm}$).

### 3.3 Metric Back-Projection Equations
For each valid pixel $(u, v)$ with metric depth $Z_{\text{cam}} > 0$:

$$X_{\text{cam}} = \frac{u - c_x}{f_x} \cdot Z_{\text{cam}}$$

$$Y_{\text{cam}} = \frac{v - c_y}{f_y} \cdot Z_{\text{cam}}$$

$$Z_{\text{cam}} = Z_{\text{cam}}$$

In compact vectorized matrix form:

$$p_{\text{cam}}(u, v) = Z_{\text{cam}}(u, v) \cdot K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$

where:

$$K^{-1} = \begin{bmatrix} \frac{1}{f_x} & 0 & -\frac{c_x}{f_x} \\ 0 & \frac{1}{f_y} & -\frac{c_y}{f_y} \\ 0 & 0 & 1 \end{bmatrix}$$

---

## 4. Validity Filtering & Range Thresholds

Raw depth images exhibit missing measurements, multipath noise, specular reflection dropouts, and peripheral edge distortion. Back-projection must apply strict filtering before spatial ingestion:

1. **Zero and Non-Finite Filter:**
   Measurements where $D(u, v) = 0$, $\text{NaN}$, or $\infty$ represent unobserved rays. These must be segregated into an explicit **Unknown** state.
2. **Radial / Range Clipping:**
   Measurements outside the calibrated sensor operating bounds are discarded:
   $$Z_{\min} \le Z_{\text{cam}} \le Z_{\max}$$
   *(Default for RealSense D435i: $Z_{\min} = 0.25\text{ m}, Z_{\max} = 5.0\text{ m}$)*.
3. **Radial Distance Check:**
   $$\|p_{\text{cam}}\|_2 = \sqrt{X_{\text{cam}}^2 + Y_{\text{cam}}^2 + Z_{\text{cam}}^2} \le d_{\max}$$
4. **Boundary Clipping:**
   Exclude outer perimeter pixels (e.g., margin of 4 pixels) where lens distortion correction introduces interpolation artifacts.

---

## 5. End-to-End Coordinate Transformation Chain

The complete transformation from raw sensor pixel $(u, v, D)$ to 3D base coordinates and 2D BEV grid cell $(r, c)$ proceeds deterministically:

$$\begin{aligned}
(u, v, D) &\xrightarrow[\text{De-quantize}]{s_{\text{depth}}} Z_{\text{cam}} \\
(u, v, Z_{\text{cam}}) &\xrightarrow[\text{Back-Project}]{K^{-1}} p_{\text{cam}} = [X_{\text{cam}}, Y_{\text{cam}}, Z_{\text{cam}}]^T \\
p_{\text{cam}} &\xrightarrow[\text{Extrinsics}]{T_{\text{base\_cam}}} p_{\text{base}} = [X_{\text{base}}, Y_{\text{base}}, Z_{\text{base}}]^T \\
p_{\text{base}} &\xrightarrow[\text{Rasterize}]{\Delta_{\text{res}}, [X_{\min}, X_{\max}], [Y_{\min}, Y_{\max}]} \text{Cell}(r, c)
\end{aligned}$$

This rigorous chain guarantees that every physical obstacle point preserves its true metric location, avoiding the catastrophic radial distortion inherent in 2D homography approaches.
