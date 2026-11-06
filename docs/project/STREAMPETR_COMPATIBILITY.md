# StreamPETR checkpoint compatibility report

Gate 2 report. Sources: `exiawsh/StreamPETR` commit 95f6470, config
`projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.py`, and the checkpoint itself.

## Checkpoint record

| Item | Value |
|---|---|
| Source | https://github.com/exiawsh/storage/releases/download/v1.0/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth |
| Uploaded | 2023-04-06 by `exiawsh` |
| Size | 154,846,243 bytes, matching the release API |
| SHA-256 | `72900a1c9bfbbd65dd52165b46e2de4931d5a4a7a04e58bf68a9add80cdd2c5f` |
| Published digest | None. The hash records our copy; it does not authenticate it against the author. |
| Load | `weights_only=True` succeeded. Pickle globals: `OrderedDict`, tensor rebuild, Float and Long storage only. |
| Contents | `state_dict` only. 591 tensors, 38,657,762 parameters. No embedded config or metadata. |
| Weights licence | **Unverified.** No independent weights licence found. Treated as subject to the non-commercial nuScenes and nuImages terms. Not redistributed; excluded from Git. |

## Architecture and configuration

| Part | Configuration |
|---|---|
| Backbone | ResNet-50, `style='pytorch'`, frozen BatchNorm, stages 3 and 4 output |
| Neck | CPFPN, 256 channels. Stride-16 level used (`position_level=0`). |
| Head | StreamPETRHead: 300 learned queries, 128 propagated, memory 512 |
| Decoder | 6 layers: self-attention, norm, cross-attention, norm, FFN, norm. 8 heads, 256 dims, FFN 2048. |
| 3D position | 64 depth bins (`LID=True`, 1 to about 59 m), range ±61.2 m, z ±10 m |
| Classes | car, truck, construction_vehicle, bus, trailer, barrier, motorcycle, bicycle, pedestrian, traffic_cone |
| Input | Six images, 704x256 |
| Detection range | `pc_range` ±51.2 m, z -5 to 3 m |

## Keys and shapes

| Group | Tensors | Status |
|---|---:|---|
| `img_backbone.*` | 318 (incl. 53 `num_batches_tracked`) | Needed. 53 equals 16 blocks x 3 + 4 downsample + 1. |
| `img_neck.*` | 6 | Needed. `lateral_convs.0/1`, `fpn_convs.0`. |
| `pts_bbox_head.*` | 251 | Needed, except `code_weights` and `match_costs`, which are loss weights. |
| `img_roi_head.*` | 16 | **Not used at inference.** `aux_2d_only=True` skips the 2D head. |

Key shapes: `reference_points (300,3)`, `pseudo_reference_points (128,3)`, `position_encoder.0 (1024,192)`,
`query_embedding.0 (256,384)`, `spatial_alignment.reduce.0 (256,8)`, `ego_pose_pe.reduce.0 (256,180)`, attention
`in_proj_weight (768,256)`, `cls_branches.*.6 (10,256)`, `reg_branches.*.4 (10,256)`, `coords_d (64)`.
The six classification and six regression branches hold identical values: the official code shares one module.
The full key list is in `.tools/streampetr_keys.json` and is enforced by a strict load.

## Required custom operations

All are expressible in plain PyTorch. No compiled operator is needed.

| Operation | Plan |
|---|---|
| Flash attention | Same math as multi-head attention over the same packed `in_proj_*` weights: `nn.MultiheadAttention` |
| MLN, SELayer_Linear | Small modules, copied semantics |
| `pos2posemb3d`, `pos2posemb1d`, `nerf_positional_encoding` | Pure tensor functions |
| `inverse_sigmoid` | With mmdet's clamps, `eps=1e-5` |
| GridMask | Training only |
| Denoising queries | Training only |
| `LiDARInstance3DBoxes` | Replaced by the explicit conversion below |

## Six-camera preprocessing

1. Order: CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT (mmdet3d's `camera_types`). Note this differs from our label order.
2. Resize 1600x900 to 704x396 with PIL's default filter (BICUBIC).
3. Crop rows 140 to 396, keeping the bottom: output 704x256.
4. Normalise RGB with mean (123.675, 116.28, 103.53), std (58.395, 57.12, 57.375).
5. Intrinsics: `K' = ida @ K`, scale 0.44, translate v by -140. `lidar2img = K' @ lidar2cam`.

## Temporal memory and ego motion

- Reset at scene start. The first frame uses zero memory plus `pseudo_reference_points`, with identity poses.
- Before each frame: reference points and stored poses move into the current frame, ages increase, entries truncate to 512.
- After each frame: top 128 queries by score enter memory with position, feature, velocity, age 0 and identity pose.
- Ego motion enters the model through a 15-value code (velocity 2, age 1, pose 12) with 12 frequencies, giving 180 dims, used by the two MLN layers.
- Inference is causal: only current and past frames are used.

## Coordinates and output

The model works in the **LIDAR_TOP sensor frame** (x right, y forward, z up, about 1.8 m above ground). This is a coordinate
frame only.

```
T_global_lidar = T_global_ego(reference) @ T_ego_lidar      # static LIDAR_TOP extrinsic
lidar2cam_i    = inverse(T_global_ego(cam_i) @ T_ego_cam_i) @ T_global_lidar
ego_pose       = T_global_lidar
```

The per-camera pose compensates ego motion between exposures. It is taken from each camera's own ego pose.

Decode: sigmoid class scores, top 300 over all class-query pairs, labels `index % 10`, query `index // 10`. A box is
`[x, y, z_centre, log dx, log dy, log dz, sin yaw, cos yaw, vx, vy]`. `dx` is along the heading, so `dx`=length and
`dy`=width (mmdet3d maps `dims[[1,0,2]]` to nuScenes wlh). Yaw and velocity are in the LIDAR frame. Keep boxes inside
±61.2 m, z ±10 m. To ego: `p_ego = R_el p + t_el`, `yaw_ego = yaw + yaw(R_el)`, `v_ego = R_el v`.

## Faithfulness assessment

The model can be reconstructed from configuration and checkpoint. Planned deviations:

| # | Deviation | Effect |
|---|---|---|
| 1 | `nn.MultiheadAttention` in place of `flash_attn` | None intended: same computation |
| 2 | Relative transforms between frames in place of global poses, float32 on MPS | Numerically safer. MPS has no float64. Same math. |
| 3 | Reference ego pose and time from CAM_FRONT, not the LIDAR_TOP keyframe row | Avoids any LiDAR-timed input. Moves the reference by tens of milliseconds. |
| 4 | float32, not official test-time autocast | Tiny numerical differences |
| 5 | Only the last decoder layer is decoded | Same output: earlier layers feed nothing forward through the branches |
| 6 | The official intrinsic tiling quirk is **kept** | Intentional: this is what the weights were trained with |

Not possible here: a numeric comparison with the official code. Verification relies on strict loading and on checks
against real annotations.

## Port verification result

573 of 573 model tensors loaded with strict shape agreement, 0 missing, 0 unexpected. 18 checkpoint tensors are unused:
16 from the 2D head and 2 loss weights. Evidence: `experiments/end_to_end/results/streampetr/port_verification.json`.

Measured on mini_val: 81 frames, 2,460 annotated objects within 40 m, 2 m matching, score at least 0.3, before the scorer's
low-visibility allowance. Each control breaks one thing, so a meaningful check must degrade.

| Check | Port | No temporal memory | Calibration scrambled across cameras |
|---|---:|---:|---:|
| Recall | 0.877 | 0.803 | 0.045 |
| Precision | 0.636 | 0.511 | 0.034 |
| Centre error (m) | 0.56 | 0.75 | 1.30 |
| Vertical centre error (m) | 0.083 | 0.091 | 0.456 |
| Median yaw error, long objects (deg) | 2.9 | 3.5 | 108.8 |
| Velocity error (m/s) | 0.17 | 0.40 | 2.45 |
| Length / width / height ratio | 0.98 / 1.03 / 0.98 | 0.99 / 1.04 / 0.99 | 1.01 / 1.07 / 1.00 |

These show camera order, calibration, frame conversion, size order, yaw, velocity and temporal memory are functioning.
They do not prove bit-level equality with the official implementation.
