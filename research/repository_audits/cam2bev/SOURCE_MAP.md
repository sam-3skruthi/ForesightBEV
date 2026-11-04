# Source Map: Cam2BEV

**Target Repository:** `Cam2BEV` (RWTH Aachen University, Institute for Automotive Engineering - ika)
**Repository URL:** [https://github.com/ika-rwth-aachen/Cam2BEV](https://github.com/ika-rwth-aachen/Cam2BEV)
**Commit Audited:** `5691703b2979d6af88ce8dd44abd6a49130b79e4` (HEAD -> master)
**Associated Publication:** *"A Sim2Real Deep Learning Approach for the Transformation of Images from Multiple Vehicle-Mounted Cameras to a Semantically Segmented Image in Bird’s Eye View"* (IEEE ITSC 2020, [arXiv:2005.04078](https://arxiv.org/abs/2005.04078))
**Auditor Mode:** Read-Only Static Reverse-Engineering & Code Inspection

---

## 1. Directory Tree & Structural Mapping

```text
Cam2BEV/
├── .gitignore                                 # Git ignore patterns
├── CITATION.bib                               # BibTeX citation entry for IEEE ITSC 2020 paper
├── CITATION.cff                               # Citation File Format metadata v1.2.0
├── LICENSE                                    # MIT License (2020 Institute for Automotive Engineering, RWTH Aachen)
├── README.md                                  # Upstream project overview, setup, data & training guide
├── requirements.original.txt                  # Original frozen Python 3.7 / TensorFlow 2.1.0 dependencies (51 packages)
├── requirements.txt                           # Downgraded / minimal Python 3.7 / TensorFlow 2.5.0 dependencies (9 packages)
├── assets/                                    # Documentation media
│   ├── logo.png                               # Project logo
│   ├── teaser.gif                             # Demonstration animation of surround camera to BEV
│   └── video_preview.jpg                      # Preview image linking to YouTube presentation
├── data/
│   └── download.sh                            # Bash script to download dataset archives from RWTH Aachen GitLab
├── model/
│   ├── train.py                               # Model training loop with tf.data, Adam optimizer & callbacks
│   ├── evaluate.py                            # Model evaluation script: confusion matrix, per-class IoU, mIoU
│   ├── predict.py                             # Inference script: generates segmented BEV PNG images
│   ├── utils.py                               # Image I/O, palette conversion, loss functions, custom metrics
│   ├── config.1_FRLR.unetxst.yml              # Config: 4 cameras -> uNetXST (Surround rig)
│   ├── config.1_FRLR.deeplab-mobilenet.yml     # Config: Stitched IPM -> DeepLab MobileNetV2 (Surround rig)
│   ├── config.1_FRLR.deeplab-xception.yml      # Config: Stitched IPM -> DeepLab Xception (Surround rig)
│   ├── config.2_F.unetxst.yml                 # Config: 1 camera -> uNetXST (Front-only rig)
│   ├── config.2_F.deeplab-mobilenet.yml       # Config: Stitched IPM -> DeepLab MobileNetV2 (Front-only rig)
│   ├── config.2_F.deeplab-xception.yml        # Config: Stitched IPM -> DeepLab Xception (Front-only rig)
│   ├── architecture/
│   │   ├── uNetXST.py                         # Multi-stream U-Net with in-network Spatial Transformers
│   │   ├── deeplab_mobilenet.py               # DeepLabv3+ wrapper with MobileNetV2 backbone
│   │   ├── deeplab_xception.py                # DeepLabv3+ wrapper with Xception backbone
│   │   └── third_party/
│   │       ├── deeplabv3plus.py               # Keras DeepLabv3+ implementation by Emil Zakirov
│   │       └── spatial_transformer.py         # Keras BilinearInterpolation STN layer by Octavio Arriaga
│   └── one_hot_conversion/
│       ├── convert_10.xml                     # 10-class Cityscapes reduction (input segmentation palette)
│       ├── convert_9+occl.xml                 # 9-class + occlusion (10 total classes, target label for 1_FRLR)
│       ├── convert_3+occl.xml                 # 3-class + occlusion (4 total classes, target label for 2_F)
│       ├── convert_3.xml                      # 3-class palette without occlusion
│       ├── convert_9.xml                      # 9-class palette without occlusion
│       └── convert_30.xml                     # Full 30-class Cityscapes palette
└── preprocessing/
    ├── camera_configs/
    │   ├── 1_FRLR/                            # 4-camera surround rig calibration files
    │   │   ├── front.yaml                     # Front camera intrinsics/extrinsics
    │   │   ├── rear.yaml                      # Rear camera intrinsics/extrinsics
    │   │   ├── left.yaml                      # Left camera intrinsics/extrinsics
    │   │   ├── right.yaml                     # Right camera intrinsics/extrinsics
    │   │   └── drone.yaml                     # Top-down virtual drone camera configuration (centered)
    │   └── 2_F/                               # Single front camera rig calibration files
    │       ├── front.yaml                     # Front camera intrinsics/extrinsics
    │       └── drone.yaml                     # Top-down virtual drone camera configuration (offset)
    ├── homography_converter/
    │   ├── README.md                          # Guide on adapting homographies for resolution & STN
    │   ├── homography_converter.py            # Converts OpenCV homography to SpatialTransformer format
    │   └── uNetXST_homographies/
    │       ├── 1_FRLR.py                      # Pre-calculated 3x3 homographies for 4-camera setup
    │       └── 2_F.py                         # Pre-calculated 3x3 homography for 1-camera setup
    ├── ipm/
    │   ├── README.md                          # Mathematical background and CLI usage for IPM
    │   ├── ipm.py                             # Classical Inverse Perspective Mapping and stitching
    │   └── assets/example.png                 # Reference visualization of stitched IPM image
    └── occlusion/
        ├── README.md                          # Documentation of occlusion ray casting methodology
        ├── occlusion.py                       # 2D line-of-sight ray casting script (multiprocessing)
        └── assets/
            ├── example-original.png           # Raw synthetic drone view
            └── example-occluded.png           # Ray-cast occluded ground truth
```

---

## 2. File-by-File Module Inventory

### 2.1 Repository Metadata & Packaging

| File Path | Lines | Purpose & Contents | License / Authorship |
|---|---|---|---|
| [LICENSE](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/LICENSE) | 21 | MIT License terms | Copyright 2020 Institute for Automotive Engineering (ika), RWTH Aachen University |
| [CITATION.bib](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/CITATION.bib) | 7 | BibTeX citation for IEEE ITSC 2020 publication | Lennart Reiher, Bastian Lampe, Lutz Eckstein |
| [CITATION.cff](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/CITATION.cff) | 27 | YAML Citation File Format v1.2.0 metadata | Lennart Reiher, Bastian Lampe, Lutz Eckstein |
| [README.md](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/README.md) | 240 | Setup, architecture descriptions, usage workflows, citations | RWTH Aachen University (ika) |
| [requirements.original.txt](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/requirements.original.txt) | 51 | Pinned dependency manifest for original paper results (Python 3.7, TensorFlow 2.1.0, CUDA 10.1) | Upstream project freeze |
| [requirements.txt](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/requirements.txt) | 9 | Minimal dependency manifest downgraded in commit `bfcaa28` (Python 3.7, TensorFlow 2.5.0) | Upstream project freeze |

### 2.2 Data Ingestion Scripts

| File Path | Lines | Purpose & Contents | License / Authorship |
|---|---|---|---|
| [data/download.sh](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/data/download.sh) | 26 | Shell script downloading `cam2bev-data-master.tar.gz` (3.7 GB) from RWTH Aachen GitLab repository and extracting nested archives | MIT License / RWTH Aachen |

### 2.3 Camera Calibration & Preprocessing Utilities

| File Path | Lines | Purpose & Contents | License / Authorship |
|---|---|---|---|
| [preprocessing/camera_configs/1_FRLR/front.yaml](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/preprocessing/camera_configs/1_FRLR/front.yaml) | 15 | Intrinsics ($f_x=278.283, f_y=408.1295, c_x=482, c_y=302$), extrinsics ($X=1.7, Y=0.0, Z=1.4$, Yaw=$0^\circ$) | RWTH Aachen |
| [preprocessing/camera_configs/1_FRLR/rear.yaml](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/preprocessing/camera_configs/1_FRLR/rear.yaml) | 15 | Intrinsics ($f_x=278.283, f_y=408.1295, c_x=482, c_y=302$), extrinsics ($X=-0.6, Y=0.0, Z=1.4$, Yaw=$180^\circ$) | RWTH Aachen |
| [preprocessing/camera_configs/1_FRLR/left.yaml](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/preprocessing/camera_configs/1_FRLR/left.yaml) | 15 | Intrinsics ($f_x=278.283, f_y=408.1295, c_x=482, c_y=302$), extrinsics ($X=0.5, Y=0.5, Z=1.5$, Yaw=$90^\circ$) | RWTH Aachen |
| [preprocessing/camera_configs/1_FRLR/right.yaml](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/preprocessing/camera_configs/1_FRLR/right.yaml) | 15 | Intrinsics ($f_x=278.283, f_y=408.1295, c_x=482, c_y=302$), extrinsics ($X=0.5, Y=-0.5, Z=1.5$, Yaw=$-90^\circ$) | RWTH Aachen |
| [preprocessing/camera_configs/1_FRLR/drone.yaml](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/preprocessing/camera_configs/1_FRLR/drone.yaml) | 15 | Intrinsics ($f_x=f_y=682.578, c_x=482, c_y=302$), extrinsics ($X=0, Y=0, Z=50.0$, Pitch=$90^\circ$, Roll=$90^\circ$) | RWTH Aachen |
| [preprocessing/camera_configs/2_F/front.yaml](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/preprocessing/camera_configs/2_F/front.yaml) | 15 | Single camera front intrinsics ($f_x=2166.11, f_y=2167.96, c_x=969.36, c_y=615.40$), extrinsics ($X=1.84, Y=-0.096, Z=1.13$) | RWTH Aachen |
| [preprocessing/camera_configs/2_F/drone.yaml](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/preprocessing/camera_configs/2_F/drone.yaml) | 15 | Single camera drone intrinsics ($f_x=f_y=3872.0, c_x=968, c_y=484$), extrinsics ($X=28.84, Y=0.0, Z=100.0$) | RWTH Aachen |
| [preprocessing/ipm/ipm.py](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/preprocessing/ipm/ipm.py) | 201 | Computes analytical camera matrices $K, R, t, P, M$, inverts to $H=(PM)^{-1}$, performs `cv2.warpPerspective`, masks invalid FOVs, stitches multi-camera BEV | MIT License / RWTH Aachen |
| [preprocessing/homography_converter/homography_converter.py](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/preprocessing/homography_converter/homography_converter.py) | 131 | Transforms OpenCV pixel homographies to aspect-ratio adjusted SpatialTransformer $[-1, 1]$ grid matrices ($H_{stn} = (S_o H_{cv} S_i)^{-1}$) | MIT License / RWTH Aachen |
| [preprocessing/homography_converter/uNetXST_homographies/1_FRLR.py](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/preprocessing/homography_converter/uNetXST_homographies/1_FRLR.py) | 33 | Pre-computed list of four $3 \times 3$ numpy matrices $H$ for front, rear, left, right cameras | MIT License / RWTH Aachen |
| [preprocessing/homography_converter/uNetXST_homographies/2_F.py](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/preprocessing/homography_converter/uNetXST_homographies/2_F.py) | 30 | Pre-computed list containing single $3 \times 3$ numpy matrix $H$ for front camera | MIT License / RWTH Aachen |
| [preprocessing/occlusion/occlusion.py](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/preprocessing/occlusion/occlusion.py) | 255 | 2D ray casting over drone semantic maps; handles blocking classes, flood fill, ego vehicle recoloring | MIT License / RWTH Aachen |

### 2.4 Model Architectures & Third-Party Layers

| File Path | Lines | Purpose & Contents | License / Authorship |
|---|---|---|---|
| [model/architecture/uNetXST.py](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/model/architecture/uNetXST.py) | 158 | Proposed architecture: $N$ parallel encoders, in-network `SpatialTransformer` feature warping ($\theta_{const}=\text{True}$), channel concatenation, U-Net decoder | MIT License / RWTH Aachen |
| [model/architecture/deeplab_mobilenet.py](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/model/architecture/deeplab_mobilenet.py) | 41 | DeepLabv3+ with MobileNetV2 backbone wrapper taking single stitched IPM image input | MIT License / RWTH Aachen |
| [model/architecture/deeplab_xception.py](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/model/architecture/deeplab_xception.py) | 41 | DeepLabv3+ with Xception backbone wrapper taking single stitched IPM image input | MIT License / RWTH Aachen |
| [model/architecture/third_party/spatial_transformer.py](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/model/architecture/third_party/spatial_transformer.py) | 247 | Keras `BilinearInterpolation` layer and `SpatialTransformer` module. Normalized grid generation $[-1, 1]$ and backward bilinear sampling | MIT License / Octavio Arriaga (2017) |
| [model/architecture/third_party/deeplabv3plus.py](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/model/architecture/third_party/deeplabv3plus.py) | 502 | Complete Keras DeepLabv3+ implementation with Atrous Spatial Pyramid Pooling (ASPP), depthwise convolutions | MIT License / Emil Zakirov (2018) |

### 2.5 Training, Inference, and Metrics Scripts

| File Path | Lines | Purpose & Contents | License / Authorship |
|---|---|---|---|
| [model/utils.py](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/model/utils.py) | 258 | Image I/O, resizing with aspect-ratio crop, one-hot encode/decode, XML parsing, class distribution calculator, `weighted_categorical_crossentropy`, `MeanIoUWithOneHotLabels` | MIT License / RWTH Aachen |
| [model/train.py](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/model/train.py) | 185 | Training execution script using `configargparse`, `tf.data.Dataset`, Adam optimizer, callbacks (`TensorBoard`, `ModelCheckpoint`, `EarlyStopping`) | MIT License / RWTH Aachen |
| [model/evaluate.py](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/model/evaluate.py) | 199 | Evaluates model on validation data: accumulates confusion matrix, computes per-class IoU and overall mIoU, plots heatmap via Seaborn | MIT License / RWTH Aachen |
| [model/predict.py](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/model/predict.py) | 136 | Inference script: feeds test camera images, outputs decoded color-coded BEV PNGs | MIT License / RWTH Aachen |

### 2.6 Class Palette Definitions (XML)

| File Path | Lines | Purpose & Contents | License / Authorship |
|---|---|---|---|
| [model/one_hot_conversion/convert_10.xml](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/model/one_hot_conversion/convert_10.xml) | 43 | Maps 30 Cityscapes RGB colors into 10 semantic input classes (`road`, `sidewalk`, `person`, `car`, `truck`, `bus`, `two-wheeler`, `obstacle`, `vegetation`, `sky`) | RWTH Aachen |
| [model/one_hot_conversion/convert_9+occl.xml](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/model/one_hot_conversion/convert_9+occl.xml) | 44 | Maps 30 Cityscapes RGB colors into 9 semantic classes + `occluded` (Class 10: RGB `150, 150, 150`). Used for `1_FRLR` labels | RWTH Aachen |
| [model/one_hot_conversion/convert_3+occl.xml](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/model/one_hot_conversion/convert_3+occl.xml) | 38 | Maps Cityscapes colors into 3 coarse classes (`road`, `vehicle`, `obstacle`) + `occluded` (Class 4). Used for `2_F` labels | RWTH Aachen |
| [model/one_hot_conversion/convert_3.xml](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/model/one_hot_conversion/convert_3.xml) | 37 | 3 classes (`road`, `vehicle`, `obstacle`) without occlusion class | RWTH Aachen |
| [model/one_hot_conversion/convert_9.xml](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/model/one_hot_conversion/convert_9.xml) | 43 | 9 classes without occlusion class | RWTH Aachen |
| [model/one_hot_conversion/convert_30.xml](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/model/one_hot_conversion/convert_30.xml) | 35 | Full 30 raw Cityscapes classes | RWTH Aachen |

### 2.7 Training Configurations (YAML)

| File Path | Lines | Target Rig | Model Architecture | Input Mode | Classes | Batch / Epochs |
|---|---|---|---|---|---|---|
| [model/config.1_FRLR.unetxst.yml](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/model/config.1_FRLR.unetxst.yml) | 30 | `1_FRLR` (4 cams) | `uNetXST.py` | 4 raw perspective streams | 10 classes (`convert_9+occl.xml`) | Batch: 5, Epochs: 100, LR: $10^{-4}$ |
| [model/config.1_FRLR.deeplab-mobilenet.yml](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/model/config.1_FRLR.deeplab-mobilenet.yml) | 30 | `1_FRLR` (4 cams) | `deeplab_mobilenet.py` | 1 stitched IPM image | 10 classes (`convert_9+occl.xml`) | Batch: 5, Epochs: 100, LR: $10^{-4}$ |
| [model/config.1_FRLR.deeplab-xception.yml](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/model/config.1_FRLR.deeplab-xception.yml) | 30 | `1_FRLR` (4 cams) | `deeplab_xception.py` | 1 stitched IPM image | 10 classes (`convert_9+occl.xml`) | Batch: 5, Epochs: 100, LR: $10^{-4}$ |
| [model/config.2_F.unetxst.yml](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/model/config.2_F.unetxst.yml) | 30 | `2_F` (1 cam) | `uNetXST.py` | 1 raw perspective stream | 4 classes (`convert_3+occl.xml`) | Batch: 5, Epochs: 100, LR: $10^{-4}$ |
| [model/config.2_F.deeplab-mobilenet.yml](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/model/config.2_F.deeplab-mobilenet.yml) | 30 | `2_F` (1 cam) | `deeplab_mobilenet.py` | 1 IPM image | 4 classes (`convert_3+occl.xml`) | Batch: 5, Epochs: 100, LR: $10^{-4}$ |
| [model/config.2_F.deeplab-xception.yml](https://github.com/ika-rwth-aachen/Cam2BEV/blob/5691703b2979d6af88ce8dd44abd6a49130b79e4/model/config.2_F.deeplab-xception.yml) | 30 | `2_F` (1 cam) | `deeplab_xception.py` | 1 IPM image | 4 classes (`convert_3+occl.xml`) | Batch: 5, Epochs: 100, LR: $10^{-4}$ |

---

## 3. Absence of Pretrained Weights Verification

A filesystem search confirmed that **zero** weight checkpoints (`.h5`, `.hdf5`, `.pt`, `.pth`, `.ckpt`) are stored within the upstream repository tree:

```bash
find . -type f \( -name "*.h5" -o -name "*.hdf5" -o -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" -o -name "*.bin" \)
# Return: 0 files found
```

The repository is structured strictly for training from scratch using synthetic data downloaded from GitLab. In contrast, the student derivative repository `Birds_Eye_View` included a 37.4 MB checkpoint `model/best_weights.hdf5` obtained from their training run of `uNetXST` on `1_FRLR`.
