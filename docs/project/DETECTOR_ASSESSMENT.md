# Pretrained detector assessment

Prepared 2026-11-05 for approval. Nothing has been downloaded or installed. Sources: official repositories at
`fundamentalvision/BEVFormer` 66b65f3 and `exiawsh/StreamPETR` 95f6470, release listings, the nuScenes devkit
`splits.py`, and the licence file shipped with the dataset.

## Recommendation

**StreamPETR R50, 428 queries, nuImages-pretrained backbone.**
Checkpoint: `exiawsh/storage` release v1.0, `stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth`, 154 MB.

## Comparison

| | StreamPETR R50 428q | BEVFormer-tiny | BEVFormer-base |
|---|---|---|---|
| Published nuScenes val NDS / mAP | 54.6 / 44.9 | 35.4 / 25.2 | 51.7 / 41.6 |
| Input | 704x256, six cameras | Six cameras, smaller | 1600x900, six cameras |
| Download | 154 MB | 400 MB | 276 MB, plus pretrain weights for training only |
| Published memory | Not stated | 6.5 GB (training) | 28.5 GB (training) |
| Metadata needed | Calibration, ego pose, timestamps | Also CAN bus | Also CAN bus |
| Custom compiled ops | None at inference | Deformable attention in Python | DCN backbone needs compiled `mmcv.ops` |
| Runs on Apple M5 | **Yes, after a torch-only port** | Possible with a larger port | Not practical |
| Code licence | Apache-2.0 | Apache-2.0 | Apache-2.0 |

BEVFormer's `use_can_bus=True` needs the nuScenes CAN expansion, which is a separate download and is not in the mini
archive.

## Feasibility on Apple Silicon

The official stacks cannot be installed here: Python 3.8, PyTorch 1.9 on CUDA 11, `mmcv-full`, `mmdet3d` 1.0.0rc6.
That route needs NVIDIA hardware. It is also unnecessary for inference.

At inference, StreamPETR needs a ResNet-50, an FPN neck and a transformer decoder. `aux_2d_only=True` skips the 2D
focal head. Flash attention is an optional speedup for ordinary multi-head attention, and the weights are still
ordinary linear layers. `mmcv` here provides training plumbing and module wrappers. A small inference-only port in plain
PyTorch can load the checkpoint, with no new dependencies.

Measured on this M5 with a random-weight stand-in of the same dimensions (34.5 M parameters, six 704x256 images, 428
queries, 512 memory tokens, six decoder layers): **about 280 ms per frame, 1.3 GB GPU memory.** The real model will
differ somewhat. The target of 2 Hz keyframes is met.

## Constraints

**LiDAR.** Inference reads no LiDAR data. The model's output frame is the LIDAR_TOP sensor frame, and its calibration
uses `lidar2cam`. The fixed `LIDAR_TOP` extrinsic from `calibrated_sensor` is therefore needed as a coordinate transform.
That is metadata, one number set per scene. Point clouds are never opened. Training labels and the backbone pretraining
are offline.

**Causality.** Streaming inference uses only current and past frames, with temporal memory reset at scene start.

**Licences.** Code is Apache-2.0, and a port must retain attribution. Weights trained on nuScenes and nuImages fall under
those datasets' terms, which the shipped licence states are CC BY-NC-SA 4.0 with additional terms: non-commercial use.
No separate weights licence was found.

**Checkpoint format.** A pickle `.pth`. It will be loaded with `weights_only=True`, or with a restricted unpickler if
its metadata needs one. It will never be loaded with full pickle.

## Evaluation validity: training overlap

**Corrected.** The official devkit `splits.py` defines `mini_val` as scene-0103 and scene-0916, and the other eight mini
scenes as `mini_train`. Only `mini_val` is used for headline pretrained-model evaluation. Dataset tables carry no split
field, so the devkit is the authority. An earlier draft of this section proposed four scenes by reading the full
`val` split. That is superseded.

scene-0103 and scene-0916 were both used earlier: 0103 for tuning and 0916 in the reported held-out set. They are
**previously seen evaluation data**. Two scenes give preliminary validation only and support no claim about
generalisation. Both are daytime, so night performance is not tested.

## Integration plan, if approved

1. Torch-only port under `src/foresightbev/perception/streampetr.py`, with the source commit recorded.
2. Strict checkpoint load: every key must match. A silent partial load is a failure.
3. Verify frames: project predicted boxes into all six cameras on real frames and inspect them, then check the LIDAR_TOP to ego conversion numerically against annotations.
4. `detector: streampetr` in configuration. Both existing detectors stay available.
5. Compare on the existing evaluator: per-class precision and recall, position error, tracking, latency, memory, and the existing safety measures.
6. Keep all safety thresholds as they are. The only new setting, a detection score cutoff, is fixed at 0.3 before evaluation, and a precision-recall sweep is reported rather than a chosen best point.

## Risks

- **No numeric reference run.** The official stack cannot run here, so port fidelity rests on strict key loading and on matching visual and annotated checks. Published val scores cover 6,019 frames and are not directly comparable to four mini scenes.
- **Detector output can outrun the tracker.** Noise settings were measured for the old detector and may need re-measuring on scene 0103.
- **The filter may stay poor.** Better detection does not fix static false-free cells from flat-ground occupancy.
