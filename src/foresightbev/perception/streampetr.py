"""Inference-only PyTorch port of StreamPETR (R50, 428 queries), for pretrained weights.

Ported from exiawsh/StreamPETR commit 95f6470, Apache-2.0, which builds on PETR
(Megvii), DETR3D and OpenMMLab code. See docs/project/STREAMPETR_COMPATIBILITY.md.
Pretrained on nuScenes train and nuImages; the weights licence is unverified and is
treated as non-commercial. Training code and LiDAR data are absent. The model works in the
LIDAR_TOP coordinate frame, so a fixed extrinsic is needed as a transform only.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from foresightbev.errors import ModelUnavailable

CLASS_NAMES = ("car", "truck", "construction_vehicle", "bus", "trailer", "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone")
CAMERA_ORDER = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
CHECKPOINT_SHA256 = "72900a1c9bfbbd65dd52165b46e2de4931d5a4a7a04e58bf68a9add80cdd2c5f"
IMAGE_MEAN = np.array([123.675, 116.28, 103.53], np.float32)
IMAGE_STD = np.array([58.395, 57.12, 57.375], np.float32)
INPUT_WIDTH, INPUT_HEIGHT = 704, 256
RESIZED_HEIGHT = 396
UNUSED_PREFIXES = ("img_roi_head.",)
UNUSED_KEYS = ("pts_bbox_head.code_weights", "pts_bbox_head.match_costs")


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(0, 1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


def pos2posemb3d(pos: torch.Tensor, features: int = 128, temperature: float = 10000) -> torch.Tensor:
    pos = pos * (2 * math.pi)
    dim = torch.arange(features, dtype=torch.float32, device=pos.device)
    dim = temperature ** (2 * torch.div(dim, 2, rounding_mode="floor") / features)

    def encode(axis):
        v = pos[..., axis, None] / dim
        return torch.stack((v[..., 0::2].sin(), v[..., 1::2].cos()), dim=-1).flatten(-2)

    return torch.cat((encode(1), encode(0), encode(2)), dim=-1)


def pos2posemb1d(pos: torch.Tensor, features: int = 256, temperature: float = 10000) -> torch.Tensor:
    pos = pos * (2 * math.pi)
    dim = torch.arange(features, dtype=torch.float32, device=pos.device)
    dim = temperature ** (2 * torch.div(dim, 2, rounding_mode="floor") / features)
    v = pos[..., 0, None] / dim
    return torch.stack((v[..., 0::2].sin(), v[..., 1::2].cos()), dim=-1).flatten(-2)


def nerf_encoding(tensor: torch.Tensor, functions: int = 6) -> torch.Tensor:
    bands = 2.0 ** torch.linspace(0.0, functions - 1, functions, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([f(tensor * b) for b in bands for f in (torch.sin, torch.cos)], dim=-1)


class Bottleneck(nn.Module):
    """ResNet bottleneck, ``style='pytorch'``: the stride sits on the 3x3 convolution."""

    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = None
        if stride != 1 or inplanes != planes * 4:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes * 4, 1, stride, bias=False), nn.BatchNorm2d(planes * 4))

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        return self.relu(y + (x if self.downsample is None else self.downsample(x)))


class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._stage(64, 64, 3, 1)
        self.layer2 = self._stage(256, 128, 4, 2)
        self.layer3 = self._stage(512, 256, 6, 2)
        self.layer4 = self._stage(1024, 512, 3, 2)

    @staticmethod
    def _stage(inplanes, planes, blocks, stride):
        return nn.Sequential(Bottleneck(inplanes, planes, stride), *[Bottleneck(planes * 4, planes) for _ in range(blocks - 1)])

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c4 = self.layer3(self.layer2(self.layer1(x)))
        return c4, self.layer4(c4)


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, padding=padding)

    def forward(self, x):
        return self.conv(x)


class CPFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lateral_convs = nn.ModuleList([ConvModule(1024, 256, 1), ConvModule(2048, 256, 1)])
        self.fpn_convs = nn.ModuleList([ConvModule(256, 256, 3, 1)])

    def forward(self, c4, c5):
        p4 = self.lateral_convs[0](c4)
        p5 = self.lateral_convs[1](c5)
        return self.fpn_convs[0](p4 + F.interpolate(p5, size=p4.shape[2:], mode="nearest"))


class SELayerLinear(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_reduce = nn.Linear(channels, channels)
        self.conv_expand = nn.Linear(channels, channels)

    def forward(self, x, x_se):
        return x * torch.sigmoid(self.conv_expand(torch.relu(self.conv_reduce(x_se))))


class MLN(nn.Module):
    def __init__(self, c_dim, f_dim=256):
        super().__init__()
        self.reduce = nn.Sequential(nn.Linear(c_dim, f_dim), nn.ReLU())
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)

    def forward(self, x, c):
        c = self.reduce(c)
        return self.gamma(c) * self.ln(x) + self.beta(c)


class AttentionWrapper(nn.Module):
    """Holds ``attn`` so state keys match. Queries add positions; values do not."""

    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(256, 8, dropout=0.1)

    def forward(self, query, key, value, query_pos, key_pos):
        out = self.attn(query + query_pos, key + key_pos, value, need_weights=False)[0]
        return query + out


class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Sequential(nn.Linear(256, 2048), nn.ReLU(), nn.Dropout(0.1)), nn.Linear(2048, 256), nn.Dropout(0.1))

    def forward(self, x):
        return x + self.layers(x)


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attentions = nn.ModuleList([AttentionWrapper(), AttentionWrapper()])
        self.ffns = nn.ModuleList([FFN()])
        self.norms = nn.ModuleList([nn.LayerNorm(256) for _ in range(3)])

    def forward(self, query, key, value, query_pos, key_pos, temp_memory, temp_pos):
        both = torch.cat([query, temp_memory], dim=0)
        query = self.norms[0](self.attentions[0](query, both, both, query_pos, torch.cat([query_pos, temp_pos], dim=0)))
        query = self.norms[1](self.attentions[1](query, key, value, query_pos, key_pos))
        return self.norms[2](self.ffns[0](query))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(6)])
        self.post_norm = nn.LayerNorm(256)

    def forward(self, query, *args):
        for layer in self.layers:
            query = layer(query, *args)
        return self.post_norm(query)


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = Decoder()


class StreamPETRHead(nn.Module):
    def __init__(self, num_query=300, num_propagated=128, memory_len=512, topk=128, depth_num=64):
        super().__init__()
        self.num_query, self.num_propagated, self.memory_len, self.topk = num_query, num_propagated, memory_len, topk
        self.register_buffer("pc_range", torch.zeros(6))
        self.register_buffer("position_range", torch.zeros(6))
        self.register_buffer("coords_d", torch.zeros(depth_num))
        fc_cls = nn.Sequential(nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(), nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(), nn.Linear(256, 10))
        fc_reg = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))
        self.cls_branches = nn.ModuleList([fc_cls for _ in range(6)])
        self.reg_branches = nn.ModuleList([fc_reg for _ in range(6)])
        self.position_encoder = nn.Sequential(nn.Linear(depth_num * 3, 1024), nn.ReLU(), nn.Linear(1024, 256))
        self.memory_embed = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256))
        self.featurized_pe = SELayerLinear(256)
        self.reference_points = nn.Embedding(num_query, 3)
        self.pseudo_reference_points = nn.Embedding(num_propagated, 3)
        self.query_embedding = nn.Sequential(nn.Linear(384, 256), nn.ReLU(), nn.Linear(256, 256))
        self.spatial_alignment = MLN(8)
        self.time_embedding = nn.Sequential(nn.Linear(256, 256), nn.LayerNorm(256))
        self.ego_pose_pe = MLN(180)
        self.ego_pose_memory = MLN(180)
        self.transformer = Transformer()
        self.reset_memory()

    def reset_memory(self):
        self.memory_embedding = None

    def _prepare_memory(self, device, relative_pose: torch.Tensor | None, dt: float) -> None:
        """Bring memory into the current frame. ``relative_pose`` maps previous to current."""
        if self.memory_embedding is None:
            self.memory_embedding = torch.zeros(1, self.memory_len, 256, device=device)
            self.memory_reference_point = torch.zeros(1, self.memory_len, 3, device=device)
            self.memory_timestamp = torch.zeros(1, self.memory_len, 1, device=device)
            self.memory_egopose = torch.zeros(1, self.memory_len, 4, 4, device=device)
            self.memory_velo = torch.zeros(1, self.memory_len, 2, device=device)
            pseudo = self.pseudo_reference_points.weight * (self.pc_range[3:6] - self.pc_range[:3]) + self.pc_range[:3]
            self.memory_reference_point[:, : self.num_propagated] += pseudo
            self.memory_egopose[:, : self.num_propagated] += torch.eye(4, device=device)
            return
        n = self.memory_len
        self.memory_timestamp = (self.memory_timestamp + dt)[:, :n]
        self.memory_egopose = (relative_pose[None, None] @ self.memory_egopose)[:, :n]
        points = torch.cat([self.memory_reference_point, torch.ones_like(self.memory_reference_point[..., :1])], dim=-1)
        self.memory_reference_point = (relative_pose[None, None] @ points.unsqueeze(-1)).squeeze(-1)[..., :3][:, :n]
        self.memory_embedding = self.memory_embedding[:, :n]
        self.memory_velo = self.memory_velo[:, :n]

    def _position_embedding(self, height: int, width: int, intrinsics: torch.Tensor, lidar2img: torch.Tensor):
        device = lidar2img.device
        n = lidar2img.shape[0]
        tokens = n * height * width
        focal = (torch.stack([intrinsics[:, 0, 0], intrinsics[:, 1, 1]], dim=-1).abs() / 1e3)[None]
        # The official tiling repeats cameras per token rather than per image. Kept: the weights saw it.
        focal = focal.repeat(1, height * width, 1).view(1, -1, 2)
        ys = torch.arange(height, dtype=torch.float32, device=device) * 16 + 8
        xs = torch.arange(width, dtype=torch.float32, device=device) * 16 + 8
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        centers = torch.stack([grid_x, grid_y], dim=-1).view(1, height * width, 1, 2).repeat(n, 1, self.coords_d.numel(), 1)
        depth = self.coords_d.view(1, 1, -1, 1).repeat(n, height * width, 1, 1)
        coords = torch.cat([centers * depth, depth, torch.ones_like(depth)], dim=-1).unsqueeze(-1)
        img2lidar = lidar2img.inverse()[:, None, None]
        points = (img2lidar @ coords).squeeze(-1)[..., :3].reshape(1, tokens, -1, 3)
        low, high = self.position_range[:3], self.position_range[3:6]
        points = ((points - low) / (high - low)).reshape(1, tokens, -1)
        embedding = self.position_encoder(inverse_sigmoid(points))
        cone = torch.cat([focal, points[..., -3:], points[..., -90:-87]], dim=-1)
        return embedding, cone

    def _temporal_alignment(self, query_pos, tgt, reference_points):
        device = query_pos.device
        low, high = self.pc_range[:3], self.pc_range[3:6]
        temp_reference = (self.memory_reference_point - low) / (high - low)
        temp_pos = self.query_embedding(pos2posemb3d(temp_reference))
        count = query_pos.shape[1]
        identity = torch.eye(4, device=device)[None, None].repeat(1, count, 1, 1)
        motion = nerf_encoding(torch.cat([torch.zeros_like(reference_points), identity[..., :3, :].flatten(-2)], dim=-1))
        tgt = self.ego_pose_memory(tgt, motion)
        query_pos = self.ego_pose_pe(query_pos, motion)
        memory_motion = nerf_encoding(torch.cat([self.memory_velo, self.memory_timestamp, self.memory_egopose[..., :3, :].flatten(-2)], dim=-1))
        temp_pos = self.ego_pose_pe(temp_pos, memory_motion)
        temp_memory = self.ego_pose_memory(self.memory_embedding, memory_motion)
        query_pos = query_pos + self.time_embedding(pos2posemb1d(torch.zeros_like(reference_points[..., :1])))
        temp_pos = temp_pos + self.time_embedding(pos2posemb1d(self.memory_timestamp))
        p = self.num_propagated
        return (torch.cat([tgt, temp_memory[:, :p]], dim=1), torch.cat([query_pos, temp_pos[:, :p]], dim=1), torch.cat([reference_points, temp_reference[:, :p]], dim=1),
                temp_memory[:, p:], temp_pos[:, p:])

    def _update_memory(self, scores, boxes, output):
        best = scores.sigmoid().max(dim=-1).values
        index = best.topk(self.topk, dim=1).indices

        def gather(x):
            return torch.gather(x, 1, index.view(1, -1, *[1] * (x.dim() - 2)).expand(1, -1, *x.shape[2:]))

        device = scores.device
        self.memory_embedding = torch.cat([gather(output).detach(), self.memory_embedding], dim=1)
        self.memory_timestamp = torch.cat([torch.zeros(1, self.topk, 1, device=device), self.memory_timestamp], dim=1)
        pose = torch.eye(4, device=device)[None, None].repeat(1, self.topk, 1, 1)
        self.memory_egopose = torch.cat([pose, self.memory_egopose], dim=1)
        self.memory_reference_point = torch.cat([gather(boxes[..., :3]).detach(), self.memory_reference_point], dim=1)
        self.memory_velo = torch.cat([gather(boxes[..., -2:]).detach(), self.memory_velo], dim=1)

    def forward(self, features, intrinsics, lidar2img, relative_pose, dt):
        self._prepare_memory(features.device, relative_pose, dt)
        n, channels, height, width = features.shape
        memory = features.permute(0, 2, 3, 1).reshape(1, n * height * width, channels)
        pos_embed, cone = self._position_embedding(height, width, intrinsics, lidar2img)
        memory = self.spatial_alignment(self.memory_embed(memory), cone)
        pos_embed = self.featurized_pe(pos_embed, memory)
        reference_points = self.reference_points.weight[None]
        query_pos = self.query_embedding(pos2posemb3d(reference_points))
        tgt, query_pos, reference_points, temp_memory, temp_pos = self._temporal_alignment(query_pos, torch.zeros_like(query_pos), reference_points)
        output = self.transformer.decoder(tgt.transpose(0, 1), memory.transpose(0, 1), memory.transpose(0, 1), query_pos.transpose(0, 1), pos_embed.transpose(0, 1),
                                          temp_memory.transpose(0, 1), temp_pos.transpose(0, 1)).transpose(0, 1)
        output = torch.nan_to_num(output)
        scores = self.cls_branches[-1](output)
        boxes = self.reg_branches[-1](output)
        boxes = torch.cat([(boxes[..., :3] + inverse_sigmoid(reference_points)).sigmoid(), boxes[..., 3:]], dim=-1)
        low, high = self.pc_range[:3], self.pc_range[3:6]
        boxes = torch.cat([boxes[..., :3] * (high - low) + low, boxes[..., 3:]], dim=-1)
        self._update_memory(scores, boxes, output)
        return scores[0], boxes[0]


class StreamPETR(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_backbone = ResNet50()
        self.img_neck = CPFPN()
        self.pts_bbox_head = StreamPETRHead()

    def forward(self, images, intrinsics, lidar2img, relative_pose, dt):
        return self.pts_bbox_head(self.img_neck(*self.img_backbone(images)), intrinsics, lidar2img, relative_pose, dt)


@dataclass(frozen=True)
class LoadReport:
    sha256: str
    checkpoint_tensors: int
    loaded_tensors: int
    unused_tensors: list[str]
    missing: list[str]
    unexpected: list[str]


def load_checkpoint(model: StreamPETR, path: Path) -> LoadReport:
    """Strict, restricted load. Anything unexplained is an error, never a partial model."""
    if not path.is_file():
        raise ModelUnavailable(f"no StreamPETR checkpoint at {path}. See docs/project/DETECTOR_ASSESSMENT.md.")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise ModelUnavailable(f"checkpoint hash {digest} is not the recorded {CHECKPOINT_SHA256}")
    state = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
    unused = sorted(k for k in state if k.startswith(UNUSED_PREFIXES) or k in UNUSED_KEYS)
    used = {k: v for k, v in state.items() if k not in unused}
    result = model.load_state_dict(used, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise ModelUnavailable(f"checkpoint mismatch: missing {result.missing_keys[:5]}, unexpected {result.unexpected_keys[:5]}")
    return LoadReport(digest, len(state), len(used), unused, list(result.missing_keys), list(result.unexpected_keys))


def preprocess_image(rgb: np.ndarray) -> np.ndarray:
    """Official test transform for a 1600x900 RGB image, as a CHW float32 array."""
    if rgb.shape[:2] != (900, 1600):
        raise ValueError(f"expected a 900x1600 image, got {rgb.shape[:2]}")
    resized = Image.fromarray(rgb).resize((INPUT_WIDTH, RESIZED_HEIGHT), Image.Resampling.BICUBIC)
    top = RESIZED_HEIGHT - INPUT_HEIGHT
    array = np.asarray(resized.crop((0, top, INPUT_WIDTH, RESIZED_HEIGHT)), np.float32)
    return ((array - IMAGE_MEAN) / IMAGE_STD).transpose(2, 0, 1)


def adjust_intrinsics(K: np.ndarray) -> np.ndarray:
    """4x4 intrinsics after the official resize and crop."""
    ida = np.eye(4)
    ida[0, 0] = ida[1, 1] = INPUT_WIDTH / 1600
    ida[1, 2] = -(RESIZED_HEIGHT - INPUT_HEIGHT)
    pad = np.eye(4)
    pad[:3, :3] = K
    return ida @ pad


@dataclass
class Boxes:
    """Decoded boxes in the LIDAR_TOP frame. ``scores`` are uncalibrated network scores."""

    center: np.ndarray
    size: np.ndarray
    yaw: np.ndarray
    velocity: np.ndarray
    scores: np.ndarray
    labels: np.ndarray


def decode(scores: torch.Tensor, boxes: torch.Tensor, max_num: int = 300, limit: float = 61.2, z_limit: float = 10.0) -> Boxes:
    probabilities = scores.sigmoid().view(-1)
    values, index = probabilities.topk(max_num)
    labels = index % 10
    chosen = boxes[torch.div(index, 10, rounding_mode="floor")]
    center = chosen[:, :3]
    keep = (center[:, 0].abs() <= limit) & (center[:, 1].abs() <= limit) & (center[:, 2].abs() <= z_limit)
    chosen, values, labels = chosen[keep].float().cpu().numpy(), values[keep].float().cpu().numpy(), labels[keep].cpu().numpy()
    return Boxes(chosen[:, :3], np.exp(chosen[:, 3:6]), np.arctan2(chosen[:, 6], chosen[:, 7]), chosen[:, 8:10], values, labels)


class StreamPETRInference:
    """Stateful per scene. Call ``reset`` at each new scene."""

    name = "streampetr_r50_428q_nuimg"

    def __init__(self, checkpoint: str | Path, device: str | None = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = StreamPETR()
        self.report = load_checkpoint(self.model, Path(checkpoint))
        self.model.to(self.device).eval()
        self.reset()

    def reset(self) -> None:
        self.model.pts_bbox_head.reset_memory()
        self.previous_pose: np.ndarray | None = None
        self.previous_time: float | None = None

    @torch.inference_mode()
    def step(self, images: list[np.ndarray], intrinsics: np.ndarray, lidar2cam: np.ndarray, T_global_lidar: np.ndarray, timestamp: float) -> Boxes:
        """``images`` and matrices follow CAMERA_ORDER; ``intrinsics`` are 3x3 originals."""
        if len(images) != len(CAMERA_ORDER):
            raise ValueError(f"expected {len(CAMERA_ORDER)} images, got {len(images)}")
        if self.previous_time is not None and timestamp <= self.previous_time:
            raise ValueError(f"non-increasing timestamp {timestamp}")
        K = np.stack([adjust_intrinsics(k) for k in intrinsics])
        lidar2img = K @ lidar2cam
        relative, dt = None, 0.0
        if self.previous_pose is not None:
            relative = torch.from_numpy(np.linalg.inv(T_global_lidar) @ self.previous_pose).float().to(self.device)
            dt = timestamp - self.previous_time
        batch = torch.from_numpy(np.stack([preprocess_image(i) for i in images])).to(self.device)
        scores, boxes = self.model(batch, torch.from_numpy(K).float().to(self.device), torch.from_numpy(lidar2img).float().to(self.device), relative, dt)
        self.previous_pose, self.previous_time = T_global_lidar.copy(), timestamp
        return decode(scores, boxes)
