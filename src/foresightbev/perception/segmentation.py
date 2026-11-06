"""Image-to-labels stage. The pipeline consumes SemanticClass labels only.

``SegFormerSegmenter`` is a pretrained Cityscapes model applied to another dataset.
It is never retrained here, and its domain gap is measured, not assumed away. It
loads only from a local directory, so nothing downloads at runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from foresightbev.contracts import CameraFrame, CameraStatus, FrameBundle, Provenance, SemanticClass as C
from foresightbev.errors import ModelUnavailable

# Cityscapes train ids 0-18 mapped to SemanticClass.
CITYSCAPES_TO_CLASS = np.array([C.ROAD, C.SIDEWALK, C.STATIC, C.STATIC, C.STATIC, C.STATIC, C.STATIC, C.STATIC, C.STATIC,
                                C.OTHER_GROUND, C.SKY, C.PERSON, C.PERSON, C.VEHICLE, C.VEHICLE, C.VEHICLE, C.VEHICLE,
                                C.VEHICLE, C.VEHICLE], np.uint8)


class Segmenter(Protocol):
    name: str
    provenance: Provenance

    def segment(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """RGB uint8 HxWx3 images to same-sized SemanticClass label arrays."""
        ...


class SegFormerSegmenter:
    name = "segformer_b2_cityscapes"
    provenance = Provenance.PRETRAINED

    def __init__(self, model_dir: str | Path, device: str | None = None, batch_size: int = 6):
        path = Path(model_dir)
        if not (path / "config.json").is_file():
            raise ModelUnavailable(f"no segmentation model at {path}. See docs/project/RUN_THE_SYSTEM.md.")
        try:
            import torch
            from transformers import SegformerForSemanticSegmentation
        except ImportError as error:
            raise ModelUnavailable(f"PyTorch and transformers are required: {error}. Run: uv sync --group ml") from error
        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.batch_size = batch_size
        self.model = SegformerForSemanticSegmentation.from_pretrained(path, local_files_only=True, use_safetensors=False).to(self.device).eval()
        config = _preprocessor(path)
        self.mean = np.array(config["image_mean"], np.float32)
        self.std = np.array(config["image_std"], np.float32)

    def segment(self, images: list[np.ndarray]) -> list[np.ndarray]:
        torch = self.torch
        out: list[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            batch = images[start:start + self.batch_size]
            array = (np.stack(batch).astype(np.float32) / 255.0 - self.mean) / self.std
            tensor = torch.from_numpy(array).permute(0, 3, 1, 2).to(self.device)
            with torch.inference_mode():
                logits = self.model(pixel_values=tensor).logits
                logits = torch.nn.functional.interpolate(logits, size=batch[0].shape[:2], mode="bilinear", align_corners=False)
            ids = logits.argmax(1).cpu().numpy()
            out.extend(CITYSCAPES_TO_CLASS[ids])
        return out


def _preprocessor(path: Path) -> dict:
    import json

    return json.loads((path / "preprocessor_config.json").read_text())


def apply_segmentation(bundle: FrameBundle, segmenter: Segmenter) -> FrameBundle:
    """Fill labels for every camera that has an image. Lost cameras stay lost."""
    names = [n for n, f in bundle.frames.items() if f.image is not None]
    labels = segmenter.segment([bundle.frames[n].image for n in names]) if names else []
    frames = dict(bundle.frames)
    for name, array in zip(names, labels):
        frame = bundle.frames[name]
        frames[name] = CameraFrame(name, frame.timestamp, array, frame.status, image=frame.image)
    return FrameBundle(bundle.sequence_id, bundle.frame_index, bundle.timestamp, frames, bundle.calibration, bundle.ego, bundle.source)


def resize_labels(labels: np.ndarray, width: int, height: int) -> np.ndarray:
    """Nearest-neighbour resize, so class ids are never blended."""
    return cv2.resize(labels, (width, height), interpolation=cv2.INTER_NEAREST)


def degrade_check(frame: CameraFrame) -> CameraStatus:
    """A uniform image is treated as a failed camera rather than as an observation."""
    if frame.image is None:
        return CameraStatus.DROPPED
    return CameraStatus.DROPPED if float(frame.image.std()) < 1.0 else frame.status
