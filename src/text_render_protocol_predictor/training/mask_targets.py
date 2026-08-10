"""Dense instance-mask and continuous geometry target preparation."""

from __future__ import annotations

import math
from dataclasses import dataclass
import torch
from PIL import Image

from ..data.dataset import ProtocolDatasetRecord
from ..data.instance_masks import (
    load_mask_as_grayscale,
    mask_image_to_tensor,
    rasterize_protocol_instance_masks,
)


@dataclass(frozen=True)
class PreparedInstanceTargets:
    masks: torch.Tensor
    mask_supervision: torch.Tensor
    modes: torch.Tensor
    boxes: torch.Tensor
    rotations: torch.Tensor
    beziers: torch.Tensor


def _resize_mask(image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    if image.size != size:
        image = image.resize(size, Image.Resampling.BOX)
    return mask_image_to_tensor(image)


def prepare_instance_targets(
    record: ProtocolDatasetRecord,
    *,
    image_grid_thw: torch.Tensor,
    patch_size: int = 16,
    output_stride: int = 4,
) -> PreparedInstanceTargets:
    """Prepare targets aligned with one processor-resized Qwen image grid."""
    if image_grid_thw.numel() != 3:
        raise ValueError(f"expected one image grid triple, got {tuple(image_grid_thw.shape)}")
    temporal, grid_h, grid_w = (int(value) for value in image_grid_thw.tolist())
    if temporal != 1:
        raise ValueError("grounded mask training supports static images only")
    if patch_size % output_stride:
        raise ValueError("vision patch size must be divisible by mask output stride")
    scale = patch_size // output_stride
    target_size = (grid_w * scale, grid_h * scale)

    text_objects = [
        obj
        for obj in sorted(record.protocol.objects, key=lambda item: (item.z_order, item.id))
        if hasattr(obj, "text")
    ]
    if tuple(obj.id for obj in text_objects) != record.text_object_ids:
        raise RuntimeError("record text-object order differs from canonical target order")

    geometry_masks: dict[str, Image.Image] = {}
    if record.mask_supervision is not None and record.mask_supervision.source == "geometry":
        geometry_masks = rasterize_protocol_instance_masks(record.protocol)

    masks: list[torch.Tensor] = []
    supervised: list[bool] = []
    modes: list[int] = []
    boxes: list[list[float]] = []
    rotations: list[list[float]] = []
    beziers: list[list[float]] = []
    width = float(record.canvas_width)
    height = float(record.canvas_height)
    for obj in text_objects:
        mask_image: Image.Image | None = None
        declaration = record.mask_supervision
        if declaration is not None and declaration.source == "geometry":
            mask_image = geometry_masks[obj.id]
        elif declaration is not None and obj.id in declaration.object_paths:
            mask_image = load_mask_as_grayscale(declaration.object_paths[obj.id])
        masks.append(
            _resize_mask(mask_image, target_size)
            if mask_image is not None
            else torch.zeros((target_size[1], target_size[0]), dtype=torch.float32)
        )
        supervised.append(mask_image is not None)

        geometry = obj.geometry
        box = geometry.box
        modes.append(1 if geometry.mode == "bezier" else 0)
        boxes.append(
            [
                (float(box.x) + float(box.width) / 2) / width,
                (float(box.y) + float(box.height) / 2) / height,
                float(box.width) / width,
                float(box.height) / height,
            ]
        )
        radians = math.radians(float(geometry.rotation_degrees))
        rotations.append([math.sin(radians), math.cos(radians)])
        baseline = geometry.baseline
        if baseline is None:
            beziers.append([0.0] * 8)
        else:
            values: list[float] = []
            for point in (baseline.p0, baseline.p1, baseline.p2, baseline.p3):
                values.extend((float(point.x) / width, float(point.y) / height))
            beziers.append(values)

    count = len(text_objects)
    return PreparedInstanceTargets(
        masks=(
            torch.stack(masks)
            if masks
            else torch.empty((0, target_size[1], target_size[0]), dtype=torch.float32)
        ),
        mask_supervision=torch.tensor(supervised, dtype=torch.bool),
        modes=torch.tensor(modes, dtype=torch.long),
        boxes=torch.tensor(boxes, dtype=torch.float32).reshape(count, 4),
        rotations=torch.tensor(rotations, dtype=torch.float32).reshape(count, 2),
        beziers=torch.tensor(beziers, dtype=torch.float32).reshape(count, 8),
    )
