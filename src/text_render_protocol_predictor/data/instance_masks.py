"""Per-object filled-slot mask loading and protocol rasterization."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw


SUPPORTED_MASK_MODES = frozenset({"1", "L", "I", "I;16"})


def _paste_rotated_box(canvas: Image.Image, geometry: Any) -> None:
    box = geometry.box
    width = max(1, math.ceil(float(box.width)))
    height = max(1, math.ceil(float(box.height)))
    local = Image.new("L", (width, height), 255)
    rotation = float(geometry.rotation_degrees)
    if rotation:
        # STRP angles are clockwise; Pillow's positive angles are counter-clockwise.
        local = local.rotate(-rotation, resample=Image.Resampling.NEAREST, expand=True)
    center_x = float(box.x) + float(box.width) / 2
    center_y = float(box.y) + float(box.height) / 2
    canvas.paste(
        255,
        (round(center_x - local.width / 2), round(center_y - local.height / 2)),
        local,
    )


def bezier_points(baseline: Any, samples: int = 129) -> list[tuple[float, float]]:
    if samples < 2:
        raise ValueError("Bezier mask rasterization requires at least two samples")
    points = (baseline.p0, baseline.p1, baseline.p2, baseline.p3)
    result: list[tuple[float, float]] = []
    for index in range(samples):
        t = index / (samples - 1)
        u = 1.0 - t
        coefficients = (u**3, 3 * u * u * t, 3 * u * t * t, t**3)
        result.append(
            (
                sum(c * float(point.x) for c, point in zip(coefficients, points, strict=True)),
                sum(c * float(point.y) for c, point in zip(coefficients, points, strict=True)),
            )
        )
    return result


def rasterize_text_slot_mask(
    obj: Any,
    *,
    canvas_size: tuple[int, int],
    bezier_samples: int = 129,
) -> Image.Image:
    """Rasterize one filled text slot, independently of every other object."""
    canvas = Image.new("L", canvas_size, 0)
    geometry = obj.geometry
    if geometry.mode == "straight":
        _paste_rotated_box(canvas, geometry)
        return canvas
    if geometry.baseline is None:
        raise ValueError(f"Bezier object {obj.id!r} has no baseline")
    if bezier_samples < 129:
        raise ValueError("filled-slot Bezier masks require at least 129 samples")
    points = bezier_points(geometry.baseline, samples=bezier_samples)
    width = max(1, round(float(geometry.box.height)))
    radius = width / 2
    draw = ImageDraw.Draw(canvas)
    draw.line(points, fill=255, width=width, joint="curve")
    for x, y in (points[0], points[-1]):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    return canvas


def rasterize_protocol_instance_masks(
    protocol: Any,
    *,
    bezier_samples: int = 129,
) -> dict[str, Image.Image]:
    """Return one independent filled-slot mask for every protocol text object."""
    canvas_size = (int(protocol.canvas.width), int(protocol.canvas.height))
    return {
        obj.id: rasterize_text_slot_mask(
            obj,
            canvas_size=canvas_size,
            bezier_samples=bezier_samples,
        )
        for obj in sorted(protocol.objects, key=lambda item: (item.z_order, item.id))
        if hasattr(obj, "text")
    }


def validate_mask_file(
    path: str | Path,
    *,
    expected_size: tuple[int, int],
) -> None:
    """Validate a generator-supplied grayscale instance-mask file."""
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError(f"instance mask must be a lossless PNG: {path}")
    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError(f"instance mask must contain PNG data: {path}")
        if image.mode not in SUPPORTED_MASK_MODES:
            raise ValueError(
                f"instance mask {path} must be grayscale, got mode {image.mode!r}"
            )
        if image.size != expected_size:
            raise ValueError(
                f"instance mask {path} size {image.size} does not match canvas "
                f"{expected_size}"
            )
        grayscale = image.convert("L")
        extrema = grayscale.getextrema()
        if extrema is None or extrema[1] == 0:
            raise ValueError(f"instance mask {path} has no foreground pixels")


def load_mask_as_grayscale(path: str | Path) -> Image.Image:
    """Load a mask without retaining an open file descriptor."""
    with Image.open(path) as image:
        return image.convert("L").copy()


def mask_image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a grayscale mask to a CPU float tensor in ``[0, 1]``."""
    grayscale = image.convert("L")
    values = torch.tensor(bytearray(grayscale.tobytes()), dtype=torch.uint8)
    return values.reshape(grayscale.height, grayscale.width).float().div_(255.0)


def load_record_instance_masks(record: Any) -> dict[str, torch.Tensor]:
    """Load only the supervised original-canvas instance masks for a record."""
    declaration = record.mask_supervision
    if declaration is None:
        return {}
    if declaration.source == "geometry":
        images = rasterize_protocol_instance_masks(record.protocol)
    else:
        images = {
            object_id: load_mask_as_grayscale(path)
            for object_id, path in declaration.object_paths.items()
        }
    return {object_id: mask_image_to_tensor(image) for object_id, image in images.items()}
