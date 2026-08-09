"""Union-mask rasterization and overlap metrics for coarse text layout."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


@dataclass(frozen=True)
class LayoutMaskMetrics:
    """Strict and dilation-tolerant overlap for one predicted protocol."""

    strict_iou: float
    dilated_iou: float
    precision: float
    recall: float
    target_pixels: int
    predicted_pixels: int


def threshold_layout_mask(mask: Image.Image, *, threshold: float) -> np.ndarray:
    """Threshold an arbitrary mask into a two-dimensional boolean array."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    grayscale = np.asarray(mask.convert("L"), dtype=np.uint8)
    return grayscale >= round(threshold * 255)


def dilate_layout_mask(
    mask: np.ndarray,
    *,
    kernel_size: int,
    iterations: int,
) -> np.ndarray:
    """Apply repeated square binary dilation using Pillow's maximum filter."""
    if mask.ndim != 2:
        raise ValueError(f"layout mask must be two-dimensional, got {mask.shape}")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("layout dilation kernel_size must be a positive odd integer")
    if iterations < 0:
        raise ValueError("layout dilation iterations must be non-negative")
    result = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")
    for _ in range(iterations):
        result = result.filter(ImageFilter.MaxFilter(kernel_size))
    return np.asarray(result, dtype=np.uint8) > 0


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
    left = round(center_x - local.width / 2)
    top = round(center_y - local.height / 2)
    canvas.paste(255, (left, top), local)


def _bezier_points(baseline: Any, samples: int) -> list[tuple[float, float]]:
    points = (baseline.p0, baseline.p1, baseline.p2, baseline.p3)
    result: list[tuple[float, float]] = []
    for index in range(samples):
        t = index / (samples - 1)
        one_minus_t = 1.0 - t
        coefficients = (
            one_minus_t**3,
            3 * one_minus_t**2 * t,
            3 * one_minus_t * t**2,
            t**3,
        )
        result.append(
            (
                sum(
                    coefficient * float(point.x)
                    for coefficient, point in zip(coefficients, points, strict=True)
                ),
                sum(
                    coefficient * float(point.y)
                    for coefficient, point in zip(coefficients, points, strict=True)
                ),
            )
        )
    return result


def _draw_bezier_band(canvas: Image.Image, geometry: Any, *, samples: int) -> None:
    baseline = geometry.baseline
    if baseline is None:
        return
    points = _bezier_points(baseline, samples)
    width = max(1, round(float(geometry.box.height)))
    draw = ImageDraw.Draw(canvas)
    draw.line(points, fill=255, width=width, joint="curve")
    radius = width / 2
    for x, y in (points[0], points[-1]):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)


def rasterize_protocol_layout_mask(
    prediction: Any,
    *,
    canvas_size: tuple[int, int],
    bezier_samples: int = 129,
) -> np.ndarray:
    """Rasterize the union of text boxes and curved baseline bands."""
    if bezier_samples < 2:
        raise ValueError("bezier_samples must be at least two")
    canvas = Image.new("L", canvas_size, 0)
    for obj in prediction.objects:
        # Protocol 2.x can include shapes; layout reward concerns text only.
        if not hasattr(obj, "text"):
            continue
        geometry = obj.geometry
        if geometry.mode == "bezier":
            _draw_bezier_band(canvas, geometry, samples=bezier_samples)
        else:
            _paste_rotated_box(canvas, geometry)
    return np.asarray(canvas, dtype=np.uint8) > 0


def calculate_layout_mask_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    dilation_kernel_size: int,
    dilation_iterations: int,
    dilated_target: np.ndarray | None = None,
) -> LayoutMaskMetrics:
    """Calculate strict IoU and IoU/precision/recall after symmetric dilation."""
    predicted = np.asarray(predicted, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError(
            f"predicted and target layout masks must have equal 2-D shapes, got "
            f"{predicted.shape} and {target.shape}"
        )

    strict_intersection = int(np.logical_and(predicted, target).sum())
    strict_union = int(np.logical_or(predicted, target).sum())
    strict_iou = strict_intersection / strict_union if strict_union else 1.0

    dilated_prediction = dilate_layout_mask(
        predicted,
        kernel_size=dilation_kernel_size,
        iterations=dilation_iterations,
    )
    if dilated_target is None:
        dilated_target = dilate_layout_mask(
            target,
            kernel_size=dilation_kernel_size,
            iterations=dilation_iterations,
        )
    else:
        dilated_target = np.asarray(dilated_target, dtype=bool)
        if dilated_target.shape != target.shape:
            raise ValueError(
                f"dilated target shape {dilated_target.shape} does not match "
                f"target {target.shape}"
            )
    intersection = int(np.logical_and(dilated_prediction, dilated_target).sum())
    union = int(np.logical_or(dilated_prediction, dilated_target).sum())
    predicted_pixels = int(dilated_prediction.sum())
    target_pixels = int(dilated_target.sum())
    return LayoutMaskMetrics(
        strict_iou=float(strict_iou),
        dilated_iou=float(intersection / union if union else 1.0),
        precision=float(intersection / predicted_pixels if predicted_pixels else 0.0),
        recall=float(intersection / target_pixels if target_pixels else 0.0),
        target_pixels=target_pixels,
        predicted_pixels=predicted_pixels,
    )
