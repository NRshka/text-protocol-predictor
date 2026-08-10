"""Oriented-box and cubic-Bézier geometry metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch


@dataclass(frozen=True)
class AuxiliaryGeometryMetrics:
    instance_count: int = 0
    mode_correct_count: int = 0
    oriented_box_iou_sum: float = 0.0
    angle_absolute_error_sum: float = 0.0
    bezier_count: int = 0
    bezier_centerline_error_sum: float = 0.0


def oriented_box_corners(geometry: Any) -> list[tuple[float, float]]:
    box = geometry.box
    center_x = float(box.x) + float(box.width) / 2
    center_y = float(box.y) + float(box.height) / 2
    radians = math.radians(float(geometry.rotation_degrees))
    cosine = math.cos(radians)
    sine = math.sin(radians)
    corners = []
    for local_x, local_y in (
        (-float(box.width) / 2, -float(box.height) / 2),
        (float(box.width) / 2, -float(box.height) / 2),
        (float(box.width) / 2, float(box.height) / 2),
        (-float(box.width) / 2, float(box.height) / 2),
    ):
        corners.append(
            (
                center_x + local_x * cosine - local_y * sine,
                center_y + local_x * sine + local_y * cosine,
            )
        )
    return corners


def _polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(
        sum(
            x1 * y2 - y1 * x2
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1], strict=True)
        )
    ) / 2


def _inside(point: tuple[float, float], edge_start: tuple[float, float], edge_end: tuple[float, float]) -> bool:
    return (
        (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1])
        - (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])
    ) >= -1.0e-9


def _intersection(
    start: tuple[float, float],
    end: tuple[float, float],
    clip_start: tuple[float, float],
    clip_end: tuple[float, float],
) -> tuple[float, float]:
    x1, y1 = start
    x2, y2 = end
    x3, y3 = clip_start
    x4, y4 = clip_end
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1.0e-12:
        return end
    determinant1 = x1 * y2 - y1 * x2
    determinant2 = x3 * y4 - y3 * x4
    return (
        (determinant1 * (x3 - x4) - (x1 - x2) * determinant2) / denominator,
        (determinant1 * (y3 - y4) - (y1 - y2) * determinant2) / denominator,
    )


def _clip_polygon(
    subject: list[tuple[float, float]], clip: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    output = subject
    for clip_start, clip_end in zip(clip, clip[1:] + clip[:1], strict=True):
        input_points = output
        output = []
        if not input_points:
            break
        previous = input_points[-1]
        for current in input_points:
            if _inside(current, clip_start, clip_end):
                if not _inside(previous, clip_start, clip_end):
                    output.append(_intersection(previous, current, clip_start, clip_end))
                output.append(current)
            elif _inside(previous, clip_start, clip_end):
                output.append(_intersection(previous, current, clip_start, clip_end))
            previous = current
    return output


def oriented_box_iou(left: Any, right: Any) -> float:
    left_polygon = oriented_box_corners(left)
    right_polygon = oriented_box_corners(right)
    intersection = _polygon_area(_clip_polygon(left_polygon, right_polygon))
    union = _polygon_area(left_polygon) + _polygon_area(right_polygon) - intersection
    return float(intersection / union if union > 0 else 0.0)


def angle_error_degrees(left: float, right: float) -> float:
    difference = abs((float(left) - float(right)) % 360.0)
    return float(min(difference, 360.0 - difference))


def sample_bezier(baseline: Any, samples: int = 129) -> list[tuple[float, float]]:
    points = (baseline.p0, baseline.p1, baseline.p2, baseline.p3)
    result = []
    for index in range(samples):
        t = index / (samples - 1)
        u = 1.0 - t
        coefficients = (u**3, 3 * u * u * t, 3 * u * t * t, t**3)
        result.append(
            tuple(
                sum(coefficient * float(getattr(point, axis)) for coefficient, point in zip(coefficients, points, strict=True))
                for axis in ("x", "y")
            )
        )
    return result


def bezier_centerline_error(left: Any, right: Any, *, samples: int = 129) -> float:
    left_points = sample_bezier(left, samples)
    right_points = sample_bezier(right, samples)
    return float(
        sum(math.hypot(lx - rx, ly - ry) for (lx, ly), (rx, ry) in zip(left_points, right_points, strict=True))
        / samples
    )


def evaluate_auxiliary_geometry(
    prediction: dict[str, torch.Tensor],
    *,
    object_ids: tuple[str, ...],
    target_protocol: Any,
) -> AuxiliaryGeometryMetrics:
    """Compare contextual-query geometry with target objects by semantic ID."""
    target_by_id = {
        obj.id: obj
        for obj in target_protocol.objects
        if hasattr(obj, "text")
    }
    count = len(object_ids)
    required = ("mode_logits", "boxes", "rotations", "beziers")
    if any(prediction[name].shape[0] != count for name in required):
        raise ValueError("auxiliary geometry predictions do not match object IDs")
    canvas_width = float(target_protocol.canvas.width)
    canvas_height = float(target_protocol.canvas.height)
    mode_correct = 0
    box_iou_sum = 0.0
    angle_sum = 0.0
    bezier_count = 0
    centerline_sum = 0.0
    compared = 0
    modes = prediction["mode_logits"].argmax(dim=-1).detach().cpu()
    boxes = prediction["boxes"].detach().float().cpu()
    rotations = prediction["rotations"].detach().float().cpu()
    beziers = prediction["beziers"].detach().float().cpu()
    for index, object_id in enumerate(object_ids):
        target = target_by_id.get(object_id)
        if target is None:
            continue
        compared += 1
        predicted_mode = "bezier" if int(modes[index].item()) == 1 else "straight"
        mode_correct += int(predicted_mode == target.geometry.mode)
        center_x, center_y, width, height = boxes[index].tolist()
        predicted_geometry = SimpleNamespace(
            box=SimpleNamespace(
                x=(center_x - width / 2) * canvas_width,
                y=(center_y - height / 2) * canvas_height,
                width=width * canvas_width,
                height=height * canvas_height,
            ),
            rotation_degrees=math.degrees(
                math.atan2(
                    float(rotations[index, 0]),
                    float(rotations[index, 1]),
                )
            ),
        )
        box_iou_sum += oriented_box_iou(predicted_geometry, target.geometry)
        angle_sum += angle_error_degrees(
            predicted_geometry.rotation_degrees,
            target.geometry.rotation_degrees,
        )
        if target.geometry.baseline is not None:
            values = beziers[index].tolist()
            predicted_baseline = SimpleNamespace(
                **{
                    f"p{point_index}": SimpleNamespace(
                        x=values[2 * point_index] * canvas_width,
                        y=values[2 * point_index + 1] * canvas_height,
                    )
                    for point_index in range(4)
                }
            )
            centerline_sum += bezier_centerline_error(
                predicted_baseline,
                target.geometry.baseline,
            )
            bezier_count += 1
    return AuxiliaryGeometryMetrics(
        instance_count=compared,
        mode_correct_count=mode_correct,
        oriented_box_iou_sum=box_iou_sum,
        angle_absolute_error_sum=angle_sum,
        bezier_count=bezier_count,
        bezier_centerline_error_sum=centerline_sum,
    )
