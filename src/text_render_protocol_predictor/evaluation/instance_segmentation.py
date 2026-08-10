"""Instance-mask quality and attached-versus-oracle association metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class InstanceSegmentationMetrics:
    target_count: int
    prediction_count: int
    attached_iou: float
    oracle_iou: float
    association_gap: float
    dice: float
    boundary_fscore: float
    ap50: float
    ap75: float


def mask_iou(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.bool()
    right = right.bool()
    intersection = left.logical_and(right).sum().item()
    union = left.logical_or(right).sum().item()
    return float(intersection / union if union else 1.0)


def mask_dice(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.bool()
    right = right.bool()
    intersection = left.logical_and(right).sum().item()
    denominator = left.sum().item() + right.sum().item()
    return float(2.0 * intersection / denominator if denominator else 1.0)


def boundary_fscore(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    tolerance: int = 2,
) -> float:
    if tolerance < 0:
        raise ValueError("boundary tolerance must be non-negative")

    def boundary(mask: torch.Tensor) -> torch.Tensor:
        value = mask.float().unsqueeze(0).unsqueeze(0)
        eroded = -F.max_pool2d(-value, kernel_size=3, stride=1, padding=1)
        return value.bool().logical_and(~eroded.bool()).squeeze(0).squeeze(0)

    predicted_boundary = boundary(predicted.bool())
    target_boundary = boundary(target.bool())
    if not predicted_boundary.any() and not target_boundary.any():
        return 1.0
    kernel = 2 * tolerance + 1

    def dilate(mask: torch.Tensor) -> torch.Tensor:
        if tolerance == 0:
            return mask
        return (
            F.max_pool2d(
                mask.float().unsqueeze(0).unsqueeze(0),
                kernel_size=kernel,
                stride=1,
                padding=tolerance,
            )
            .bool()
            .squeeze(0)
            .squeeze(0)
        )

    target_dilated = dilate(target_boundary)
    predicted_dilated = dilate(predicted_boundary)
    precision = (
        predicted_boundary.logical_and(target_dilated).sum().item()
        / predicted_boundary.sum().item()
        if predicted_boundary.any()
        else 0.0
    )
    recall = (
        target_boundary.logical_and(predicted_dilated).sum().item()
        / target_boundary.sum().item()
        if target_boundary.any()
        else 0.0
    )
    return float(2 * precision * recall / (precision + recall) if precision + recall else 0.0)


def _minimum_cost_assignment(cost: list[list[float]]) -> list[tuple[int, int]]:
    """Rectangular Hungarian assignment in O(n^3), without SciPy."""
    if not cost or not cost[0]:
        return []
    rows = len(cost)
    columns = len(cost[0])
    transposed = rows > columns
    matrix = [list(row) for row in cost]
    if transposed:
        matrix = [[cost[row][column] for row in range(rows)] for column in range(columns)]
        rows, columns = columns, rows

    u = [0.0] * (rows + 1)
    v = [0.0] * (columns + 1)
    p = [0] * (columns + 1)
    way = [0] * (columns + 1)
    for row in range(1, rows + 1):
        p[0] = row
        column0 = 0
        minimum = [float("inf")] * (columns + 1)
        used = [False] * (columns + 1)
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = float("inf")
            column1 = 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                current = matrix[row0 - 1][column - 1] - u[row0] - v[column]
                if current < minimum[column]:
                    minimum[column] = current
                    way[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(columns + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    pairs = [(p[column] - 1, column - 1) for column in range(1, columns + 1) if p[column]]
    if transposed:
        return [(column, row) for row, column in pairs]
    return pairs


def _average_precision(
    predictions: list[tuple[str, torch.Tensor, float]],
    targets: Mapping[str, torch.Tensor],
    *,
    threshold: float,
) -> float:
    if not targets:
        return 1.0 if not predictions else 0.0
    ordered = sorted(predictions, key=lambda item: item[2], reverse=True)
    unmatched = set(targets)
    true_positives: list[int] = []
    false_positives: list[int] = []
    for _object_id, predicted, _score in ordered:
        candidates = [
            (mask_iou(predicted, targets[target_id]), target_id)
            for target_id in unmatched
        ]
        best_iou, best_id = max(candidates, default=(0.0, None))
        matched = best_id is not None and best_iou >= threshold
        true_positives.append(int(matched))
        false_positives.append(int(not matched))
        if matched:
            unmatched.remove(best_id)
    if not ordered:
        return 0.0
    recalls: list[float] = []
    precisions: list[float] = []
    tp = fp = 0
    for positive, negative in zip(true_positives, false_positives, strict=True):
        tp += positive
        fp += negative
        recalls.append(tp / len(targets))
        precisions.append(tp / (tp + fp))
    return float(
        sum(
            max(
                (precision for recall, precision in zip(recalls, precisions, strict=True) if recall >= level),
                default=0.0,
            )
            for level in (index / 100 for index in range(101))
        )
        / 101
    )


def evaluate_instance_segmentation(
    predicted: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    *,
    scores: Mapping[str, float] | None = None,
    threshold: float = 0.5,
    boundary_tolerance: int = 2,
) -> InstanceSegmentationMetrics:
    """Evaluate ID-attached masks and mask-only oracle association."""
    target_ids = list(target)
    predicted_ids = list(predicted)
    target_count = len(target_ids)
    attached_ious: list[float] = []
    dices: list[float] = []
    boundaries: list[float] = []
    for object_id in target_ids:
        target_mask = target[object_id].ge(threshold)
        predicted_mask = predicted.get(object_id)
        if predicted_mask is None:
            attached_ious.append(0.0)
            dices.append(0.0)
            boundaries.append(0.0)
            continue
        predicted_mask = predicted_mask.ge(threshold)
        attached_ious.append(mask_iou(predicted_mask, target_mask))
        dices.append(mask_dice(predicted_mask, target_mask))
        boundaries.append(
            boundary_fscore(
                predicted_mask,
                target_mask,
                tolerance=boundary_tolerance,
            )
        )

    pairwise = [
        [
            1.0
            - mask_iou(predicted[predicted_id].ge(threshold), target[target_id].ge(threshold))
            for target_id in target_ids
        ]
        for predicted_id in predicted_ids
    ]
    oracle_sum = sum(1.0 - pairwise[row][column] for row, column in _minimum_cost_assignment(pairwise))
    attached_iou = sum(attached_ious) / target_count if target_count else float(not predicted)
    oracle_iou = oracle_sum / target_count if target_count else float(not predicted)
    scored_predictions = [
        (
            object_id,
            predicted[object_id].ge(threshold),
            float(scores.get(object_id, 1.0) if scores else 1.0),
        )
        for object_id in predicted_ids
    ]
    binary_targets = {
        object_id: target[object_id].ge(threshold) for object_id in target_ids
    }
    return InstanceSegmentationMetrics(
        target_count=target_count,
        prediction_count=len(predicted),
        attached_iou=float(attached_iou),
        oracle_iou=float(oracle_iou),
        association_gap=float(max(0.0, oracle_iou - attached_iou)),
        dice=float(sum(dices) / target_count if target_count else float(not predicted)),
        boundary_fscore=float(
            sum(boundaries) / target_count if target_count else float(not predicted)
        ),
        ap50=_average_precision(scored_predictions, binary_targets, threshold=0.50),
        ap75=_average_precision(scored_predictions, binary_targets, threshold=0.75),
    )


def evaluate_instance_slices(
    predicted: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    *,
    scores: Mapping[str, float] | None,
    record: Any,
    small_area_fraction: float = 0.01,
    rotated_degrees: float = 1.0,
    crowded_instances: int = 5,
) -> dict[str, InstanceSegmentationMetrics]:
    """Evaluate deterministic source, geometry, and difficulty slices."""
    declaration = record.mask_supervision
    source = "real" if declaration.source == "files" else "synthetic"
    objects = {
        obj.id: obj
        for obj in record.protocol.objects
        if hasattr(obj, "text") and obj.id in target
    }
    target_binary = {object_id: mask.ge(0.5) for object_id, mask in target.items()}
    overlapping: set[str] = set()
    target_ids = list(target_binary)
    for left_index, left_id in enumerate(target_ids):
        for right_id in target_ids[left_index + 1 :]:
            if target_binary[left_id].logical_and(target_binary[right_id]).any():
                overlapping.update((left_id, right_id))
    canvas_area = float(record.canvas_width * record.canvas_height)
    ids_by_slice: dict[str, set[str]] = {f"source/{source}": set(target)}
    for object_id, obj in objects.items():
        ids_by_slice.setdefault(f"mode/{obj.geometry.mode}", set()).add(object_id)
        box = obj.geometry.box
        if float(box.width * box.height) / canvas_area <= small_area_fraction:
            ids_by_slice.setdefault("challenge/small", set()).add(object_id)
        angle = abs((float(obj.geometry.rotation_degrees) + 180.0) % 360.0 - 180.0)
        if angle >= rotated_degrees:
            ids_by_slice.setdefault("challenge/rotated", set()).add(object_id)
        if object_id in overlapping:
            ids_by_slice.setdefault("challenge/overlapping", set()).add(object_id)
    if len(objects) >= crowded_instances:
        ids_by_slice["challenge/crowded"] = set(target)

    results: dict[str, InstanceSegmentationMetrics] = {}
    for name, object_ids in ids_by_slice.items():
        if not object_ids:
            continue
        predicted_subset = {
            object_id: predicted[object_id]
            for object_id in object_ids
            if object_id in predicted
        }
        target_subset = {object_id: target[object_id] for object_id in object_ids}
        score_subset = (
            {
                object_id: scores[object_id]
                for object_id in predicted_subset
                if object_id in scores
            }
            if scores is not None
            else None
        )
        results[name] = evaluate_instance_segmentation(
            predicted_subset,
            target_subset,
            scores=score_subset,
        )
    return results
