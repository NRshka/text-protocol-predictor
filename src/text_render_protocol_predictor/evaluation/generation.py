"""Strict validity and task metrics for autoregressive generations."""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field

from pydantic import ValidationError

from .geometry import angle_error_degrees, bezier_centerline_error, oriented_box_iou
from ..protocol.schema import (
    BoundingBox,
    LinearGradientFill,
    PredictionObjectV1,
    PredictionProtocol,
    PredictionTextObjectV2,
    SolidFill,
    UnsupportedProtocolVersion,
)


@dataclass(frozen=True)
class GenerationValidityMetrics:
    evaluated_count: int
    valid_json_count: int
    schema_valid_count: int
    ground_truth_object_count: int = 0
    ground_truth_text_object_count: int = 0
    semantic_id_true_positive_count: int = 0
    semantic_id_false_positive_count: int = 0
    semantic_id_false_negative_count: int = 0
    semantic_id_exact_match_count: int = 0
    box_iou_sum: float = 0.0
    font_correct_count: int = 0
    character_error_count: int = 0
    reference_character_count: int = 0
    word_error_count: int = 0
    reference_word_count: int = 0
    bezier_squared_error_sum: float = 0.0
    bezier_coordinate_count: int = 0
    color_absolute_error_sum: float = 0.0
    color_channel_count: int = 0
    has_ground_truth: bool = False
    grounding_evaluated_count: int = 0
    grounding_valid_count: int = 0
    mask_evaluated_count: int = 0
    mask_target_count: int = 0
    mask_prediction_count: int = 0
    mask_attached_iou_sum: float = 0.0
    mask_oracle_iou_sum: float = 0.0
    mask_dice_sum: float = 0.0
    mask_boundary_fscore_sum: float = 0.0
    mask_ap50_sum: float = 0.0
    mask_ap75_sum: float = 0.0
    oriented_box_iou_sum: float = 0.0
    oriented_box_count: int = 0
    angle_absolute_error_sum: float = 0.0
    angle_count: int = 0
    geometry_mode_correct_count: int = 0
    geometry_mode_count: int = 0
    bezier_centerline_error_sum: float = 0.0
    bezier_centerline_count: int = 0
    serialized_geometry_mask_iou_sum: float = 0.0
    serialized_geometry_mask_count: int = 0
    auxiliary_geometry_count: int = 0
    auxiliary_geometry_mode_correct_count: int = 0
    auxiliary_oriented_box_iou_sum: float = 0.0
    auxiliary_angle_absolute_error_sum: float = 0.0
    auxiliary_bezier_count: int = 0
    auxiliary_bezier_centerline_error_sum: float = 0.0
    grounding_slice_metrics: dict[str, int | float] = field(default_factory=dict)
    generation_latency_seconds_sum: float = 0.0
    generation_latency_count: int = 0
    peak_memory_bytes: int = 0

    @property
    def valid_json_percent(self) -> float:
        if self.evaluated_count == 0:
            return 0.0
        return 100.0 * self.valid_json_count / self.evaluated_count

    @property
    def schema_valid_percent(self) -> float:
        if self.evaluated_count == 0:
            return 0.0
        return 100.0 * self.schema_valid_count / self.evaluated_count

    @property
    def grounding_valid_percent(self) -> float:
        if self.grounding_evaluated_count == 0:
            return 0.0
        return 100.0 * self.grounding_valid_count / self.grounding_evaluated_count

    @property
    def mask_attached_iou(self) -> float:
        if self.mask_target_count == 0:
            return 0.0
        return self.mask_attached_iou_sum / self.mask_target_count

    @property
    def mask_oracle_iou(self) -> float:
        if self.mask_target_count == 0:
            return 0.0
        return self.mask_oracle_iou_sum / self.mask_target_count

    @property
    def mask_association_gap(self) -> float:
        return max(0.0, self.mask_oracle_iou - self.mask_attached_iou)

    @property
    def oriented_box_iou(self) -> float:
        if self.oriented_box_count == 0:
            return 0.0
        return self.oriented_box_iou_sum / self.oriented_box_count

    @property
    def angle_mae(self) -> float:
        if self.angle_count == 0:
            return 0.0
        return self.angle_absolute_error_sum / self.angle_count

    @property
    def geometry_mode_accuracy(self) -> float:
        if self.geometry_mode_count == 0:
            return 0.0
        return self.geometry_mode_correct_count / self.geometry_mode_count

    @property
    def bezier_centerline_error(self) -> float:
        if self.bezier_centerline_count == 0:
            return 0.0
        return self.bezier_centerline_error_sum / self.bezier_centerline_count

    @property
    def serialized_geometry_mask_iou(self) -> float:
        if self.serialized_geometry_mask_count == 0:
            return 0.0
        return (
            self.serialized_geometry_mask_iou_sum
            / self.serialized_geometry_mask_count
        )

    @property
    def auxiliary_geometry_mode_accuracy(self) -> float:
        if self.auxiliary_geometry_count == 0:
            return 0.0
        return (
            self.auxiliary_geometry_mode_correct_count
            / self.auxiliary_geometry_count
        )

    @property
    def auxiliary_oriented_box_iou(self) -> float:
        if self.auxiliary_geometry_count == 0:
            return 0.0
        return self.auxiliary_oriented_box_iou_sum / self.auxiliary_geometry_count

    @property
    def auxiliary_angle_mae(self) -> float:
        if self.auxiliary_geometry_count == 0:
            return 0.0
        return (
            self.auxiliary_angle_absolute_error_sum
            / self.auxiliary_geometry_count
        )

    @property
    def auxiliary_bezier_centerline_error(self) -> float:
        if self.auxiliary_bezier_count == 0:
            return 0.0
        return (
            self.auxiliary_bezier_centerline_error_sum
            / self.auxiliary_bezier_count
        )

    @property
    def generation_latency_seconds(self) -> float:
        if self.generation_latency_count == 0:
            return 0.0
        return self.generation_latency_seconds_sum / self.generation_latency_count

    @property
    def box_iou(self) -> float:
        if self.ground_truth_object_count == 0:
            return 0.0
        return self.box_iou_sum / self.ground_truth_object_count

    @property
    def character_error_rate(self) -> float:
        if self.reference_character_count == 0:
            return 0.0
        return self.character_error_count / self.reference_character_count

    @property
    def word_error_rate(self) -> float:
        if self.reference_word_count == 0:
            return 0.0
        return self.word_error_count / self.reference_word_count

    @property
    def font_accuracy(self) -> float:
        if self.ground_truth_text_object_count == 0:
            return 0.0
        return self.font_correct_count / self.ground_truth_text_object_count

    @property
    def bezier_mse(self) -> float:
        """MSE over the eight cubic-Bezier control-point coordinates compared."""
        if self.bezier_coordinate_count == 0:
            return 0.0
        return self.bezier_squared_error_sum / self.bezier_coordinate_count

    @property
    def color_mae(self) -> float:
        """MAE over RGB fill channels, in the protocol's 0--255 color scale."""
        if self.color_channel_count == 0:
            return 0.0
        return self.color_absolute_error_sum / self.color_channel_count

    @property
    def semantic_id_precision(self) -> float:
        denominator = (
            self.semantic_id_true_positive_count + self.semantic_id_false_positive_count
        )
        if denominator:
            return self.semantic_id_true_positive_count / denominator
        return 1.0 if self.semantic_id_false_negative_count == 0 else 0.0

    @property
    def semantic_id_recall(self) -> float:
        denominator = (
            self.semantic_id_true_positive_count + self.semantic_id_false_negative_count
        )
        if denominator:
            return self.semantic_id_true_positive_count / denominator
        return 1.0 if self.semantic_id_false_positive_count == 0 else 0.0

    @property
    def semantic_id_exact_match(self) -> float:
        if self.evaluated_count == 0:
            return 0.0
        return self.semantic_id_exact_match_count / self.evaluated_count

    def as_log_dict(self) -> dict[str, int | float]:
        metrics: dict[str, int | float] = {
            "generation/evaluated_count": self.evaluated_count,
            "generation/valid_json_count": self.valid_json_count,
            "generation/schema_valid_count": self.schema_valid_count,
            "generation/valid_json_percent": self.valid_json_percent,
            "generation/schema_valid_percent": self.schema_valid_percent,
        }
        if self.has_ground_truth:
            metrics.update(
                {
                    "generation/ground_truth_object_count": self.ground_truth_object_count,
                    "generation/ground_truth_text_object_count": (
                        self.ground_truth_text_object_count
                    ),
                    "generation/box_iou": self.box_iou,
                    "generation/cer": self.character_error_rate,
                    "generation/wer": self.word_error_rate,
                    "generation/font_accuracy": self.font_accuracy,
                    "generation/bezier_mse": self.bezier_mse,
                    "generation/bezier_coordinate_count": self.bezier_coordinate_count,
                    "generation/color_mae": self.color_mae,
                    "generation/color_channel_count": self.color_channel_count,
                    "generation/oriented_box_iou": self.oriented_box_iou,
                    "generation/oriented_box_count": self.oriented_box_count,
                    "generation/angle_mae": self.angle_mae,
                    "generation/angle_count": self.angle_count,
                    "generation/geometry_mode_accuracy": self.geometry_mode_accuracy,
                    "generation/geometry_mode_count": self.geometry_mode_count,
                    "generation/bezier_centerline_error": self.bezier_centerline_error,
                    "generation/bezier_centerline_count": self.bezier_centerline_count,
                    "generation/semantic_id_precision": self.semantic_id_precision,
                    "generation/semantic_id_recall": self.semantic_id_recall,
                    "generation/semantic_id_exact_match": self.semantic_id_exact_match,
                    "generation/semantic_id_true_positive_count": (
                        self.semantic_id_true_positive_count
                    ),
                    "generation/semantic_id_false_positive_count": (
                        self.semantic_id_false_positive_count
                    ),
                    "generation/semantic_id_false_negative_count": (
                        self.semantic_id_false_negative_count
                    ),
                }
            )
        if self.grounding_evaluated_count:
            metrics.update(
                {
                    "generation/grounding_evaluated_count": self.grounding_evaluated_count,
                    "generation/grounding_valid_count": self.grounding_valid_count,
                    "generation/grounding_valid_percent": self.grounding_valid_percent,
                }
            )
        if self.mask_evaluated_count:
            image_denominator = self.mask_evaluated_count
            instance_denominator = self.mask_target_count
            metrics.update(
                {
                    "generation/mask_evaluated_count": image_denominator,
                    "generation/mask_target_count": self.mask_target_count,
                    "generation/mask_prediction_count": self.mask_prediction_count,
                    "generation/mask_attached_iou": self.mask_attached_iou,
                    "generation/mask_oracle_iou": self.mask_oracle_iou,
                    "generation/mask_association_gap": self.mask_association_gap,
                    "generation/mask_dice": (
                        self.mask_dice_sum / instance_denominator
                        if instance_denominator
                        else 0.0
                    ),
                    "generation/mask_boundary_fscore": (
                        self.mask_boundary_fscore_sum / instance_denominator
                        if instance_denominator
                        else 0.0
                    ),
                    "generation/mask_ap50": self.mask_ap50_sum / image_denominator,
                    "generation/mask_ap75": self.mask_ap75_sum / image_denominator,
                }
            )
        if self.serialized_geometry_mask_count:
            metrics.update(
                {
                    "generation/serialized_geometry_mask_iou": (
                        self.serialized_geometry_mask_iou
                    ),
                    "generation/serialized_geometry_mask_count": (
                        self.serialized_geometry_mask_count
                    ),
                }
            )
        if self.auxiliary_geometry_count:
            metrics.update(
                {
                    "generation/auxiliary_geometry_count": self.auxiliary_geometry_count,
                    "generation/auxiliary_geometry_mode_accuracy": (
                        self.auxiliary_geometry_mode_accuracy
                    ),
                    "generation/auxiliary_oriented_box_iou": (
                        self.auxiliary_oriented_box_iou
                    ),
                    "generation/auxiliary_angle_mae": self.auxiliary_angle_mae,
                    "generation/auxiliary_bezier_count": self.auxiliary_bezier_count,
                    "generation/auxiliary_bezier_centerline_error": (
                        self.auxiliary_bezier_centerline_error
                    ),
                }
            )
        metrics.update(self.grounding_slice_metrics)
        if self.generation_latency_count:
            metrics.update(
                {
                    "generation/latency_seconds_per_sample": (
                        self.generation_latency_seconds
                    ),
                    "generation/peak_memory_bytes": self.peak_memory_bytes,
                }
            )
        return metrics


def _edit_distance(reference: list[str], prediction: list[str]) -> int:
    """Return Levenshtein distance using memory linear in the shorter sequence."""
    if len(reference) < len(prediction):
        shorter, longer = reference, prediction
    else:
        shorter, longer = prediction, reference
    previous = list(range(len(shorter) + 1))
    for longer_index, longer_item in enumerate(longer, start=1):
        current = [longer_index]
        for shorter_index, shorter_item in enumerate(shorter, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[shorter_index] + 1,
                    previous[shorter_index - 1] + (shorter_item != longer_item),
                )
            )
        previous = current
    return previous[-1]


def _box_iou(left: BoundingBox, right: BoundingBox) -> float:
    intersection_width = max(
        0.0, min(left.x + left.width, right.x + right.width) - max(left.x, right.x)
    )
    intersection_height = max(
        0.0, min(left.y + left.height, right.y + right.height) - max(left.y, right.y)
    )
    intersection = intersection_width * intersection_height
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union > 0 else 0.0


def _rgb(color: str) -> tuple[int, int, int]:
    return (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))


def _fill_rgb_samples(
    fill: SolidFill | LinearGradientFill,
) -> list[tuple[int, int, int]]:
    """Return comparable RGB samples for a solid fill or a gradient.

    Solid fills are constant. Gradients are sampled at their declared stop
    colors. A pair is compared only when both fills expose the same number of
    samples; this avoids inventing a correspondence between unrelated stops.
    """
    if isinstance(fill, SolidFill):
        return [_rgb(fill.color)]
    return [_rgb(stop.color) for stop in fill.stops]


def _parse_prediction(output: str) -> tuple[bool, PredictionProtocol | None]:
    try:
        value = json.loads(output.strip())
    except (json.JSONDecodeError, TypeError):
        return False, None
    try:
        prediction = PredictionProtocol.model_validate(value)
    except (ValidationError, UnsupportedProtocolVersion):
        return True, None
    object_ids = [obj.id for obj in prediction.objects]
    if len(object_ids) != len(set(object_ids)):
        return True, None
    return True, prediction


def _is_text_object(value: object) -> bool:
    return isinstance(value, (PredictionObjectV1, PredictionTextObjectV2))


def evaluate_generation_validity(outputs: list[str]) -> GenerationValidityMetrics:
    """Evaluate complete, unrepaired outputs against the prediction schema."""
    valid_json = 0
    schema_valid = 0
    for output in outputs:
        is_json, prediction = _parse_prediction(output)
        valid_json += int(is_json)
        schema_valid += int(prediction is not None)
    return GenerationValidityMetrics(
        evaluated_count=len(outputs),
        valid_json_count=valid_json,
        schema_valid_count=schema_valid,
    )


def evaluate_generation_predictions(
    outputs: list[str], targets: list[str | PredictionProtocol]
) -> GenerationValidityMetrics:
    """Evaluate generated protocols against targets, matching objects by semantic ID."""
    if len(outputs) != len(targets):
        raise ValueError("outputs and targets must have the same length")

    valid_json = 0
    schema_valid = 0
    ground_truth_objects = 0
    ground_truth_text_objects = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    exact_matches = 0
    box_iou_sum = 0.0
    font_correct = 0
    character_errors = 0
    reference_characters = 0
    word_errors = 0
    reference_words = 0
    bezier_squared_error_sum = 0.0
    bezier_coordinate_count = 0
    color_absolute_error_sum = 0.0
    color_channel_count = 0
    oriented_box_iou_sum = 0.0
    oriented_box_count = 0
    angle_absolute_error_sum = 0.0
    angle_count = 0
    geometry_mode_correct_count = 0
    geometry_mode_count = 0
    bezier_centerline_error_sum = 0.0
    bezier_centerline_count = 0

    for output, raw_target in zip(outputs, targets, strict=True):
        target = (
            PredictionProtocol.model_validate(raw_target)
            if isinstance(raw_target, PredictionProtocol)
            else PredictionProtocol.model_validate_json(raw_target)
        )
        is_json, prediction = _parse_prediction(output)
        valid_json += int(is_json)
        schema_valid += int(prediction is not None)

        target_by_id = {obj.id: obj for obj in target.objects}
        prediction_by_id = (
            {obj.id: obj for obj in prediction.objects} if prediction is not None else {}
        )
        target_ids = set(target_by_id)
        prediction_ids = set(prediction_by_id)
        matched_ids = target_ids & prediction_ids
        true_positives += len(matched_ids)
        false_positives += len(prediction_ids - target_ids)
        false_negatives += len(target_ids - prediction_ids)
        exact_matches += int(prediction is not None and target_ids == prediction_ids)
        ground_truth_objects += len(target_ids)
        ground_truth_text_objects += sum(_is_text_object(obj) for obj in target.objects)

        for object_id in target_ids | prediction_ids:
            target_obj = target_by_id.get(object_id)
            prediction_obj = prediction_by_id.get(object_id)
            reference_text = (
                unicodedata.normalize("NFC", target_obj.text)
                if target_obj is not None and _is_text_object(target_obj)
                else ""
            )
            predicted_text = (
                unicodedata.normalize("NFC", prediction_obj.text)
                if prediction_obj is not None and _is_text_object(prediction_obj)
                else ""
            )
            character_errors += _edit_distance(list(reference_text), list(predicted_text))
            reference_characters += len(reference_text)
            reference_tokens = reference_text.split()
            predicted_tokens = predicted_text.split()
            word_errors += _edit_distance(reference_tokens, predicted_tokens)
            reference_words += len(reference_tokens)

            if target_obj is None or prediction_obj is None:
                continue
            box_iou_sum += _box_iou(
                target_obj.geometry.box, prediction_obj.geometry.box
            )
            if _is_text_object(target_obj) and _is_text_object(prediction_obj):
                oriented_box_iou_sum += oriented_box_iou(
                    target_obj.geometry, prediction_obj.geometry
                )
                oriented_box_count += 1
                angle_absolute_error_sum += angle_error_degrees(
                    target_obj.geometry.rotation_degrees,
                    prediction_obj.geometry.rotation_degrees,
                )
                angle_count += 1
                geometry_mode_correct_count += int(
                    target_obj.geometry.mode == prediction_obj.geometry.mode
                )
                geometry_mode_count += 1
                font_correct += int(
                    target_obj.style.font_id == prediction_obj.style.font_id
                )

            # A straight prediction has no curve and is intentionally excluded,
            # as is a straight target for which no ground-truth curve exists.
            target_baseline = getattr(target_obj.geometry, "baseline", None)
            prediction_baseline = getattr(prediction_obj.geometry, "baseline", None)
            if target_baseline is not None and prediction_baseline is not None:
                bezier_centerline_error_sum += bezier_centerline_error(
                    target_baseline, prediction_baseline
                )
                bezier_centerline_count += 1
                for point_name in ("p0", "p1", "p2", "p3"):
                    target_point = getattr(target_baseline, point_name)
                    prediction_point = getattr(prediction_baseline, point_name)
                    for coordinate in ("x", "y"):
                        difference = getattr(prediction_point, coordinate) - getattr(
                            target_point, coordinate
                        )
                        bezier_squared_error_sum += difference * difference
                        bezier_coordinate_count += 1

            target_colors = _fill_rgb_samples(target_obj.style.fill)
            prediction_colors = _fill_rgb_samples(prediction_obj.style.fill)
            if len(target_colors) == len(prediction_colors):
                for target_color, prediction_color in zip(
                    target_colors, prediction_colors, strict=True
                ):
                    color_absolute_error_sum += sum(
                        abs(predicted - expected)
                        for expected, predicted in zip(
                            target_color, prediction_color, strict=True
                        )
                    )
                    color_channel_count += 3

    return GenerationValidityMetrics(
        evaluated_count=len(outputs),
        valid_json_count=valid_json,
        schema_valid_count=schema_valid,
        ground_truth_object_count=ground_truth_objects,
        ground_truth_text_object_count=ground_truth_text_objects,
        semantic_id_true_positive_count=true_positives,
        semantic_id_false_positive_count=false_positives,
        semantic_id_false_negative_count=false_negatives,
        semantic_id_exact_match_count=exact_matches,
        box_iou_sum=box_iou_sum,
        font_correct_count=font_correct,
        character_error_count=character_errors,
        reference_character_count=reference_characters,
        word_error_count=word_errors,
        reference_word_count=reference_words,
        bezier_squared_error_sum=bezier_squared_error_sum,
        bezier_coordinate_count=bezier_coordinate_count,
        color_absolute_error_sum=color_absolute_error_sum,
        color_channel_count=color_channel_count,
        oriented_box_iou_sum=oriented_box_iou_sum,
        oriented_box_count=oriented_box_count,
        angle_absolute_error_sum=angle_absolute_error_sum,
        angle_count=angle_count,
        geometry_mode_correct_count=geometry_mode_correct_count,
        geometry_mode_count=geometry_mode_count,
        bezier_centerline_error_sum=bezier_centerline_error_sum,
        bezier_centerline_count=bezier_centerline_count,
        has_ground_truth=True,
    )
