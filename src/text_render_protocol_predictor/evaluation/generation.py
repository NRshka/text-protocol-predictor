"""Strict validity and task metrics for autoregressive generations."""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass

from pydantic import ValidationError

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
                font_correct += int(
                    target_obj.style.font_id == prediction_obj.style.font_id
                )

            # A straight prediction has no curve and is intentionally excluded,
            # as is a straight target for which no ground-truth curve exists.
            target_baseline = getattr(target_obj.geometry, "baseline", None)
            prediction_baseline = getattr(prediction_obj.geometry, "baseline", None)
            if target_baseline is not None and prediction_baseline is not None:
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
        has_ground_truth=True,
    )
