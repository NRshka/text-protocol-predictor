import json

import pytest

from text_render_protocol_predictor.evaluation import (
    evaluate_generation_predictions,
    evaluate_generation_validity,
)
from text_render_protocol_predictor.protocol import canonicalize


def test_generation_validity_uses_all_outputs_as_denominator() -> None:
    valid = '{"protocol_version":"1.0","canvas":{"width":100,"height":50},"objects":[]}'
    json_only = '{"protocol_version":"2.0"}'
    invalid = "```json\n{}\n```"

    metrics = evaluate_generation_validity([valid, json_only, invalid])

    assert metrics.evaluated_count == 3
    assert metrics.valid_json_count == 2
    assert metrics.schema_valid_count == 1
    assert metrics.valid_json_percent == 200 / 3
    assert metrics.schema_valid_percent == 100 / 3


def test_generation_task_metrics_are_perfect_for_exact_prediction(
    protocol_dict: dict,
) -> None:
    target = canonicalize(protocol_dict)
    metrics = evaluate_generation_predictions([target], [target])

    assert metrics.box_iou == 1.0
    assert metrics.character_error_rate == 0.0
    assert metrics.word_error_rate == 0.0
    assert metrics.font_accuracy == 1.0
    assert metrics.semantic_id_precision == 1.0
    assert metrics.semantic_id_recall == 1.0
    assert metrics.semantic_id_exact_match == 1.0


def test_generation_task_metrics_penalize_missing_and_extra_ids(
    protocol_dict: dict,
) -> None:
    target = canonicalize(protocol_dict)
    prediction = json.loads(target)
    prediction["objects"][0]["text"] = "Fist"
    prediction["objects"][0]["style"]["font_id"] = "WrongFont"
    prediction["objects"][1]["id"] = "wrong-id"

    metrics = evaluate_generation_predictions(
        [json.dumps(prediction, ensure_ascii=False)], [target]
    )

    assert metrics.semantic_id_true_positive_count == 1
    assert metrics.semantic_id_false_positive_count == 1
    assert metrics.semantic_id_false_negative_count == 1
    assert metrics.semantic_id_precision == 0.5
    assert metrics.semantic_id_recall == 0.5
    assert metrics.semantic_id_exact_match == 0.0
    assert metrics.box_iou == 0.5
    assert metrics.font_accuracy == 0.0
    assert metrics.character_error_rate == 1.0
    assert metrics.word_error_rate == 1.5


def test_invalid_prediction_counts_as_all_objects_missing(protocol_dict: dict) -> None:
    metrics = evaluate_generation_predictions(["not json"], [canonicalize(protocol_dict)])

    assert metrics.valid_json_count == 0
    assert metrics.schema_valid_count == 0
    assert metrics.semantic_id_precision == 0.0
    assert metrics.semantic_id_recall == 0.0
    assert metrics.box_iou == 0.0
    assert metrics.character_error_rate == pytest.approx(1.0)
    assert metrics.word_error_rate == pytest.approx(1.0)


def test_duplicate_semantic_ids_are_schema_invalid(protocol_dict: dict) -> None:
    target = canonicalize(protocol_dict)
    prediction = json.loads(target)
    prediction["objects"][1]["id"] = prediction["objects"][0]["id"]

    metrics = evaluate_generation_predictions([json.dumps(prediction)], [target])

    assert metrics.valid_json_count == 1
    assert metrics.schema_valid_count == 0
    assert metrics.semantic_id_exact_match == 0.0
