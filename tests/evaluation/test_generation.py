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


def test_generation_validity_rejects_future_versions_without_guessing() -> None:
    future = '{"protocol_version":"3.0","canvas":{"width":1,"height":1},"objects":[]}'
    metrics = evaluate_generation_validity([future])

    assert metrics.valid_json_count == 1
    assert metrics.schema_valid_count == 0


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
    assert metrics.color_mae == 0.0
    assert metrics.color_channel_count == 6
    assert metrics.bezier_mse == 0.0
    assert metrics.bezier_coordinate_count == 0


def test_generation_task_metrics_measure_bezier_and_rgb_errors(protocol_dict: dict) -> None:
    target = json.loads(canonicalize(protocol_dict))
    prediction = json.loads(canonicalize(protocol_dict))
    for document in (target, prediction):
        geometry = document["objects"][0]["geometry"]
        geometry["mode"] = "bezier"
        geometry["baseline"] = {
            "p0": {"x": 0, "y": 0},
            "p1": {"x": 1, "y": 1},
            "p2": {"x": 2, "y": 2},
            "p3": {"x": 3, "y": 3},
        }
    prediction["objects"][0]["geometry"]["baseline"]["p2"]["x"] += 4
    prediction["objects"][0]["style"]["fill"]["color"] = "#F00A14FF"
    target["objects"][0]["style"]["fill"]["color"] = "#FA141EFF"

    metrics = evaluate_generation_predictions([json.dumps(prediction)], [json.dumps(target)])

    assert metrics.bezier_mse == 2.0
    assert metrics.bezier_coordinate_count == 8
    assert metrics.color_mae == pytest.approx(5.0)
    assert metrics.color_channel_count == 6


def test_bezier_mse_excludes_straight_predictions(protocol_dict: dict) -> None:
    target = json.loads(canonicalize(protocol_dict))
    geometry = target["objects"][0]["geometry"]
    geometry["mode"] = "bezier"
    geometry["baseline"] = {
        "p0": {"x": 0, "y": 0},
        "p1": {"x": 1, "y": 1},
        "p2": {"x": 2, "y": 2},
        "p3": {"x": 3, "y": 3},
    }

    metrics = evaluate_generation_predictions([canonicalize(protocol_dict)], [json.dumps(target)])

    assert metrics.bezier_mse == 0.0
    assert metrics.bezier_coordinate_count == 0


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


def test_version_21_shapes_are_evaluated_without_text_fields(
    protocol_21_dict: dict,
) -> None:
    target = canonicalize(protocol_21_dict)
    metrics = evaluate_generation_predictions([target], [target])

    assert metrics.schema_valid_count == 1
    assert metrics.ground_truth_object_count == 2
    assert metrics.ground_truth_text_object_count == 1
    assert metrics.box_iou == 1.0
    assert metrics.font_accuracy == 1.0
    assert metrics.character_error_rate == 0.0
