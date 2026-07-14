from text_render_protocol_predictor.evaluation import evaluate_generation_validity


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
