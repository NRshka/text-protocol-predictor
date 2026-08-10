from text_render_protocol_predictor.evaluation import evaluate_grounding_promotion


def test_promotion_gate_accepts_positive_paired_improvement() -> None:
    baseline = [0.40, 0.50, 0.60, 0.55] * 10
    grounded = [value + 0.08 for value in baseline]

    decision = evaluate_grounding_promotion(
        baseline_geometry_mask_iou=baseline,
        grounded_geometry_mask_iou=grounded,
        attached_miou=0.72,
        oracle_miou=0.73,
        baseline_cer=0.10,
        grounded_cer=0.105,
        baseline_schema_validity=0.95,
        grounded_schema_validity=0.945,
        bootstrap_resamples=500,
    )

    assert decision.promoted is True
    assert decision.geometry_relative_improvement.lower > 0


def test_promotion_gate_reports_failed_thresholds() -> None:
    decision = evaluate_grounding_promotion(
        baseline_geometry_mask_iou=[0.5, 0.5],
        grounded_geometry_mask_iou=[0.5, 0.5],
        attached_miou=0.5,
        oracle_miou=0.6,
        baseline_cer=0.1,
        grounded_cer=0.12,
        baseline_schema_validity=0.95,
        grounded_schema_validity=0.93,
        bootstrap_resamples=50,
    )

    assert decision.promoted is False
    assert len(decision.failures) == 5
