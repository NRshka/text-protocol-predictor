"""Paired-bootstrap decision gate for the contextual mask-query hypothesis."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PairedBootstrapInterval:
    point_estimate: float
    lower: float
    upper: float
    confidence: float
    resamples: int


@dataclass(frozen=True)
class GroundingPromotionDecision:
    promoted: bool
    geometry_relative_improvement: PairedBootstrapInterval
    association_gap: float
    cer_regression: float
    schema_validity_regression: float
    failures: tuple[str, ...]


def _relative_improvement(baseline: Sequence[float], grounded: Sequence[float]) -> float:
    baseline_mean = sum(baseline) / len(baseline)
    grounded_mean = sum(grounded) / len(grounded)
    if baseline_mean <= 0:
        return math.inf if grounded_mean > baseline_mean else 0.0
    return (grounded_mean - baseline_mean) / baseline_mean


def paired_bootstrap_relative_improvement(
    baseline: Sequence[float],
    grounded: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 2026,
) -> PairedBootstrapInterval:
    """Bootstrap paired per-sample geometry-to-mask IoUs."""
    if len(baseline) != len(grounded) or not baseline:
        raise ValueError("paired metric arrays must have the same non-zero length")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if resamples < 2:
        raise ValueError("resamples must be at least two")
    if any(not math.isfinite(float(value)) for value in (*baseline, *grounded)):
        raise ValueError("paired metrics must be finite")
    random_generator = random.Random(seed)
    count = len(baseline)
    estimates = []
    for _ in range(resamples):
        indices = [random_generator.randrange(count) for _ in range(count)]
        estimates.append(
            _relative_improvement(
                [float(baseline[index]) for index in indices],
                [float(grounded[index]) for index in indices],
            )
        )
    estimates.sort()
    alpha = (1.0 - confidence) / 2.0
    lower_index = max(0, math.floor(alpha * (resamples - 1)))
    upper_index = min(resamples - 1, math.ceil((1.0 - alpha) * (resamples - 1)))
    return PairedBootstrapInterval(
        point_estimate=_relative_improvement(baseline, grounded),
        lower=estimates[lower_index],
        upper=estimates[upper_index],
        confidence=confidence,
        resamples=resamples,
    )


def evaluate_grounding_promotion(
    *,
    baseline_geometry_mask_iou: Sequence[float],
    grounded_geometry_mask_iou: Sequence[float],
    attached_miou: float,
    oracle_miou: float,
    baseline_cer: float,
    grounded_cer: float,
    baseline_schema_validity: float,
    grounded_schema_validity: float,
    bootstrap_resamples: int = 10_000,
    seed: int = 2026,
) -> GroundingPromotionDecision:
    """Apply the documented Rank-1 promotion thresholds."""
    interval = paired_bootstrap_relative_improvement(
        baseline_geometry_mask_iou,
        grounded_geometry_mask_iou,
        resamples=bootstrap_resamples,
        seed=seed,
    )
    association_gap = max(0.0, oracle_miou - attached_miou)
    cer_regression = grounded_cer - baseline_cer
    schema_regression = baseline_schema_validity - grounded_schema_validity
    failures = []
    if interval.point_estimate < 0.05:
        failures.append("serialized geometry-to-mask IoU improvement is below 5%")
    if interval.lower <= 0:
        failures.append("paired-bootstrap confidence interval is not strictly positive")
    if association_gap > 0.02:
        failures.append("attached-versus-oracle mIoU gap exceeds 0.02")
    if cer_regression > 0.01:
        failures.append("CER regression exceeds one absolute percentage point")
    if schema_regression > 0.01:
        failures.append("schema-validity regression exceeds one absolute percentage point")
    return GroundingPromotionDecision(
        promoted=not failures,
        geometry_relative_improvement=interval,
        association_gap=association_gap,
        cer_regression=cer_regression,
        schema_validity_regression=schema_regression,
        failures=tuple(failures),
    )
