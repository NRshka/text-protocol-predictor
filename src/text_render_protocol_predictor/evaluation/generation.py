"""Strict protocol validity metrics for autoregressive generations."""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import ValidationError

from ..protocol.schema import PredictionProtocol


@dataclass(frozen=True)
class GenerationValidityMetrics:
    evaluated_count: int
    valid_json_count: int
    schema_valid_count: int

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

    def as_log_dict(self) -> dict[str, int | float]:
        return {
            "generation/evaluated_count": self.evaluated_count,
            "generation/valid_json_count": self.valid_json_count,
            "generation/schema_valid_count": self.schema_valid_count,
            "generation/valid_json_percent": self.valid_json_percent,
            "generation/schema_valid_percent": self.schema_valid_percent,
        }


def evaluate_generation_validity(outputs: list[str]) -> GenerationValidityMetrics:
    """Evaluate complete, unrepaired outputs against the prediction schema."""
    valid_json = 0
    schema_valid = 0
    for output in outputs:
        try:
            value = json.loads(output.strip())
        except (json.JSONDecodeError, TypeError):
            continue
        valid_json += 1
        try:
            PredictionProtocol.model_validate(value)
        except ValidationError:
            continue
        schema_valid += 1
    return GenerationValidityMetrics(
        evaluated_count=len(outputs),
        valid_json_count=valid_json,
        schema_valid_count=schema_valid,
    )

