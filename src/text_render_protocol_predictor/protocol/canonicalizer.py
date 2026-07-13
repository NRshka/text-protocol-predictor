"""Canonical model-target projection and JSON serialization."""

from __future__ import annotations

import json
import unicodedata
from decimal import Decimal, ROUND_HALF_EVEN
from typing import Any, Mapping

from .schema import DatasetProtocol, PredictionObject, PredictionProtocol
from .validator import validate_dataset_protocol

CANONICALIZER_VERSION = "1.0.0"
DEFAULT_DECIMAL_PLACES = 3


def project_protocol(value: DatasetProtocol | Mapping[str, Any]) -> PredictionProtocol:
    """Remove dataset-only fields and deterministically assign target object IDs."""
    protocol = validate_dataset_protocol(value)
    ordered = sorted(protocol.objects, key=lambda obj: (obj.z_order, obj.id))
    objects = []
    for index, obj in enumerate(ordered):
        data = obj.model_dump(exclude={"tight_bbox"})
        data["id"] = f"text_{index:03d}"
        data["text"] = unicodedata.normalize("NFC", data["text"])
        objects.append(PredictionObject.model_validate(data))
    return PredictionProtocol(
        protocol_version=protocol.protocol_version,
        canvas=protocol.canvas,
        objects=objects,
    )


def _round_number(value: float, decimal_places: int) -> int | float:
    quantum = Decimal(1).scaleb(-decimal_places)
    rounded = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_EVEN)
    if rounded == 0:
        return 0
    if rounded == rounded.to_integral_value():
        return int(rounded)
    return float(rounded)


def _normalize(value: Any, decimal_places: int) -> Any:
    if isinstance(value, dict):
        return {key: _normalize(item, decimal_places) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize(item, decimal_places) for item in value]
    if isinstance(value, float):
        return _round_number(value, decimal_places)
    return value


def canonicalize(
    value: DatasetProtocol | PredictionProtocol | Mapping[str, Any] | str,
    *,
    decimal_places: int = DEFAULT_DECIMAL_PLACES,
) -> str:
    """Return the one canonical JSON representation of a prediction target."""
    if decimal_places < 0:
        raise ValueError("decimal_places must be non-negative")
    if isinstance(value, str):
        value = json.loads(value)

    if isinstance(value, DatasetProtocol) or (
        isinstance(value, Mapping) and "sample_id" in value
    ):
        target = project_protocol(value)
    elif isinstance(value, PredictionProtocol):
        target = value
    else:
        target = PredictionProtocol.model_validate(value)

    normalized = _normalize(target.model_dump(mode="python"), decimal_places)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )

