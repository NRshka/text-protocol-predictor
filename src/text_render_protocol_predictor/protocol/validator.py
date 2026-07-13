"""Structural and renderer-independent semantic protocol validation."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

from pydantic import ValidationError

from .schema import DatasetProtocol


class ProtocolValidationError(ValueError):
    """Raised when a protocol violates structural or semantic invariants."""


def validate_dataset_protocol(
    value: DatasetProtocol | Mapping[str, Any],
    *,
    font_ids: Collection[str] | None = None,
) -> DatasetProtocol:
    try:
        protocol = value if isinstance(value, DatasetProtocol) else DatasetProtocol.model_validate(value)
    except ValidationError as exc:
        raise ProtocolValidationError(str(exc)) from exc

    ids = [obj.id for obj in protocol.objects]
    if len(ids) != len(set(ids)):
        raise ProtocolValidationError("object identifiers must be unique")

    if font_ids is not None:
        unknown = sorted({obj.style.font_id for obj in protocol.objects} - set(font_ids))
        if unknown:
            raise ProtocolValidationError(f"unknown font identifiers: {', '.join(unknown)}")

    return protocol

