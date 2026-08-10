"""Training-only mask grounding for otherwise strict STRP JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping


MASK_REFERENCE_FIELD = "mask_ref"
MASK_TOKEN = "<MASK>"
MASK_TOKENIZER_SURFACE = json.dumps(MASK_TOKEN)
GROUNDING_FORMAT_VERSION = "1.0.0"


@dataclass(frozen=True)
class GroundingStripResult:
    """A best-effort clean protocol plus strict grounding diagnostics."""

    clean_json: str
    object_ids: tuple[str, ...]
    valid: bool
    errors: tuple[str, ...] = ()


def _loads_object(value: str | Mapping[str, Any]) -> dict[str, Any]:
    raw = json.loads(value) if isinstance(value, str) else dict(value)
    if not isinstance(raw, dict):
        raise ValueError("grounded protocol must be a JSON object")
    return raw


def _is_text_object(obj: Mapping[str, Any]) -> bool:
    # STRP 1.0 has no object_type and contains text-only objects. In 2.x the
    # discriminator is authoritative; checking text also keeps malformed
    # generations recoverable for clean-schema diagnostics.
    return obj.get("object_type", "text") == "text" or "text" in obj


def ground_protocol_json(
    value: str | Mapping[str, Any],
    *,
    mask_token: str = MASK_TOKEN,
) -> str:
    """Insert one mask reference after ``direction`` in every text object."""
    protocol = _loads_object(value)
    objects = protocol.get("objects")
    if not isinstance(objects, list):
        raise ValueError("protocol objects must be a list")

    grounded_objects: list[Any] = []
    for index, value_obj in enumerate(objects):
        if not isinstance(value_obj, dict):
            raise ValueError(f"protocol object {index} must be an object")
        if MASK_REFERENCE_FIELD in value_obj:
            raise ValueError(
                f"source protocol object {value_obj.get('id', index)!r} already contains "
                f"{MASK_REFERENCE_FIELD!r}"
            )
        if not _is_text_object(value_obj):
            grounded_objects.append(value_obj)
            continue
        if "direction" not in value_obj:
            raise ValueError(
                f"text object {value_obj.get('id', index)!r} has no direction field"
            )
        grounded: dict[str, Any] = {}
        for key, item in value_obj.items():
            grounded[key] = item
            if key == "direction":
                grounded[MASK_REFERENCE_FIELD] = mask_token
        grounded_objects.append(grounded)

    protocol["objects"] = grounded_objects
    return json.dumps(
        protocol,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def strip_mask_references(
    value: str | Mapping[str, Any],
    *,
    mask_token: str = MASK_TOKEN,
) -> GroundingStripResult:
    """Remove mask references while reporting whether grounding was exact.

    The clean JSON is returned even when a reference is missing or malformed,
    allowing ordinary STRP validation to remain an independent metric.
    """
    protocol = _loads_object(value)
    objects = protocol.get("objects")
    if not isinstance(objects, list):
        raise ValueError("protocol objects must be a list")

    errors: list[str] = []
    object_ids: list[str] = []
    clean_objects: list[Any] = []
    seen_ids: set[str] = set()
    for index, value_obj in enumerate(objects):
        if not isinstance(value_obj, dict):
            errors.append(f"object {index} is not a JSON object")
            clean_objects.append(value_obj)
            continue
        clean = dict(value_obj)
        object_id = clean.get("id")
        if isinstance(object_id, str):
            if object_id in seen_ids:
                errors.append(f"duplicate object id {object_id!r}")
            seen_ids.add(object_id)
        if _is_text_object(clean):
            object_ids.append(object_id if isinstance(object_id, str) else str(index))
            keys = list(clean)
            if MASK_REFERENCE_FIELD in clean:
                reference_index = keys.index(MASK_REFERENCE_FIELD)
                direction_index = keys.index("direction") if "direction" in clean else -2
                geometry_index = keys.index("geometry") if "geometry" in clean else -1
                if (
                    reference_index != direction_index + 1
                    or geometry_index != reference_index + 1
                ):
                    errors.append(
                        f"text object {object_id if object_id is not None else index!r} "
                        "must place mask_ref immediately after direction and before geometry"
                    )
            reference = clean.pop(MASK_REFERENCE_FIELD, None)
            if reference != mask_token:
                errors.append(
                    f"text object {object_id if object_id is not None else index!r} has "
                    f"mask_ref={reference!r}; expected {mask_token!r}"
                )
        elif MASK_REFERENCE_FIELD in clean:
            clean.pop(MASK_REFERENCE_FIELD)
            errors.append(
                f"non-text object {object_id if object_id is not None else index!r} "
                "contains mask_ref"
            )
        clean_objects.append(clean)

    protocol["objects"] = clean_objects
    return GroundingStripResult(
        clean_json=json.dumps(
            protocol,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ),
        object_ids=tuple(object_ids),
        valid=not errors,
        errors=tuple(errors),
    )
