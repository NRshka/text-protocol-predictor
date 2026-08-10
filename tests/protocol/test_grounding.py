import json

from text_render_protocol_predictor.protocol import (
    ground_protocol_json,
    strip_mask_references,
)


def test_grounding_inserts_mask_after_direction_and_round_trips(protocol_dict: dict) -> None:
    clean = {
        "protocol_version": "1.0",
        "canvas": protocol_dict["canvas"],
        "objects": [
            {key: value for key, value in obj.items() if key not in {"tight_bbox", "object_type"}}
            for obj in protocol_dict["objects"]
        ],
    }

    grounded = ground_protocol_json(clean)
    parsed = json.loads(grounded)

    keys = list(parsed["objects"][0])
    assert keys.index("direction") + 1 == keys.index("mask_ref")
    assert keys.index("mask_ref") + 1 == keys.index("geometry")
    assert parsed["objects"][0]["mask_ref"] == "<MASK>"
    stripped = strip_mask_references(grounded)
    assert stripped.valid is True
    assert json.loads(stripped.clean_json) == clean


def test_grounding_skips_shapes_and_reports_missing_text_reference(
    protocol_21_dict: dict,
) -> None:
    prediction = {
        "protocol_version": "2.1",
        "canvas": protocol_21_dict["canvas"],
        "objects": [
            {
                key: value
                for key, value in obj.items()
                if key not in {"tight_bbox", "annotation"}
            }
            for obj in protocol_21_dict["objects"]
        ],
    }
    grounded = json.loads(ground_protocol_json(prediction))
    shape = next(obj for obj in grounded["objects"] if obj["object_type"] == "shape")
    text = next(obj for obj in grounded["objects"] if obj["object_type"] == "text")
    assert "mask_ref" not in shape
    assert text["mask_ref"] == "<MASK>"

    del text["mask_ref"]
    stripped = strip_mask_references(grounded)
    assert stripped.valid is False
    assert "expected '<MASK>'" in stripped.errors[0]
    assert "mask_ref" not in stripped.clean_json


def test_grounding_reports_misplaced_reference_and_duplicate_ids(
    protocol_dict: dict,
) -> None:
    clean = json.loads(ground_protocol_json({
        "protocol_version": "1.0",
        "canvas": protocol_dict["canvas"],
        "objects": [
            {
                key: value
                for key, value in obj.items()
                if key not in {"tight_bbox", "object_type"}
            }
            for obj in protocol_dict["objects"]
        ],
    }))
    first = clean["objects"][0]
    first["mask_ref"] = first.pop("mask_ref")
    clean["objects"][1]["id"] = first["id"]

    result = strip_mask_references(clean)

    assert result.valid is False
    assert any("immediately after direction" in error for error in result.errors)
    assert any("duplicate object id" in error for error in result.errors)


def test_grounding_supports_zero_objects() -> None:
    clean = '{"protocol_version":"1.0","canvas":{"width":1,"height":1},"objects":[]}'

    grounded = ground_protocol_json(clean)
    result = strip_mask_references(grounded)

    assert result.valid is True
    assert result.object_ids == ()
    assert json.loads(result.clean_json) == json.loads(clean)
