import json

from text_render_protocol_predictor.protocol import canonicalize


def test_projection_and_canonicalization(protocol_dict: dict) -> None:
    serialized = canonicalize(protocol_dict)
    target = json.loads(serialized)

    assert list(target) == ["protocol_version", "canvas", "objects"]
    assert [obj["id"] for obj in target["objects"]] == ["text_000", "text_001"]
    assert target["objects"][1]["text"] == "Café"
    assert target["objects"][1]["geometry"]["box"]["x"] == 0
    assert target["objects"][1]["geometry"]["box"]["y"] == 20.124
    assert "sample_id" not in serialized
    assert "tight_bbox" not in serialized


def test_canonicalization_is_idempotent(protocol_dict: dict) -> None:
    once = canonicalize(protocol_dict)
    assert canonicalize(once) == once

