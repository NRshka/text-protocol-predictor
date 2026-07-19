import json

from text_render_protocol_predictor.protocol import canonicalize


def test_projection_and_canonicalization(protocol_dict: dict) -> None:
    serialized = canonicalize(protocol_dict)
    target = json.loads(serialized)

    assert list(target) == ["protocol_version", "canvas", "objects"]
    assert [obj["id"] for obj in target["objects"]] == ["source-a", "source-b"]
    assert target["objects"][1]["text"] == "Café"
    assert target["objects"][1]["geometry"]["box"]["x"] == 0
    assert target["objects"][1]["geometry"]["box"]["y"] == 20.124
    assert "sample_id" not in serialized
    assert "tight_bbox" not in serialized


def test_canonicalization_is_idempotent(protocol_dict: dict) -> None:
    once = canonicalize(protocol_dict)
    assert canonicalize(once) == once


def test_version_21_projection_keeps_shapes_and_removes_evidence(
    protocol_21_dict: dict,
) -> None:
    target = json.loads(canonicalize(protocol_21_dict))

    assert target["protocol_version"] == "2.1"
    assert [obj["object_type"] for obj in target["objects"]] == ["shape", "text"]
    assert "purpose" not in target
    assert all("annotation" not in obj for obj in target["objects"])
    assert all("tight_bbox" not in obj for obj in target["objects"])
    assert canonicalize(json.dumps(target)) == canonicalize(protocol_21_dict)
