import pytest

from text_render_protocol_predictor.protocol import (
    DatasetProtocolV21,
    ProtocolValidationError,
    detect_protocol_version,
    validate_dataset_protocol,
)


def test_duplicate_ids_are_rejected(copied_protocol: dict) -> None:
    copied_protocol["objects"][1]["id"] = copied_protocol["objects"][0]["id"]
    with pytest.raises(ProtocolValidationError, match="unique"):
        validate_dataset_protocol(copied_protocol)


def test_unknown_fonts_are_rejected(protocol_dict: dict) -> None:
    with pytest.raises(ProtocolValidationError, match="unknown font"):
        validate_dataset_protocol(protocol_dict, font_ids={"DejaVuSans"})


def test_straight_baseline_is_explicitly_null(copied_protocol: dict) -> None:
    copied_protocol["objects"][0]["geometry"]["baseline"] = {
        "p0": {"x": 0, "y": 0},
        "p1": {"x": 1, "y": 0},
        "p2": {"x": 2, "y": 0},
        "p3": {"x": 3, "y": 0},
    }
    with pytest.raises(ProtocolValidationError, match="baseline=null"):
        validate_dataset_protocol(copied_protocol)


def test_version_21_dispatches_to_its_schema(protocol_21_dict: dict) -> None:
    protocol = validate_dataset_protocol(protocol_21_dict, font_ids={"Inter"})

    assert isinstance(protocol, DatasetProtocolV21)
    assert detect_protocol_version(protocol) == "2.1"
    assert protocol.purpose == "annotation"
    assert protocol.objects[1].annotation.geometry_confidence == 0.95
    assert protocol.objects[1].annotation.text_confidence == 1.0


def test_version_20_rejects_21_annotation_fields(protocol_21_dict: dict) -> None:
    protocol_21_dict["protocol_version"] = "2.0"
    protocol_21_dict["purpose"] = "render"
    with pytest.raises(ProtocolValidationError, match="annotation"):
        validate_dataset_protocol(protocol_21_dict)


def test_annotation_purpose_requires_version_21(protocol_21_dict: dict) -> None:
    protocol_21_dict["protocol_version"] = "2.0"
    for obj in protocol_21_dict["objects"]:
        obj.pop("annotation", None)
    with pytest.raises(ProtocolValidationError, match="purpose"):
        validate_dataset_protocol(protocol_21_dict)


def test_unknown_versions_are_not_guessed(protocol_dict: dict) -> None:
    protocol_dict["protocol_version"] = "3.0"
    with pytest.raises(ProtocolValidationError, match="unsupported protocol_version"):
        validate_dataset_protocol(protocol_dict)
