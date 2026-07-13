import pytest

from text_render_protocol_predictor.protocol import (
    ProtocolValidationError,
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

