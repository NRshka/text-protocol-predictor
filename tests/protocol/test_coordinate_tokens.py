from __future__ import annotations

import copy
import json

import pytest

from text_render_protocol_predictor.protocol import CoordinateTokenCodec


def test_coordinate_codec_quantizes_geometry_and_round_trips(
    protocol_dict: dict,
) -> None:
    protocol = copy.deepcopy(protocol_dict)
    geometry = protocol["objects"][0]["geometry"]
    geometry["mode"] = "bezier"
    geometry["baseline"] = {
        "p0": {"x": 0, "y": 0},
        "p1": {"x": 320, "y": 180},
        "p2": {"x": 640, "y": 360},
        "p3": {"x": 1280, "y": 720},
    }
    codec = CoordinateTokenCodec(bins=512)

    encoded = codec.encode_json(protocol)
    target = json.loads(encoded)
    encoded_geometry = target["objects"][1]["geometry"]

    assert encoded_geometry["box"] == {
        "x": "<coord_000>",
        "y": "<coord_014>",
        "width": "<coord_120>",
        "height": "<coord_057>",
    }
    assert encoded_geometry["baseline"]["p1"] == {
        "x": "<coord_128>",
        "y": "<coord_128>",
    }
    assert encoded_geometry["baseline"]["p3"] == {
        "x": "<coord_511>",
        "y": "<coord_511>",
    }
    assert target["objects"][1]["style"]["font_size"] == 54
    assert target["objects"][1]["geometry"]["rotation_degrees"] == 0

    decoded = json.loads(codec.decode_json(encoded))
    decoded_geometry = decoded["objects"][1]["geometry"]
    assert decoded_geometry["box"]["x"] == 0
    assert decoded_geometry["box"]["width"] == pytest.approx(300.587, abs=0.001)
    assert decoded_geometry["baseline"]["p1"]["x"] == pytest.approx(320.626, abs=0.001)


def test_coordinate_codec_quantizes_shape_boxes(protocol_21_dict: dict) -> None:
    encoded = json.loads(CoordinateTokenCodec().encode_json(protocol_21_dict))
    shape = next(obj for obj in encoded["objects"] if obj["object_type"] == "shape")

    assert shape["geometry"]["box"] == {
        "x": "<coord_002>",
        "y": "<coord_004>",
        "width": "<coord_160>",
        "height": "<coord_071>",
    }
    assert shape["geometry"]["corner_radius"] == 12


def test_coordinate_codec_filters_shapes_before_quantization(
    protocol_21_dict: dict,
) -> None:
    encoded = json.loads(
        CoordinateTokenCodec().encode_json(protocol_21_dict, text_only=True)
    )

    assert [obj["object_type"] for obj in encoded["objects"]] == ["text"]
    assert all(obj["id"] != "panel" for obj in encoded["objects"])


def test_positive_box_dimensions_never_encode_as_zero(protocol_dict: dict) -> None:
    protocol = copy.deepcopy(protocol_dict)
    protocol["objects"][0]["geometry"]["box"]["width"] = 0.01
    protocol["objects"][0]["geometry"]["box"]["height"] = 0.01

    encoded = json.loads(CoordinateTokenCodec().encode_json(protocol))
    box = encoded["objects"][1]["geometry"]["box"]

    assert box["width"] == "<coord_001>"
    assert box["height"] == "<coord_001>"


def test_coordinate_tokenizer_surfaces_include_json_quotes() -> None:
    codec = CoordinateTokenCodec(bins=8)

    assert codec.token(3) == "<coord_3>"
    assert codec.json_token(3) == '"<coord_3>"'
    assert codec.tokenizer_tokens[3] == '"<coord_3>"'


def test_decoder_rejects_non_coordinate_geometry_values(protocol_dict: dict) -> None:
    encoded = json.loads(CoordinateTokenCodec().encode_json(protocol_dict))
    encoded["objects"][0]["geometry"]["box"]["x"] = 10

    with pytest.raises(ValueError, match="expected a coordinate token"):
        CoordinateTokenCodec().decode_json(encoded)


def test_decoder_can_use_external_canvas_dimensions(protocol_dict: dict) -> None:
    codec = CoordinateTokenCodec()
    encoded = codec.encode_json(protocol_dict)

    decoded = json.loads(codec.decode_json(encoded, canvas_size=(640, 360)))

    assert decoded["canvas"] == {"width": 640, "height": 360}
    assert decoded["objects"][0]["geometry"]["box"]["width"] == pytest.approx(
        100.196, abs=0.001
    )
