from __future__ import annotations

import json

import pytest

from infer import (
    decode_prediction_coordinates,
    resolve_coordinate_codec,
    write_prediction,
)
from text_render_protocol_predictor.protocol import CoordinateTokenCodec


class FakeTokenizer:
    def __init__(self, tokens: tuple[str, ...] = ()) -> None:
        self.tokens = tokens

    def get_vocab(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}


def test_inference_auto_detects_atomic_coordinate_vocabulary() -> None:
    expected = CoordinateTokenCodec(bins=8)

    codec = resolve_coordinate_codec(
        FakeTokenizer(expected.tokenizer_tokens),
        mode="auto",
        bins=8,
        token_prefix="coord",
    )

    assert codec is not None
    assert codec.bins == expected.bins
    assert codec.prefix == expected.prefix


def test_inference_auto_preserves_legacy_float_checkpoint() -> None:
    assert (
        resolve_coordinate_codec(
            FakeTokenizer(),
            mode="auto",
            bins=512,
            token_prefix="coord",
        )
        is None
    )


def test_inference_rejects_partial_coordinate_vocabulary() -> None:
    codec = CoordinateTokenCodec(bins=8)

    with pytest.raises(ValueError, match="only 2/8"):
        resolve_coordinate_codec(
            FakeTokenizer(codec.tokenizer_tokens[:2]),
            mode="auto",
            bins=8,
            token_prefix="coord",
        )


def test_inference_decodes_using_actual_image_resolution(protocol_dict: dict) -> None:
    codec = CoordinateTokenCodec()
    encoded = codec.encode_json(protocol_dict)

    decoded, error = decode_prediction_coordinates(
        encoded,
        codec,
        image_size=(640, 360),
    )
    prediction = json.loads(decoded)

    assert error is None
    assert prediction["canvas"] == {"width": 640, "height": 360}
    assert prediction["objects"][0]["geometry"]["box"] == {
        "x": pytest.approx(5.01, abs=0.001),
        "y": pytest.approx(4.932, abs=0.001),
        "width": pytest.approx(100.196, abs=0.001),
        "height": pytest.approx(19.726, abs=0.001),
    }
    assert "<coord_" not in decoded


def test_inference_preserves_invalid_generation_and_reports_decode_error() -> None:
    output, error = decode_prediction_coordinates(
        "not json",
        CoordinateTokenCodec(),
        image_size=(900, 1200),
    )

    assert output == "not json"
    assert error is not None


def test_write_prediction_pretty_prints_json(tmp_path) -> None:
    path = tmp_path / "prediction.json"

    write_prediction(path, '{"x":1.5}')

    assert json.loads(path.read_text(encoding="utf-8")) == {"x": 1.5}
    assert "\n    \"x\"" in path.read_text(encoding="utf-8")
