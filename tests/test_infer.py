from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from infer import (
    decode_and_write_masks,
    decode_prediction_coordinates,
    is_grounded_checkpoint,
    resolve_coordinate_codec,
    write_mask_sidecar,
    write_prediction,
)
from text_render_protocol_predictor.protocol import CoordinateTokenCodec
from text_render_protocol_predictor.training import ProtocolPromptTemplate


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


def test_inference_decodes_tokens_while_preserving_numeric_coordinates(
    protocol_dict: dict,
) -> None:
    codec = CoordinateTokenCodec()
    encoded = json.loads(codec.encode_json(protocol_dict))
    encoded["objects"][0]["geometry"]["box"]["width"] = 1

    decoded, error = decode_prediction_coordinates(
        json.dumps(encoded),
        codec,
        image_size=(1280, 720),
    )
    prediction = json.loads(decoded)

    assert error is None
    assert prediction["objects"][0]["geometry"]["box"]["width"] == 1
    assert prediction["objects"][0]["geometry"]["box"]["x"] == pytest.approx(
        10.02, abs=0.001
    )
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


def test_grounded_checkpoint_detection_rejects_partial_export(tmp_path) -> None:
    (tmp_path / "grounding_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="grounding_model.safetensors"):
        is_grounded_checkpoint(tmp_path)


def test_mask_sidecar_writes_original_size_probability_png(tmp_path) -> None:
    class Batch(dict):
        def to(self, _device):
            return self

    class Processor:
        def apply_chat_template(self, conversations, **_kwargs):
            has_target = conversations[0][-1]["role"] == "assistant"
            ids = [1, 99, 2, 99, 3] if has_target else [1, 99, 2]
            return Batch(
                input_ids=torch.tensor([ids]),
                attention_mask=torch.ones((1, len(ids)), dtype=torch.long),
                image_grid_thw=torch.tensor([[1, 2, 2]]),
            )

    class Model:
        mask_token_id = 99

        def __call__(self, **_kwargs):
            return SimpleNamespace(
                mask_logits=[torch.zeros(1, 8, 8)],
                geometry_predictions=[{"quality_logits": torch.tensor([0.0])}],
            )

    image_path = tmp_path / "image.png"
    Image.new("RGB", (23, 17)).save(image_path)
    output_dir = tmp_path / "masks"
    decode_and_write_masks(
        model=Model(),
        processor=Processor(),
        prompt_template=ProtocolPromptTemplate(grounding_enabled=True),
        image_path=image_path,
        raw_output='{"mask_ref":"<MASK>"}',
        object_ids=("title",),
        canvas_size=(23, 17),
        protocol_version="1.0",
        device="cpu",
        output_dir=output_dir,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["instances"][0]["quality_score"] == pytest.approx(0.5)
    with Image.open(output_dir / manifest["instances"][0]["mask"]) as mask:
        assert mask.size == (23, 17)
        assert mask.getpixel((0, 0)) == 128


def test_error_mask_manifest_contains_no_instances(tmp_path) -> None:
    write_mask_sidecar(
        tmp_path,
        status="error",
        canvas_size=(20, 10),
        error="malformed grounding",
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "error"
    assert manifest["instances"] == []
    assert manifest["error"] == "malformed grounding"
