from __future__ import annotations

import json

import pytest
from PIL import Image

from text_render_protocol_predictor.data import ProtocolManifestDataset


def test_dataset_resolves_root_relative_paths(tmp_path, protocol_dict: dict) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "protocols").mkdir()
    Image.new("RGB", (1280, 720)).save(tmp_path / "images" / "sample.jpg")
    (tmp_path / "protocols" / "sample.json").write_text(
        json.dumps(protocol_dict), encoding="utf-8"
    )
    (tmp_path / "train.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "image": "images/sample.jpg",
                "protocol": "protocols/sample.json",
                "seed": 17,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = ProtocolManifestDataset(
        dataset_root=tmp_path,
        manifest_path="train.jsonl",
        font_ids={"Inter"},
    )
    record = dataset[0]
    assert record.canvas_width == 1280
    assert record.image_path == tmp_path / "images" / "sample.jpg"
    assert record.canonical_protocol.startswith('{"protocol_version":"1.0"')


def test_dataset_rejects_path_escape(tmp_path) -> None:
    (tmp_path / "train.jsonl").write_text(
        json.dumps(
            {"sample_id": "x", "image": "../x.jpg", "protocol": "x.json", "seed": 1}
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = ProtocolManifestDataset(dataset_root=tmp_path, manifest_path="train.jsonl")
    with pytest.raises(ValueError, match="escapes"):
        _ = dataset[0]
