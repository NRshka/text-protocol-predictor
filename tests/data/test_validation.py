from __future__ import annotations

import json

import pytest

from text_render_protocol_predictor.data import ProtocolManifestDataset, validate_dataset


def test_preflight_collects_failures(tmp_path) -> None:
    (tmp_path / "train.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "missing",
                "image": "images/missing.jpg",
                "protocol": "protocols/missing.json",
                "seed": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = ProtocolManifestDataset(dataset_root=tmp_path, manifest_path="train.jsonl")
    report = validate_dataset(dataset)
    assert report.total == 1
    assert report.invalid == 1
    with pytest.raises(ValueError, match="rejected 1/1"):
        report.raise_for_errors()
