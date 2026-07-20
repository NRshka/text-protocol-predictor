from __future__ import annotations

import json

import pytest
from PIL import Image

from text_render_protocol_predictor.data import GRPOManifestDataset, validate_dataset


def _save_webp(path, *, size=(12, 8), color=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=color).save(path, format="WEBP", lossless=True)


def _dataset(tmp_path, *, mask_color=255, background_size=(12, 8)):
    split = tmp_path / "splits"
    _save_webp(tmp_path / "images" / "original.webp")
    _save_webp(tmp_path / "images" / "background.webp", size=background_size)
    _save_webp(tmp_path / "images" / "mask.webp", color=mask_color)
    split.mkdir()
    (split / "train.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "image": "../images/original.webp",
                "background": "../images/background.webp",
                "text_mask": "../images/mask.webp",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return GRPOManifestDataset(
        dataset_root=tmp_path,
        manifest_path="splits/train.jsonl",
        minimum_mask_coverage=0.01,
    )


def test_loads_relative_paths_and_canvas(tmp_path):
    record = _dataset(tmp_path)[0]

    assert record.sample_id == "sample-1"
    assert (record.canvas_width, record.canvas_height) == (12, 8)
    assert record.mask_coverage == pytest.approx(1.0)
    assert record.background_path == (tmp_path / "images/background.webp").resolve()


def test_preflight_reports_mismatched_dimensions(tmp_path):
    dataset = _dataset(tmp_path, background_size=(10, 8))

    report = validate_dataset(dataset)

    assert report.invalid == 1
    assert "dimensions differ" in report.failures[0].error


def test_rejects_empty_text_mask(tmp_path):
    dataset = _dataset(tmp_path, mask_color=0)

    with pytest.raises(ValueError, match="mask coverage"):
        dataset[0]
