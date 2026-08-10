from __future__ import annotations

import json

import pytest
import torch
from PIL import Image

from text_render_protocol_predictor.data import (
    ProtocolManifestDataset,
    rasterize_protocol_instance_masks,
)
from text_render_protocol_predictor.training.mask_targets import prepare_instance_targets


def _write_sample(tmp_path, protocol_dict: dict, *, mask_supervision: dict | None) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "protocols").mkdir()
    (tmp_path / "masks").mkdir()
    Image.new("RGB", (1280, 720)).save(tmp_path / "images" / "sample.png")
    (tmp_path / "protocols" / "sample.json").write_text(
        json.dumps(protocol_dict), encoding="utf-8"
    )
    row = {
        "sample_id": "sample-1",
        "image": "images/sample.png",
        "protocol": "protocols/sample.json",
        "seed": 17,
    }
    if mask_supervision is not None:
        row["mask_supervision"] = mask_supervision
    (tmp_path / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_file_mask_supervision_is_partial_and_grounded(tmp_path, protocol_dict: dict) -> None:
    _write_sample(
        tmp_path,
        protocol_dict,
        mask_supervision={"source": "files", "objects": {"source-a": "masks/a.png"}},
    )
    Image.new("L", (1280, 720), 0).save(tmp_path / "masks" / "a.png")
    mask = Image.open(tmp_path / "masks" / "a.png")
    mask.putpixel((10, 10), 255)
    mask.save(tmp_path / "masks" / "a.png")

    record = ProtocolManifestDataset(
        dataset_root=tmp_path,
        manifest_path="train.jsonl",
        grounding_enabled=True,
    )[0]

    assert record.text_object_ids == ("source-a", "source-b")
    assert record.mask_supervision.source == "files"
    assert set(record.mask_supervision.object_paths) == {"source-a"}
    assert record.grounded_protocol.count('"mask_ref":"<MASK>"') == 2
    targets = prepare_instance_targets(
        record,
        image_grid_thw=torch.tensor([1, 8, 8]),
    )
    assert targets.masks.shape == (2, 32, 32)
    assert targets.mask_supervision.tolist() == [True, False]


def test_geometry_supervision_rasterizes_independent_objects(tmp_path, protocol_dict: dict) -> None:
    _write_sample(
        tmp_path,
        protocol_dict,
        mask_supervision={"source": "geometry"},
    )
    record = ProtocolManifestDataset(
        dataset_root=tmp_path,
        manifest_path="train.jsonl",
        grounding_enabled=True,
    )[0]

    masks = rasterize_protocol_instance_masks(record.protocol)

    assert set(masks) == {"source-a", "source-b"}
    assert masks["source-a"].getbbox() is not None
    assert masks["source-b"].getbbox() is not None
    assert masks["source-a"].getpixel((20, 25)) == 255
    assert masks["source-b"].getpixel((20, 25)) == 255


def test_mask_dimensions_are_validated(tmp_path, protocol_dict: dict) -> None:
    _write_sample(
        tmp_path,
        protocol_dict,
        mask_supervision={"source": "files", "objects": {"source-a": "masks/a.png"}},
    )
    Image.new("L", (10, 10), 255).save(tmp_path / "masks" / "a.png")
    dataset = ProtocolManifestDataset(
        dataset_root=tmp_path,
        manifest_path="train.jsonl",
        grounding_enabled=True,
    )

    with pytest.raises(ValueError, match="does not match canvas"):
        _ = dataset[0]


def test_soft_grayscale_mask_and_secure_path_validation(
    tmp_path,
    protocol_dict: dict,
) -> None:
    _write_sample(
        tmp_path,
        protocol_dict,
        mask_supervision={"source": "files", "objects": {"source-a": "masks/a.png"}},
    )
    soft = Image.new("L", (1280, 720), 0)
    soft.putpixel((10, 10), 128)
    soft.save(tmp_path / "masks" / "a.png")
    dataset = ProtocolManifestDataset(
        dataset_root=tmp_path,
        manifest_path="train.jsonl",
        grounding_enabled=True,
    )
    assert dataset[0].mask_supervision is not None

    row = json.loads((tmp_path / "train.jsonl").read_text(encoding="utf-8"))
    row["mask_supervision"]["objects"]["source-a"] = "../outside.png"
    (tmp_path / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    escaped = ProtocolManifestDataset(
        dataset_root=tmp_path,
        manifest_path="train.jsonl",
        grounding_enabled=True,
    )
    with pytest.raises(ValueError, match="escapes configured root"):
        _ = escaped[0]
