from __future__ import annotations

from dataclasses import dataclass

import pytest
from PIL import Image

from text_render_protocol_predictor.training.grpo_trainer import (
    build_hf_grpo_dataset,
    grpo_conversation,
)


@dataclass
class Record:
    sample_id: str
    image_path: object
    background_path: object
    text_mask_path: object
    canvas_width: int = 16
    canvas_height: int = 12
    mask_coverage: float = 0.1


class Records:
    def __init__(self, record):
        self.record = record

    def __len__(self):
        return 1

    def __getitem__(self, index):
        assert index == 0
        return self.record


def test_grpo_conversation_leaves_image_in_separate_dataset_column():
    messages = grpo_conversation(width=16, height=12, protocol_version="1.0")

    assert [message["role"] for message in messages] == ["system", "user"]
    assert "Canvas size: 16 x 12" in messages[1]["content"]
    assert isinstance(messages[1]["content"], str)


def test_hf_dataset_exposes_original_to_policy_and_reward_paths(tmp_path):
    pytest.importorskip("datasets")
    paths = []
    for name in ("original", "background", "mask"):
        path = tmp_path / f"{name}.webp"
        Image.new("RGB", (16, 12)).save(path, format="WEBP", lossless=True)
        paths.append(path)
    dataset = build_hf_grpo_dataset(
        Records(Record("sample", *paths)),
        protocol_version="1.0",
    )

    row = dataset[0]
    assert row["image"].size == (16, 12)
    assert row["original_path"] == str(paths[0])
    assert row["background_path"] == str(paths[1])
    assert row["text_mask_path"] == str(paths[2])
