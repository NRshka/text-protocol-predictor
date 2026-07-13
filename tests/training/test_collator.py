from __future__ import annotations

from types import SimpleNamespace

import torch

from text_render_protocol_predictor.training import ProtocolPromptTemplate, ProtocolSFTCollator


class FakeProcessor:
    def apply_chat_template(self, conversations, **kwargs):
        batch_size = len(conversations)
        input_ids = torch.tensor([[10, 11, 12, 13]] * batch_size)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "assistant_masks": torch.tensor([[0, 0, 1, 1]] * batch_size),
        }


def test_collator_masks_everything_except_assistant() -> None:
    record = SimpleNamespace(
        image_path="image.png",
        canvas_width=100,
        canvas_height=50,
        canonical_protocol='{"protocol_version":"1.0","canvas":{"width":100,"height":50},"objects":[]}',
    )
    collator = ProtocolSFTCollator(FakeProcessor(), ProtocolPromptTemplate())
    batch = collator([record])
    assert batch["labels"].tolist() == [[-100, -100, 12, 13]]
    assert "assistant_masks" not in batch


def test_collator_rejects_overlength_without_truncation() -> None:
    record = SimpleNamespace(
        image_path="image.png",
        canvas_width=100,
        canvas_height=50,
        canonical_protocol="{}",
    )
    collator = ProtocolSFTCollator(
        FakeProcessor(), ProtocolPromptTemplate(), max_sequence_tokens=3
    )
    try:
        collator([record])
    except ValueError as exc:
        assert "rejected rather than truncated" in str(exc)
    else:
        raise AssertionError("expected overlength batch to be rejected")


def test_collator_enforces_output_token_limit() -> None:
    record = SimpleNamespace(
        image_path="image.png",
        canvas_width=100,
        canvas_height=50,
        canonical_protocol="{}",
    )
    collator = ProtocolSFTCollator(
        FakeProcessor(), ProtocolPromptTemplate(), max_output_tokens=1
    )
    try:
        collator([record])
    except ValueError as exc:
        assert "assistant target length" in str(exc)
    else:
        raise AssertionError("expected assistant target to exceed the output limit")
