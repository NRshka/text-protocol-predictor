from __future__ import annotations

from types import SimpleNamespace

import torch

from text_render_protocol_predictor.training import ProtocolPromptTemplate, ProtocolSFTCollator


class FakeProcessor:
    def apply_chat_template(self, conversations, **kwargs):
        if not kwargs["tokenize"]:
            texts = []
            for conversation in conversations:
                if conversation[-1]["role"] == "assistant":
                    target = conversation[-1]["content"][0]["text"]
                    texts.append("AB" + target)
                else:
                    texts.append("AB")
            return texts

        assert "padding" not in kwargs
        assert "return_assistant_tokens_mask" not in kwargs
        assert kwargs["processor_kwargs"] == {"padding": True}
        batch_size = len(conversations)
        has_assistant = conversations[0][-1]["role"] == "assistant"
        token_row = [10, 11, 12, 13] if has_assistant else [10, 11]
        input_ids = torch.tensor([token_row] * batch_size)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
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
