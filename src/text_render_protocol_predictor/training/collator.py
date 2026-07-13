"""Qwen3-VL collation with loss restricted to assistant target tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..data.dataset import ProtocolDatasetRecord
from .prompts import ProtocolPromptTemplate


@dataclass
class ProtocolSFTCollator:
    processor: Any
    prompt_template: ProtocolPromptTemplate
    max_sequence_tokens: int | None = None
    max_output_tokens: int | None = None
    ignore_index: int = -100

    def __call__(self, records: list[ProtocolDatasetRecord]) -> dict[str, torch.Tensor]:
        conversations = [
            self.prompt_template.conversation(
                image=record.image_path,
                width=record.canvas_width,
                height=record.canvas_height,
                target=record.canonical_protocol,
            )
            for record in records
        ]
        batch = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        mask_key = "assistant_masks" if "assistant_masks" in batch else "assistant_mask"
        if mask_key not in batch:
            raise RuntimeError(
                "Qwen3-VL chat template did not return an assistant token mask; "
                "assistant-only SFT loss cannot be guaranteed"
            )
        assistant_mask = batch.pop(mask_key).bool()
        input_ids = batch["input_ids"]
        if assistant_mask.shape != input_ids.shape:
            raise RuntimeError("assistant token mask shape does not match input_ids")
        if not torch.all(assistant_mask.any(dim=1)):
            raise RuntimeError("one or more examples contain no assistant target tokens")
        if self.max_output_tokens is not None:
            longest_output = int(assistant_mask.sum(dim=1).max().item())
            if longest_output > self.max_output_tokens:
                raise ValueError(
                    f"assistant target length {longest_output} exceeds configured maximum "
                    f"{self.max_output_tokens}; samples are rejected rather than truncated"
                )
        if self.max_sequence_tokens is not None and input_ids.shape[1] > self.max_sequence_tokens:
            raise ValueError(
                f"batch sequence length {input_ids.shape[1]} exceeds configured maximum "
                f"{self.max_sequence_tokens}; samples are rejected rather than truncated"
            )
        labels = input_ids.clone()
        labels.masked_fill_(~assistant_mask, self.ignore_index)
        if "attention_mask" in batch:
            labels.masked_fill_(~batch["attention_mask"].bool(), self.ignore_index)
        batch["labels"] = labels
        return dict(batch)
