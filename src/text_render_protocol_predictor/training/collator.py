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
        full_conversations = [
            self.prompt_template.conversation(
                image=record.image_path,
                width=record.canvas_width,
                height=record.canvas_height,
                protocol_version=getattr(record, "protocol_version", "1.0"),
                target=record.canonical_protocol,
            )
            for record in records
        ]
        prompt_conversations = [
            self.prompt_template.conversation(
                image=record.image_path,
                width=record.canvas_width,
                height=record.canvas_height,
                protocol_version=getattr(record, "protocol_version", "1.0"),
            )
            for record in records
        ]
        full_texts = self.processor.apply_chat_template(
            full_conversations,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_texts = self.processor.apply_chat_template(
            prompt_conversations,
            tokenize=False,
            add_generation_prompt=True,
        )
        if isinstance(full_texts, str):
            full_texts = [full_texts]
        if isinstance(prompt_texts, str):
            prompt_texts = [prompt_texts]

        for full_text, prompt_text, record in zip(
            full_texts, prompt_texts, records, strict=True
        ):
            if not full_text.startswith(prompt_text):
                raise RuntimeError("full chat rendering does not start with generation prompt")
            start = len(prompt_text)
            end = start + len(record.canonical_protocol)
            if full_text[start:end] != record.canonical_protocol:
                raise RuntimeError("canonical target is not contiguous in rendered assistant message")

        prompt_batch = self.processor.apply_chat_template(
            prompt_conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "padding": True,
            },
        )
        batch = self.processor.apply_chat_template(
            full_conversations,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "padding": True,
            },
        )
        input_ids = batch["input_ids"]
        assistant_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        full_attention = batch["attention_mask"].bool()
        prompt_attention = prompt_batch["attention_mask"].bool()
        for index in range(input_ids.shape[0]):
            full_positions = torch.nonzero(full_attention[index], as_tuple=False).squeeze(1)
            prompt_ids = prompt_batch["input_ids"][index][prompt_attention[index]]
            if len(full_positions) <= len(prompt_ids):
                raise RuntimeError("rendered assistant completion contains no target tokens")
            full_prefix = input_ids[index][full_positions[: len(prompt_ids)]]
            if not torch.equal(full_prefix, prompt_ids):
                raise RuntimeError(
                    "tokenized full conversation does not start with tokenized generation prompt"
                )
            assistant_mask[index, full_positions[len(prompt_ids) :]] = True
        if not torch.all(assistant_mask.any(dim=1)):
            raise RuntimeError("failed to map one or more assistant targets to tokens")
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


@dataclass
class ProtocolGenerationCollator:
    """Build prompt-only batches while retaining IDs for distributed deduplication."""

    processor: Any
    prompt_template: ProtocolPromptTemplate

    def __call__(self, records: list[ProtocolDatasetRecord]) -> dict[str, Any]:
        conversations = [
            self.prompt_template.conversation(
                image=record.image_path,
                width=record.canvas_width,
                height=record.canvas_height,
                protocol_version=getattr(record, "protocol_version", "1.0"),
            )
            for record in records
        ]
        batch = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
        batch["_sample_ids"] = [record.sample_id for record in records]
        batch["_targets"] = [record.canonical_protocol for record in records]
        return dict(batch)
