"""Qwen3-VL collation with loss restricted to assistant target tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..data.dataset import ProtocolDatasetRecord
from .prompts import ProtocolPromptTemplate
from .mask_targets import prepare_instance_targets


def locate_completion_token_positions(
    full_batch: Any,
    prompt_batch: Any,
    *,
    token_id: int,
) -> list[torch.Tensor]:
    """Locate a token only in assistant completion spans, excluding the prompt."""
    full_attention = full_batch["attention_mask"].bool()
    prompt_attention = prompt_batch["attention_mask"].bool()
    results: list[torch.Tensor] = []
    for index in range(full_batch["input_ids"].shape[0]):
        full_positions = torch.nonzero(
            full_attention[index],
            as_tuple=False,
        ).squeeze(1)
        prompt_ids = prompt_batch["input_ids"][index][prompt_attention[index]]
        if len(full_positions) < len(prompt_ids):
            raise RuntimeError("full conversation is shorter than its prompt")
        if not torch.equal(
            full_batch["input_ids"][index][full_positions[: len(prompt_ids)]],
            prompt_ids,
        ):
            raise RuntimeError(
                "tokenized full conversation does not start with tokenized prompt"
            )
        completion_positions = full_positions[len(prompt_ids) :]
        completion_ids = full_batch["input_ids"][index].index_select(
            0,
            completion_positions,
        )
        results.append(completion_positions[completion_ids.eq(token_id)])
    return results


@dataclass
class ProtocolSFTCollator:
    processor: Any
    prompt_template: ProtocolPromptTemplate
    max_sequence_tokens: int | None = None
    max_output_tokens: int | None = None
    ignore_index: int = -100
    mask_token_id: int | None = None
    vision_patch_size: int = 16
    mask_output_stride: int = 4

    def __call__(self, records: list[ProtocolDatasetRecord]) -> dict[str, Any]:
        full_conversations = [
            self.prompt_template.conversation(
                image=record.image_path,
                width=record.canvas_width,
                height=record.canvas_height,
                protocol_version=getattr(record, "protocol_version", "1.0"),
                target=(
                    record.grounded_protocol
                    if self.prompt_template.grounding_enabled
                    else record.canonical_protocol
                ),
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
            target = (
                record.grounded_protocol
                if self.prompt_template.grounding_enabled
                else record.canonical_protocol
            )
            end = start + len(target)
            if full_text[start:end] != target:
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
        if self.prompt_template.grounding_enabled:
            if self.mask_token_id is None:
                raise ValueError("grounded collation requires mask_token_id")
            counts = [len(record.text_object_ids) for record in records]
            maximum = max(counts, default=0)
            positions = torch.full(
                (len(records), maximum), -1, dtype=torch.long
            )
            instance_valid = torch.zeros(
                (len(records), maximum), dtype=torch.bool
            )
            prepared = []
            completion_mask_positions = locate_completion_token_positions(
                batch,
                prompt_batch,
                token_id=self.mask_token_id,
            )
            for index, (record, count) in enumerate(zip(records, counts, strict=True)):
                token_positions = completion_mask_positions[index]
                if len(token_positions) != count:
                    raise RuntimeError(
                        f"sample {record.sample_id!r} has {len(token_positions)} mask tokens "
                        f"for {count} text objects"
                    )
                if count:
                    positions[index, :count] = token_positions
                    instance_valid[index, :count] = True
                prepared.append(
                    prepare_instance_targets(
                        record,
                        image_grid_thw=batch["image_grid_thw"][index],
                        patch_size=self.vision_patch_size,
                        output_stride=self.mask_output_stride,
                    )
                )
            batch["mask_token_positions"] = positions
            batch["instance_valid"] = instance_valid
            batch["instance_masks"] = [item.masks for item in prepared]
            batch["mask_supervision"] = [
                item.mask_supervision for item in prepared
            ]
            batch["geometry_modes"] = [item.modes for item in prepared]
            batch["geometry_boxes"] = [item.boxes for item in prepared]
            batch["geometry_rotations"] = [item.rotations for item in prepared]
            batch["geometry_beziers"] = [item.beziers for item in prepared]
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
        batch["_records"] = records
        return dict(batch)
