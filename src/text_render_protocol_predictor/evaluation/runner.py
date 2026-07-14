"""Distributed generation evaluation for a fixed validation subset."""

from __future__ import annotations

from typing import Any

import torch

from .generation import GenerationValidityMetrics, evaluate_generation_validity


@torch.no_grad()
def evaluate_generation(
    *,
    accelerator: Any,
    model: Any,
    processor: Any,
    dataloader: Any,
    max_new_tokens: int,
) -> GenerationValidityMetrics:
    from accelerate.utils import gather_object

    was_training = model.training
    model.eval()
    generation_model = accelerator.unwrap_model(model)
    local_results: list[tuple[str, str]] = []
    for batch in dataloader:
        sample_ids = batch.pop("_sample_ids")
        input_length = batch["input_ids"].shape[1]
        generated = generation_model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            synced_gpus=accelerator.num_processes > 1,
        )
        completions = generated[:, input_length:]
        decoded = processor.batch_decode(
            completions,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        local_results.extend(zip(sample_ids, decoded, strict=True))

    gathered_results = gather_object(local_results)
    unique_outputs: dict[str, str] = {}
    for sample_id, output in gathered_results:
        unique_outputs.setdefault(sample_id, output)
    if was_training:
        model.train()
    return evaluate_generation_validity(list(unique_outputs.values()))

