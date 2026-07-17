"""Distributed generation evaluation for a fixed validation subset."""

from __future__ import annotations

from typing import Any

import torch

from .generation import GenerationValidityMetrics, evaluate_generation_predictions


@torch.no_grad()
def evaluate_generation(
    *,
    accelerator: Any,
    model: Any,
    processor: Any,
    dataloader: Any,
    max_new_tokens: int,
    progress_bar: bool = True,
) -> GenerationValidityMetrics:
    from accelerate.utils import gather_object
    from tqdm.auto import tqdm

    was_training = model.training
    model.eval()
    generation_model = accelerator.unwrap_model(model)
    local_results: list[tuple[str, str, str]] = []
    batches = tqdm(
        dataloader,
        desc="Generation evaluation",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
        disable=not progress_bar or not accelerator.is_local_main_process,
    )
    for batch in batches:
        sample_ids = batch.pop("_sample_ids")
        targets = batch.pop("_targets")
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
        local_results.extend(zip(sample_ids, decoded, targets, strict=True))

    gathered_results = gather_object(local_results)
    unique_results: dict[str, tuple[str, str]] = {}
    for sample_id, output, target in gathered_results:
        unique_results.setdefault(sample_id, (output, target))
    if was_training:
        model.train()
    outputs = [result[0] for result in unique_results.values()]
    targets = [result[1] for result in unique_results.values()]
    return evaluate_generation_predictions(outputs, targets)
