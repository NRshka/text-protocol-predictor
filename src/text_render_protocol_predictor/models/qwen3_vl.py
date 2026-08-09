"""Qwen3-VL loading and language-decoder LoRA selection."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..protocol.coordinate_tokens import CoordinateTokenCodec


DEFAULT_LORA_LEAVES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

REQUIRED_PEFT_WEIGHT_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "chat_template.jinja",
    "processor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
)


@dataclass(frozen=True)
class CoordinateTokenRegistration:
    """Tokenizer IDs and initialization sources for coordinate vocabulary rows."""

    token_ids: tuple[int, ...]
    source_token_ids: tuple[tuple[int, ...], ...]
    new_token_ids: tuple[int, ...]


def add_coordinate_tokens(
    tokenizer: Any, codec: CoordinateTokenCodec
) -> CoordinateTokenRegistration:
    """Register quoted coordinate values as regular, indivisible tokens."""
    from transformers import AddedToken

    source_token_ids = tuple(
        tuple(tokenizer.encode(str(index), add_special_tokens=False))
        for index in range(codec.bins)
    )
    if any(not ids for ids in source_token_ids):
        raise ValueError("tokenizer could not encode one or more coordinate bin indices")

    existing_vocabulary = tokenizer.get_vocab()
    new_surfaces = {
        surface for surface in codec.tokenizer_tokens if surface not in existing_vocabulary
    }
    added_tokens = [
        AddedToken(surface, special=False, normalized=False)
        for surface in codec.tokenizer_tokens
    ]
    tokenizer.add_tokens(added_tokens, special_tokens=False)

    token_ids: list[int] = []
    special_ids = set(getattr(tokenizer, "all_special_ids", ()))
    for surface in codec.tokenizer_tokens:
        ids = tokenizer.encode(surface, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"coordinate tokenizer surface {surface!r} encoded as {len(ids)} tokens"
            )
        token_id = int(ids[0])
        if token_id in special_ids:
            raise ValueError(
                f"coordinate tokenizer surface {surface!r} was registered as a special token"
            )
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if decoded != surface:
            raise ValueError(
                f"coordinate tokenizer surface {surface!r} decodes as {decoded!r}"
            )
        token_ids.append(token_id)
    if len(set(token_ids)) != codec.bins:
        raise ValueError("coordinate tokenizer surfaces do not have unique token IDs")
    new_token_ids = tuple(
        token_id
        for token_id, surface in zip(token_ids, codec.tokenizer_tokens, strict=True)
        if surface in new_surfaces
    )
    return CoordinateTokenRegistration(
        tuple(token_ids), source_token_ids, new_token_ids
    )


def resize_and_initialize_coordinate_embeddings(
    model: Any,
    tokenizer: Any,
    registration: CoordinateTokenRegistration,
    *,
    initialize_all: bool = False,
) -> None:
    """Resize vocabulary layers and initialize new rows from their numeric spelling."""
    import torch

    input_embeddings = model.get_input_embeddings()
    old_vocab_size = int(input_embeddings.weight.shape[0])
    target_vocab_size = max(len(tokenizer), old_vocab_size)
    if target_vocab_size != old_vocab_size:
        model.resize_token_embeddings(target_vocab_size, mean_resizing=False)

    rows_to_initialize = (
        set(registration.token_ids)
        if initialize_all
        else set(registration.new_token_ids)
    )
    new_token_indices = [
        index
        for index, token_id in enumerate(registration.token_ids)
        if token_id in rows_to_initialize
    ]
    if not new_token_indices:
        return

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    with torch.no_grad():
        for index in new_token_indices:
            token_id = registration.token_ids[index]
            source_ids = torch.tensor(
                registration.source_token_ids[index],
                device=input_embeddings.weight.device,
            )
            input_embeddings.weight[token_id].copy_(
                input_embeddings.weight.index_select(0, source_ids).mean(dim=0)
            )

        if output_embeddings is not None and (
            output_embeddings.weight.data_ptr() != input_embeddings.weight.data_ptr()
        ):
            for index in new_token_indices:
                token_id = registration.token_ids[index]
                source_ids = torch.tensor(
                    registration.source_token_ids[index],
                    device=output_embeddings.weight.device,
                )
                output_embeddings.weight[token_id].copy_(
                    output_embeddings.weight.index_select(0, source_ids).mean(dim=0)
                )


def resize_model_to_tokenizer_vocabulary(model: Any, tokenizer: Any) -> bool:
    """Ensure PEFT can address every token row stored by a checkpoint tokenizer."""
    model_vocab_size = int(model.get_input_embeddings().weight.shape[0])
    target_vocab_size = max(len(tokenizer), model_vocab_size)
    if target_vocab_size == model_vocab_size:
        return False
    model.resize_token_embeddings(target_vocab_size, mean_resizing=False)
    return True


def coordinate_trainable_token_indices(
    model: Any, registration: CoordinateTokenRegistration
) -> list[int] | dict[str, list[int]]:
    """Select both vocabulary layers when Qwen's input/output weights are untied."""
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    indices = list(registration.token_ids)
    if output_embeddings is None or (
        output_embeddings.weight.data_ptr() == input_embeddings.weight.data_ptr()
    ):
        return indices
    return {"embed_tokens": indices, "lm_head": indices}


def _validate_adapter_coordinate_tokens(
    weights_path: Path,
    expected: list[int] | dict[str, list[int]],
) -> None:
    adapter_config = json.loads(
        (weights_path / "adapter_config.json").read_text(encoding="utf-8")
    )
    configured = adapter_config.get("trainable_token_indices")
    if configured != expected:
        raise ValueError(
            "the PEFT checkpoint coordinate-token rows do not match the configured "
            "coordinate vocabulary; start from the base model or use the checkpoint's "
            "coordinate settings"
        )


def inspect_peft_weights_directory(path: str | Path) -> tuple[Path, str]:
    """Validate a local PEFT export and return its base model identifier."""
    weights_path = Path(path).expanduser().resolve()
    if not weights_path.is_dir():
        raise ValueError(f"PEFT weights path is not a directory: {weights_path}")
    missing = [name for name in REQUIRED_PEFT_WEIGHT_FILES if not (weights_path / name).is_file()]
    if missing:
        raise ValueError(
            f"PEFT weights directory {weights_path} is missing required files: "
            f"{', '.join(missing)}"
        )
    try:
        adapter_config = json.loads(
            (weights_path / "adapter_config.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid PEFT adapter_config.json in {weights_path}: {exc}") from exc
    if str(adapter_config.get("peft_type", "")).upper() != "LORA":
        raise ValueError("weights-only initialization currently requires a LoRA PEFT adapter")
    base_model = adapter_config.get("base_model_name_or_path")
    if not isinstance(base_model, str) or not base_model.strip():
        raise ValueError("adapter_config.json must contain base_model_name_or_path")
    return weights_path, base_model


def _language_model_with_prefix(model: Any) -> tuple[str, Any]:
    candidates = [model, getattr(model, "model", None)]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "language_model"):
            language_model = candidate.language_model
            for name, module in model.named_modules():
                if module is language_model:
                    return name, language_model
    raise ValueError("could not locate Qwen3-VL language_model module")


def language_lora_targets(
    model: Any,
    leaves: Sequence[str] = DEFAULT_LORA_LEAVES,
) -> list[str]:
    """Return exact decoder-only linear module names for PEFT."""
    import torch

    prefix, language_model = _language_model_with_prefix(model)
    allowed = set(leaves)
    targets = [
        f"{prefix}.{name}"
        for name, module in language_model.named_modules()
        if name.rsplit(".", 1)[-1] in allowed and isinstance(module, torch.nn.Linear)
    ]
    if not targets:
        raise ValueError(f"no language-decoder LoRA targets found for {sorted(allowed)}")
    return targets


def load_qwen3_vl_for_sft(
    model_cfg: Any,
    lora_cfg: Any,
    coordinate_codec: CoordinateTokenCodec | None = None,
) -> tuple[Any, Any]:
    """Load the processor and model, optionally attaching decoder-only LoRA."""
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dtype_name = str(model_cfg.precision).lower()
    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if dtype_name not in dtypes:
        raise ValueError(f"unsupported precision: {model_cfg.precision}")

    weights_path = getattr(model_cfg, "weights_path", None)
    if weights_path and not bool(lora_cfg.enabled):
        raise ValueError("model.weights_path contains PEFT weights, so lora.enabled must be true")
    if weights_path:
        resolved_weights_path, base_model_name = inspect_peft_weights_directory(weights_path)
        processor_source = str(resolved_weights_path)
    else:
        resolved_weights_path = None
        base_model_name = str(model_cfg.name_or_path)
        processor_source = base_model_name

    processor_kwargs = {}
    if getattr(model_cfg, "image_min_pixels", None) is not None:
        processor_kwargs["min_pixels"] = int(model_cfg.image_min_pixels)
    if getattr(model_cfg, "image_max_pixels", None) is not None:
        processor_kwargs["max_pixels"] = int(model_cfg.image_max_pixels)
    processor = AutoProcessor.from_pretrained(processor_source, **processor_kwargs)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_name,
        dtype=dtypes[dtype_name],
        attn_implementation=getattr(model_cfg, "attn_implementation", "sdpa"),
    )
    coordinate_registration = None
    trainable_token_indices = None
    if coordinate_codec is not None:
        coordinate_registration = add_coordinate_tokens(
            processor.tokenizer, coordinate_codec
        )
        resize_and_initialize_coordinate_embeddings(
            model,
            processor.tokenizer,
            coordinate_registration,
            initialize_all=resolved_weights_path is not None,
        )
        trainable_token_indices = coordinate_trainable_token_indices(
            model, coordinate_registration
        )
    else:
        # A PEFT export may carry an expanded tokenizer even when the caller
        # does not use the corresponding codec directly. PEFT installs its
        # TrainableTokens wrappers before loading adapter state, so the base
        # vocabulary must already contain every referenced row.
        resize_model_to_tokenizer_vocabulary(model, processor.tokenizer)
    if bool(model_cfg.gradient_checkpointing):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if resolved_weights_path is not None:
        from peft import PeftModel

        if trainable_token_indices is not None:
            _validate_adapter_coordinate_tokens(
                resolved_weights_path, trainable_token_indices
            )

        model = PeftModel.from_pretrained(
            model,
            str(resolved_weights_path),
            is_trainable=True,
        )
    elif bool(lora_cfg.enabled):
        from peft import LoraConfig, get_peft_model

        if str(lora_cfg.target_scope) != "language_decoder":
            raise ValueError("the initial SFT implementation supports language_decoder LoRA only")
        targets = language_lora_targets(model)
        peft_config = LoraConfig(
            r=int(lora_cfg.rank),
            lora_alpha=int(lora_cfg.alpha),
            lora_dropout=float(lora_cfg.dropout),
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
            trainable_token_indices=trainable_token_indices,
        )
        model = get_peft_model(model, peft_config)
    else:
        base = getattr(model, "model", model)
        if bool(model_cfg.freeze_vision_tower) and hasattr(base, "visual"):
            base.visual.requires_grad_(False)

    return model, processor
