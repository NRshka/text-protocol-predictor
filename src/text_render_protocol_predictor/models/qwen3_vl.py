"""Qwen3-VL loading and language-decoder LoRA selection."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


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


def load_qwen3_vl_for_sft(model_cfg: Any, lora_cfg: Any) -> tuple[Any, Any]:
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
    if bool(model_cfg.gradient_checkpointing):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if resolved_weights_path is not None:
        from peft import PeftModel

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
        )
        model = get_peft_model(model, peft_config)
    else:
        base = getattr(model, "model", model)
        if bool(model_cfg.freeze_vision_tower) and hasattr(base, "visual"):
            base.visual.requires_grad_(False)

    return model, processor
