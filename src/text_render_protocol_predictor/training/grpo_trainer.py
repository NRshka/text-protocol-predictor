"""TRL GRPO dataset conversion, configuration, and training entry point."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .prompts import ProtocolPromptTemplate


def _append_rendered_images_to_rows(
    image_rows: Any,
    rendered_images: list[Any],
) -> int:
    """Append candidates to TRL's per-completion ``[original]`` image lists."""
    if not image_rows or not rendered_images:
        return 0
    target_rows = list(image_rows)[-len(rendered_images) :]
    appended = 0
    for image_row, rendered_image in zip(target_rows, rendered_images, strict=True):
        if rendered_image is not None and isinstance(image_row, list):
            image_row.append(rendered_image)
            appended += 1
    return appended


def _append_finite_metric(metric_store: dict[str, list[float]], name: str, value: float) -> bool:
    """Skip undefined all-invalid batch diagnostics instead of poisoning an epoch mean."""
    if not math.isfinite(value):
        return False
    metric_store.setdefault(name, []).append(value)
    return True


def grpo_conversation(
    *,
    width: int,
    height: int,
    protocol_version: str,
    prompt_template: ProtocolPromptTemplate | None = None,
) -> list[dict[str, str]]:
    """Return text-only messages; TRL injects the separate image column."""
    template = prompt_template or ProtocolPromptTemplate()
    return [
        {"role": "system", "content": template.system},
        {
            "role": "user",
            "content": template.user_text(width, height, protocol_version),
        },
    ]


def build_hf_grpo_dataset(
    dataset: Any,
    *,
    protocol_version: str,
    prompt_template: ProtocolPromptTemplate | None = None,
) -> Any:
    """Materialize validated records as the column layout expected by TRL."""
    from datasets import Dataset, Image as DatasetImage

    rows = []
    for index in range(len(dataset)):
        record = dataset[index]
        rows.append(
            {
                "prompt": grpo_conversation(
                    width=record.canvas_width,
                    height=record.canvas_height,
                    protocol_version=protocol_version,
                    prompt_template=prompt_template,
                ),
                # The policy sees the original image. The erased image is only
                # the renderer substrate used by the reward.
                "image": str(record.image_path),
                "sample_id": record.sample_id,
                "original_path": str(record.image_path),
                "background_path": str(record.background_path),
                "text_mask_path": str(record.text_mask_path),
                "canvas_width": record.canvas_width,
                "canvas_height": record.canvas_height,
                "mask_coverage": record.mask_coverage,
                "reference_words": [
                    word.model_dump(mode="json") for word in record.words
                ],
            }
        )
    return Dataset.from_list(rows).cast_column("image", DatasetImage(decode=True))


def _report_to(tracking_cfg: Any) -> str:
    provider = str(getattr(tracking_cfg, "provider", "none")).strip().lower()
    if provider in {"", "none", "null", "disabled"}:
        return "none"
    if provider not in {"wandb", "clearml"}:
        raise ValueError(
            f"unsupported tracking provider {provider!r}; expected wandb, clearml, or none"
        )
    return provider


def build_grpo_config(cfg: Any) -> Any:
    from trl import GRPOConfig

    precision = str(cfg.model.precision).lower()
    if precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError(f"unsupported precision: {cfg.model.precision}")
    evaluation_enabled = bool(cfg.evaluation.enabled)
    return GRPOConfig(
        output_dir=str(cfg.grpo.output_dir),
        seed=int(cfg.grpo.seed),
        data_seed=int(cfg.grpo.seed),
        max_steps=int(cfg.grpo.max_steps),
        per_device_train_batch_size=int(cfg.grpo.per_device_batch_size),
        per_device_eval_batch_size=int(cfg.evaluation.per_device_batch_size),
        gradient_accumulation_steps=int(cfg.grpo.gradient_accumulation_steps),
        learning_rate=float(cfg.grpo.learning_rate),
        weight_decay=float(cfg.grpo.weight_decay),
        warmup_steps=round(float(cfg.grpo.warmup_ratio) * int(cfg.grpo.max_steps)),
        lr_scheduler_type=str(cfg.grpo.scheduler),
        max_grad_norm=float(cfg.grpo.gradient_clip_norm),
        bf16=precision == "bf16",
        fp16=precision == "fp16",
        gradient_checkpointing=bool(cfg.model.gradient_checkpointing),
        use_cache=False,
        disable_dropout=True,
        num_generations=int(cfg.grpo.num_generations),
        num_generations_eval=int(cfg.evaluation.num_generations),
        max_completion_length=int(cfg.model.max_output_tokens),
        temperature=float(cfg.grpo.temperature),
        top_p=float(cfg.grpo.top_p),
        top_k=int(cfg.grpo.top_k),
        repetition_penalty=float(cfg.grpo.repetition_penalty),
        beta=float(cfg.grpo.beta),
        epsilon=float(cfg.grpo.epsilon),
        num_iterations=int(cfg.grpo.num_iterations),
        scale_rewards=str(cfg.grpo.scale_rewards),
        loss_type="grpo",
        mask_truncated_completions=False,
        use_vllm=bool(cfg.grpo.use_vllm),
        logging_steps=int(cfg.grpo.logging_steps),
        logging_first_step=True,
        log_completions=bool(cfg.grpo.log_completions),
        save_strategy="steps",
        save_steps=int(cfg.grpo.save_steps),
        save_total_limit=int(cfg.grpo.save_total_limit),
        eval_strategy="steps" if evaluation_enabled else "no",
        eval_steps=int(cfg.evaluation.eval_steps) if evaluation_enabled else None,
        remove_unused_columns=False,
        dataloader_num_workers=int(cfg.dataset.num_workers),
        dataloader_pin_memory=True,
        dataloader_persistent_workers=int(cfg.dataset.num_workers) > 0,
        report_to=_report_to(cfg.tracking),
        project=str(cfg.tracking.project),
        run_name=cfg.tracking.run_name,
    )


def train_grpo(
    *,
    cfg: Any,
    model: Any,
    processor: Any,
    train_dataset: Any,
    validation_dataset: Any | None,
    reward: Any,
    renderer_metadata: dict[str, Any],
) -> Any:
    import torch
    from accelerate.utils import gather_object
    from omegaconf import OmegaConf
    from trl import GRPOTrainer

    from ..rendering import RenderStatus

    class ProtocolGRPOTrainer(GRPOTrainer):
        """Add renderer diagnostics without changing the optimized reward."""

        def _generate_and_score_completions(
            self, inputs: Any, *args: Any, **kwargs: Any
        ) -> Any:
            # TRL releases differ here: some pass an explicit `mode`, while
            # others infer it internally and call this method with only
            # `inputs`. Forward the exact call shape to remain compatible with
            # both APIs.
            result = super()._generate_and_score_completions(inputs, *args, **kwargs)
            mode = kwargs.get("mode")
            if mode is None and args:
                mode = args[0]
            if mode is None:
                mode = "train" if self.model.training else "eval"
            breakdowns = reward.last_breakdowns
            if not breakdowns:
                return result
            statuses = list(RenderStatus)
            rows = []
            for item in breakdowns:
                rows.append(
                    [
                        item.background_masked_mae,
                        item.masked_mae if item.masked_mae is not None else float("nan"),
                        item.outside_mae if item.outside_mae is not None else float("nan"),
                        item.restoration_delta
                        if item.restoration_delta is not None
                        else float("nan"),
                        item.word_precision
                        if item.word_precision is not None
                        else float("nan"),
                        item.word_recall
                        if item.word_recall is not None
                        else float("nan"),
                        item.word_score if item.word_score is not None else float("nan"),
                        float(item.reference_word_count),
                        float(item.predicted_word_count),
                        float(item.matched_word_count),
                        1.0 if item.reference_word_count > 0 else 0.0,
                        (
                            float(item.empty_word_prediction)
                            if item.empty_word_prediction is not None
                            else float("nan")
                        ),
                        item.layout_strict_iou
                        if item.layout_strict_iou is not None
                        else float("nan"),
                        item.layout_iou
                        if item.layout_iou is not None
                        else float("nan"),
                        item.layout_precision
                        if item.layout_precision is not None
                        else float("nan"),
                        item.layout_recall
                        if item.layout_recall is not None
                        else float("nan"),
                        item.layout_target_coverage
                        if item.layout_target_coverage is not None
                        else float("nan"),
                        item.layout_predicted_coverage
                        if item.layout_predicted_coverage is not None
                        else float("nan"),
                        *(1.0 if item.status is status else 0.0 for status in statuses),
                    ]
                )
            local = torch.tensor(rows, dtype=torch.float32, device=self.accelerator.device)
            gathered = self.accelerator.gather(local)
            names = [
                "reconstruction/background_masked_mae",
                "reconstruction/masked_mae",
                "reconstruction/outside_mae",
                "reconstruction/restoration_delta",
                "content/word_precision",
                "content/word_recall",
                "content/word_score",
                "content/reference_word_count",
                "content/predicted_word_count",
                "content/matched_word_count",
                "content/reference_available_rate",
                "content/empty_prediction_rate",
                "layout/strict_iou",
                "layout/dilated_iou",
                "layout/precision",
                "layout/recall",
                "layout/dilated_target_coverage",
                "layout/dilated_predicted_coverage",
                *(f"render_status/{status.value}_rate" for status in statuses),
            ]
            mode_metrics = self._metrics.get(mode)
            metric_store = mode_metrics if isinstance(mode_metrics, dict) else self._metrics
            for column, name in enumerate(names):
                value = torch.nanmean(gathered[:, column]).item()
                _append_finite_metric(metric_store, name, value)
            if bool(cfg.grpo.log_completions) and bool(cfg.grpo.log_rendered_images):
                rendered_images = gather_object(
                    [item.rendered_image for item in breakdowns]
                )
                _append_rendered_images_to_rows(
                    self._logs.get("images"),
                    rendered_images,
                )
            return result

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "left"
    tracking_provider = _report_to(cfg.tracking)
    if tracking_provider == "wandb":
        os.environ.setdefault("WANDB_PROJECT", str(cfg.tracking.project))
    elif tracking_provider == "clearml":
        os.environ.setdefault("CLEARML_PROJECT", str(cfg.tracking.project))
        if cfg.tracking.run_name:
            os.environ.setdefault("CLEARML_TASK", str(cfg.tracking.run_name))
    arguments = build_grpo_config(cfg)
    trainer = ProtocolGRPOTrainer(
        model=model,
        reward_funcs=reward,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=processor,
    )

    # TRL 0.29 clones an already-loaded PEFT adapter to a frozen `ref`
    # adapter when beta is nonzero. Fail loudly if an incompatible TRL
    # release silently loses the intended SFT reference policy.
    if float(cfg.grpo.beta) > 0:
        peft_config = getattr(trainer.model, "peft_config", {})
        if "ref" not in peft_config:
            raise RuntimeError(
                "GRPO beta is nonzero, but TRL did not create the frozen `ref` PEFT adapter; "
                "use the supported TRL version or set beta=0 explicitly"
            )

    trainer.train(resume_from_checkpoint=cfg.grpo.resume_from)
    final_dir = Path(cfg.grpo.output_dir) / "final_model"
    trainer.save_model(str(final_dir))
    if trainer.is_world_process_zero():
        processor.save_pretrained(final_dir)
        metadata = {
            "resolved_config": OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            "reward_config": asdict(reward.config),
            "renderer": renderer_metadata,
        }
        final_dir.mkdir(parents=True, exist_ok=True)
        (final_dir / "grpo_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return trainer.state
