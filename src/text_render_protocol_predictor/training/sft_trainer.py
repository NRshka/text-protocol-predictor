"""Accelerate-native SFT loop with resumable state and W&B tracking."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from ..evaluation.runner import evaluate_generation
from .collator import ProtocolGenerationCollator
from .prompts import ProtocolPromptTemplate


@dataclass
class TrainerState:
    global_step: int = 0
    epoch: int = 0
    batch_in_epoch: int = 0


def _trainable_parameters(model: Any) -> list[torch.nn.Parameter]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    return parameters


def _step_scheduler_after_optimizer(
    *, scheduler: Any, sync_gradients: bool, optimizer_step_was_skipped: bool
) -> bool:
    """Advance once per successful global optimizer update, independent of world size."""
    if not sync_gradients or optimizer_step_was_skipped:
        return False
    scheduler.step()
    return True


def _save_checkpoint(
    accelerator: Any,
    model: Any,
    processor: Any,
    state: TrainerState,
    output_dir: Path,
    resolved_config: dict[str, Any],
) -> None:
    checkpoint_dir = output_dir / f"checkpoint-{state.global_step:08d}"
    accelerator.wait_for_everyone()
    accelerator.save_state(str(checkpoint_dir / "accelerate_state"))
    if accelerator.is_main_process:
        artifact_dir = checkpoint_dir / "model"
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(
            artifact_dir,
            is_main_process=True,
            save_function=accelerator.save,
            safe_serialization=True,
        )
        processor.save_pretrained(artifact_dir)
        (checkpoint_dir / "trainer_state.json").write_text(
            json.dumps(asdict(state), indent=2) + "\n", encoding="utf-8"
        )
        (checkpoint_dir / "resolved_config.json").write_text(
            json.dumps(resolved_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    accelerator.wait_for_everyone()


def _load_checkpoint(accelerator: Any, checkpoint: Path) -> TrainerState:
    accelerator.load_state(str(checkpoint / "accelerate_state"))
    with (checkpoint / "trainer_state.json").open("r", encoding="utf-8") as stream:
        return TrainerState(**json.load(stream))


def _token_accuracy_counts(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    predictions = logits[:, :-1].argmax(dim=-1)
    shifted_labels = labels[:, 1:]
    valid = shifted_labels.ne(-100)
    correct = predictions.eq(shifted_labels).logical_and(valid).sum()
    return torch.stack((correct, valid.sum())).to(dtype=torch.float64)


@torch.no_grad()
def evaluate_teacher_forced(
    accelerator: Any, model: Any, dataloader: DataLoader
) -> tuple[float, float]:
    model.eval()
    losses = []
    accuracy_counts = torch.zeros(2, dtype=torch.float64, device=accelerator.device)
    for batch in dataloader:
        output = model(**batch)
        gathered = accelerator.gather_for_metrics(output.loss.detach().repeat(batch["input_ids"].shape[0]))
        losses.append(gathered.float().cpu())
        accuracy_counts += _token_accuracy_counts(output.logits, batch["labels"])
    accuracy_counts = accelerator.reduce(accuracy_counts, reduction="sum")
    model.train()
    if not losses:
        return math.nan, math.nan
    token_accuracy = (
        (accuracy_counts[0] / accuracy_counts[1]).item()
        if accuracy_counts[1].item() > 0
        else math.nan
    )
    return torch.cat(losses).mean().item(), token_accuracy


def train_sft(
    *,
    cfg: Any,
    model: Any,
    processor: Any,
    train_dataset: Any,
    validation_dataset: Any,
    collator: Any,
) -> TrainerState:
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from omegaconf import OmegaConf
    from tqdm.auto import tqdm
    from transformers import get_scheduler

    accelerator = Accelerator(
        gradient_accumulation_steps=int(cfg.training.gradient_accumulation_steps),
        mixed_precision=str(cfg.model.precision),
        log_with="wandb" if cfg.tracking.provider == "wandb" else None,
        # The schedule is configured in global optimizer steps. Accelerate's
        # default prepared scheduler advances once per process when batches are
        # not split, which makes a 4-GPU schedule run 4x too fast.
        step_scheduler_with_optimizer=False,
    )
    set_seed(int(cfg.training.seed), device_specific=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.training.per_device_batch_size),
        shuffle=True,
        collate_fn=collator,
        num_workers=int(cfg.dataset.num_workers),
        pin_memory=True,
        persistent_workers=int(cfg.dataset.num_workers) > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(cfg.evaluation.per_device_batch_size),
        shuffle=False,
        collate_fn=collator,
        num_workers=int(cfg.dataset.num_workers),
        pin_memory=True,
        persistent_workers=int(cfg.dataset.num_workers) > 0,
    )
    generation_loader = None
    if bool(cfg.evaluation.generation.enabled):
        generation_size = min(int(cfg.evaluation.generation.num_samples), len(validation_dataset))
        generation_dataset = Subset(validation_dataset, range(generation_size))
        generation_loader = DataLoader(
            generation_dataset,
            batch_size=int(cfg.evaluation.generation.per_device_batch_size),
            shuffle=False,
            collate_fn=ProtocolGenerationCollator(
                processor=processor,
                prompt_template=ProtocolPromptTemplate(),
            ),
            num_workers=int(cfg.dataset.num_workers),
            pin_memory=True,
            persistent_workers=int(cfg.dataset.num_workers) > 0,
        )
    optimizer = torch.optim.AdamW(
        _trainable_parameters(model),
        lr=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
    )
    scheduler = get_scheduler(
        str(cfg.training.scheduler),
        optimizer=optimizer,
        num_warmup_steps=round(float(cfg.training.warmup_ratio) * int(cfg.training.max_steps)),
        num_training_steps=int(cfg.training.max_steps),
    )
    model, optimizer, train_loader, validation_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, validation_loader, scheduler
    )
    if generation_loader is not None:
        generation_loader = accelerator.prepare(generation_loader)

    resolved_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    resolved_config["training"]["effective_global_batch_size"] = (
        int(cfg.training.per_device_batch_size)
        * accelerator.num_processes
        * int(cfg.training.gradient_accumulation_steps)
    )
    accelerator.init_trackers(
        project_name=str(cfg.tracking.project),
        config=resolved_config,
        init_kwargs={
            "wandb": {
                "entity": cfg.tracking.entity,
                "name": cfg.tracking.run_name,
                "mode": str(cfg.tracking.mode),
            }
        },
    )
    output_dir = Path(cfg.training.output_dir)
    state = TrainerState()
    if cfg.training.resume_from:
        state = _load_checkpoint(accelerator, Path(cfg.training.resume_from))

    progress = tqdm(
        total=int(cfg.training.max_steps),
        initial=state.global_step,
        desc="SFT",
        unit="step",
        dynamic_ncols=True,
        disable=(
            not bool(cfg.training.progress_bar)
            or not accelerator.is_local_main_process
        ),
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    while state.global_step < int(cfg.training.max_steps):
        active_loader = train_loader
        if state.batch_in_epoch:
            active_loader = accelerator.skip_first_batches(train_loader, state.batch_in_epoch)
        for batch_index, batch in enumerate(active_loader, start=state.batch_in_epoch):
            with accelerator.accumulate(model):
                output = model(**batch)
                loss = output.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), float(cfg.training.gradient_clip_norm)
                    )
                optimizer.step()
                _step_scheduler_after_optimizer(
                    scheduler=scheduler,
                    sync_gradients=accelerator.sync_gradients,
                    optimizer_step_was_skipped=optimizer.step_was_skipped,
                )
                optimizer.zero_grad(set_to_none=True)

            state.batch_in_epoch = batch_index + 1
            if not accelerator.sync_gradients:
                continue
            if optimizer.step_was_skipped:
                accelerator.log(
                    {"train/skipped_optimizer_steps": 1},
                    step=state.global_step,
                )
                continue
            state.global_step += 1
            mean_loss = accelerator.gather(loss.detach()).float().mean().item()
            train_accuracy_counts = accelerator.reduce(
                _token_accuracy_counts(output.logits.detach(), batch["labels"]),
                reduction="sum",
            )
            train_token_accuracy = (
                (train_accuracy_counts[0] / train_accuracy_counts[1]).item()
                if train_accuracy_counts[1].item() > 0
                else math.nan
            )
            learning_rate = scheduler.get_last_lr()[0]
            progress.update(1)
            progress.set_postfix(
                epoch=state.epoch + 1,
                loss=f"{mean_loss:.4f}",
                lr=f"{learning_rate:.3e}",
                refresh=True,
            )
            accelerator.log(
                {
                    "train/loss": mean_loss,
                    "train/token_accuracy": train_token_accuracy,
                    "train/learning_rate": learning_rate,
                    "train/epoch": state.epoch,
                },
                step=state.global_step,
            )
            if state.global_step % int(cfg.training.eval_steps) == 0:
                validation_loss, validation_token_accuracy = evaluate_teacher_forced(
                    accelerator, model, validation_loader
                )
                evaluation_metrics = {
                    "validation/loss": validation_loss,
                    "validation/token_accuracy": validation_token_accuracy,
                }
                if generation_loader is not None:
                    validity = evaluate_generation(
                        accelerator=accelerator,
                        model=model,
                        processor=processor,
                        dataloader=generation_loader,
                        max_new_tokens=int(cfg.evaluation.generation.max_new_tokens),
                    )
                    evaluation_metrics.update(validity.as_log_dict())
                accelerator.log(evaluation_metrics, step=state.global_step)
            if state.global_step % int(cfg.training.save_steps) == 0:
                _save_checkpoint(
                    accelerator, model, processor, state, output_dir, resolved_config
                )
            if state.global_step >= int(cfg.training.max_steps):
                break
        else:
            state.epoch += 1
            state.batch_in_epoch = 0
            continue
        break

    _save_checkpoint(accelerator, model, processor, state, output_dir, resolved_config)
    progress.close()
    accelerator.end_training()
    return state
