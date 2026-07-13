"""Accelerate-native SFT loop with resumable state and W&B tracking."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


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


@torch.no_grad()
def evaluate_loss(accelerator: Any, model: Any, dataloader: DataLoader) -> float:
    model.eval()
    losses = []
    for batch in dataloader:
        output = model(**batch)
        gathered = accelerator.gather_for_metrics(output.loss.detach().repeat(batch["input_ids"].shape[0]))
        losses.append(gathered.float().cpu())
    model.train()
    if not losses:
        return math.nan
    return torch.cat(losses).mean().item()


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
    from transformers import get_scheduler

    accelerator = Accelerator(
        gradient_accumulation_steps=int(cfg.training.gradient_accumulation_steps),
        mixed_precision=str(cfg.model.precision),
        log_with="wandb" if cfg.tracking.provider == "wandb" else None,
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
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            state.batch_in_epoch = batch_index + 1
            if not accelerator.sync_gradients:
                continue
            state.global_step += 1
            mean_loss = accelerator.gather(loss.detach()).float().mean().item()
            accelerator.log(
                {
                    "train/loss": mean_loss,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/epoch": state.epoch,
                },
                step=state.global_step,
            )
            if state.global_step % int(cfg.training.eval_steps) == 0:
                validation_loss = evaluate_loss(accelerator, model, validation_loader)
                accelerator.log({"validation/loss": validation_loss}, step=state.global_step)
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
    accelerator.end_training()
    return state
