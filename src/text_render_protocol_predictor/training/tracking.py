"""Shared experiment-tracker support for resolved Hydra configurations."""

from __future__ import annotations

from typing import Any

from transformers import TrainerCallback


def store_hydra_config(provider: str, tracker: Any, config: dict[str, Any]) -> None:
    """Store the complete resolved Hydra config in a provider-specific namespace."""
    if provider == "wandb":
        tracker.config.update({"hydra": config}, allow_val_change=True)
        return
    if provider == "clearml":
        tracker.connect_configuration(
            config,
            name="Hydra",
            description="Complete resolved Hydra configuration used for this run.",
            ignore_remote_overrides=True,
        )
        return
    raise ValueError(f"unsupported tracking provider {provider!r}")


def _active_tracker(provider: str) -> Any:
    """Return the tracker initialized by the Transformers reporting callback."""
    if provider == "wandb":
        import wandb

        tracker = wandb.run
    elif provider == "clearml":
        from clearml import Task

        tracker = Task.current_task()
    else:
        raise ValueError(f"unsupported tracking provider {provider!r}")
    if tracker is None:
        raise RuntimeError(f"{provider} did not initialize an experiment tracker")
    return tracker


class HydraConfigCallback(TrainerCallback):
    """Log the resolved Hydra config after Transformers initializes its tracker."""

    def __init__(self, provider: str, config: dict[str, Any]) -> None:
        self.provider = provider
        self.config = config

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if state.is_world_process_zero:
            store_hydra_config(
                self.provider,
                _active_tracker(self.provider),
                self.config,
            )
