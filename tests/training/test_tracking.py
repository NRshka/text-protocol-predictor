from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from text_render_protocol_predictor.training.sft_trainer import _tracking_configuration
from text_render_protocol_predictor.training.tracking import (
    HydraConfigCallback,
    store_hydra_config,
)


def tracking_config(provider: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        provider=provider,
        project="strp",
        run_name="sft-run",
        wandb=SimpleNamespace(entity="team", mode="offline"),
        clearml=SimpleNamespace(task_type="training", reuse_last_task_id=False),
    )


def test_wandb_tracking_configuration() -> None:
    tracker, init_kwargs = _tracking_configuration(tracking_config("wandb"))

    assert tracker == "wandb"
    assert init_kwargs == {
        "wandb": {"entity": "team", "name": "sft-run", "mode": "offline"}
    }


def test_clearml_tracking_configuration() -> None:
    tracker, init_kwargs = _tracking_configuration(tracking_config("clearml"))

    assert tracker == "clearml"
    assert init_kwargs == {
        "clearml": {
            "project_name": "strp",
            "task_name": "sft-run",
            "task_type": "training",
            "reuse_last_task_id": False,
        }
    }


@pytest.mark.parametrize("provider", [None, "none", "disabled"])
def test_tracking_can_be_disabled(provider: str | None) -> None:
    assert _tracking_configuration(tracking_config(provider)) == (None, {})


def test_unknown_tracking_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported tracking provider"):
        _tracking_configuration(tracking_config("other"))


def test_resolved_hydra_config_is_stored_in_wandb_namespace() -> None:
    tracker = Mock()
    config = {"model": {"name": "test-model"}, "training": {"seed": 7}}

    store_hydra_config("wandb", tracker, config)

    tracker.config.update.assert_called_once_with(
        {"hydra": config},
        allow_val_change=True,
    )


def test_resolved_hydra_config_is_stored_in_clearml_section() -> None:
    tracker = Mock()
    config = {"model": {"name": "test-model"}, "training": {"seed": 7}}

    store_hydra_config("clearml", tracker, config)

    tracker.connect_configuration.assert_called_once_with(
        config,
        name="Hydra",
        description="Complete resolved Hydra configuration used for this run.",
        ignore_remote_overrides=True,
    )


def test_grpo_hydra_callback_only_logs_on_world_process_zero(monkeypatch) -> None:
    tracker = Mock()
    active_tracker = Mock(return_value=tracker)
    monkeypatch.setattr(
        "text_render_protocol_predictor.training.tracking._active_tracker",
        active_tracker,
    )
    callback = HydraConfigCallback("wandb", {"reward": {"outside_weight": 0.1}})

    callback.on_train_begin(None, SimpleNamespace(is_world_process_zero=False), None)
    active_tracker.assert_not_called()

    callback.on_train_begin(None, SimpleNamespace(is_world_process_zero=True), None)
    active_tracker.assert_called_once_with("wandb")
    tracker.config.update.assert_called_once_with(
        {"hydra": {"reward": {"outside_weight": 0.1}}},
        allow_val_change=True,
    )
