from types import SimpleNamespace

import pytest

from text_render_protocol_predictor.training.sft_trainer import _tracking_configuration


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
