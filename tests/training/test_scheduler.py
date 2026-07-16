from text_render_protocol_predictor.training.sft_trainer import (
    _step_scheduler_after_optimizer,
)


class CountingScheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


def test_scheduler_steps_once_per_synchronized_optimizer_update() -> None:
    scheduler = CountingScheduler()

    for micro_step in range(8):
        stepped = _step_scheduler_after_optimizer(
            scheduler=scheduler,
            sync_gradients=micro_step == 7,
            optimizer_step_was_skipped=False,
        )
        assert stepped is (micro_step == 7)

    assert scheduler.steps == 1


def test_scheduler_does_not_step_when_optimizer_update_is_skipped() -> None:
    scheduler = CountingScheduler()

    stepped = _step_scheduler_after_optimizer(
        scheduler=scheduler,
        sync_gradients=True,
        optimizer_step_was_skipped=True,
    )

    assert not stepped
    assert scheduler.steps == 0
