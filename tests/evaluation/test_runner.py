import sys
from types import ModuleType, SimpleNamespace

import torch

from text_render_protocol_predictor.evaluation.runner import evaluate_generation


def test_generation_evaluation_reports_batch_progress(monkeypatch) -> None:
    progress_args = {}

    def fake_tqdm(iterable, **kwargs):
        progress_args.update(kwargs)
        return iterable

    accelerate = ModuleType("accelerate")
    accelerate_utils = ModuleType("accelerate.utils")
    accelerate_utils.gather_object = lambda results: results
    accelerate.utils = accelerate_utils
    monkeypatch.setitem(sys.modules, "accelerate", accelerate)
    monkeypatch.setitem(sys.modules, "accelerate.utils", accelerate_utils)
    monkeypatch.setattr("tqdm.auto.tqdm", fake_tqdm)

    class Model:
        training = True

        def eval(self) -> None:
            self.training = False

        def train(self) -> None:
            self.training = True

        def generate(self, **batch):
            return torch.cat((batch["input_ids"], torch.tensor([[2]])), dim=1)

    model = Model()
    accelerator = SimpleNamespace(
        is_local_main_process=True,
        num_processes=1,
        unwrap_model=lambda wrapped: wrapped,
    )
    processor = SimpleNamespace(
        batch_decode=lambda *args, **kwargs: [
            '{"protocol_version":"1.0","canvas":{"width":1,"height":1},"objects":[]}'
        ]
    )
    target = '{"protocol_version":"1.0","canvas":{"width":1,"height":1},"objects":[]}'
    dataloader = [
        {
            "_sample_ids": ["sample-1"],
            "_targets": [target],
            "input_ids": torch.tensor([[1]]),
        }
    ]

    metrics = evaluate_generation(
        accelerator=accelerator,
        model=model,
        processor=processor,
        dataloader=dataloader,
        max_new_tokens=8,
        progress_bar=True,
    )

    assert progress_args["desc"] == "Generation evaluation"
    assert progress_args["unit"] == "batch"
    assert progress_args["disable"] is False
    assert metrics.evaluated_count == 1
    assert metrics.semantic_id_exact_match == 1.0
    assert model.training is True
