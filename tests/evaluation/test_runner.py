import sys
from types import ModuleType, SimpleNamespace

import torch

from text_render_protocol_predictor.evaluation.runner import evaluate_generation
from text_render_protocol_predictor.protocol import CoordinateTokenCodec


def test_generation_evaluation_reports_batch_progress(monkeypatch, protocol_dict: dict) -> None:
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
    coordinate_codec = CoordinateTokenCodec()
    target = coordinate_codec.encode_json(protocol_dict)
    processor = SimpleNamespace(batch_decode=lambda *args, **kwargs: [target])
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
        coordinate_codec=coordinate_codec,
    )

    assert progress_args["desc"] == "Generation evaluation"
    assert progress_args["unit"] == "batch"
    assert progress_args["disable"] is False
    assert metrics.evaluated_count == 1
    assert metrics.schema_valid_count == 1
    assert metrics.ground_truth_object_count == 2
    assert metrics.box_iou == 1.0
    assert metrics.semantic_id_exact_match == 1.0
    assert model.training is True
