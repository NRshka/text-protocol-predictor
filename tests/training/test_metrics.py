import torch

from text_render_protocol_predictor.training.sft_trainer import _token_accuracy_counts


def test_token_accuracy_is_causally_shifted_and_ignores_prompt() -> None:
    logits = torch.zeros((1, 5, 8))
    logits[0, 1, 3] = 1
    logits[0, 2, 4] = 1
    logits[0, 3, 0] = 1
    labels = torch.tensor([[-100, -100, 3, 4, 5]])

    correct, total = _token_accuracy_counts(logits, labels)

    assert correct.item() == 2
    assert total.item() == 3
