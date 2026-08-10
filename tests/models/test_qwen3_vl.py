from __future__ import annotations

import json

import pytest
import torch

from text_render_protocol_predictor.models.qwen3_vl import (
    REQUIRED_PEFT_WEIGHT_FILES,
    add_mask_token,
    add_coordinate_tokens,
    coordinate_trainable_token_indices,
    inspect_peft_weights_directory,
    resize_and_initialize_coordinate_embeddings,
    resize_model_to_tokenizer_vocabulary,
    vision_lora_targets,
)
from text_render_protocol_predictor.protocol import CoordinateTokenCodec


def _write_peft_export(path, *, peft_type: str = "LORA") -> None:
    path.mkdir()
    config = {
        "peft_type": peft_type,
        "base_model_name_or_path": "Qwen/Qwen3-VL-8B-Instruct",
    }
    for name in REQUIRED_PEFT_WEIGHT_FILES:
        content = json.dumps(config) if name == "adapter_config.json" else "placeholder"
        (path / name).write_text(content, encoding="utf-8")


def test_inspect_peft_weights_directory_reads_base_model(tmp_path) -> None:
    weights_path = tmp_path / "adapter"
    _write_peft_export(weights_path)

    resolved, base_model = inspect_peft_weights_directory(weights_path)

    assert resolved == weights_path
    assert base_model == "Qwen/Qwen3-VL-8B-Instruct"


def test_inspect_peft_weights_directory_reports_missing_files(tmp_path) -> None:
    weights_path = tmp_path / "adapter"
    weights_path.mkdir()
    (weights_path / "adapter_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="adapter_model.safetensors"):
        inspect_peft_weights_directory(weights_path)


def test_inspect_peft_weights_directory_rejects_non_lora(tmp_path) -> None:
    weights_path = tmp_path / "adapter"
    _write_peft_export(weights_path, peft_type="IA3")

    with pytest.raises(ValueError, match="LoRA"):
        inspect_peft_weights_directory(weights_path)


class _FakeTokenizer:
    def __init__(self) -> None:
        self._size = 4
        self._added: dict[str, int] = {}
        self._decoded: dict[int, str] = {}
        self.all_special_ids: list[int] = []

    def __len__(self) -> int:
        return self._size

    def get_vocab(self) -> dict[str, int]:
        return dict(self._added)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        if text in self._added:
            return [self._added[text]]
        if text.isdigit():
            return [int(text) % 4]
        return [0, 1]

    def add_tokens(self, tokens, *, special_tokens: bool) -> int:
        assert special_tokens is False
        added = 0
        for token in tokens:
            surface = token.content
            if surface not in self._added:
                token_id = self._size
                self._added[surface] = token_id
                self._decoded[token_id] = surface
                self._size += 1
                added += 1
        return added

    def decode(self, ids, **kwargs) -> str:
        del kwargs
        return self._decoded[ids[0]]


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(4, 2)
        self.lm_head = torch.nn.Linear(2, 4, bias=False)
        with torch.no_grad():
            values = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
            self.embed_tokens.weight.copy_(values)
            self.lm_head.weight.copy_(values + 10)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def resize_token_embeddings(self, size: int, *, mean_resizing: bool):
        assert mean_resizing is False
        old_input = self.embed_tokens.weight.detach().clone()
        old_output = self.lm_head.weight.detach().clone()
        self.embed_tokens = torch.nn.Embedding(size, 2)
        self.lm_head = torch.nn.Linear(2, size, bias=False)
        with torch.no_grad():
            self.embed_tokens.weight[: len(old_input)].copy_(old_input)
            self.lm_head.weight[: len(old_output)].copy_(old_output)
        return self.embed_tokens


def test_coordinate_tokens_resize_and_initialize_both_vocabulary_layers() -> None:
    tokenizer = _FakeTokenizer()
    codec = CoordinateTokenCodec(bins=4)
    registration = add_coordinate_tokens(tokenizer, codec)
    model = _FakeModel()

    resize_and_initialize_coordinate_embeddings(model, tokenizer, registration)

    assert registration.token_ids == (4, 5, 6, 7)
    assert registration.new_token_ids == (4, 5, 6, 7)
    assert torch.equal(model.embed_tokens.weight[4], torch.tensor([1.0, 2.0]))
    assert torch.equal(model.embed_tokens.weight[6], torch.tensor([5.0, 6.0]))
    assert torch.equal(model.lm_head.weight[4], torch.tensor([11.0, 12.0]))
    assert torch.equal(model.lm_head.weight[6], torch.tensor([15.0, 16.0]))
    assert coordinate_trainable_token_indices(model, registration) == {
        "embed_tokens": [4, 5, 6, 7],
        "lm_head": [4, 5, 6, 7],
    }


def test_coordinate_json_value_is_one_regular_token() -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "0": 1, "1": 2, "2": 3, "3": 4},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]"
    )

    registration = add_coordinate_tokens(tokenizer, CoordinateTokenCodec(bins=4))
    coordinate_id = registration.token_ids[2]

    assert tokenizer.encode('"<coord_2>"', add_special_tokens=False) == [coordinate_id]
    assert tokenizer.decode([coordinate_id], skip_special_tokens=True) == '"<coord_2>"'
    assert coordinate_id not in tokenizer.all_special_ids


def test_mask_json_value_is_one_regular_token() -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(
        models.WordLevel({"[UNK]": 0, "mask": 1}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
    )

    registration = add_mask_token(tokenizer)

    assert tokenizer.encode('"<MASK>"', add_special_tokens=False) == [
        registration.token_id
    ]
    assert tokenizer.decode(
        [registration.token_id], skip_special_tokens=True
    ) == '"<MASK>"'
    assert registration.token_id not in tokenizer.all_special_ids


def test_new_tokens_initialize_inside_preallocated_model_vocabulary() -> None:
    tokenizer = _FakeTokenizer()
    registration = add_coordinate_tokens(tokenizer, CoordinateTokenCodec(bins=4))
    model = _FakeModel()
    model.embed_tokens = torch.nn.Embedding(8, 2)
    model.lm_head = torch.nn.Linear(2, 8, bias=False)
    with torch.no_grad():
        model.embed_tokens.weight.zero_()
        model.lm_head.weight.zero_()
        model.embed_tokens.weight[0] = torch.tensor([1.0, 2.0])
        model.lm_head.weight[0] = torch.tensor([11.0, 12.0])

    resize_and_initialize_coordinate_embeddings(model, tokenizer, registration)

    assert torch.equal(model.embed_tokens.weight[4], torch.tensor([1.0, 2.0]))
    assert torch.equal(model.lm_head.weight[4], torch.tensor([11.0, 12.0]))


def test_expanded_checkpoint_tokenizer_resizes_base_before_peft_loading() -> None:
    tokenizer = _FakeTokenizer()
    tokenizer._size = 8
    model = _FakeModel()

    resized = resize_model_to_tokenizer_vocabulary(model, tokenizer)

    assert resized is True
    assert model.embed_tokens.weight.shape[0] == 8
    assert model.lm_head.weight.shape[0] == 8
    assert resize_model_to_tokenizer_vocabulary(model, tokenizer) is False


def test_vision_lora_targets_only_final_requested_blocks() -> None:
    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = torch.nn.Module()
            self.attn.qkv = torch.nn.Linear(2, 2)
            self.attn.proj = torch.nn.Linear(2, 2)
            self.mlp = torch.nn.Module()
            self.mlp.linear_fc1 = torch.nn.Linear(2, 2)
            self.mlp.linear_fc2 = torch.nn.Linear(2, 2)

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self.model.visual = torch.nn.Module()
            self.model.visual.blocks = torch.nn.ModuleList(
                [Block() for _ in range(4)]
            )

    targets = vision_lora_targets(Model(), final_blocks=2)

    assert len(targets) == 8
    assert all("blocks.2" in name or "blocks.3" in name for name in targets)
