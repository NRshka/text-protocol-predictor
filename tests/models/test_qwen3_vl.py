from __future__ import annotations

import json

import pytest

from text_render_protocol_predictor.models.qwen3_vl import (
    REQUIRED_PEFT_WEIGHT_FILES,
    inspect_peft_weights_directory,
)


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
