"""Hydra entry point for Qwen3-VL supervised fine-tuning."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from text_render_protocol_predictor.data import ProtocolManifestDataset, validate_dataset
from text_render_protocol_predictor.models import load_qwen3_vl_for_sft
from text_render_protocol_predictor.training import ProtocolPromptTemplate, ProtocolSFTCollator
from text_render_protocol_predictor.training.sft_trainer import train_sft


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.model.weights_path and cfg.training.resume_from:
        raise ValueError(
            "model.weights_path initializes PEFT weights with fresh training state; "
            "it cannot be combined with training.resume_from"
        )
    common = {
        "dataset_root": cfg.dataset.root_dir,
        "decimal_places": int(cfg.protocol.decimal_places),
        "verify_image_dimensions": bool(cfg.dataset.verify_image_dimensions),
        "max_objects": int(cfg.protocol.max_objects),
    }
    train_dataset = ProtocolManifestDataset(
        manifest_path=cfg.dataset.manifests.train,
        **common,
    )
    validation_dataset = ProtocolManifestDataset(
        manifest_path=cfg.dataset.manifests.validation,
        **common,
    )
    if cfg.dataset.preflight_validation:
        validate_dataset(train_dataset).raise_for_errors()
        validate_dataset(validation_dataset).raise_for_errors()
    model, processor = load_qwen3_vl_for_sft(cfg.model, cfg.lora)
    collator = ProtocolSFTCollator(
        processor=processor,
        prompt_template=ProtocolPromptTemplate(),
        max_sequence_tokens=int(cfg.model.max_sequence_tokens),
        max_output_tokens=int(cfg.model.max_output_tokens),
    )
    train_sft(
        cfg=cfg,
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        collator=collator,
    )


if __name__ == "__main__":
    main()
