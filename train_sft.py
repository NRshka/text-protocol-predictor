"""Hydra entry point for Qwen3-VL supervised fine-tuning."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.text_render_protocol_predictor.data import (
    ProtocolManifestDataset,
    StructuralNoiseConfig,
    validate_dataset,
)
from src.text_render_protocol_predictor.models import load_qwen3_vl_for_sft
from src.text_render_protocol_predictor.protocol import coordinate_codec_from_config
from src.text_render_protocol_predictor.training import ProtocolPromptTemplate, ProtocolSFTCollator
from src.text_render_protocol_predictor.training.sft_trainer import train_sft


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.model.weights_path and cfg.training.resume_from:
        raise ValueError(
            "model.weights_path initializes PEFT weights with fresh training state; "
            "it cannot be combined with training.resume_from"
        )
    coordinate_codec = coordinate_codec_from_config(cfg.protocol)
    prompt_template = ProtocolPromptTemplate(coordinate_codec=coordinate_codec)
    common = {
        "dataset_root": cfg.dataset.root_dir,
        "decimal_places": int(cfg.protocol.decimal_places),
        "verify_image_dimensions": bool(cfg.dataset.verify_image_dimensions),
        "max_objects": int(cfg.protocol.max_objects),
        "coordinate_codec": coordinate_codec,
    }
    train_dataset = ProtocolManifestDataset(
        manifest_path=cfg.dataset.manifests.train,
        structural_noise=StructuralNoiseConfig.from_mapping(
            cfg.augmentation.structural_noise
        ),
        **common,
    )
    validation_dataset = ProtocolManifestDataset(
        manifest_path=cfg.dataset.manifests.validation,
        **common,
    )
    if cfg.dataset.preflight_validation:
        validate_dataset(train_dataset).raise_for_errors()
        validate_dataset(validation_dataset).raise_for_errors()
    model, processor = load_qwen3_vl_for_sft(
        cfg.model, cfg.lora, coordinate_codec=coordinate_codec
    )
    collator = ProtocolSFTCollator(
        processor=processor,
        prompt_template=prompt_template,
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
        prompt_template=prompt_template,
    )


if __name__ == "__main__":
    main()
