"""Hydra entry point for reconstruction-reward GRPO fine-tuning."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.text_render_protocol_predictor.data import GRPOManifestDataset, validate_dataset
from src.text_render_protocol_predictor.models import load_qwen3_vl_for_sft
from src.text_render_protocol_predictor.rendering import SyntextPredictionRenderer
from src.text_render_protocol_predictor.rewards import PixelMAEReward, PixelMAERewardConfig
from src.text_render_protocol_predictor.training import build_hf_grpo_dataset
from src.text_render_protocol_predictor.training.grpo_trainer import train_grpo


@hydra.main(version_base="1.3", config_path="configs", config_name="grpo")
def main(cfg: DictConfig) -> None:
    if not cfg.model.weights_path:
        raise ValueError(
            "model.weights_path must point to an SFT PEFT export; milestone-one GRPO "
            "is intentionally initialized from the supervised policy"
        )
    if not bool(cfg.lora.enabled):
        raise ValueError("GRPO from SFT PEFT weights requires lora.enabled=true")
    if str(cfg.protocol.version) != "1.0":
        raise ValueError(
            "the erased-text milestone supports protocol 1.0 only because non-text shapes "
            "remain in the background"
        )

    dataset_kwargs = {
        "dataset_root": cfg.dataset.root_dir,
        "mask_threshold": float(cfg.reward.mask_threshold),
        "minimum_mask_coverage": float(cfg.dataset.minimum_mask_coverage),
        "maximum_mask_coverage": float(cfg.dataset.maximum_mask_coverage),
        "require_webp": bool(cfg.dataset.require_webp),
    }
    train_source = GRPOManifestDataset(
        manifest_path=cfg.dataset.manifests.train,
        **dataset_kwargs,
    )
    validation_source = None
    if bool(cfg.evaluation.enabled):
        validation_source = GRPOManifestDataset(
            manifest_path=cfg.dataset.manifests.validation,
            **dataset_kwargs,
        )
    if bool(cfg.dataset.preflight_validation):
        validate_dataset(train_source).raise_for_errors()
        if validation_source is not None:
            validate_dataset(validation_source).raise_for_errors()

    renderer = SyntextPredictionRenderer(
        list(cfg.renderer.font_paths),
        protocol_version=str(cfg.protocol.version),
        max_objects=int(cfg.protocol.max_objects),
        max_text_characters=int(cfg.renderer.max_text_characters),
        max_font_size=float(cfg.renderer.max_font_size),
        max_geometry_scale=float(cfg.renderer.max_geometry_scale),
    )
    reward = PixelMAEReward(
        renderer,
        PixelMAERewardConfig(
            mask_threshold=float(cfg.reward.mask_threshold),
            mask_dilation_radius=int(cfg.reward.mask_dilation_radius),
            mask_blur_radius=float(cfg.reward.mask_blur_radius),
            outside_weight=float(cfg.reward.outside_weight),
            cache_size=int(cfg.reward.cache_size),
            invalid_json_reward=float(cfg.reward.invalid_json_reward),
            invalid_schema_reward=float(cfg.reward.invalid_schema_reward),
            invalid_semantics_reward=float(cfg.reward.invalid_semantics_reward),
            unknown_font_reward=float(cfg.reward.unknown_font_reward),
            renderer_failure_reward=float(cfg.reward.renderer_failure_reward),
        ),
    )
    train_dataset = build_hf_grpo_dataset(
        train_source,
        protocol_version=str(cfg.protocol.version),
    )
    validation_dataset = (
        build_hf_grpo_dataset(
            validation_source,
            protocol_version=str(cfg.protocol.version),
        )
        if validation_source is not None
        else None
    )

    # Allocate the large model only after dataset and renderer preflight pass.
    model, processor = load_qwen3_vl_for_sft(cfg.model, cfg.lora)
    train_grpo(
        cfg=cfg,
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        reward=reward,
        renderer_metadata={
            "package": "synthetic-text-protocol",
            "version": renderer.renderer_version,
            "font_registry_fingerprint": renderer.font_registry_fingerprint,
            "font_ids": sorted(renderer.font_ids),
        },
    )


if __name__ == "__main__":
    main()
