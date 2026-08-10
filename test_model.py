"""Evaluate exported PEFT weights on a text-render-protocol dataset split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from src.text_render_protocol_predictor.data import ProtocolManifestDataset
from src.text_render_protocol_predictor.evaluation.runner import evaluate_generation
from src.text_render_protocol_predictor.models.grounded_qwen3_vl import (
    GROUNDING_CONFIG_FILE,
    GROUNDING_WEIGHTS_FILE,
    GroundedQwen3VL,
    load_grounding_config,
)
from src.text_render_protocol_predictor.models.qwen3_vl import (
    inspect_peft_weights_directory,
    resize_model_to_tokenizer_vocabulary,
)
from src.text_render_protocol_predictor.protocol import (
    MASK_TOKENIZER_SURFACE,
    CoordinateTokenCodec,
)
from src.text_render_protocol_predictor.training import (
    ProtocolGenerationCollator,
    ProtocolPromptTemplate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="PEFT adapter directory")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("test.jsonl"),
        help="Manifest path, relative to the dataset root unless absolute (default: test.jsonl)",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, help="Evaluate only the first N records")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--image-min-pixels", type=int, default=200704)
    parser.add_argument("--image-max-pixels", type=int, default=1003520)
    parser.add_argument("--decimal-places", type=int, default=3)
    parser.add_argument("--max-objects", type=int, default=64)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--text-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate text objects only (default: true)",
    )
    parser.add_argument(
        "--coordinate-encoding",
        choices=("auto", "enabled", "disabled"),
        default="auto",
    )
    parser.add_argument("--coordinate-bins", type=int, default=512)
    parser.add_argument("--coordinate-token-prefix", default="coord")
    parser.add_argument(
        "--no-verify-image-dimensions",
        action="store_false",
        dest="verify_image_dimensions",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        help="Optionally save sample IDs, raw predictions, and targets as JSONL",
    )
    return parser.parse_args()


def resolve_dtype(name: str, device: str) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.startswith("cuda") else torch.float32
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def validate_args(args: argparse.Namespace) -> None:
    for name in ("batch_size", "max_new_tokens", "image_min_pixels", "image_max_pixels"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")


def print_metrics(metrics: object) -> None:
    rows = (
        ("semantic_id_precision", metrics.semantic_id_precision),
        ("semantic_id_recall", metrics.semantic_id_recall),
        ("font_accuracy", metrics.font_accuracy),
        ("wer", metrics.word_error_rate),
        ("cer", metrics.character_error_rate),
        ("box_iou", metrics.box_iou),
        ("oriented_box_iou", metrics.oriented_box_iou),
        ("angle_mae", metrics.angle_mae),
        ("geometry_mode_accuracy", metrics.geometry_mode_accuracy),
        ("bezier_centerline_error", metrics.bezier_centerline_error),
        ("bezier_mse", metrics.bezier_mse),
        ("color_mae", metrics.color_mae),
    )
    for name, value in rows:
        print(f"{name}: {value:.6f}")
    print(f"evaluated_samples: {metrics.evaluated_count}")
    print(f"valid_json: {metrics.valid_json_count}/{metrics.evaluated_count}")
    print(f"schema_valid: {metrics.schema_valid_count}/{metrics.evaluated_count}")
    print(f"bezier_coordinates_compared: {metrics.bezier_coordinate_count}")
    print(f"color_channels_compared: {metrics.color_channel_count}")
    if metrics.grounding_evaluated_count:
        print(
            "grounding_valid: "
            f"{metrics.grounding_valid_count}/{metrics.grounding_evaluated_count}"
        )
    if metrics.mask_evaluated_count:
        print(f"mask_attached_iou: {metrics.mask_attached_iou:.6f}")
        print(f"mask_oracle_iou: {metrics.mask_oracle_iou:.6f}")
        print(f"mask_association_gap: {metrics.mask_association_gap:.6f}")
        logged = metrics.as_log_dict()
        for name in (
            "generation/mask_dice",
            "generation/mask_boundary_fscore",
            "generation/mask_ap50",
            "generation/mask_ap75",
            "generation/serialized_geometry_mask_iou",
            "generation/auxiliary_geometry_mode_accuracy",
            "generation/auxiliary_oriented_box_iou",
            "generation/auxiliary_angle_mae",
            "generation/auxiliary_bezier_centerline_error",
        ):
            if name in logged:
                print(f"{name.removeprefix('generation/')}: {logged[name]:.6f}")
        for name, value in sorted(logged.items()):
            if name.startswith("generation/slice/"):
                print(f"{name.removeprefix('generation/')}: {value}")


def _device_batches(dataloader: DataLoader, device: str):
    for batch in dataloader:
        yield {
            name: value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
            for name, value in batch.items()
        }


def main() -> None:
    args = parse_args()
    validate_args(args)
    weights_path, base_model = inspect_peft_weights_directory(args.weights)
    processor = AutoProcessor.from_pretrained(
        str(weights_path),
        min_pixels=args.image_min_pixels,
        max_pixels=args.image_max_pixels,
    )
    coordinate_codec: CoordinateTokenCodec | None = None
    if args.coordinate_encoding != "disabled":
        candidate = CoordinateTokenCodec(
            bins=args.coordinate_bins,
            prefix=args.coordinate_token_prefix,
        )
        present = sum(
            token in processor.tokenizer.get_vocab()
            for token in candidate.tokenizer_tokens
        )
        if present == candidate.bins:
            coordinate_codec = candidate
        elif present or args.coordinate_encoding == "enabled":
            raise ValueError(
                f"checkpoint tokenizer contains {present}/{candidate.bins} configured "
                "coordinate tokens"
            )
    grounding_files = (
        weights_path / GROUNDING_CONFIG_FILE,
        weights_path / GROUNDING_WEIGHTS_FILE,
    )
    present_grounding = [path.is_file() for path in grounding_files]
    if any(present_grounding) and not all(present_grounding):
        raise ValueError("grounded checkpoint export is incomplete")
    grounded = all(present_grounding)
    dataset = ProtocolManifestDataset(
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        decimal_places=args.decimal_places,
        verify_image_dimensions=args.verify_image_dimensions,
        max_objects=args.max_objects,
        coordinate_codec=coordinate_codec,
        text_only_targets=args.text_only,
        grounding_enabled=grounded,
    )
    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    if len(dataset) == 0:
        raise ValueError("the selected test dataset is empty")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model,
        dtype=resolve_dtype(args.dtype, args.device),
        attn_implementation=args.attn_implementation,
    )
    resize_model_to_tokenizer_vocabulary(model, processor.tokenizer)
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, str(weights_path), is_trainable=False)
    if grounded:
        decoder_config, mask_token_id = load_grounding_config(weights_path)
        if processor.tokenizer.encode(
            MASK_TOKENIZER_SURFACE,
            add_special_tokens=False,
        ) != [mask_token_id]:
            raise ValueError("grounded checkpoint tokenizer has an invalid mask token")
        model = GroundedQwen3VL(
            model,
            mask_token_id=mask_token_id,
            decoder_config=decoder_config,
        )
        model.load_grounding_pretrained(weights_path)
    model.to(args.device).eval()
    model.config.use_cache = True

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=ProtocolGenerationCollator(
            processor,
            ProtocolPromptTemplate(
                coordinate_codec=coordinate_codec,
                text_only=args.text_only,
                grounding_enabled=grounded,
            ),
        ),
    )
    prediction_rows: list[dict[str, str]] = []
    prompt_template = ProtocolPromptTemplate(
        coordinate_codec=coordinate_codec,
        text_only=args.text_only,
        grounding_enabled=grounded,
    )
    metrics = evaluate_generation(
        accelerator=SimpleNamespace(
            device=torch.device(args.device),
            is_local_main_process=True,
            num_processes=1,
            unwrap_model=lambda value: value,
        ),
        model=model,
        processor=processor,
        dataloader=_device_batches(dataloader, args.device),
        max_new_tokens=args.max_new_tokens,
        coordinate_codec=coordinate_codec,
        decimal_places=args.decimal_places,
        grounding_enabled=grounded,
        prompt_template=prompt_template,
        prediction_sink=prediction_rows,
    )
    print_metrics(metrics)
    if args.predictions_output is not None:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions_output.open("w", encoding="utf-8") as stream:
            for row in prediction_rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
