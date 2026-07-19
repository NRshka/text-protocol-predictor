"""Evaluate exported PEFT weights on a text-render-protocol dataset split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from src.text_render_protocol_predictor.data import ProtocolManifestDataset
from src.text_render_protocol_predictor.evaluation import evaluate_generation_predictions
from src.text_render_protocol_predictor.models.qwen3_vl import inspect_peft_weights_directory
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


def main() -> None:
    args = parse_args()
    validate_args(args)
    weights_path, base_model = inspect_peft_weights_directory(args.weights)
    dataset = ProtocolManifestDataset(
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        decimal_places=args.decimal_places,
        verify_image_dimensions=args.verify_image_dimensions,
        max_objects=args.max_objects,
    )
    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    if len(dataset) == 0:
        raise ValueError("the selected test dataset is empty")

    processor = AutoProcessor.from_pretrained(
        str(weights_path),
        min_pixels=args.image_min_pixels,
        max_pixels=args.image_max_pixels,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model,
        dtype=resolve_dtype(args.dtype, args.device),
        attn_implementation=args.attn_implementation,
    )
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, str(weights_path), is_trainable=False)
    model.to(args.device).eval()
    model.config.use_cache = True

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        collate_fn=ProtocolGenerationCollator(processor, ProtocolPromptTemplate()),
    )
    sample_ids: list[str] = []
    outputs: list[str] = []
    targets: list[str] = []
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Testing", unit="batch", dynamic_ncols=True):
            sample_ids.extend(batch.pop("_sample_ids"))
            targets.extend(batch.pop("_targets"))
            model_batch = {
                name: value.to(args.device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
                for name, value in batch.items()
            }
            input_length = model_batch["input_ids"].shape[1]
            generated = model.generate(
                **model_batch,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
            outputs.extend(
                processor.batch_decode(
                    generated[:, input_length:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )

    metrics = evaluate_generation_predictions(outputs, targets)
    print_metrics(metrics)
    if args.predictions_output is not None:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions_output.open("w", encoding="utf-8") as stream:
            for sample_id, prediction, target in zip(
                sample_ids, outputs, targets, strict=True
            ):
                stream.write(
                    json.dumps(
                        {"sample_id": sample_id, "prediction": prediction, "target": target},
                        ensure_ascii=False,
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()
