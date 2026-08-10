from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from src.text_render_protocol_predictor.evaluation import evaluate_generation_validity
from src.text_render_protocol_predictor.models.qwen3_vl import (
    inspect_peft_weights_directory,
    resize_model_to_tokenizer_vocabulary,
)
from src.text_render_protocol_predictor.models.grounded_qwen3_vl import (
    GROUNDING_CONFIG_FILE,
    GROUNDING_WEIGHTS_FILE,
    GroundedQwen3VL,
    load_grounding_config,
)
from src.text_render_protocol_predictor.protocol import (
    MASK_TOKENIZER_SURFACE,
    CoordinateTokenCodec,
    strip_mask_references,
)
from src.text_render_protocol_predictor.training import (
    ProtocolPromptTemplate,
    locate_completion_token_positions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--weights", type=Path, help="PEFT adapter/processor directory")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument(
        "--protocol-version", choices=("1.0", "2.0", "2.1"), default="1.0"
    )
    parser.add_argument(
        "--text-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request text objects only (use --no-text-only for legacy shape-aware models)",
    )
    parser.add_argument("--image-min-pixels", type=int, default=200704)
    parser.add_argument("--image-max-pixels", type=int, default=1003520)
    parser.add_argument("--output", type=Path, help="Save the pixel-coordinate prediction here")
    parser.add_argument(
        "--raw-output",
        type=Path,
        help="Optionally save the native model generation before coordinate decoding",
    )
    parser.add_argument(
        "--mask-output-dir",
        type=Path,
        help="Decode grounded instance masks into PNG files and manifest.json",
    )
    parser.add_argument(
        "--coordinate-encoding",
        choices=("auto", "enabled", "disabled"),
        default="auto",
        help="Detect atomic coordinate tokens in the checkpoint, require them, or disable them",
    )
    parser.add_argument("--coordinate-bins", type=int, default=512)
    parser.add_argument("--coordinate-token-prefix", default="coord")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print valid JSON")
    return parser.parse_args()


def resolve_dtype(name: str, device: str) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.startswith("cuda") else torch.float32
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def resolve_coordinate_codec(
    tokenizer: object,
    *,
    mode: str,
    bins: int,
    token_prefix: str,
) -> CoordinateTokenCodec | None:
    """Resolve coordinate encoding without introducing untrained inference tokens."""
    if mode == "disabled":
        return None
    codec = CoordinateTokenCodec(bins=bins, prefix=token_prefix)
    vocabulary = tokenizer.get_vocab()
    present = sum(token in vocabulary for token in codec.tokenizer_tokens)
    if present == codec.bins:
        return codec
    if present:
        raise ValueError(
            f"checkpoint tokenizer contains only {present}/{codec.bins} expected "
            "coordinate tokens"
        )
    if mode == "enabled":
        raise ValueError(
            "coordinate encoding was explicitly enabled, but the checkpoint tokenizer "
            "does not contain the configured coordinate tokens"
        )
    return None


def decode_prediction_coordinates(
    output: str,
    codec: CoordinateTokenCodec | None,
    *,
    image_size: tuple[int, int],
) -> tuple[str, str | None]:
    """Decode atomic coordinates against the actual input image dimensions."""
    if codec is None:
        return output, None
    try:
        return (
            codec.decode_json(
                output,
                canvas_size=image_size,
                allow_numeric_coordinates=True,
            ),
            None,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return output, str(exc)


def write_prediction(path: Path, output: str) -> None:
    """Write pretty JSON when possible and preserve invalid text for diagnosis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        value = json.loads(output)
    except json.JSONDecodeError:
        path.write_text(output + "\n", encoding="utf-8")
        return
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )


def is_grounded_checkpoint(weights_path: Path | None) -> bool:
    """Detect a complete grounding export and reject partial/corrupt exports."""
    if weights_path is None:
        return False
    grounding_files = (
        weights_path / GROUNDING_CONFIG_FILE,
        weights_path / GROUNDING_WEIGHTS_FILE,
    )
    present = [path.is_file() for path in grounding_files]
    if any(present) and not all(present):
        missing = [path.name for path, exists in zip(grounding_files, present) if not exists]
        raise ValueError(
            "grounded checkpoint is incomplete; missing: " + ", ".join(missing)
        )
    return all(present)


def write_mask_sidecar(
    output_dir: Path,
    *,
    status: str,
    canvas_size: tuple[int, int],
    instances: list[dict] | None = None,
    error: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1.0",
        "status": status,
        "canvas": {"width": canvas_size[0], "height": canvas_size[1]},
        "threshold": 0.5,
        "instances": instances or [],
        "error": error,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _tensor_probability_image(value: torch.Tensor) -> Image.Image:
    encoded = value.detach().float().clamp(0, 1).mul(255).round().to(torch.uint8).cpu()
    height, width = encoded.shape
    return Image.frombytes("L", (width, height), encoded.contiguous().numpy().tobytes())


def decode_and_write_masks(
    *,
    model: GroundedQwen3VL,
    processor: object,
    prompt_template: ProtocolPromptTemplate,
    image_path: Path,
    raw_output: str,
    object_ids: tuple[str, ...],
    canvas_size: tuple[int, int],
    protocol_version: str,
    device: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    conversation = prompt_template.conversation(
        image=image_path,
        width=canvas_size[0],
        height=canvas_size[1],
        protocol_version=protocol_version,
        target=raw_output,
    )
    full_batch = processor.apply_chat_template(
        [conversation],
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"padding": True},
    ).to(device)
    prompt_conversation = prompt_template.conversation(
        image=image_path,
        width=canvas_size[0],
        height=canvas_size[1],
        protocol_version=protocol_version,
    )
    prompt_batch = processor.apply_chat_template(
        [prompt_conversation],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"padding": True},
    ).to(device)
    token_positions = locate_completion_token_positions(
        full_batch,
        prompt_batch,
        token_id=model.mask_token_id,
    )[0]
    if len(token_positions) != len(object_ids):
        raise ValueError(
            f"generated completion contains {len(token_positions)} atomic mask tokens "
            f"for {len(object_ids)} text objects"
        )
    positions = token_positions.unsqueeze(0)
    valid = torch.ones_like(positions, dtype=torch.bool)
    with torch.inference_mode():
        result = model(
            **full_batch,
            mask_token_positions=positions,
            instance_valid=valid,
        )
    logits = result.mask_logits[0]
    qualities = result.geometry_predictions[0]["quality_logits"].sigmoid()
    resized = F.interpolate(
        logits.sigmoid().unsqueeze(1),
        size=(canvas_size[1], canvas_size[0]),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    instances: list[dict] = []
    for index, (object_id, probability, quality) in enumerate(
        zip(object_ids, resized, qualities, strict=True)
    ):
        filename = f"{index:04d}-{object_id}.png"
        _tensor_probability_image(probability).save(output_dir / filename)
        instances.append(
            {
                "object_id": object_id,
                "mask": filename,
                "quality_score": float(quality.item()),
            }
        )
    write_mask_sidecar(
        output_dir,
        status="ok",
        canvas_size=canvas_size,
        instances=instances,
    )


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)

    if args.weights:
        weights_path, base_model = inspect_peft_weights_directory(args.weights)
        processor_source = str(weights_path)
    else:
        weights_path = None
        base_model = args.model
        processor_source = base_model

    processor = AutoProcessor.from_pretrained(
        processor_source,
        min_pixels=args.image_min_pixels,
        max_pixels=args.image_max_pixels,
    )
    coordinate_codec = resolve_coordinate_codec(
        processor.tokenizer,
        mode=args.coordinate_encoding,
        bins=args.coordinate_bins,
        token_prefix=args.coordinate_token_prefix,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model,
        dtype=resolve_dtype(args.dtype, args.device),
        attn_implementation="sdpa",
    )
    resize_model_to_tokenizer_vocabulary(model, processor.tokenizer)
    if weights_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(weights_path), is_trainable=False)
    grounded = is_grounded_checkpoint(weights_path)
    if args.mask_output_dir is not None and not grounded:
        raise ValueError("--mask-output-dir requires a grounded checkpoint")
    if grounded:
        decoder_config, saved_mask_token_id = load_grounding_config(weights_path)
        token_ids = processor.tokenizer.encode(
            MASK_TOKENIZER_SURFACE, add_special_tokens=False
        )
        if token_ids != [saved_mask_token_id]:
            raise ValueError("grounded checkpoint tokenizer has an invalid mask token")
        model = GroundedQwen3VL(
            model,
            mask_token_id=saved_mask_token_id,
            decoder_config=decoder_config,
        )
        model.load_grounding_pretrained(weights_path)
    model.to(args.device).eval()
    model.config.use_cache = True

    with Image.open(args.image) as image:
        width, height = image.size
    prompt_template = ProtocolPromptTemplate(
        coordinate_codec=coordinate_codec,
        text_only=args.text_only,
        grounding_enabled=grounded,
    )
    conversation = prompt_template.conversation(
        image=args.image,
        width=width,
        height=height,
        protocol_version=args.protocol_version,
    )
    batch = processor.apply_chat_template(
        [conversation],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"padding": True},
    ).to(args.device)

    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
    generation_started_at = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    generation_latency_seconds = time.perf_counter() - generation_started_at
    peak_memory_bytes = (
        int(torch.cuda.max_memory_allocated(args.device))
        if args.device.startswith("cuda")
        else 0
    )
    completion = generated[:, batch["input_ids"].shape[1] :]
    raw_output = processor.batch_decode(
        completion,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    grounding_result = None
    native_clean_output = raw_output
    grounding_error = None
    if grounded:
        try:
            grounding_result = strip_mask_references(raw_output)
            native_clean_output = grounding_result.clean_json
            if not grounding_result.valid:
                grounding_error = "; ".join(grounding_result.errors)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            grounding_error = str(exc)
    output, coordinate_decode_error = decode_prediction_coordinates(
        native_clean_output,
        coordinate_codec,
        image_size=(width, height),
    )

    metrics = evaluate_generation_validity([output])
    print(
        f"valid_json={bool(metrics.valid_json_count)} "
        f"schema_valid={bool(metrics.schema_valid_count)} "
        f"generated_tokens={completion.shape[1]} "
        f"generation_latency_seconds={generation_latency_seconds:.3f} "
        f"peak_memory_bytes={peak_memory_bytes}",
        file=sys.stderr,
    )
    if coordinate_decode_error is not None:
        print(
            f"coordinate_decode_error={coordinate_decode_error}",
            file=sys.stderr,
        )
    if args.raw_output:
        write_prediction(args.raw_output, raw_output)
    if args.mask_output_dir is not None:
        try:
            if grounding_result is None or not grounding_result.valid:
                raise ValueError(grounding_error or "generated grounding is invalid")
            if not metrics.schema_valid_count:
                raise ValueError("clean generated STRP is schema-invalid")
            decode_and_write_masks(
                model=model,
                processor=processor,
                prompt_template=prompt_template,
                image_path=args.image,
                raw_output=raw_output,
                object_ids=grounding_result.object_ids,
                canvas_size=(width, height),
                protocol_version=args.protocol_version,
                device=args.device,
                output_dir=args.mask_output_dir,
            )
        except Exception as exc:
            write_mask_sidecar(
                args.mask_output_dir,
                status="error",
                canvas_size=(width, height),
                error=f"{type(exc).__name__}: {exc}",
            )
            print(f"mask_decode_error={type(exc).__name__}: {exc}", file=sys.stderr)
    if args.output:
        write_prediction(args.output, output)
    if args.pretty and metrics.valid_json_count:
        print(json.dumps(json.loads(output), ensure_ascii=False, indent=4))
    else:
        print(output)


if __name__ == "__main__":
    main()
