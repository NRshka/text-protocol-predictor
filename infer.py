from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from src.text_render_protocol_predictor.evaluation import evaluate_generation_validity
from src.text_render_protocol_predictor.models.qwen3_vl import (
    inspect_peft_weights_directory,
    resize_model_to_tokenizer_vocabulary,
)
from src.text_render_protocol_predictor.protocol import CoordinateTokenCodec
from src.text_render_protocol_predictor.training import ProtocolPromptTemplate


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
    parser.add_argument("--image-min-pixels", type=int, default=200704)
    parser.add_argument("--image-max-pixels", type=int, default=1003520)
    parser.add_argument("--output", type=Path, help="Save the pixel-coordinate prediction here")
    parser.add_argument(
        "--raw-output",
        type=Path,
        help="Optionally save the native model generation before coordinate decoding",
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
    model.to(args.device).eval()
    model.config.use_cache = True

    with Image.open(args.image) as image:
        width, height = image.size
    conversation = ProtocolPromptTemplate(coordinate_codec=coordinate_codec).conversation(
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
    completion = generated[:, batch["input_ids"].shape[1] :]
    raw_output = processor.batch_decode(
        completion,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    output, coordinate_decode_error = decode_prediction_coordinates(
        raw_output,
        coordinate_codec,
        image_size=(width, height),
    )

    metrics = evaluate_generation_validity([output])
    print(
        f"valid_json={bool(metrics.valid_json_count)} "
        f"schema_valid={bool(metrics.schema_valid_count)} "
        f"generated_tokens={completion.shape[1]} "
        f"generation_latency_seconds={generation_latency_seconds:.3f}",
        file=sys.stderr,
    )
    if coordinate_decode_error is not None:
        print(
            f"coordinate_decode_error={coordinate_decode_error}",
            file=sys.stderr,
        )
    if args.raw_output:
        write_prediction(args.raw_output, raw_output)
    if args.output:
        write_prediction(args.output, output)
    if args.pretty and metrics.valid_json_count:
        print(json.dumps(json.loads(output), ensure_ascii=False, indent=4))
    else:
        print(output)


if __name__ == "__main__":
    main()
