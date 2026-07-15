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
from src.text_render_protocol_predictor.models.qwen3_vl import inspect_peft_weights_directory
from src.text_render_protocol_predictor.training import ProtocolPromptTemplate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--weights", type=Path, help="PEFT adapter/processor directory")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--image-min-pixels", type=int, default=200704)
    parser.add_argument("--image-max-pixels", type=int, default=1003520)
    parser.add_argument("--output", type=Path, help="Also save the raw generation here")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print valid JSON")
    return parser.parse_args()


def resolve_dtype(name: str, device: str) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.startswith("cuda") else torch.float32
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


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
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model,
        dtype=resolve_dtype(args.dtype, args.device),
        attn_implementation="sdpa",
    )
    if weights_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(weights_path), is_trainable=False)
    model.to(args.device).eval()
    model.config.use_cache = True

    with Image.open(args.image) as image:
        width, height = image.size
    conversation = ProtocolPromptTemplate().conversation(
        image=args.image,
        width=width,
        height=height,
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
    output = processor.batch_decode(
        completion,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    metrics = evaluate_generation_validity([output])
    print(
        f"valid_json={bool(metrics.valid_json_count)} "
        f"schema_valid={bool(metrics.schema_valid_count)} "
        f"generated_tokens={completion.shape[1]} "
        f"generation_latency_seconds={generation_latency_seconds:.3f}",
        file=sys.stderr,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # args.output.write_text(output + "\n", encoding="utf-8")
        with open(args.output, "w") as file:
            json.dump(json.loads(output), file, ensure_ascii=False, indent=4)
    if args.pretty and metrics.valid_json_count:
        print(json.dumps(json.loads(output), ensure_ascii=False, indent=4))
    else:
        print(output)


if __name__ == "__main__":
    main()
