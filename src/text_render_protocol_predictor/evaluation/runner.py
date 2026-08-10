"""Distributed generation evaluation for a fixed validation subset."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import torch
import torch.nn.functional as F

from .generation import GenerationValidityMetrics, evaluate_generation_predictions
from .geometry import AuxiliaryGeometryMetrics, evaluate_auxiliary_geometry
from .instance_segmentation import (
    InstanceSegmentationMetrics,
    evaluate_instance_segmentation,
    evaluate_instance_slices,
)
from ..data.instance_masks import (
    load_record_instance_masks,
    mask_image_to_tensor,
    rasterize_text_slot_mask,
)
from ..protocol.coordinate_tokens import (
    CoordinateTokenCodec,
    decode_coordinate_json_or_original,
)
from ..protocol.grounding import strip_mask_references
from ..protocol.schema import PredictionProtocol
from ..training.collator import locate_completion_token_positions


def _serialized_geometry_mask_iou(
    clean_output: str,
    *,
    coordinate_codec: CoordinateTokenCodec | None,
    decimal_places: int,
    record: Any,
    target_masks: dict[str, torch.Tensor],
) -> tuple[float, int]:
    decoded = decode_coordinate_json_or_original(
        clean_output,
        coordinate_codec,
        decimal_places=decimal_places,
    )
    prediction = PredictionProtocol.model_validate_json(decoded)
    predicted_masks = {
        obj.id: mask_image_to_tensor(
            rasterize_text_slot_mask(
                obj,
                canvas_size=(record.canvas_width, record.canvas_height),
            )
        )
        for obj in prediction.objects
        if hasattr(obj, "text") and obj.id in target_masks
    }
    metrics = evaluate_instance_segmentation(predicted_masks, target_masks)
    return metrics.attached_iou * metrics.target_count, metrics.target_count


def _summarize_instance_slices(
    samples: list[dict[str, InstanceSegmentationMetrics]],
) -> dict[str, int | float]:
    grouped: dict[str, list[InstanceSegmentationMetrics]] = {}
    for sample in samples:
        for name, metrics in sample.items():
            grouped.setdefault(name, []).append(metrics)
    result: dict[str, int | float] = {}
    for name, values in grouped.items():
        image_count = len(values)
        target_count = sum(item.target_count for item in values)
        prefix = f"generation/slice/{name}"
        result[f"{prefix}/image_count"] = image_count
        result[f"{prefix}/target_count"] = target_count
        if target_count:
            attached = sum(item.attached_iou * item.target_count for item in values)
            oracle = sum(item.oracle_iou * item.target_count for item in values)
            result[f"{prefix}/attached_iou"] = attached / target_count
            result[f"{prefix}/oracle_iou"] = oracle / target_count
            result[f"{prefix}/association_gap"] = max(
                0.0,
                (oracle - attached) / target_count,
            )
            result[f"{prefix}/dice"] = (
                sum(item.dice * item.target_count for item in values) / target_count
            )
            result[f"{prefix}/boundary_fscore"] = (
                sum(item.boundary_fscore * item.target_count for item in values)
                / target_count
            )
        result[f"{prefix}/ap50"] = sum(item.ap50 for item in values) / image_count
        result[f"{prefix}/ap75"] = sum(item.ap75 for item in values) / image_count
    return result


@torch.no_grad()
def evaluate_generation(
    *,
    accelerator: Any,
    model: Any,
    processor: Any,
    dataloader: Any,
    max_new_tokens: int,
    progress_bar: bool = True,
    coordinate_codec: CoordinateTokenCodec | None = None,
    decimal_places: int = 3,
    grounding_enabled: bool = False,
    prompt_template: Any | None = None,
    prediction_sink: list[dict[str, str]] | None = None,
) -> GenerationValidityMetrics:
    from accelerate.utils import gather_object
    from tqdm.auto import tqdm

    was_training = model.training
    model.eval()
    generation_model = accelerator.unwrap_model(model)
    accelerator_device = getattr(accelerator, "device", torch.device("cpu"))
    local_generation_seconds = 0.0
    local_generation_count = 0
    local_peak_memory = 0
    local_results: list[
        tuple[
            str,
            str,
            str,
            InstanceSegmentationMetrics | None,
            tuple[float, int] | None,
            AuxiliaryGeometryMetrics | None,
            dict[str, InstanceSegmentationMetrics],
        ]
    ] = []
    batches = tqdm(
        dataloader,
        desc="Generation evaluation",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
        disable=not progress_bar or not accelerator.is_local_main_process,
    )
    for batch in batches:
        sample_ids = batch.pop("_sample_ids")
        targets = batch.pop("_targets")
        records = batch.pop("_records", None)
        input_length = batch["input_ids"].shape[1]
        if accelerator_device.type == "cuda":
            torch.cuda.synchronize(accelerator_device)
            torch.cuda.reset_peak_memory_stats(accelerator_device)
        started_at = time.perf_counter()
        generated = generation_model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            synced_gpus=accelerator.num_processes > 1,
        )
        if accelerator_device.type == "cuda":
            torch.cuda.synchronize(accelerator_device)
            local_peak_memory = max(
                local_peak_memory,
                int(torch.cuda.max_memory_allocated(accelerator_device)),
            )
        local_generation_seconds += time.perf_counter() - started_at
        local_generation_count += len(sample_ids)
        completions = generated[:, input_length:]
        decoded = processor.batch_decode(
            completions,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        mask_metrics: list[InstanceSegmentationMetrics | None] = [None] * len(decoded)
        serialized_metrics: list[tuple[float, int] | None] = [None] * len(decoded)
        auxiliary_geometry_metrics: list[AuxiliaryGeometryMetrics | None] = [
            None
        ] * len(decoded)
        slice_metrics: list[dict[str, InstanceSegmentationMetrics]] = [
            {} for _ in decoded
        ]
        if grounding_enabled:
            if prompt_template is None or records is None:
                raise ValueError(
                    "grounded generation evaluation requires prompt_template and records"
                )
            for index, (record, raw_output) in enumerate(
                zip(records, decoded, strict=True)
            ):
                target_masks = load_record_instance_masks(record)
                try:
                    stripped = strip_mask_references(raw_output)
                    if not stripped.valid:
                        continue
                    if target_masks:
                        serialized_metrics[index] = _serialized_geometry_mask_iou(
                            stripped.clean_json,
                            coordinate_codec=coordinate_codec,
                            decimal_places=decimal_places,
                            record=record,
                            target_masks=target_masks,
                        )
                    conversation = prompt_template.conversation(
                        image=record.image_path,
                        width=record.canvas_width,
                        height=record.canvas_height,
                        protocol_version=record.protocol_version,
                        target=raw_output,
                    )
                    full_batch = processor.apply_chat_template(
                        [conversation],
                        tokenize=True,
                        add_generation_prompt=False,
                        return_dict=True,
                        return_tensors="pt",
                        processor_kwargs={"padding": True},
                    ).to(accelerator_device)
                    prompt_conversation = prompt_template.conversation(
                        image=record.image_path,
                        width=record.canvas_width,
                        height=record.canvas_height,
                        protocol_version=record.protocol_version,
                    )
                    prompt_batch = processor.apply_chat_template(
                        [prompt_conversation],
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt",
                        processor_kwargs={"padding": True},
                    ).to(accelerator_device)
                    positions = locate_completion_token_positions(
                        full_batch,
                        prompt_batch,
                        token_id=generation_model.mask_token_id,
                    )[0]
                    if len(positions) != len(stripped.object_ids):
                        continue
                    grounded_output = generation_model(
                        **full_batch,
                        mask_token_positions=positions.unsqueeze(0),
                        instance_valid=torch.ones_like(
                            positions,
                            dtype=torch.bool,
                        ).unsqueeze(0),
                    )
                    probabilities = F.interpolate(
                        grounded_output.mask_logits[0].sigmoid().unsqueeze(1),
                        size=(record.canvas_height, record.canvas_width),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1).cpu()
                    qualities = grounded_output.geometry_predictions[0][
                        "quality_logits"
                    ].sigmoid().cpu()
                    auxiliary_geometry_metrics[index] = evaluate_auxiliary_geometry(
                        grounded_output.geometry_predictions[0],
                        object_ids=stripped.object_ids,
                        target_protocol=record.protocol,
                    )
                    predicted_masks = dict(
                        (
                            (object_id, probability)
                            for object_id, probability in zip(
                                stripped.object_ids, probabilities, strict=True
                            )
                            if object_id in target_masks
                        )
                    )
                    scores = {
                        object_id: float(score.item())
                        for object_id, score in zip(
                            stripped.object_ids, qualities, strict=True
                        )
                        if object_id in target_masks
                    }
                    if target_masks:
                        mask_metrics[index] = evaluate_instance_segmentation(
                            predicted_masks,
                            target_masks,
                            scores=scores,
                        )
                        slice_metrics[index] = evaluate_instance_slices(
                            predicted_masks,
                            target_masks,
                            scores=scores,
                            record=record,
                        )
                except (RuntimeError, TypeError, ValueError):
                    continue
        local_results.extend(
            zip(
                sample_ids,
                decoded,
                targets,
                mask_metrics,
                serialized_metrics,
                auxiliary_geometry_metrics,
                slice_metrics,
                strict=True,
            )
        )

    gathered_results = gather_object(local_results)
    gathered_performance = gather_object(
        [(local_generation_seconds, local_generation_count, local_peak_memory)]
    )
    unique_results: dict[
        str,
        tuple[
            str,
            str,
            InstanceSegmentationMetrics | None,
            tuple[float, int] | None,
            AuxiliaryGeometryMetrics | None,
            dict[str, InstanceSegmentationMetrics],
        ],
    ] = {}
    for (
        sample_id,
        output,
        target,
        instance_metrics,
        serialized_metric,
        auxiliary_metric,
        per_slice_metrics,
    ) in gathered_results:
        unique_results.setdefault(
            sample_id,
            (
                output,
                target,
                instance_metrics,
                serialized_metric,
                auxiliary_metric,
                per_slice_metrics,
            ),
        )
    if was_training:
        model.train()
    if prediction_sink is not None:
        prediction_sink.extend(
            {
                "sample_id": sample_id,
                "prediction": result[0],
                "target": result[1],
            }
            for sample_id, result in unique_results.items()
        )
    native_outputs: list[str] = []
    grounding_valid_count = 0
    for (
        output,
        _target,
        _instance_metrics,
        _serialized,
        _auxiliary,
        _slices,
    ) in unique_results.values():
        if not grounding_enabled:
            native_outputs.append(output)
            continue
        try:
            stripped = strip_mask_references(output)
        except (TypeError, ValueError):
            native_outputs.append(output)
        else:
            native_outputs.append(stripped.clean_json)
            grounding_valid_count += int(stripped.valid)
    outputs = [
        decode_coordinate_json_or_original(
            output, coordinate_codec, decimal_places=decimal_places
        )
        for output in native_outputs
    ]
    targets = [
        decode_coordinate_json_or_original(
            result[1], coordinate_codec, decimal_places=decimal_places
        )
        for result in unique_results.values()
    ]
    metrics = evaluate_generation_predictions(outputs, targets)
    metrics = replace(
        metrics,
        generation_latency_seconds_sum=sum(item[0] for item in gathered_performance),
        generation_latency_count=sum(item[1] for item in gathered_performance),
        peak_memory_bytes=max((item[2] for item in gathered_performance), default=0),
    )
    if grounding_enabled:
        instance_results = [
            result[2] for result in unique_results.values() if result[2] is not None
        ]
        metrics = replace(
            metrics,
            grounding_evaluated_count=len(outputs),
            grounding_valid_count=grounding_valid_count,
            mask_evaluated_count=len(instance_results),
            mask_target_count=sum(item.target_count for item in instance_results),
            mask_prediction_count=sum(item.prediction_count for item in instance_results),
            mask_attached_iou_sum=sum(
                item.attached_iou * item.target_count for item in instance_results
            ),
            mask_oracle_iou_sum=sum(
                item.oracle_iou * item.target_count for item in instance_results
            ),
            mask_dice_sum=sum(item.dice * item.target_count for item in instance_results),
            mask_boundary_fscore_sum=sum(
                item.boundary_fscore * item.target_count for item in instance_results
            ),
            mask_ap50_sum=sum(item.ap50 for item in instance_results),
            mask_ap75_sum=sum(item.ap75 for item in instance_results),
            serialized_geometry_mask_iou_sum=sum(
                item[0]
                for result in unique_results.values()
                if (item := result[3]) is not None
            ),
            serialized_geometry_mask_count=sum(
                item[1]
                for result in unique_results.values()
                if (item := result[3]) is not None
            ),
            auxiliary_geometry_count=sum(
                item.instance_count
                for result in unique_results.values()
                if (item := result[4]) is not None
            ),
            auxiliary_geometry_mode_correct_count=sum(
                item.mode_correct_count
                for result in unique_results.values()
                if (item := result[4]) is not None
            ),
            auxiliary_oriented_box_iou_sum=sum(
                item.oriented_box_iou_sum
                for result in unique_results.values()
                if (item := result[4]) is not None
            ),
            auxiliary_angle_absolute_error_sum=sum(
                item.angle_absolute_error_sum
                for result in unique_results.values()
                if (item := result[4]) is not None
            ),
            auxiliary_bezier_count=sum(
                item.bezier_count
                for result in unique_results.values()
                if (item := result[4]) is not None
            ),
            auxiliary_bezier_centerline_error_sum=sum(
                item.bezier_centerline_error_sum
                for result in unique_results.values()
                if (item := result[4]) is not None
            ),
            grounding_slice_metrics=_summarize_instance_slices(
                [result[5] for result in unique_results.values()]
            ),
        )
    return metrics
