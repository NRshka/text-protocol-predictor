from types import SimpleNamespace

import pytest
import torch

from text_render_protocol_predictor.evaluation import (
    angle_error_degrees,
    evaluate_instance_segmentation,
    evaluate_instance_slices,
    oriented_box_iou,
)


def test_instance_metrics_expose_attached_oracle_gap() -> None:
    left = torch.zeros(8, 8)
    right = torch.zeros(8, 8)
    left[1:4, 1:4] = 1
    right[4:7, 4:7] = 1
    target = {"a": left, "b": right}
    predicted = {"a": right, "b": left}

    metrics = evaluate_instance_segmentation(predicted, target)

    assert metrics.attached_iou == 0.0
    assert metrics.oracle_iou == 1.0
    assert metrics.association_gap == 1.0
    assert metrics.ap50 == pytest.approx(1.0)


def test_oriented_box_and_angle_metrics() -> None:
    box = SimpleNamespace(x=10, y=20, width=30, height=12)
    geometry = SimpleNamespace(box=box, rotation_degrees=25)

    assert oriented_box_iou(geometry, geometry) == pytest.approx(1.0)
    assert angle_error_degrees(359, 1) == 2


def test_instance_slices_include_source_mode_and_overlap() -> None:
    left = torch.zeros(16, 16)
    right = torch.zeros(16, 16)
    left[2:8, 2:9] = 1
    right[5:11, 6:13] = 1
    objects = [
        SimpleNamespace(
            id="a",
            text="A",
            geometry=SimpleNamespace(
                mode="straight",
                rotation_degrees=12,
                box=SimpleNamespace(x=2, y=2, width=7, height=6),
            ),
        ),
        SimpleNamespace(
            id="b",
            text="B",
            geometry=SimpleNamespace(
                mode="bezier",
                rotation_degrees=0,
                box=SimpleNamespace(x=6, y=5, width=7, height=6),
            ),
        ),
    ]
    record = SimpleNamespace(
        mask_supervision=SimpleNamespace(source="files"),
        protocol=SimpleNamespace(objects=objects),
        canvas_width=64,
        canvas_height=64,
    )

    slices = evaluate_instance_slices(
        {"a": left, "b": right},
        {"a": left, "b": right},
        scores={"a": 1.0, "b": 1.0},
        record=record,
    )

    assert slices["source/real"].attached_iou == 1.0
    assert slices["mode/straight"].target_count == 1
    assert slices["mode/bezier"].target_count == 1
    assert slices["challenge/rotated"].target_count == 1
    assert slices["challenge/overlapping"].target_count == 2
