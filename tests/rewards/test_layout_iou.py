from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from text_render_protocol_predictor.rewards import (
    calculate_layout_mask_metrics,
    dilate_layout_mask,
    rasterize_protocol_layout_mask,
)


def _point(x, y):
    return SimpleNamespace(x=x, y=y)


def _geometry(*, x, y, width, height, rotation=0, mode="straight", baseline=None):
    return SimpleNamespace(
        mode=mode,
        box=SimpleNamespace(x=x, y=y, width=width, height=height),
        rotation_degrees=rotation,
        baseline=baseline,
    )


def _prediction(*geometries):
    return SimpleNamespace(
        objects=[SimpleNamespace(text="text", geometry=geometry) for geometry in geometries]
    )


def test_rasterizes_union_of_straight_boxes_with_clockwise_rotation():
    prediction = _prediction(
        _geometry(x=4, y=5, width=10, height=4),
        _geometry(x=20, y=5, width=6, height=2, rotation=90),
    )

    mask = rasterize_protocol_layout_mask(prediction, canvas_size=(32, 20))

    assert mask[5:9, 4:14].all()
    assert mask[3:9, 22:24].all()
    assert mask.sum() == 52


def test_rasterizes_bezier_as_a_band_using_box_height():
    baseline = SimpleNamespace(
        p0=_point(4, 14),
        p1=_point(10, 4),
        p2=_point(20, 4),
        p3=_point(26, 14),
    )
    prediction = _prediction(
        _geometry(
            x=4,
            y=4,
            width=22,
            height=5,
            mode="bezier",
            baseline=baseline,
        )
    )

    mask = rasterize_protocol_layout_mask(prediction, canvas_size=(32, 20))

    assert mask[14, 4]
    assert mask[7, 15]
    assert mask[14, 26]
    assert not mask[0, 0]


def test_repeated_dilation_and_tolerant_iou():
    point = np.zeros((15, 15), dtype=bool)
    point[7, 7] = True
    assert dilate_layout_mask(point, kernel_size=5, iterations=1).sum() == 25
    assert dilate_layout_mask(point, kernel_size=5, iterations=2).sum() == 81

    target = np.zeros((20, 20), dtype=bool)
    target[6:10, 5:13] = True
    predicted = np.zeros_like(target)
    predicted[6:10, 8:16] = True
    metrics = calculate_layout_mask_metrics(
        predicted,
        target,
        dilation_kernel_size=5,
        dilation_iterations=2,
    )

    assert metrics.strict_iou == pytest.approx(5 / 11)
    assert metrics.dilated_iou > metrics.strict_iou
    assert 0 < metrics.precision < 1
    assert 0 < metrics.recall < 1


def test_empty_prediction_has_zero_overlap_with_nonempty_target():
    predicted = np.zeros((8, 8), dtype=bool)
    target = np.zeros_like(predicted)
    target[2:4, 2:4] = True

    metrics = calculate_layout_mask_metrics(
        predicted,
        target,
        dilation_kernel_size=1,
        dilation_iterations=0,
    )

    assert metrics.strict_iou == 0
    assert metrics.dilated_iou == 0
    assert metrics.precision == 0
    assert metrics.recall == 0
