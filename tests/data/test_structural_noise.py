from __future__ import annotations

import pytest

from text_render_protocol_predictor.data import (
    StructuralNoiseConfig,
    apply_structural_noise,
)
from text_render_protocol_predictor.protocol.schema import DatasetProtocol


def _config(**overrides: object) -> StructuralNoiseConfig:
    values = {
        "enabled": True,
        "group_scale_min": 1.0,
        "group_scale_max": 1.0,
        "group_rotation_probability": 0.0,
        "local_position_min": 0.0,
        "local_position_max": 0.0,
        "local_size_min": 0.0,
        "local_size_max": 0.0,
        "local_rotation_probability": 0.0,
        "bend_probability": 0.0,
    }
    values.update(overrides)
    return StructuralNoiseConfig(**values)


def test_structural_noise_is_deterministic_and_does_not_mutate_input(
    protocol_dict: dict,
) -> None:
    protocol = DatasetProtocol.model_validate(protocol_dict)
    first = apply_structural_noise(protocol, config=_config(), seed=41)
    second = apply_structural_noise(protocol, config=_config(), seed=41)

    assert first == second
    assert first is not protocol
    assert protocol.objects[0].geometry.box.y == pytest.approx(20.1236)
    assert first.objects[0].tight_bbox is None


def test_explicit_group_members_share_group_translation(protocol_dict: dict) -> None:
    protocol = DatasetProtocol.model_validate(protocol_dict)
    original_delta_x = (
        protocol.objects[1].geometry.box.x - protocol.objects[0].geometry.box.x
    )
    augmented = apply_structural_noise(
        protocol,
        config=_config(),
        seed=8,
        object_groups={"source-a": "headline", "source-b": "headline"},
    )
    augmented_delta_x = (
        augmented.objects[1].geometry.box.x - augmented.objects[0].geometry.box.x
    )

    assert augmented_delta_x == pytest.approx(original_delta_x)


def test_bending_converts_straight_geometry_to_bezier(protocol_dict: dict) -> None:
    protocol = DatasetProtocol.model_validate(protocol_dict)
    augmented = apply_structural_noise(
        protocol,
        config=_config(
            group_translation_std_x=0.0,
            group_translation_std_y=0.0,
            bend_probability=1.0,
            bend_min_fraction=0.1,
            bend_max_fraction=0.1,
        ),
        seed=3,
    )

    assert all(obj.geometry.mode == "bezier" for obj in augmented.objects)
    assert all(obj.geometry.baseline is not None for obj in augmented.objects)
    assert all(obj.geometry.rotation_degrees == 0 for obj in augmented.objects)


def test_structural_noise_transforms_shapes_without_bending(
    protocol_21_dict: dict,
) -> None:
    protocol = DatasetProtocol.model_validate(protocol_21_dict)
    augmented = apply_structural_noise(
        protocol,
        config=_config(
            group_translation_std_x=0.01,
            group_translation_std_y=0.01,
            bend_probability=1.0,
            bend_min_fraction=0.1,
            bend_max_fraction=0.1,
        ),
        seed=3,
    )

    shape = next(obj for obj in augmented.objects if obj.object_type == "shape")
    text = next(obj for obj in augmented.objects if obj.object_type == "text")
    assert not hasattr(shape.geometry, "mode")
    assert shape.tight_bbox is None
    assert text.geometry.mode == "bezier"
