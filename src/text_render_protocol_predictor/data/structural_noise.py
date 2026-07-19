"""Deterministic structural noise for protocol geometry."""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, fields
from typing import Any, Mapping

from ..protocol.schema import (
    BezierBaseline,
    BoundingBox,
    DatasetProtocol,
    DatasetObjectV1,
    DatasetTextObjectV20,
    Point,
)


_SEMANTIC_GROUP_PATTERNS = {
    "headline": re.compile(r"(?:^|[_.-])(headline|heading|title)(?:$|[_.-])", re.I),
    "feature_list": re.compile(r"(?:^|[_.-])(feature|features|benefit)(?:$|[_.-])", re.I),
    "attribute_grid": re.compile(
        r"(?:^|[_.-])(attribute|attributes|spec|specification)(?:$|[_.-])", re.I
    ),
    "badge": re.compile(r"(?:^|[_.-])(badge|label|ribbon)(?:$|[_.-])", re.I),
    "footer": re.compile(r"(?:^|[_.-])(footer|legal|disclaimer)(?:$|[_.-])", re.I),
    "cta": re.compile(r"(?:^|[_.-])(cta|call.to.action|button)(?:$|[_.-])", re.I),
}


@dataclass(frozen=True)
class StructuralNoiseConfig:
    """Sampling ranges expressed as fractions of the original canvas or box."""

    enabled: bool = False
    probability: float = 1.0
    seed: int = 0

    group_translation_std_x: float = 0.015
    group_translation_std_y: float = 0.015
    group_scale_min: float = 0.95
    group_scale_max: float = 1.05
    group_rotation_probability: float = 0.15
    group_rotation_min_degrees: float = 0.5
    group_rotation_max_degrees: float = 2.0

    local_position_min: float = 0.003
    local_position_max: float = 0.010
    local_size_min: float = 0.02
    local_size_max: float = 0.06
    local_rotation_probability: float = 0.15
    local_rotation_min_degrees: float = 0.5
    local_rotation_max_degrees: float = 2.0

    bend_probability: float = 0.05
    bend_min_fraction: float = 0.03
    bend_max_fraction: float = 0.12

    def __post_init__(self) -> None:
        probabilities = (
            self.probability,
            self.group_rotation_probability,
            self.local_rotation_probability,
            self.bend_probability,
        )
        if any(value < 0 or value > 1 for value in probabilities):
            raise ValueError("structural-noise probabilities must be in [0, 1]")
        ranges = (
            ("group_scale", self.group_scale_min, self.group_scale_max),
            ("local_position", self.local_position_min, self.local_position_max),
            ("local_size", self.local_size_min, self.local_size_max),
            (
                "group_rotation_degrees",
                self.group_rotation_min_degrees,
                self.group_rotation_max_degrees,
            ),
            (
                "local_rotation_degrees",
                self.local_rotation_min_degrees,
                self.local_rotation_max_degrees,
            ),
            ("bend_fraction", self.bend_min_fraction, self.bend_max_fraction),
        )
        for name, minimum, maximum in ranges:
            if minimum < 0 or maximum < minimum:
                raise ValueError(f"invalid structural-noise range {name}: [{minimum}, {maximum}]")
        if self.group_scale_min <= 0:
            raise ValueError("group_scale_min must be positive")
        if self.local_size_max >= 1:
            raise ValueError("local_size_max must be less than 1")
        if self.group_translation_std_x < 0 or self.group_translation_std_y < 0:
            raise ValueError("group translation standard deviations must be non-negative")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "StructuralNoiseConfig":
        if value is None:
            return cls()
        known = {field.name for field in fields(cls)}
        values = {key: value[key] for key in known if key in value}
        return cls(**values)


@dataclass(frozen=True)
class _Affine:
    origin_x: float
    origin_y: float
    scale_x: float
    scale_y: float
    rotation_degrees: float
    translate_x: float
    translate_y: float

    def point(self, point: Point) -> Point:
        x = (point.x - self.origin_x) * self.scale_x
        y = (point.y - self.origin_y) * self.scale_y
        radians = math.radians(self.rotation_degrees)
        cos_angle = math.cos(radians)
        sin_angle = math.sin(radians)
        return Point(
            x=self.origin_x + x * cos_angle - y * sin_angle + self.translate_x,
            y=self.origin_y + x * sin_angle + y * cos_angle + self.translate_y,
        )


def _signed_magnitude(rng: random.Random, minimum: float, maximum: float) -> float:
    if maximum == 0:
        return 0.0
    magnitude = rng.uniform(minimum, maximum)
    return magnitude if rng.random() < 0.5 else -magnitude


def _occasional_angle(
    rng: random.Random, *, probability: float, minimum: float, maximum: float
) -> float:
    if rng.random() >= probability:
        return 0.0
    return _signed_magnitude(rng, minimum, maximum)


def _group_name(object_id: str, explicit_groups: Mapping[str, str]) -> str:
    if object_id in explicit_groups:
        return explicit_groups[object_id]
    for group_name, pattern in _SEMANTIC_GROUP_PATTERNS.items():
        if pattern.search(object_id):
            return group_name
    return f"object:{object_id}"


def _group_bounds(objects: list[Any]) -> BoundingBox:
    left = min(obj.geometry.box.x for obj in objects)
    top = min(obj.geometry.box.y for obj in objects)
    right = max(obj.geometry.box.x + obj.geometry.box.width for obj in objects)
    bottom = max(obj.geometry.box.y + obj.geometry.box.height for obj in objects)
    return BoundingBox(x=left, y=top, width=right - left, height=bottom - top)


def _transform_object(obj: Any, affine: _Affine) -> None:
    box = obj.geometry.box
    center = affine.point(Point(x=box.x + box.width / 2, y=box.y + box.height / 2))
    baseline = getattr(obj.geometry, "baseline", None)
    if baseline is not None:
        corners = [
            affine.point(Point(x=box.x, y=box.y)),
            affine.point(Point(x=box.x + box.width, y=box.y)),
            affine.point(Point(x=box.x, y=box.y + box.height)),
            affine.point(Point(x=box.x + box.width, y=box.y + box.height)),
        ]
        left = min(point.x for point in corners)
        top = min(point.y for point in corners)
        right = max(point.x for point in corners)
        bottom = max(point.y for point in corners)
        obj.geometry.box = BoundingBox(
            x=left,
            y=top,
            width=right - left,
            height=bottom - top,
        )
        obj.geometry.baseline = BezierBaseline(
            p0=affine.point(baseline.p0),
            p1=affine.point(baseline.p1),
            p2=affine.point(baseline.p2),
            p3=affine.point(baseline.p3),
        )
    else:
        obj.geometry.box = BoundingBox(
            x=center.x - box.width * affine.scale_x / 2,
            y=center.y - box.height * affine.scale_y / 2,
            width=box.width * affine.scale_x,
            height=box.height * affine.scale_y,
        )
        obj.geometry.rotation_degrees += affine.rotation_degrees
    obj.tight_bbox = None


def _bend_object(
    obj: DatasetObjectV1 | DatasetTextObjectV20,
    rng: random.Random,
    config: StructuralNoiseConfig,
) -> None:
    box = obj.geometry.box
    bend = _signed_magnitude(rng, config.bend_min_fraction, config.bend_max_fraction)
    bend_pixels = bend * box.height
    if obj.geometry.baseline is not None:
        baseline = obj.geometry.baseline
        baseline.p1.y += bend_pixels
        baseline.p2.y += bend_pixels
        return

    center_x = box.x + box.width / 2
    center_y = box.y + box.height / 2
    half_width = box.width * 0.45
    baseline = BezierBaseline(
        p0=Point(x=center_x - half_width, y=center_y),
        p1=Point(x=center_x - half_width / 3, y=center_y + bend_pixels),
        p2=Point(x=center_x + half_width / 3, y=center_y + bend_pixels),
        p3=Point(x=center_x + half_width, y=center_y),
    )
    rotation = obj.geometry.rotation_degrees
    if rotation:
        rotation_affine = _Affine(center_x, center_y, 1, 1, rotation, 0, 0)
        baseline = BezierBaseline(
            p0=rotation_affine.point(baseline.p0),
            p1=rotation_affine.point(baseline.p1),
            p2=rotation_affine.point(baseline.p2),
            p3=rotation_affine.point(baseline.p3),
        )
    obj.geometry.mode = "bezier"
    obj.geometry.rotation_degrees = 0.0
    obj.geometry.baseline = baseline


def apply_structural_noise(
    protocol: DatasetProtocol,
    *,
    config: StructuralNoiseConfig,
    seed: int,
    object_groups: Mapping[str, str] | None = None,
) -> DatasetProtocol:
    """Return a perturbed copy, preserving coherent motion within semantic groups."""
    if not config.enabled:
        return protocol
    rng = random.Random(seed)
    if rng.random() >= config.probability:
        return protocol

    result = protocol.model_copy(deep=True)
    explicit_groups = object_groups or {}
    groups: dict[str, list[Any]] = {}
    for obj in result.objects:
        groups.setdefault(_group_name(obj.id, explicit_groups), []).append(obj)

    canvas_width = result.canvas.width
    canvas_height = result.canvas.height
    for objects in groups.values():
        bounds = _group_bounds(objects)
        group_affine = _Affine(
            origin_x=bounds.x + bounds.width / 2,
            origin_y=bounds.y + bounds.height / 2,
            scale_x=rng.uniform(config.group_scale_min, config.group_scale_max),
            scale_y=rng.uniform(config.group_scale_min, config.group_scale_max),
            rotation_degrees=_occasional_angle(
                rng,
                probability=config.group_rotation_probability,
                minimum=config.group_rotation_min_degrees,
                maximum=config.group_rotation_max_degrees,
            ),
            translate_x=rng.gauss(0, config.group_translation_std_x * canvas_width),
            translate_y=rng.gauss(0, config.group_translation_std_y * canvas_height),
        )
        for obj in objects:
            _transform_object(obj, group_affine)

    for obj in result.objects:
        box = obj.geometry.box
        local_affine = _Affine(
            origin_x=box.x + box.width / 2,
            origin_y=box.y + box.height / 2,
            scale_x=1 + _signed_magnitude(
                rng, config.local_size_min, config.local_size_max
            ),
            scale_y=1 + _signed_magnitude(
                rng, config.local_size_min, config.local_size_max
            ),
            rotation_degrees=_occasional_angle(
                rng,
                probability=config.local_rotation_probability,
                minimum=config.local_rotation_min_degrees,
                maximum=config.local_rotation_max_degrees,
            ),
            translate_x=_signed_magnitude(
                rng, config.local_position_min, config.local_position_max
            )
            * canvas_width,
            translate_y=_signed_magnitude(
                rng, config.local_position_min, config.local_position_max
            )
            * canvas_height,
        )
        _transform_object(obj, local_affine)
        if isinstance(obj, (DatasetObjectV1, DatasetTextObjectV20)) and (
            rng.random() < config.bend_probability
        ):
            _bend_object(obj, rng, config)

    return result
