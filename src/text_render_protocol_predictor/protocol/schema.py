"""Strict, version-dispatched Pydantic models for STRP 1.0, 2.0, and 2.1."""

from __future__ import annotations

import json
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


SUPPORTED_PROTOCOL_VERSIONS = frozenset({"1.0", "2.0", "2.1"})


class UnsupportedProtocolVersion(ValueError):
    """Raised when no local schema exists for a declared protocol version."""


class ProtocolModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


Color = Annotated[str, StringConstraints(pattern=r"^#[0-9A-F]{8}$")]
ObjectId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.-]+$")]


class Canvas(ProtocolModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class Background(ProtocolModel):
    source: str
    sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class BoundingBox(ProtocolModel):
    x: float
    y: float
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class Point(ProtocolModel):
    x: float
    y: float


class BezierBaseline(ProtocolModel):
    p0: Point
    p1: Point
    p2: Point
    p3: Point


class Geometry(ProtocolModel):
    mode: Literal["straight", "bezier"]
    box: BoundingBox
    rotation_degrees: float
    baseline: BezierBaseline | None

    @model_validator(mode="after")
    def baseline_matches_mode(self) -> "Geometry":
        if self.mode == "bezier" and self.baseline is None:
            raise ValueError("bezier geometry requires a baseline")
        if self.mode == "straight" and self.baseline is not None:
            raise ValueError("straight geometry requires baseline=null")
        return self


class GeometryV2(Geometry):
    # 2.x explicitly permits an omitted baseline for straight text.
    baseline: BezierBaseline | None = None


class SolidFill(ProtocolModel):
    type: Literal["solid"]
    color: Color


class GradientStop(ProtocolModel):
    offset: float = Field(ge=0, le=1)
    color: Color


class LinearGradientFill(ProtocolModel):
    type: Literal["linear_gradient"]
    angle_degrees: float
    stops: list[GradientStop] = Field(min_length=2)

    @model_validator(mode="after")
    def stops_span_unit_interval(self) -> "LinearGradientFill":
        offsets = [stop.offset for stop in self.stops]
        if offsets != sorted(offsets):
            raise ValueError("gradient stops must be sorted by offset")
        if offsets[0] != 0.0 or offsets[-1] != 1.0:
            raise ValueError("gradient stops must start at 0 and end at 1")
        return self


Fill = Annotated[SolidFill | LinearGradientFill, Field(discriminator="type")]


class Stroke(ProtocolModel):
    width: float = Field(ge=0)
    color: Color


class Shadow(ProtocolModel):
    color: Color
    offset_x: float
    offset_y: float
    blur_radius: float = Field(ge=0)


class TextStyle(ProtocolModel):
    font_id: str = Field(min_length=1)
    font_size: float = Field(gt=0)
    fill: Fill
    stroke: Stroke
    shadow: Shadow | None
    character_spacing: float
    line_height: float = Field(gt=0)
    bold: bool
    italic: bool
    underline: bool
    alignment: Literal["left", "center", "right"]


# Backwards-compatible public name used by existing callers.
Style = TextStyle


class AnnotationEvidence(ProtocolModel):
    text_confidence: float = Field(default=1.0, ge=0, le=1)
    geometry_confidence: float = Field(default=1.0, ge=0, le=1)
    style_confidence: float = Field(default=1.0, ge=0, le=1)
    font_match: Literal["known", "nearest", "unknown"] = "known"
    notes: str | None = None


class ShapeGeometry(ProtocolModel):
    box: BoundingBox
    rotation_degrees: float
    corner_radius: float = Field(ge=0)


class ShapeStyle(ProtocolModel):
    fill: Fill
    stroke: Stroke
    shadow: Shadow | None


class PredictionObjectV1(ProtocolModel):
    id: ObjectId
    text: str = Field(min_length=1)
    language: str = Field(min_length=1)
    direction: Literal["ltr", "rtl", "ttb"]
    geometry: Geometry
    style: TextStyle
    z_order: int


class DatasetObjectV1(PredictionObjectV1):
    tight_bbox: BoundingBox | None
    object_type: Literal["text"] | None = None


class PredictionTextObjectV2(ProtocolModel):
    object_type: Literal["text"]
    id: ObjectId
    text: str = Field(min_length=1)
    language: str = Field(min_length=1)
    direction: Literal["ltr", "rtl", "ttb"]
    geometry: GeometryV2
    style: TextStyle
    z_order: int


class DatasetTextObjectV20(PredictionTextObjectV2):
    tight_bbox: BoundingBox | None


class DatasetTextObjectV21(DatasetTextObjectV20):
    annotation: AnnotationEvidence | None = None


class PredictionShapeObjectV2(ProtocolModel):
    object_type: Literal["shape"]
    id: ObjectId
    shape: Literal["rectangle", "ellipse"]
    geometry: ShapeGeometry
    style: ShapeStyle
    z_order: int


class DatasetShapeObjectV20(PredictionShapeObjectV2):
    tight_bbox: BoundingBox | None


class DatasetShapeObjectV21(DatasetShapeObjectV20):
    annotation: AnnotationEvidence | None = None


PredictionObjectV2 = Annotated[
    PredictionTextObjectV2 | PredictionShapeObjectV2,
    Field(discriminator="object_type"),
]
DatasetObjectV20 = Annotated[
    DatasetTextObjectV20 | DatasetShapeObjectV20,
    Field(discriminator="object_type"),
]
DatasetObjectV21 = Annotated[
    DatasetTextObjectV21 | DatasetShapeObjectV21,
    Field(discriminator="object_type"),
]

# Compatibility aliases. These remain the 1.0 text object because older code
# imported the unversioned names before tagged objects existed.
PredictionObject = PredictionObjectV1
DatasetObject = DatasetObjectV1


def _declared_version(value: Any) -> str:
    if isinstance(value, BaseModel):
        version = getattr(value, "protocol_version", None)
    elif isinstance(value, dict):
        version = value.get("protocol_version")
    else:
        version = None
    if version not in SUPPORTED_PROTOCOL_VERSIONS:
        raise UnsupportedProtocolVersion(f"unsupported protocol_version: {version!r}")
    return version


def detect_protocol_version(value: Any) -> str:
    """Return a supported declared version without guessing future semantics."""
    return _declared_version(value)


class PredictionProtocol(ProtocolModel):
    """Dispatch facade and common base for model-facing protocol versions."""

    _version_models: ClassVar[dict[str, type["PredictionProtocol"]]] = {}
    protocol_version: str
    canvas: Canvas
    objects: list[Any]

    @classmethod
    def model_validate(cls, obj: Any, **kwargs: Any) -> "PredictionProtocol":
        if cls is PredictionProtocol:
            return cls._version_models[_declared_version(obj)].model_validate(obj, **kwargs)
        return super().model_validate(obj, **kwargs)

    @classmethod
    def model_validate_json(cls, json_data: str | bytes | bytearray, **kwargs: Any) -> "PredictionProtocol":
        if cls is PredictionProtocol:
            return cls.model_validate(json.loads(json_data), **kwargs)
        return super().model_validate_json(json_data, **kwargs)


class PredictionProtocolV1(PredictionProtocol):
    protocol_version: Literal["1.0"]
    objects: list[PredictionObjectV1]


class PredictionProtocolV20(PredictionProtocol):
    protocol_version: Literal["2.0"]
    objects: list[PredictionObjectV2]


class PredictionProtocolV21(PredictionProtocol):
    protocol_version: Literal["2.1"]
    objects: list[PredictionObjectV2]


PredictionProtocol._version_models = {
    "1.0": PredictionProtocolV1,
    "2.0": PredictionProtocolV20,
    "2.1": PredictionProtocolV21,
}


class DatasetProtocol(ProtocolModel):
    """Dispatch facade and common base for complete dataset protocols."""

    _version_models: ClassVar[dict[str, type["DatasetProtocol"]]] = {}
    protocol_version: str
    sample_id: str = Field(min_length=1)
    seed: int
    canvas: Canvas
    background: Background
    objects: list[Any]

    @classmethod
    def model_validate(cls, obj: Any, **kwargs: Any) -> "DatasetProtocol":
        if cls is DatasetProtocol:
            return cls._version_models[_declared_version(obj)].model_validate(obj, **kwargs)
        return super().model_validate(obj, **kwargs)

    @classmethod
    def model_validate_json(cls, json_data: str | bytes | bytearray, **kwargs: Any) -> "DatasetProtocol":
        if cls is DatasetProtocol:
            return cls.model_validate(json.loads(json_data), **kwargs)
        return super().model_validate_json(json_data, **kwargs)


class DatasetProtocolV1(DatasetProtocol):
    protocol_version: Literal["1.0"]
    objects: list[DatasetObjectV1]


class DatasetProtocolV20(DatasetProtocol):
    protocol_version: Literal["2.0"]
    purpose: Literal["render"] = "render"
    objects: list[DatasetObjectV20]


class DatasetProtocolV21(DatasetProtocol):
    protocol_version: Literal["2.1"]
    purpose: Literal["render", "annotation"] = "render"
    objects: list[DatasetObjectV21]


DatasetProtocol._version_models = {
    "1.0": DatasetProtocolV1,
    "2.0": DatasetProtocolV20,
    "2.1": DatasetProtocolV21,
}
