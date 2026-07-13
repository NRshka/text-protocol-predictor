"""Strict Pydantic models for STRP 1.0 and the model-facing projection."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


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


class Style(ProtocolModel):
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


class PredictionObject(ProtocolModel):
    id: ObjectId
    text: str = Field(min_length=1)
    language: str = Field(min_length=1)
    direction: Literal["ltr", "rtl", "ttb"]
    geometry: Geometry
    style: Style
    z_order: int


class DatasetObject(PredictionObject):
    tight_bbox: BoundingBox | None


class PredictionProtocol(ProtocolModel):
    protocol_version: Literal["1.0"]
    canvas: Canvas
    objects: list[PredictionObject]


class DatasetProtocol(ProtocolModel):
    protocol_version: Literal["1.0"]
    sample_id: str = Field(min_length=1)
    seed: int
    canvas: Canvas
    background: Background
    objects: list[DatasetObject]

