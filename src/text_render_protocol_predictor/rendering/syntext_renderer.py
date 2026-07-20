"""Strict bridge from model projections to the synthetic protocol renderer."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import ValidationError

from ..protocol.schema import PredictionProtocol


class RenderStatus(str, Enum):
    OK = "ok"
    INVALID_JSON = "invalid_json"
    INVALID_SCHEMA = "invalid_schema"
    INVALID_SEMANTICS = "invalid_semantics"
    UNKNOWN_FONT = "unknown_font"
    RENDERER_FAILURE = "renderer_failure"


@dataclass(frozen=True)
class RenderOutcome:
    status: RenderStatus
    image: Image.Image | None = None
    error: str | None = None


class SyntextPredictionRenderer:
    """Render strict prediction JSON onto an erased-text background.

    Importing ``syntext`` is delayed until construction so the SFT-only path
    does not require the sibling renderer package.
    """

    def __init__(
        self,
        font_paths: list[str | Path],
        *,
        protocol_version: str = "1.0",
        max_objects: int = 64,
        max_text_characters: int = 4096,
        max_font_size: float = 2048.0,
        max_geometry_scale: float = 2.0,
    ) -> None:
        if protocol_version != "1.0":
            raise ValueError(
                "reconstruction GRPO currently requires protocol 1.0: erased-text "
                "backgrounds already retain non-text shapes"
            )
        if not font_paths:
            raise ValueError("renderer.font_paths must contain at least one font path")
        try:
            import syntext
            from syntext import DocumentProtocol, ProtocolRenderer
            from syntext.fonts import FontRegistry
        except ImportError as exc:
            raise ImportError(
                "synthetic-text-protocol is required for GRPO; install the sibling repo "
                "with `pip install -e ../synthetic-text-protocol`"
            ) from exc

        self.protocol_version = protocol_version
        self.renderer_version = str(getattr(syntext, "__version__", "unknown"))
        self.max_objects = int(max_objects)
        self.max_text_characters = int(max_text_characters)
        self.max_font_size = float(max_font_size)
        self.max_geometry_scale = float(max_geometry_scale)
        if self.max_geometry_scale < 1.0:
            raise ValueError("max_geometry_scale must be at least 1")
        self.fonts = FontRegistry.scan([Path(path).expanduser() for path in font_paths])
        self.font_ids = frozenset(self.fonts.ids)
        self._document_protocol_class = DocumentProtocol
        self._renderer = ProtocolRenderer(self.fonts)
        self.font_registry_fingerprint = self._fingerprint_fonts()

    def _fingerprint_fonts(self) -> str:
        digest = hashlib.sha256()
        for font_id in sorted(self.font_ids):
            path = self.fonts.path(font_id)
            digest.update(font_id.encode("utf-8"))
            digest.update(b"\0")
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
        return digest.hexdigest()

    def _validate_semantics(self, prediction: Any, canvas_size: tuple[int, int]) -> str | None:
        if prediction.protocol_version != self.protocol_version:
            return (
                f"expected protocol_version {self.protocol_version!r}, got "
                f"{prediction.protocol_version!r}"
            )
        if (prediction.canvas.width, prediction.canvas.height) != canvas_size:
            return (
                f"predicted canvas {(prediction.canvas.width, prediction.canvas.height)} "
                f"does not match background {canvas_size}"
            )
        if len(prediction.objects) > self.max_objects:
            return f"prediction has {len(prediction.objects)} objects; maximum is {self.max_objects}"
        object_ids = [obj.id for obj in prediction.objects]
        if len(object_ids) != len(set(object_ids)):
            return "prediction contains duplicate object IDs"
        total_characters = sum(len(obj.text) for obj in prediction.objects)
        if total_characters > self.max_text_characters:
            return (
                f"prediction has {total_characters} text characters; maximum is "
                f"{self.max_text_characters}"
            )
        width, height = canvas_size
        maximum_dimension = max(width, height)
        font_limit = min(self.max_font_size, maximum_dimension * self.max_geometry_scale)
        oversized = [obj.id for obj in prediction.objects if obj.style.font_size > font_limit]
        if oversized:
            return f"font_size exceeds {font_limit} for objects: {', '.join(oversized[:5])}"
        for obj in prediction.objects:
            box = obj.geometry.box
            if (
                abs(box.x) > maximum_dimension * self.max_geometry_scale
                or abs(box.y) > maximum_dimension * self.max_geometry_scale
                or box.width > width * self.max_geometry_scale
                or box.height > height * self.max_geometry_scale
            ):
                return f"geometry exceeds safe rendering bounds for object {obj.id!r}"
            if abs(obj.geometry.rotation_degrees) > 3600:
                return f"rotation exceeds safe rendering bounds for object {obj.id!r}"
            if obj.geometry.baseline is not None:
                points = (
                    obj.geometry.baseline.p0,
                    obj.geometry.baseline.p1,
                    obj.geometry.baseline.p2,
                    obj.geometry.baseline.p3,
                )
                if any(
                    abs(point.x) > maximum_dimension * self.max_geometry_scale
                    or abs(point.y) > maximum_dimension * self.max_geometry_scale
                    for point in points
                ):
                    return f"baseline exceeds safe rendering bounds for object {obj.id!r}"
            effect_limit = maximum_dimension * 0.5
            shadow = obj.style.shadow
            if (
                obj.style.stroke.width > effect_limit
                or abs(obj.style.character_spacing) > maximum_dimension
                or obj.style.line_height > 10
                or (
                    shadow is not None
                    and (
                        shadow.blur_radius > effect_limit
                        or abs(shadow.offset_x) > maximum_dimension
                        or abs(shadow.offset_y) > maximum_dimension
                    )
                )
            ):
                return f"style effects exceed safe rendering bounds for object {obj.id!r}"
        unknown = sorted({obj.style.font_id for obj in prediction.objects} - self.font_ids)
        if unknown:
            return f"unknown font_id values: {', '.join(unknown[:10])}"
        return None

    def render_prediction(
        self,
        completion: str,
        background: Image.Image,
        *,
        sample_id: str,
    ) -> RenderOutcome:
        try:
            raw = json.loads(completion)
        except (json.JSONDecodeError, TypeError) as exc:
            return RenderOutcome(RenderStatus.INVALID_JSON, error=str(exc))
        try:
            prediction = PredictionProtocol.model_validate(raw)
        except (ValidationError, ValueError, TypeError) as exc:
            return RenderOutcome(RenderStatus.INVALID_SCHEMA, error=str(exc))

        semantic_error = self._validate_semantics(prediction, background.size)
        if semantic_error is not None:
            status = (
                RenderStatus.UNKNOWN_FONT
                if semantic_error.startswith("unknown font_id")
                else RenderStatus.INVALID_SEMANTICS
            )
            return RenderOutcome(status, error=semantic_error)

        envelope = {
            "protocol_version": prediction.protocol_version,
            "purpose": "render",
            "sample_id": sample_id,
            "seed": 0,
            "canvas": prediction.canvas.model_dump(mode="json"),
            "background": {"source": str(sample_id), "sha256": "0" * 64},
            "objects": [obj.model_dump(mode="json") for obj in prediction.objects],
        }
        try:
            protocol = self._document_protocol_class.model_validate(envelope)
            rendered, _ = self._renderer.render(background, protocol)
        except Exception as exc:
            return RenderOutcome(
                RenderStatus.RENDERER_FAILURE,
                error=f"{type(exc).__name__}: {exc}",
            )
        return RenderOutcome(RenderStatus.OK, image=rendered)
