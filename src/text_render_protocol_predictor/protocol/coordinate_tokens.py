"""Canvas-normalized atomic coordinate tokens for model-facing protocols."""

from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from .canonicalizer import DEFAULT_DECIMAL_PLACES, canonicalize

COORDINATE_ENCODING_VERSION = "1.0.0"
DEFAULT_COORDINATE_BINS = 512


@dataclass(frozen=True)
class CoordinateTokenCodec:
    """Encode geometry coordinates as a fixed, normalized token vocabulary.

    Coordinate values are normalized independently by canvas width or height.
    Box widths/heights use the same axes as x/y. Style measurements, rotation,
    and shape corner radii deliberately remain ordinary JSON numbers.
    """

    bins: int = DEFAULT_COORDINATE_BINS
    prefix: str = "coord"

    def __post_init__(self) -> None:
        if self.bins < 2:
            raise ValueError("coordinate bins must be at least 2")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", self.prefix):
            raise ValueError(
                "coordinate token prefix must start with a letter and contain only "
                "letters, digits, underscores, or hyphens"
            )

    @property
    def digits(self) -> int:
        return len(str(self.bins - 1))

    def token(self, index: int) -> str:
        if not 0 <= index < self.bins:
            raise ValueError(f"coordinate token index {index} is outside [0, {self.bins - 1}]")
        return f"<{self.prefix}_{index:0{self.digits}d}>"

    def json_token(self, index: int) -> str:
        """Return the exact tokenizer surface, including the JSON string quotes."""
        return json.dumps(self.token(index), ensure_ascii=False)

    @property
    def tokens(self) -> tuple[str, ...]:
        """Logical coordinate values as they appear after JSON parsing."""
        return tuple(self.token(index) for index in range(self.bins))

    @property
    def tokenizer_tokens(self) -> tuple[str, ...]:
        """Atomic tokenizer surfaces used in compact canonical JSON."""
        return tuple(self.json_token(index) for index in range(self.bins))

    def quantize(self, value: int | float, extent: int, *, positive: bool = False) -> int:
        if extent <= 0:
            raise ValueError("coordinate extent must be positive")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("coordinate value must be finite")
        normalized = min(max(numeric / extent, 0.0), 1.0)
        index = round(normalized * (self.bins - 1))
        # PredictionProtocol requires positive box dimensions. Reserve bin zero
        # for positions/control points while keeping every encoded target valid.
        return max(1, index) if positive else index

    def dequantize(self, index: int, extent: int) -> float:
        if extent <= 0:
            raise ValueError("coordinate extent must be positive")
        if not 0 <= index < self.bins:
            raise ValueError(f"coordinate token index {index} is outside [0, {self.bins - 1}]")
        return index * extent / (self.bins - 1)

    def encode_json(
        self,
        value: Mapping[str, Any] | Any | str,
        *,
        decimal_places: int = DEFAULT_DECIMAL_PLACES,
        text_only: bool = False,
    ) -> str:
        """Return canonical prediction JSON with quantized geometry values."""
        target = json.loads(
            canonicalize(
                value,
                decimal_places=decimal_places,
                text_only=text_only,
            )
        )
        self._transform_protocol(target, encode=True)
        return json.dumps(
            target,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )

    def decode_json(
        self,
        value: Mapping[str, Any] | str,
        *,
        decimal_places: int = DEFAULT_DECIMAL_PLACES,
        canvas_size: tuple[int, int] | None = None,
        allow_numeric_coordinates: bool = False,
    ) -> str:
        """Decode model-facing JSON back to a validated canonical protocol."""
        target = json.loads(value) if isinstance(value, str) else deepcopy(dict(value))
        if canvas_size is not None:
            width, height = canvas_size
            if width <= 0 or height <= 0:
                raise ValueError("canvas_size dimensions must be positive")
            if not isinstance(target.get("canvas"), Mapping):
                raise ValueError("protocol canvas must be an object")
            target["canvas"] = {"width": int(width), "height": int(height)}
        self._transform_protocol(
            target,
            encode=False,
            allow_numeric_coordinates=allow_numeric_coordinates,
        )
        return canonicalize(target, decimal_places=decimal_places)

    def _parse_token(self, value: Any) -> int:
        if not isinstance(value, str):
            raise ValueError(f"expected a coordinate token, got {value!r}")
        match = re.fullmatch(
            rf"<{re.escape(self.prefix)}_([0-9]{{{self.digits}}})>", value
        )
        if match is None:
            raise ValueError(f"invalid coordinate token: {value!r}")
        index = int(match.group(1))
        if index >= self.bins:
            raise ValueError(f"coordinate token index {index} is outside [0, {self.bins - 1}]")
        return index

    def _transform_protocol(
        self,
        protocol: dict[str, Any],
        *,
        encode: bool,
        allow_numeric_coordinates: bool = False,
    ) -> None:
        canvas = protocol.get("canvas")
        if not isinstance(canvas, Mapping):
            raise ValueError("protocol canvas must be an object")
        width = canvas.get("width")
        height = canvas.get("height")
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise ValueError("protocol canvas width must be a positive integer")
        if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
            raise ValueError("protocol canvas height must be a positive integer")

        objects = protocol.get("objects")
        if not isinstance(objects, list):
            raise ValueError("protocol objects must be a list")
        for obj in objects:
            if not isinstance(obj, dict):
                raise ValueError("protocol object entries must be objects")
            geometry = obj.get("geometry")
            if not isinstance(geometry, dict):
                raise ValueError("protocol geometry must be an object")
            box = geometry.get("box")
            if not isinstance(box, dict):
                raise ValueError("protocol geometry box must be an object")
            self._transform_pair(
                box,
                "x",
                width,
                encode=encode,
                allow_numeric_coordinates=allow_numeric_coordinates,
            )
            self._transform_pair(
                box,
                "y",
                height,
                encode=encode,
                allow_numeric_coordinates=allow_numeric_coordinates,
            )
            self._transform_pair(
                box,
                "width",
                width,
                encode=encode,
                positive=True,
                allow_numeric_coordinates=allow_numeric_coordinates,
            )
            self._transform_pair(
                box,
                "height",
                height,
                encode=encode,
                positive=True,
                allow_numeric_coordinates=allow_numeric_coordinates,
            )

            baseline = geometry.get("baseline")
            if baseline is None:
                continue
            if not isinstance(baseline, dict):
                raise ValueError("protocol geometry baseline must be an object or null")
            for point_name in ("p0", "p1", "p2", "p3"):
                point = baseline.get(point_name)
                if not isinstance(point, dict):
                    raise ValueError(f"protocol baseline {point_name} must be an object")
                self._transform_pair(
                    point,
                    "x",
                    width,
                    encode=encode,
                    allow_numeric_coordinates=allow_numeric_coordinates,
                )
                self._transform_pair(
                    point,
                    "y",
                    height,
                    encode=encode,
                    allow_numeric_coordinates=allow_numeric_coordinates,
                )

    def _transform_pair(
        self,
        container: dict[str, Any],
        key: str,
        extent: int,
        *,
        encode: bool,
        positive: bool = False,
        allow_numeric_coordinates: bool = False,
    ) -> None:
        if key not in container:
            raise ValueError(f"missing geometry coordinate {key!r}")
        if encode:
            index = self.quantize(container[key], extent, positive=positive)
            container[key] = self.token(index)
        else:
            value = container[key]
            if allow_numeric_coordinates and isinstance(value, (int, float)) and not isinstance(
                value, bool
            ):
                return
            index = self._parse_token(value)
            container[key] = self.dequantize(index, extent)


def coordinate_codec_from_config(protocol_cfg: Any) -> CoordinateTokenCodec | None:
    """Construct the optional codec from a Hydra-like protocol configuration."""
    coordinate_cfg = getattr(protocol_cfg, "coordinate_encoding", None)
    if coordinate_cfg is None or not bool(getattr(coordinate_cfg, "enabled", False)):
        return None
    return CoordinateTokenCodec(
        bins=int(getattr(coordinate_cfg, "bins", DEFAULT_COORDINATE_BINS)),
        prefix=str(getattr(coordinate_cfg, "token_prefix", "coord")),
    )


def decode_coordinate_json_or_original(
    value: str,
    codec: CoordinateTokenCodec | None,
    *,
    decimal_places: int = DEFAULT_DECIMAL_PLACES,
) -> str:
    """Decode a generated target, leaving malformed outputs invalid downstream."""
    if codec is None:
        return value
    try:
        return codec.decode_json(value, decimal_places=decimal_places)
    except (TypeError, ValueError, json.JSONDecodeError):
        return value
