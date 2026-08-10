"""Versioned multimodal prompt construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..protocol.coordinate_tokens import CoordinateTokenCodec


@dataclass(frozen=True)
class ProtocolPromptTemplate:
    version: str = "1.2.0"
    system: str = "You extract editable text rendering protocols from images."
    coordinate_codec: CoordinateTokenCodec | None = None
    text_only: bool = False
    grounding_enabled: bool = False

    def user_text(
        self,
        width: int,
        height: int,
        protocol_version: str = "1.0",
        *,
        text_only: bool = False,
    ) -> str:
        effective_text_only = self.text_only or text_only
        object_description = (
            "text objects"
            if effective_text_only or protocol_version == "1.0"
            else "text and shape objects"
        )
        shape_instruction = (
            'Do not include shape objects; every object must have object_type "text".\n'
            if effective_text_only and protocol_version != "1.0"
            else ""
        )
        if self.coordinate_codec is None:
            coordinate_instruction = (
                "Coordinates and font sizes must use original canvas pixels.\n"
            )
        else:
            first_token = self.coordinate_codec.json_token(0)
            last_token = self.coordinate_codec.json_token(
                self.coordinate_codec.bins - 1
            )
            coordinate_instruction = (
                f"Use normalized coordinate tokens {first_token} through {last_token} for "
                "every geometry box x, y, width, and height, and for every Bezier baseline "
                "point x and y. Normalize horizontal values by canvas width and vertical "
                "values by canvas height. Font sizes and all other numeric fields must use "
                "their original units.\n"
            )
        grounding_instruction = (
            'For every text object, emit exactly "mask_ref":"<MASK>" immediately '
            'after "direction" and before "geometry". Do not emit mask_ref for shapes.\n'
            if self.grounding_enabled
            else ""
        )
        return (
            f"Extract all visible {object_description} from the image and return their editable "
            "rendering protocol.\n"
            "Return valid JSON only.\n"
            f"Use protocol version {protocol_version}.\n"
            f"{shape_instruction}"
            f"Canvas size: {width} x {height}.\n"
            f"{coordinate_instruction}"
            f"{grounding_instruction}"
            "Do not include explanations or Markdown."
        )

    def conversation(
        self,
        *,
        image: str | Path | Any,
        width: int,
        height: int,
        protocol_version: str = "1.0",
        target: str | None = None,
    ) -> list[dict[str, Any]]:
        # Transformers image processors accept local paths as strings, but do
        # not accept pathlib.Path objects. Keep this conversion at the prompt
        # boundary so dataset records can retain strongly typed paths.
        processor_image = str(image) if isinstance(image, Path) else image
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": processor_image},
                    {
                        "type": "text",
                        "text": self.user_text(width, height, protocol_version),
                    },
                ],
            },
        ]
        if target is not None:
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": target}]}
            )
        return messages
