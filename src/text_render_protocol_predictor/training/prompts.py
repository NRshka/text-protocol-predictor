"""Versioned multimodal prompt construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProtocolPromptTemplate:
    version: str = "1.0.0"
    system: str = "You extract editable text rendering protocols from images."

    def user_text(self, width: int, height: int, protocol_version: str = "1.0") -> str:
        object_description = (
            "text objects" if protocol_version == "1.0" else "text and shape objects"
        )
        return (
            f"Extract all visible {object_description} from the image and return their editable "
            "rendering protocol.\n"
            "Return valid JSON only.\n"
            f"Use protocol version {protocol_version}.\n"
            f"Canvas size: {width} x {height}.\n"
            "Coordinates and font sizes must use original canvas pixels.\n"
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
