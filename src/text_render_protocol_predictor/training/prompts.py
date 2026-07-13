"""Versioned multimodal prompt construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProtocolPromptTemplate:
    version: str = "1.0.0"
    system: str = "You extract editable text rendering protocols from images."

    def user_text(self, width: int, height: int) -> str:
        return (
            "Extract all visible text objects from the image and return their editable "
            "rendering protocol.\n"
            "Return valid JSON only.\n"
            "Use protocol version 1.0.\n"
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
        target: str | None = None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.user_text(width, height)},
                ],
            },
        ]
        if target is not None:
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": target}]}
            )
        return messages

