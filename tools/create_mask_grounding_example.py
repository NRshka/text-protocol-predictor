#!/usr/bin/env python3
"""Create a tiny, self-contained mask-grounded dataset example."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


def _text_object(
    object_id: str,
    text: str,
    box: dict[str, float],
    *,
    rotation: float,
    z_order: int,
) -> dict:
    return {
        "object_type": "text",
        "id": object_id,
        "text": text,
        "language": "en",
        "direction": "ltr",
        "geometry": {
            "mode": "straight",
            "box": box,
            "rotation_degrees": rotation,
            "baseline": None,
        },
        "style": {
            "font_id": "ExampleFont",
            "font_size": 28,
            "fill": {"type": "solid", "color": "#FFFFFFFF"},
            "stroke": {"width": 0, "color": "#000000FF"},
            "shadow": None,
            "character_spacing": 0,
            "line_height": 1,
            "bold": False,
            "italic": False,
            "underline": False,
            "alignment": "center",
        },
        "z_order": z_order,
        "tight_bbox": None,
    }


def _slot_mask(size: tuple[int, int], box: dict[str, float], rotation: float) -> Image.Image:
    local = Image.new("L", (round(box["width"]), round(box["height"])), 255)
    if rotation:
        local = local.rotate(-rotation, resample=Image.Resampling.NEAREST, expand=True)
    mask = Image.new("L", size, 0)
    center = (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    mask.paste(
        255,
        (round(center[0] - local.width / 2), round(center[1] - local.height / 2)),
        local,
    )
    return mask


def create_example(output: Path) -> Path:
    size = (320, 180)
    image_dir = output / "images"
    protocol_dir = output / "protocols"
    mask_dir = output / "masks" / "example-0001"
    for directory in (image_dir, protocol_dir, mask_dir):
        directory.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGB", size, "#26384A")
    draw = ImageDraw.Draw(image)
    draw.rectangle((42, 47, 232, 99), fill="#5C7188")
    draw.text((65, 58), "OVERLAPPING SLOTS", fill="white")
    image_path = image_dir / "example-0001.png"
    image.save(image_path)

    title_box = {"x": 42.0, "y": 47.0, "width": 190.0, "height": 52.0}
    badge_box = {"x": 180.0, "y": 72.0, "width": 105.0, "height": 42.0}
    title_mask = _slot_mask(size, title_box, 0)
    badge_mask = _slot_mask(size, badge_box, -8)
    title_mask.save(mask_dir / "title.png")
    badge_mask.save(mask_dir / "badge.png")

    protocol = {
        "protocol_version": "2.1",
        "purpose": "render",
        "sample_id": "example-0001",
        "seed": 2026,
        "canvas": {"width": size[0], "height": size[1]},
        "background": {
            "source": "generated",
            "sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        },
        "objects": [
            _text_object("title", "OVERLAPPING SLOTS", title_box, rotation=0, z_order=0),
            _text_object("badge", "NEW", badge_box, rotation=-8, z_order=1),
        ],
    }
    (protocol_dir / "example-0001.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "sample_id": "example-0001",
        "image": "images/example-0001.png",
        "protocol": "protocols/example-0001.json",
        "seed": 2026,
        "mask_supervision": {
            "source": "files",
            "objects": {
                "title": "masks/example-0001/title.png",
                "badge": "masks/example-0001/badge.png",
            },
        },
    }
    manifest_path = output / "manifest.jsonl"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(create_example(args.output))


if __name__ == "__main__":
    main()
