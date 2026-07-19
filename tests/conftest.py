from __future__ import annotations

import copy

import pytest


@pytest.fixture
def protocol_dict() -> dict:
    return {
        "protocol_version": "1.0",
        "sample_id": "sample-1",
        "seed": 17,
        "canvas": {"width": 1280, "height": 720},
        "background": {"source": "background.png", "sha256": "a" * 64},
        "objects": [
            {
                "id": "source-b",
                "text": "Cafe\u0301",
                "language": "fr",
                "direction": "ltr",
                "geometry": {
                    "mode": "straight",
                    "box": {"x": -0.0, "y": 20.1236, "width": 300.0, "height": 80.0},
                    "rotation_degrees": 0.0,
                    "baseline": None,
                },
                "style": {
                    "font_id": "Inter",
                    "font_size": 54.0,
                    "fill": {"type": "solid", "color": "#FFFFFFFF"},
                    "stroke": {"width": 0.0, "color": "#000000FF"},
                    "shadow": None,
                    "character_spacing": 0.0,
                    "line_height": 1.2,
                    "bold": False,
                    "italic": False,
                    "underline": False,
                    "alignment": "center",
                },
                "z_order": 1,
                "tight_bbox": None,
            },
            {
                "id": "source-a",
                "text": "First",
                "language": "en",
                "direction": "ltr",
                "geometry": {
                    "mode": "straight",
                    "box": {"x": 10, "y": 10, "width": 200, "height": 40},
                    "rotation_degrees": 0,
                    "baseline": None,
                },
                "style": {
                    "font_id": "Inter",
                    "font_size": 30,
                    "fill": {"type": "solid", "color": "#FFFFFFFF"},
                    "stroke": {"width": 0, "color": "#000000FF"},
                    "shadow": None,
                    "character_spacing": 0,
                    "line_height": 1,
                    "bold": False,
                    "italic": False,
                    "underline": False,
                    "alignment": "left",
                },
                "z_order": 0,
                "tight_bbox": None,
            },
        ],
    }


@pytest.fixture
def copied_protocol(protocol_dict: dict) -> dict:
    return copy.deepcopy(protocol_dict)


@pytest.fixture
def protocol_21_dict(protocol_dict: dict) -> dict:
    text = copy.deepcopy(protocol_dict["objects"][0])
    text["object_type"] = "text"
    text["annotation"] = {
        "text_confidence": 0.9,
        "geometry_confidence": 0.8,
        "style_confidence": 0.7,
        "font_match": "nearest",
        "notes": None,
    }
    return {
        "protocol_version": "2.1",
        "purpose": "annotation",
        "sample_id": "sample-21",
        "seed": 0,
        "canvas": copy.deepcopy(protocol_dict["canvas"]),
        "background": copy.deepcopy(protocol_dict["background"]),
        "objects": [
            text,
            {
                "object_type": "shape",
                "id": "panel",
                "shape": "rectangle",
                "geometry": {
                    "box": {"x": 5, "y": 6, "width": 400, "height": 100},
                    "rotation_degrees": 0,
                    "corner_radius": 12,
                },
                "style": {
                    "fill": {"type": "solid", "color": "#102030FF"},
                    "stroke": {"width": 1, "color": "#FFFFFFFF"},
                    "shadow": None,
                },
                "z_order": -1,
                "tight_bbox": None,
                "annotation": {"geometry_confidence": 0.95},
            },
        ],
    }
