from __future__ import annotations

import json

from PIL import Image

from text_render_protocol_predictor.rendering import RenderStatus, SyntextPredictionRenderer


def _prediction(font_id="Inter"):
    return {
        "protocol_version": "1.0",
        "canvas": {"width": 32, "height": 24},
        "objects": [
            {
                "id": "headline",
                "text": "Sale",
                "language": "en",
                "direction": "ltr",
                "geometry": {
                    "mode": "straight",
                    "box": {"x": 2, "y": 3, "width": 20, "height": 8},
                    "rotation_degrees": 0,
                    "baseline": None,
                },
                "style": {
                    "font_id": font_id,
                    "font_size": 8,
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
            }
        ],
    }


class FakeDocumentProtocol:
    envelope = None

    @classmethod
    def model_validate(cls, envelope):
        cls.envelope = envelope
        return envelope


class FakeProtocolRenderer:
    def render(self, background, protocol):
        return background.copy(), protocol


def _renderer_without_syntext():
    renderer = object.__new__(SyntextPredictionRenderer)
    renderer.protocol_version = "1.0"
    renderer.max_objects = 64
    renderer.max_text_characters = 4096
    renderer.max_font_size = 2048
    renderer.max_geometry_scale = 2.0
    renderer.font_ids = frozenset({"Inter"})
    renderer._document_protocol_class = FakeDocumentProtocol
    renderer._renderer = FakeProtocolRenderer()
    return renderer


def test_builds_render_envelope_from_strict_projection():
    renderer = _renderer_without_syntext()

    outcome = renderer.render_prediction(
        json.dumps(_prediction()),
        Image.new("RGB", (32, 24)),
        sample_id="real-1",
    )

    assert outcome.status is RenderStatus.OK
    assert FakeDocumentProtocol.envelope["sample_id"] == "real-1"
    assert FakeDocumentProtocol.envelope["purpose"] == "render"
    assert FakeDocumentProtocol.envelope["objects"][0]["text"] == "Sale"
    assert outcome.prediction is not None


def test_rejects_json_schema_canvas_and_unknown_font_separately():
    renderer = _renderer_without_syntext()
    background = Image.new("RGB", (32, 24))

    assert renderer.render_prediction("nope", background, sample_id="x").status is RenderStatus.INVALID_JSON
    assert renderer.render_prediction("{}", background, sample_id="x").status is RenderStatus.INVALID_SCHEMA
    wrong_canvas = _prediction()
    wrong_canvas["canvas"]["width"] = 33
    assert (
        renderer.render_prediction(json.dumps(wrong_canvas), background, sample_id="x").status
        is RenderStatus.INVALID_SEMANTICS
    )
    assert (
        renderer.render_prediction(json.dumps(_prediction("missing")), background, sample_id="x").status
        is RenderStatus.UNKNOWN_FONT
    )
