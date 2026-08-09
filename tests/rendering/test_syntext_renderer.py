from __future__ import annotations

import json

from PIL import Image

from text_render_protocol_predictor.rendering import RenderStatus, SyntextPredictionRenderer


def _prediction(font_id="Inter", protocol_version="1.0"):
    prediction = {
        "protocol_version": protocol_version,
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
    if protocol_version == "2.1":
        prediction["objects"][0]["object_type"] = "text"
    return prediction


class FakeDocumentProtocol:
    envelope = None

    @classmethod
    def model_validate(cls, envelope):
        cls.envelope = envelope
        return envelope


class FakeProtocolRenderer:
    def render(self, background, protocol):
        return background.copy(), protocol


def _renderer_without_syntext(protocol_version="1.0"):
    renderer = object.__new__(SyntextPredictionRenderer)
    renderer.protocol_version = protocol_version
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


def test_renders_text_only_protocol_21():
    renderer = _renderer_without_syntext("2.1")

    outcome = renderer.render_prediction(
        json.dumps(_prediction(protocol_version="2.1")),
        Image.new("RGB", (32, 24)),
        sample_id="real-21",
    )

    assert outcome.status is RenderStatus.OK
    assert FakeDocumentProtocol.envelope["protocol_version"] == "2.1"
    assert FakeDocumentProtocol.envelope["objects"][0]["object_type"] == "text"


def test_rejects_protocol_21_shape_objects():
    renderer = _renderer_without_syntext("2.1")
    prediction = _prediction(protocol_version="2.1")
    prediction["objects"].append(
        {
            "object_type": "shape",
            "id": "panel",
            "shape": "rectangle",
            "geometry": {
                "box": {"x": 1, "y": 1, "width": 30, "height": 10},
                "rotation_degrees": 0,
                "corner_radius": 2,
            },
            "style": {
                "fill": {"type": "solid", "color": "#102030FF"},
                "stroke": {"width": 0, "color": "#00000000"},
                "shadow": None,
            },
            "z_order": -1,
        }
    )

    outcome = renderer.render_prediction(
        json.dumps(prediction),
        Image.new("RGB", (32, 24)),
        sample_id="shape",
    )

    assert outcome.status is RenderStatus.INVALID_SEMANTICS
    assert outcome.error is not None
    assert "shape objects are not allowed" in outcome.error
    assert outcome.predicted_texts == ("Sale",)
