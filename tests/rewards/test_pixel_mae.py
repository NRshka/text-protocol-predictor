from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from text_render_protocol_predictor.rendering import RenderOutcome, RenderStatus
from text_render_protocol_predictor.rewards import (
    PixelMAEReward,
    PixelMAERewardConfig,
    masked_rgb_mae,
    prepare_text_mask,
)


class FakeRenderer:
    def __init__(self, images):
        self.images = images
        self.calls = 0

    def render_prediction(self, completion, background, *, sample_id):
        self.calls += 1
        value = self.images.get(completion)
        if isinstance(value, RenderStatus):
            return RenderOutcome(value, error="expected test failure")
        if isinstance(value, tuple):
            image, prediction = value
            return RenderOutcome(
                RenderStatus.OK,
                image=image.copy(),
                predicted_texts=tuple(obj.text for obj in prediction.objects),
                prediction=prediction,
            )
        return RenderOutcome(RenderStatus.OK, image=value.copy())


def _assets(tmp_path):
    original = Image.new("RGB", (5, 5), "black")
    original.putpixel((2, 2), (255, 255, 255))
    background = Image.new("RGB", (5, 5), "black")
    mask = Image.new("L", (5, 5), 0)
    mask.putpixel((2, 2), 255)
    paths = []
    for name, image in (("original", original), ("background", background), ("mask", mask)):
        path = tmp_path / f"{name}.webp"
        image.save(path, format="WEBP", lossless=True)
        paths.append(path)
    return original, background, paths


def _reward(renderer):
    return PixelMAEReward(
        renderer,
        PixelMAERewardConfig(
            mask_dilation_radius=0,
            mask_blur_radius=0,
            outside_weight=0.1,
        ),
    )


def test_masked_mae_and_mask_preparation():
    source = Image.new("L", (5, 5), 0)
    source.putpixel((2, 2), 255)
    mask = prepare_text_mask(source, threshold=0.5, dilation_radius=1, blur_radius=0)

    assert mask.sum() == pytest.approx(9.0)
    black = np.zeros((5, 5, 3), dtype=np.float32)
    white = np.ones((5, 5, 3), dtype=np.float32)
    assert masked_rgb_mae(black, white, mask) == pytest.approx(1.0)


def test_reward_prefers_exact_reconstruction_and_penalizes_outside_changes(tmp_path):
    original, background, paths = _assets(tmp_path)
    all_white = Image.new("RGB", (5, 5), "white")
    reward = _reward(FakeRenderer({"exact": original, "empty": background, "outside": all_white}))
    common = dict(
        sample_id="sample",
        original_path=paths[0],
        background_path=paths[1],
        text_mask_path=paths[2],
    )

    exact = reward.score("exact", **common)
    empty = reward.score("empty", **common)
    outside = reward.score("outside", **common)

    assert exact.reward == pytest.approx(1.0)
    assert exact.rendered_image is not None
    assert exact.rendered_image.getpixel((2, 2)) == (255, 255, 255)
    assert empty.reward == pytest.approx(0.0)
    assert exact.restoration_delta == pytest.approx(1.0)
    assert outside.reward == pytest.approx(0.9)
    assert outside.outside_mae == pytest.approx(1.0)


def test_invalid_completion_gets_floor_and_duplicates_render_once(tmp_path):
    original, _, paths = _assets(tmp_path)
    renderer = FakeRenderer(
        {"bad": RenderStatus.INVALID_SCHEMA, "same": original}
    )
    reward = _reward(renderer)
    kwargs = {
        "sample_id": ["sample", "sample"],
        "original_path": [str(paths[0])] * 2,
        "background_path": [str(paths[1])] * 2,
        "text_mask_path": [str(paths[2])] * 2,
    }

    assert reward(["bad", "bad"], **kwargs) == [-0.9, -0.9]
    assert renderer.calls == 1
    assert reward([[{"role": "assistant", "content": "same"}]] * 2, **kwargs) == [1.0, 1.0]
    assert renderer.calls == 2


def test_word_reward_prefers_correct_content_and_rejects_empty_protocol(tmp_path):
    original, background, paths = _assets(tmp_path)
    exact_completion = json.dumps(
        {"objects": [{"text": "SUMMER SALE"}]}, separators=(",", ":")
    )
    empty_completion = json.dumps({"objects": []}, separators=(",", ":"))
    reward = _reward(
        FakeRenderer(
            {
                exact_completion: original,
                empty_completion: background,
            }
        )
    )
    common = dict(
        sample_id="sample",
        original_path=paths[0],
        background_path=paths[1],
        text_mask_path=paths[2],
        reference_words=[
            {"text": "summer", "confidence": 0.95},
            {"text": "sale", "confidence": 0.9},
        ],
    )

    exact = reward.score(exact_completion, **common)
    empty = reward.score(empty_completion, **common)

    assert exact.reward == pytest.approx(1.4)
    assert exact.word_score == pytest.approx(1.0)
    assert exact.matched_word_count == 2
    assert empty.status is RenderStatus.INVALID_SEMANTICS
    assert empty.reward == pytest.approx(-0.85)
    assert empty.predicted_word_count == 0
    assert empty.empty_word_prediction is True


def test_no_ocr_words_preserves_pixel_only_reward(tmp_path):
    original, _, paths = _assets(tmp_path)
    completion = json.dumps({"objects": []})
    reward = _reward(FakeRenderer({completion: original}))

    result = reward.score(
        completion,
        sample_id="sample",
        original_path=paths[0],
        background_path=paths[1],
        text_mask_path=paths[2],
    )

    assert result.status is RenderStatus.OK
    assert result.reward == pytest.approx(1.0)
    assert result.word_score is None


def test_layout_iou_is_added_to_valid_pixel_reward(tmp_path):
    original, _, paths = _assets(tmp_path)
    geometry = SimpleNamespace(
        mode="straight",
        box=SimpleNamespace(x=2, y=2, width=1, height=1),
        rotation_degrees=0,
        baseline=None,
    )
    prediction = SimpleNamespace(
        objects=[SimpleNamespace(text="text", geometry=geometry)]
    )
    reward = PixelMAEReward(
        FakeRenderer({"layout": (original, prediction)}),
        PixelMAERewardConfig(
            mask_dilation_radius=0,
            mask_blur_radius=0,
            layout_dilation_kernel_size=1,
            layout_dilation_iterations=0,
        ),
    )

    result = reward.score(
        "layout",
        sample_id="sample",
        original_path=paths[0],
        background_path=paths[1],
        text_mask_path=paths[2],
    )

    assert result.layout_strict_iou == pytest.approx(1.0)
    assert result.layout_iou == pytest.approx(1.0)
    assert result.layout_precision == pytest.approx(1.0)
    assert result.layout_recall == pytest.approx(1.0)
    assert result.reward == pytest.approx(1.2)
