from __future__ import annotations

import pytest

from text_render_protocol_predictor.rewards import match_word_content, tokenize_words


def test_tokenization_normalizes_unicode_case_and_ignores_object_grouping():
    assert tokenize_words("  СКИДКА—Café's 50%! ") == ["скидка", "café's", "50"]

    metrics = match_word_content(
        ["Summer", "SALE today"],
        [
            {"text": "summer", "confidence": 1.0},
            {"text": "sale", "confidence": 1.0},
        ],
    )

    assert metrics.recall == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.score == pytest.approx(0.7 + 0.3 * 2 / 3)


def test_matching_is_one_to_one_for_duplicate_words():
    metrics = match_word_content(
        ["sale sale sale"],
        [{"text": "sale", "confidence": 1.0}],
    )

    assert metrics.matched_count == 1
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(1 / 3)


def test_fuzzy_matching_and_confidence_filtering():
    metrics = match_word_content(
        ["Sumer"],
        [
            {"text": "Summer", "confidence": 0.9},
            {"text": "noise", "confidence": 0.2},
        ],
        fuzzy_threshold=0.8,
        minimum_confidence=0.5,
    )

    similarity = 5 / 6
    assert metrics.reference_count == 1
    assert metrics.matched_count == 1
    assert metrics.precision == pytest.approx(similarity)
    assert metrics.recall == pytest.approx(similarity)
    assert metrics.score == pytest.approx(similarity)


def test_no_reference_disables_content_reward():
    metrics = match_word_content(["anything"], [])

    assert not metrics.active
    assert metrics.score is None
    assert metrics.predicted_count == 1
