"""Box-free, confidence-aware word matching for protocol completions."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


_WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", flags=re.UNICODE)


@dataclass(frozen=True)
class WordMatchMetrics:
    """One-to-one multiset matching diagnostics."""

    precision: float | None
    recall: float | None
    score: float | None
    reference_count: int
    predicted_count: int
    matched_count: int

    @property
    def active(self) -> bool:
        return self.reference_count > 0


def tokenize_words(text: str) -> list[str]:
    """Return normalized Unicode lexical words without imposing OCR grouping."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return _WORD_PATTERN.findall(normalized)


def _reference_tokens(
    words: Sequence[Any] | None,
    *,
    minimum_confidence: float,
) -> list[tuple[str, float]]:
    tokens: list[tuple[str, float]] = []
    for word in words or ():
        if isinstance(word, str):
            text = word
            confidence = 1.0
        elif isinstance(word, Mapping):
            text = str(word.get("text", ""))
            confidence = float(word.get("confidence", 1.0))
        else:
            text = str(getattr(word, "text", ""))
            confidence = float(getattr(word, "confidence", 1.0))
        if confidence < minimum_confidence:
            continue
        tokens.extend((token, confidence) for token in tokenize_words(text))
    return tokens


def _normalized_edit_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1]
                    + (left_character != right_character),
                )
            )
        previous = current
    return 1.0 - previous[-1] / max(len(left), len(right))


def match_word_content(
    predicted_texts: Sequence[str],
    reference_words: Sequence[Any] | None,
    *,
    recall_weight: float = 0.7,
    fuzzy_threshold: float = 0.8,
    minimum_confidence: float = 0.5,
) -> WordMatchMetrics:
    """Match flattened predicted and OCR words one-to-one, ignoring regions and boxes."""
    if not 0.0 <= recall_weight <= 1.0:
        raise ValueError("recall_weight must be between 0 and 1")
    if not 0.0 <= fuzzy_threshold <= 1.0:
        raise ValueError("fuzzy_threshold must be between 0 and 1")
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be between 0 and 1")

    references = _reference_tokens(
        reference_words,
        minimum_confidence=minimum_confidence,
    )
    predictions = [token for text in predicted_texts for token in tokenize_words(text)]
    if not references:
        return WordMatchMetrics(
            precision=None,
            recall=None,
            score=None,
            reference_count=0,
            predicted_count=len(predictions),
            matched_count=0,
        )
    if not predictions:
        return WordMatchMetrics(
            precision=0.0,
            recall=0.0,
            score=0.0,
            reference_count=len(references),
            predicted_count=0,
            matched_count=0,
        )

    unmatched_references = set(range(len(references)))
    unmatched_predictions = set(range(len(predictions)))
    matches: list[tuple[int, int, float]] = []

    # Resolve exact duplicates against the highest-confidence references first.
    exact_by_token: dict[str, list[int]] = {}
    for reference_index, (token, confidence) in sorted(
        enumerate(references), key=lambda item: (-item[1][1], item[0])
    ):
        exact_by_token.setdefault(token, []).append(reference_index)
    for prediction_index, token in enumerate(predictions):
        candidates = exact_by_token.get(token)
        if candidates:
            reference_index = candidates.pop(0)
            unmatched_references.remove(reference_index)
            unmatched_predictions.remove(prediction_index)
            matches.append((reference_index, prediction_index, 1.0))

    # Greedy maximum-similarity matching is deterministic and avoids a heavy
    # assignment dependency for the small word sets in marketing images.
    fuzzy_candidates: list[tuple[float, float, int, int]] = []
    for reference_index in unmatched_references:
        reference, confidence = references[reference_index]
        for prediction_index in unmatched_predictions:
            similarity = _normalized_edit_similarity(
                reference, predictions[prediction_index]
            )
            if similarity >= fuzzy_threshold:
                fuzzy_candidates.append(
                    (similarity, confidence, reference_index, prediction_index)
                )
    fuzzy_candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    for similarity, _, reference_index, prediction_index in fuzzy_candidates:
        if (
            reference_index not in unmatched_references
            or prediction_index not in unmatched_predictions
        ):
            continue
        unmatched_references.remove(reference_index)
        unmatched_predictions.remove(prediction_index)
        matches.append((reference_index, prediction_index, similarity))

    precision_credit = sum(similarity for _, _, similarity in matches)
    recall_credit = sum(
        references[reference_index][1] * similarity
        for reference_index, _, similarity in matches
    )
    precision = precision_credit / len(predictions)
    recall = recall_credit / sum(confidence for _, confidence in references)
    score = recall_weight * recall + (1.0 - recall_weight) * precision
    return WordMatchMetrics(
        precision=float(precision),
        recall=float(recall),
        score=float(score),
        reference_count=len(references),
        predicted_count=len(predictions),
        matched_count=len(matches),
    )
