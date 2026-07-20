from pathlib import Path

import pytest

from tools.download_cyrillic_fonts import _existing_ids, _representative_variant


def test_representative_variant_prefers_regular_400_and_extended_subset():
    detail = {
        "variants": {
            "700": {"normal": {"cyrillic": {"url": {"ttf": "bold.ttf"}}}},
            "400": {
                "italic": {"cyrillic": {"url": {"ttf": "italic.ttf"}}},
                "normal": {
                    "cyrillic": {"url": {"ttf": "base.ttf"}},
                    "cyrillic-ext": {"url": {"ttf": "extended.ttf"}},
                },
            },
        }
    }

    assert _representative_variant(detail) == (400, "normal", "cyrillic-ext", "extended.ttf")


def test_representative_variant_falls_back_to_nearest_weight():
    detail = {
        "variants": {
            "300": {"italic": {"cyrillic": {"url": {"ttf": "font.ttf"}}}},
            "700": {"normal": {"latin": {"url": {"ttf": "latin.ttf"}}}},
        }
    }

    assert _representative_variant(detail) == (300, "italic", "cyrillic", "font.ttf")


def test_representative_variant_rejects_missing_cyrillic_ttf():
    detail = {"variants": {"400": {"normal": {"latin": {"url": {"ttf": "font.ttf"}}}}}}

    with pytest.raises(ValueError, match="no Cyrillic TTF"):
        _representative_variant(detail)


def test_existing_ids_reads_jsonl(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"id":"one"}\n{"id":"two"}\n', encoding="utf-8")

    assert _existing_ids(manifest) == {"one", "two"}
