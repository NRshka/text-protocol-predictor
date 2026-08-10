#!/usr/bin/env python3
"""Download one open-license Cyrillic TTF per Fontsource family.

The script deliberately uses Fontsource's documented API/CDN instead of scraping
font catalogue websites. Downloads are resumable and accompanied by licenses and
a JSONL manifest containing source URLs and SHA-256 digests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API_URL = "https://api.fontsource.org/v1/fonts"
USER_AGENT = "text-render-protocol-predictor-font-fetcher/1.0"
RUSSIAN_ALPHABET_TEXT = (
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
)
RUSSIAN_ALPHABET = frozenset(ord(char) for char in RUSSIAN_ALPHABET_TEXT)


def _get_bytes(url: str, *, timeout: float, attempts: int = 4) -> bytes:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed HTTPS API
                return response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def _get_json(url: str, *, timeout: float) -> Any:
    return json.loads(_get_bytes(url, timeout=timeout))


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".part",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    os.replace(temporary, path)


def _representative_variant(detail: dict[str, Any]) -> tuple[int, str, str, str]:
    variants = detail["variants"]
    weights = sorted(
        (int(weight) for weight in variants),
        key=lambda value: (abs(value - 400), value),
    )
    for weight in weights:
        styles = variants[str(weight)]
        for style in ("normal", "italic", *sorted(styles)):
            if style not in styles:
                continue
            subsets = styles[style]
            # Fontsource's web subsets are disjoint files: ``cyrillic-ext``
            # contains additional characters, not the base Russian alphabet.
            for subset in ("cyrillic", "cyrillic-ext"):
                url = subsets.get(subset, {}).get("url", {}).get("ttf")
                if url:
                    return weight, style, subset, url
    raise ValueError("no Cyrillic TTF variant found")


def _verify_russian_cmap(font_path: Path) -> None:
    try:
        from fontTools.ttLib import TTFont
    except ImportError as exc:
        raise RuntimeError("font verification requires `pip install fonttools`") from exc

    with TTFont(font_path, lazy=True) as font:
        codepoints: set[int] = set()
        for table in font["cmap"].tables:
            if table.isUnicode():
                codepoints.update(table.cmap)
    missing = RUSSIAN_ALPHABET - codepoints
    if missing:
        sample = "".join(chr(value) for value in sorted(missing)[:12])
        raise ValueError(f"missing {len(missing)} Russian letters (sample: {sample})")


def _download_family(font: dict[str, Any], output: Path, timeout: float) -> dict[str, Any]:
    font_id = font["id"]
    detail = _get_json(f"{API_URL}/{font_id}", timeout=timeout)
    weight, style, subset, font_url = _representative_variant(detail)
    version = _get_json(
        f"https://api.fontsource.org/v1/version/{font_id}", timeout=timeout
    )["latest"]
    font_url = font_url.replace("@latest/", f"@{version}/", 1)
    license_url = (
        f"https://cdn.jsdelivr.net/npm/@fontsource/{font_id}@{version}/LICENSE"
    )
    font_path = output / "fonts" / f"{font_id}.ttf"
    license_path = output / "licenses" / f"{font_id}.txt"

    if not font_path.exists():
        _write_atomic(font_path, _get_bytes(font_url, timeout=timeout))
    try:
        _verify_russian_cmap(font_path)
    except Exception:
        font_path.unlink(missing_ok=True)
        raise
    if not license_path.exists():
        _write_atomic(license_path, _get_bytes(license_url, timeout=timeout))

    return {
        "id": font_id,
        "family": detail["family"],
        "category": detail.get("category"),
        "license": detail.get("license"),
        "source": detail.get("source"),
        "fontsource_version": version,
        "weight": weight,
        "style": style,
        "subset": subset,
        "font_file": str(font_path.relative_to(output)),
        "license_file": str(license_path.relative_to(output)),
        "font_url": font_url,
        "license_url": license_url,
        "sha256": hashlib.sha256(font_path.read_bytes()).hexdigest(),
    }


def _existing_ids(manifest: Path) -> set[str]:
    if not manifest.exists():
        return set()
    ids: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.add(json.loads(line)["id"])
    return ids


def _compact_manifest(manifest: Path) -> None:
    if not manifest.exists():
        return
    records: dict[str, dict[str, Any]] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["id"]] = record
    content = "".join(
        json.dumps(records[font_id], ensure_ascii=False, sort_keys=True) + "\n"
        for font_id in sorted(records)
    )
    _write_atomic(manifest, content.encode())


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("fonts/cyrillic"))
    parser.add_argument("--limit", type=int, default=1000, help="maximum number of families")
    parser.add_argument("--workers", type=int, default=4, help="parallel downloads (keep modest)")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--dry-run", action="store_true", help="only report the available count")
    args = parser.parse_args()
    if args.limit < 1 or args.workers < 1 or args.timeout <= 0:
        parser.error("limit, workers, and timeout must be positive")
    return args


def main() -> int:
    args = _arguments()
    catalogue = _get_json(API_URL, timeout=args.timeout)
    fonts = sorted(
        (font for font in catalogue if "cyrillic" in font["subsets"]),
        key=lambda font: (font["family"].casefold(), font["id"]),
    )
    target = min(args.limit, len(fonts))
    print(f"Fontsource offers {len(fonts)} Cyrillic families; target for this run: {target}.")
    if args.limit > len(fonts):
        shortfall = args.limit - len(fonts)
        print(f"Requested {args.limit}, but the open catalogue is short by {shortfall}.")
    if args.dry_run:
        return 0

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.jsonl"
    _compact_manifest(manifest)
    completed = _existing_ids(manifest)
    selected = [font for font in fonts[:target] if font["id"] not in completed]
    failures: list[tuple[str, str]] = []
    with manifest.open("a", encoding="utf-8") as manifest_file:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_download_family, font, output, args.timeout): font
                for font in selected
            }
            for index, future in enumerate(as_completed(futures), start=1):
                font = futures[future]
                try:
                    record = future.result()
                except Exception as exc:  # continue corpus work after an isolated bad font
                    failures.append((font["id"], f"{type(exc).__name__}: {exc}"))
                    progress = f"[{index}/{len(selected)}]"
                    print(f"{progress} FAILED {font['family']}: {exc}", file=sys.stderr)
                    continue
                manifest_file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                manifest_file.flush()
                print(f"[{index}/{len(selected)}] {record['family']}")

    _compact_manifest(manifest)
    summary = {
        "requested": args.limit,
        "available": len(fonts),
        "selected": target,
        "completed": len(_existing_ids(manifest)),
        "failures": [{"id": font_id, "error": error} for font_id, error in failures],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Completed {summary['completed']} families in {output}; failures: {len(failures)}.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
