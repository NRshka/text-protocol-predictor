#!/usr/bin/env python3
"""Validate mask-grounded SFT manifests with the production dataset loader."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from text_render_protocol_predictor.data import (  # noqa: E402
    ProtocolManifestDataset,
    validate_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("train.jsonl"))
    parser.add_argument("--max-objects", type=int, default=64)
    parser.add_argument(
        "--no-verify-image-dimensions",
        action="store_false",
        dest="verify_image_dimensions",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = ProtocolManifestDataset(
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        max_objects=args.max_objects,
        verify_image_dimensions=args.verify_image_dimensions,
        text_only_targets=True,
        grounding_enabled=True,
    )
    report = validate_dataset(dataset)
    if report.failures:
        report.raise_for_errors()

    sources: Counter[str] = Counter()
    text_objects = 0
    supervised_objects = 0
    for index in range(len(dataset)):
        record = dataset[index]
        text_objects += len(record.text_object_ids)
        supervision = record.mask_supervision
        source = supervision.source if supervision is not None else "none"
        sources[source] += 1
        if supervision is not None:
            supervised_objects += (
                len(record.text_object_ids)
                if supervision.source == "geometry"
                else len(supervision.object_paths)
            )
    print(
        json.dumps(
            {
                "status": "ok",
                "samples": len(dataset),
                "text_objects": text_objects,
                "supervised_objects": supervised_objects,
                "samples_by_mask_source": dict(sorted(sources.items())),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
