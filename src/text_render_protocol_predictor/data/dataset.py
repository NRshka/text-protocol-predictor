"""Lazy records for image/protocol SFT datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Collection

from PIL import Image

from .manifest import ManifestEntry, load_manifest, resolve_dataset_path
from ..protocol.canonicalizer import canonicalize
from ..protocol.schema import DatasetProtocol
from ..protocol.validator import validate_dataset_protocol


@dataclass(frozen=True)
class ProtocolDatasetRecord:
    sample_id: str
    image_path: Path
    protocol_path: Path
    canvas_width: int
    canvas_height: int
    protocol: DatasetProtocol
    canonical_protocol: str
    seed: int


class ProtocolManifestDataset:
    """Load one configured split whose entry paths are relative to dataset_root."""

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        manifest_path: str | Path,
        decimal_places: int = 3,
        font_ids: Collection[str] | None = None,
        require_files: bool = True,
        verify_image_dimensions: bool = True,
        max_objects: int | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        manifest = Path(manifest_path)
        if not manifest.is_absolute():
            manifest = resolve_dataset_path(self.dataset_root, str(manifest))
        self.manifest_path = manifest
        self.entries = load_manifest(manifest)
        self.decimal_places = decimal_places
        self.font_ids = font_ids
        self.require_files = require_files
        self.verify_image_dimensions = verify_image_dimensions
        self.max_objects = max_objects

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> ProtocolDatasetRecord:
        entry: ManifestEntry = self.entries[index]
        image_path = resolve_dataset_path(self.dataset_root, entry.image)
        protocol_path = resolve_dataset_path(self.dataset_root, entry.protocol)
        if self.require_files and not image_path.is_file():
            raise FileNotFoundError(f"image does not exist: {image_path}")
        if not protocol_path.is_file():
            raise FileNotFoundError(f"protocol does not exist: {protocol_path}")

        with protocol_path.open("r", encoding="utf-8") as stream:
            raw_protocol = json.load(stream)
        protocol = validate_dataset_protocol(raw_protocol, font_ids=self.font_ids)
        self._validate_envelope(entry, protocol)
        if self.max_objects is not None and len(protocol.objects) > self.max_objects:
            raise ValueError(
                f"sample {entry.sample_id!r} has {len(protocol.objects)} objects; "
                f"configured maximum is {self.max_objects}"
            )
        if self.verify_image_dimensions and image_path.is_file():
            with Image.open(image_path) as image:
                if image.size != (protocol.canvas.width, protocol.canvas.height):
                    raise ValueError(
                        f"image dimensions {image.size} do not match protocol canvas "
                        f"{(protocol.canvas.width, protocol.canvas.height)}"
                    )
        return ProtocolDatasetRecord(
            sample_id=entry.sample_id,
            image_path=image_path,
            protocol_path=protocol_path,
            canvas_width=protocol.canvas.width,
            canvas_height=protocol.canvas.height,
            protocol=protocol,
            canonical_protocol=canonicalize(protocol, decimal_places=self.decimal_places),
            seed=entry.seed,
        )

    @staticmethod
    def _validate_envelope(entry: ManifestEntry, protocol: DatasetProtocol) -> None:
        if entry.sample_id != protocol.sample_id:
            raise ValueError(
                f"manifest sample_id {entry.sample_id!r} does not match protocol "
                f"sample_id {protocol.sample_id!r}"
            )
        if entry.seed != protocol.seed:
            raise ValueError(
                f"manifest seed {entry.seed} does not match protocol seed {protocol.seed}"
            )
