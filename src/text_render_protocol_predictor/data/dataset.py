"""Lazy records for image/protocol SFT datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Literal

from PIL import Image

from .instance_masks import rasterize_protocol_instance_masks, validate_mask_file
from .manifest import ManifestEntry, load_manifest, resolve_dataset_path
from .structural_noise import (
    StructuralNoiseConfig,
    apply_structural_noise,
)
from ..protocol.canonicalizer import canonicalize
from ..protocol.coordinate_tokens import CoordinateTokenCodec
from ..protocol.grounding import ground_protocol_json
from ..protocol.schema import DatasetProtocol
from ..protocol.validator import validate_dataset_protocol


@dataclass(frozen=True)
class InstanceMaskSupervision:
    source: Literal["files", "geometry"]
    object_paths: dict[str, Path]


@dataclass(frozen=True)
class ProtocolDatasetRecord:
    sample_id: str
    image_path: Path
    protocol_path: Path
    canvas_width: int
    canvas_height: int
    protocol_version: str
    purpose: str
    protocol: DatasetProtocol
    canonical_protocol: str
    grounded_protocol: str
    text_object_ids: tuple[str, ...]
    mask_supervision: InstanceMaskSupervision | None
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
        structural_noise: StructuralNoiseConfig | None = None,
        coordinate_codec: CoordinateTokenCodec | None = None,
        text_only_targets: bool = False,
        grounding_enabled: bool = False,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        manifest = Path(manifest_path)
        if not manifest.is_absolute():
            manifest = resolve_dataset_path(self.dataset_root, str(manifest))
        self.manifest_path = manifest
        self.entry_root = manifest.parent
        self.entries = load_manifest(manifest)
        self.decimal_places = decimal_places
        self.font_ids = font_ids
        self.require_files = require_files
        self.verify_image_dimensions = verify_image_dimensions
        self.max_objects = max_objects
        self.structural_noise = structural_noise or StructuralNoiseConfig()
        self.coordinate_codec = coordinate_codec
        self.text_only_targets = text_only_targets
        self.grounding_enabled = grounding_enabled

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> ProtocolDatasetRecord:
        entry: ManifestEntry = self.entries[index]
        relative_manifest_root = self.entry_root.relative_to(self.dataset_root)
        image_path = resolve_dataset_path(
            self.dataset_root, str(relative_manifest_root / entry.image)
        )
        protocol_path = resolve_dataset_path(
            self.dataset_root, str(relative_manifest_root / entry.protocol)
        )
        if self.require_files and not image_path.is_file():
            raise FileNotFoundError(f"image does not exist: {image_path}")
        if not protocol_path.is_file():
            raise FileNotFoundError(f"protocol does not exist: {protocol_path}")

        with protocol_path.open("r", encoding="utf-8") as stream:
            raw_protocol = json.load(stream)
        protocol = validate_dataset_protocol(raw_protocol, font_ids=self.font_ids)
        self._validate_envelope(entry, protocol)
        target_object_count = sum(
            not self.text_only_targets or hasattr(obj, "text")
            for obj in protocol.objects
        )
        if self.max_objects is not None and target_object_count > self.max_objects:
            raise ValueError(
                f"sample {entry.sample_id!r} has {target_object_count} target objects; "
                f"configured maximum is {self.max_objects}"
            )
        if self.verify_image_dimensions and image_path.is_file():
            with Image.open(image_path) as image:
                if image.size != (protocol.canvas.width, protocol.canvas.height):
                    raise ValueError(
                        f"image dimensions {image.size} do not match protocol canvas "
                        f"{(protocol.canvas.width, protocol.canvas.height)}"
                    )
        protocol = apply_structural_noise(
            protocol,
            config=self.structural_noise,
            seed=self.structural_noise.seed + protocol.seed,
            object_groups=entry.structural_groups,
        )
        if self.coordinate_codec is None:
            canonical_protocol = canonicalize(
                protocol,
                decimal_places=self.decimal_places,
                text_only=self.text_only_targets,
            )
        else:
            canonical_protocol = self.coordinate_codec.encode_json(
                protocol,
                decimal_places=self.decimal_places,
                text_only=self.text_only_targets,
            )
        grounded_protocol = (
            ground_protocol_json(canonical_protocol)
            if self.grounding_enabled
            else canonical_protocol
        )
        text_objects = tuple(
            obj
            for obj in sorted(protocol.objects, key=lambda obj: (obj.z_order, obj.id))
            if hasattr(obj, "text")
        )
        mask_supervision = self._resolve_mask_supervision(
            entry,
            protocol,
            text_object_ids={obj.id for obj in text_objects},
        )
        return ProtocolDatasetRecord(
            sample_id=entry.sample_id,
            image_path=image_path,
            protocol_path=protocol_path,
            canvas_width=protocol.canvas.width,
            canvas_height=protocol.canvas.height,
            protocol_version=protocol.protocol_version,
            purpose=getattr(protocol, "purpose", "render"),
            protocol=protocol,
            canonical_protocol=canonical_protocol,
            grounded_protocol=grounded_protocol,
            text_object_ids=tuple(obj.id for obj in text_objects),
            mask_supervision=mask_supervision,
            seed=protocol.seed,
        )

    def _resolve_mask_supervision(
        self,
        entry: ManifestEntry,
        protocol: DatasetProtocol,
        *,
        text_object_ids: set[str],
    ) -> InstanceMaskSupervision | None:
        declared = entry.mask_supervision
        if declared is None:
            return None
        if declared.source == "geometry":
            if not text_object_ids:
                raise ValueError(
                    f"sample {entry.sample_id!r} requests geometry mask supervision "
                    "but has no target text objects"
                )
            empty_ids = [
                object_id
                for object_id, mask in rasterize_protocol_instance_masks(protocol).items()
                if mask.getbbox() is None
            ]
            if empty_ids:
                raise ValueError(
                    f"sample {entry.sample_id!r} has geometry-derived masks with no "
                    f"on-canvas foreground: {', '.join(empty_ids)}"
                )
            return InstanceMaskSupervision(source="geometry", object_paths={})

        unknown_ids = sorted(set(declared.objects) - text_object_ids)
        if unknown_ids:
            raise ValueError(
                f"sample {entry.sample_id!r} has masks for unknown or non-text objects: "
                f"{', '.join(unknown_ids)}"
            )
        object_paths: dict[str, Path] = {}
        for object_id, relative_path in declared.objects.items():
            relative_manifest_root = self.entry_root.relative_to(self.dataset_root)
            path = resolve_dataset_path(
                self.dataset_root,
                str(relative_manifest_root / relative_path),
            )
            if self.require_files and not path.is_file():
                raise FileNotFoundError(f"instance mask does not exist: {path}")
            if path.is_file():
                validate_mask_file(
                    path,
                    expected_size=(protocol.canvas.width, protocol.canvas.height),
                )
            object_paths[object_id] = path
        if len(set(object_paths.values())) != len(object_paths):
            raise ValueError(
                f"sample {entry.sample_id!r} assigns one mask file to multiple objects"
            )
        return InstanceMaskSupervision(source="files", object_paths=object_paths)

    @staticmethod
    def _validate_envelope(entry: ManifestEntry, protocol: DatasetProtocol) -> None:
        if entry.sample_id != protocol.sample_id:
            raise ValueError(
                f"manifest sample_id {entry.sample_id!r} does not match protocol "
                f"sample_id {protocol.sample_id!r}"
            )
        if entry.seed is not None and entry.seed != protocol.seed:
            raise ValueError(
                f"manifest seed {entry.seed} does not match protocol seed {protocol.seed}"
            )
