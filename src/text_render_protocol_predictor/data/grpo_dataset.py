"""Manifest-backed reconstruction data used by GRPO rewards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .manifest import resolve_dataset_path


class OCRWord(BaseModel):
    """One box-free OCR word used only by the GRPO content reward."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    text: str = Field(min_length=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("text")
    @classmethod
    def text_must_contain_non_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("OCR word text must contain non-whitespace characters")
        return value


class GRPOManifestEntry(BaseModel):
    """One real-image reconstruction example."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    image: str = Field(min_length=1)
    background: str = Field(min_length=1)
    text_mask: str = Field(min_length=1)
    words: list[OCRWord] = Field(default_factory=list)


def load_grpo_manifest(path: str | Path) -> list[GRPOManifestEntry]:
    path = Path(path)
    entries: list[GRPOManifestEntry] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                entry = GRPOManifestEntry.model_validate_json(line)
            except Exception as exc:
                raise ValueError(
                    f"invalid GRPO manifest record at {path}:{line_number}: {exc}"
                ) from exc
            if entry.sample_id in seen:
                raise ValueError(
                    f"duplicate sample_id {entry.sample_id!r} at {path}:{line_number}"
                )
            seen.add(entry.sample_id)
            entries.append(entry)
    if not entries:
        raise ValueError(f"GRPO manifest is empty: {path}")
    return entries


@dataclass(frozen=True)
class GRPODatasetRecord:
    sample_id: str
    image_path: Path
    background_path: Path
    text_mask_path: Path
    canvas_width: int
    canvas_height: int
    mask_coverage: float
    words: tuple[OCRWord, ...] = ()


class GRPOManifestDataset:
    """Validate and expose original/background/mask triples lazily."""

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        manifest_path: str | Path,
        mask_threshold: float = 0.5,
        minimum_mask_coverage: float = 1e-5,
        maximum_mask_coverage: float = 1.0,
        require_webp: bool = True,
        require_words: bool = False,
        minimum_word_confidence: float = 0.0,
    ) -> None:
        if not 0.0 <= mask_threshold <= 1.0:
            raise ValueError("mask_threshold must be between 0 and 1")
        if not 0.0 <= minimum_mask_coverage <= maximum_mask_coverage <= 1.0:
            raise ValueError(
                "mask coverage bounds must satisfy 0 <= minimum <= maximum <= 1"
            )
        if not 0.0 <= minimum_word_confidence <= 1.0:
            raise ValueError("minimum_word_confidence must be between 0 and 1")

        self.dataset_root = Path(dataset_root).expanduser().resolve()
        manifest = Path(manifest_path)
        if not manifest.is_absolute():
            manifest = resolve_dataset_path(self.dataset_root, str(manifest))
        else:
            manifest = manifest.expanduser().resolve()
            if not manifest.is_relative_to(self.dataset_root):
                raise ValueError(f"manifest must be inside dataset_root: {manifest}")
        self.manifest_path = manifest
        self.entry_root = manifest.parent
        self.entries = load_grpo_manifest(manifest)
        self.mask_threshold = float(mask_threshold)
        self.minimum_mask_coverage = float(minimum_mask_coverage)
        self.maximum_mask_coverage = float(maximum_mask_coverage)
        self.require_webp = require_webp
        self.require_words = bool(require_words)
        self.minimum_word_confidence = float(minimum_word_confidence)

    def __len__(self) -> int:
        return len(self.entries)

    def _resolve_entry_path(self, relative_path: str) -> Path:
        relative_manifest_root = self.entry_root.relative_to(self.dataset_root)
        return resolve_dataset_path(
            self.dataset_root, str(relative_manifest_root / relative_path)
        )

    def __getitem__(self, index: int) -> GRPODatasetRecord:
        entry = self.entries[index]
        usable_words = tuple(
            word
            for word in entry.words
            if word.confidence >= self.minimum_word_confidence
            and any(character.isalnum() for character in word.text)
        )
        if self.require_words and not usable_words:
            raise ValueError(
                f"sample {entry.sample_id!r} has no OCR words with confidence >= "
                f"{self.minimum_word_confidence:.3f}"
            )
        image_path = self._resolve_entry_path(entry.image)
        background_path = self._resolve_entry_path(entry.background)
        text_mask_path = self._resolve_entry_path(entry.text_mask)
        paths = (image_path, background_path, text_mask_path)
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"GRPO image does not exist: {path}")
            if self.require_webp and path.suffix.lower() != ".webp":
                raise ValueError(f"GRPO image must be WebP: {path}")

        sizes: list[tuple[int, int]] = []
        for path in paths:
            try:
                with Image.open(path) as image:
                    image.load()
                    sizes.append(image.size)
            except Exception as exc:
                raise ValueError(f"cannot decode image {path}: {exc}") from exc
        if len(set(sizes)) != 1:
            raise ValueError(
                "original, background, and text mask dimensions differ: "
                f"{dict(zip(('image', 'background', 'text_mask'), sizes, strict=True))}"
            )

        with Image.open(text_mask_path) as mask_image:
            mask = mask_image.convert("L")
            cutoff = round(self.mask_threshold * 255)
            histogram = mask.histogram()
            foreground = sum(histogram[cutoff:])
            coverage = foreground / (mask.width * mask.height)
        if not self.minimum_mask_coverage <= coverage <= self.maximum_mask_coverage:
            raise ValueError(
                f"sample {entry.sample_id!r} mask coverage {coverage:.6f} is outside "
                f"[{self.minimum_mask_coverage:.6f}, {self.maximum_mask_coverage:.6f}]"
            )

        width, height = sizes[0]
        return GRPODatasetRecord(
            sample_id=entry.sample_id,
            image_path=image_path,
            background_path=background_path,
            text_mask_path=text_mask_path,
            canvas_width=width,
            canvas_height=height,
            mask_coverage=coverage,
            words=usable_words,
        )
