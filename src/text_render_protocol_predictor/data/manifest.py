"""JSONL split-manifest loading with dataset-root-relative paths."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    image: str = Field(min_length=1)
    protocol: str = Field(min_length=1)
    seed: int

    recipe: str | None = None
    template_id: str | None = None
    structural_groups: dict[str, str] | None = None


def load_manifest(path: str | Path) -> list[ManifestEntry]:
    path = Path(path)
    entries: list[ManifestEntry] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                entry = ManifestEntry.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"invalid manifest record at {path}:{line_number}: {exc}") from exc
            if entry.sample_id in seen:
                raise ValueError(f"duplicate sample_id {entry.sample_id!r} at {path}:{line_number}")
            seen.add(entry.sample_id)
            entries.append(entry)
    return entries


def resolve_dataset_path(dataset_root: Path, relative_path: str) -> Path:
    candidate_path = Path(relative_path)
    if candidate_path.is_absolute():
        raise ValueError(f"dataset paths must be relative: {relative_path!r}")
    root = dataset_root.expanduser().resolve()
    candidate = (root / candidate_path).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"dataset path escapes configured root: {relative_path!r}")
    return candidate
