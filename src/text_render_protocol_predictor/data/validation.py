"""Fail-fast split validation performed before model allocation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SampleValidationFailure:
    index: int
    sample_id: str
    error: str


@dataclass
class DatasetValidationReport:
    total: int
    valid: int = 0
    failures: list[SampleValidationFailure] = field(default_factory=list)

    @property
    def invalid(self) -> int:
        return len(self.failures)

    def raise_for_errors(self, *, max_examples: int = 20) -> None:
        if not self.failures:
            return
        examples = "\n".join(
            f"- index={failure.index} sample_id={failure.sample_id!r}: {failure.error}"
            for failure in self.failures[:max_examples]
        )
        remainder = self.invalid - min(self.invalid, max_examples)
        suffix = f"\n... and {remainder} more" if remainder else ""
        raise ValueError(
            f"dataset validation rejected {self.invalid}/{self.total} samples:\n"
            f"{examples}{suffix}"
        )


def validate_dataset(dataset: Any) -> DatasetValidationReport:
    report = DatasetValidationReport(total=len(dataset))
    for index, entry in enumerate(dataset.entries):
        try:
            dataset[index]
        except Exception as exc:
            report.failures.append(
                SampleValidationFailure(
                    index=index,
                    sample_id=entry.sample_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
        else:
            report.valid += 1
    return report
