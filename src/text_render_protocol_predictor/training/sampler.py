"""Deterministic distributed balancing by mask-supervision source."""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Mapping

from torch.utils.data import Sampler


class MaskSourceBalancedSampler(Sampler[int]):
    """Sample file/geometry/unmasked records at configured proportions."""

    SOURCES = ("files", "geometry", "none")

    def __init__(
        self,
        dataset: object,
        *,
        weights: Mapping[str, float],
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
    ) -> None:
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError("invalid distributed sampler rank or replica count")
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = math.ceil(len(dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.buckets: dict[str, list[int]] = {source: [] for source in self.SOURCES}
        for index, entry in enumerate(dataset.entries):
            declaration = entry.mask_supervision
            source = declaration.source if declaration is not None else "none"
            self.buckets[source].append(index)
        unknown = sorted(set(weights) - set(self.SOURCES))
        if unknown:
            raise ValueError(f"unknown mask sampling sources: {unknown}")
        active_weights = {
            source: float(weights.get(source, 0.0))
            for source in self.SOURCES
            if self.buckets[source]
        }
        if any(weight < 0 for weight in active_weights.values()):
            raise ValueError("mask sampling weights must be non-negative")
        total_weight = sum(active_weights.values())
        if total_weight <= 0:
            raise ValueError("at least one non-empty mask source must have positive weight")
        self.weights = {
            source: weight / total_weight for source, weight in active_weights.items()
        }

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _source_counts(self) -> dict[str, int]:
        exact = {source: self.total_size * weight for source, weight in self.weights.items()}
        counts = {source: math.floor(value) for source, value in exact.items()}
        remainder = self.total_size - sum(counts.values())
        order = sorted(
            exact,
            key=lambda source: (exact[source] - counts[source], source),
            reverse=True,
        )
        for source in order[:remainder]:
            counts[source] += 1
        return counts

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        indices: list[int] = []
        for source, count in self._source_counts().items():
            bucket = self.buckets[source]
            shuffled = list(bucket)
            rng.shuffle(shuffled)
            for offset in range(count):
                if offset and offset % len(shuffled) == 0:
                    rng.shuffle(shuffled)
                indices.append(shuffled[offset % len(shuffled)])
        rng.shuffle(indices)
        if len(indices) != self.total_size:
            raise RuntimeError("balanced sampler produced the wrong global sample count")
        return iter(indices[self.rank : self.total_size : self.num_replicas])
