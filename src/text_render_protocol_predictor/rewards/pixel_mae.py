"""Masked pixel-reconstruction reward for rendered protocol completions."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageFilter

from ..rendering import RenderStatus


@dataclass(frozen=True)
class PixelMAERewardConfig:
    mask_threshold: float = 0.5
    mask_dilation_radius: int = 4
    mask_blur_radius: float = 2.0
    outside_weight: float = 0.1
    cache_size: int = 8
    invalid_json_reward: float = -1.0
    invalid_schema_reward: float = -0.9
    invalid_semantics_reward: float = -0.85
    unknown_font_reward: float = -0.8
    renderer_failure_reward: float = -0.8

    def __post_init__(self) -> None:
        if not 0.0 <= self.mask_threshold <= 1.0:
            raise ValueError("mask_threshold must be between 0 and 1")
        if self.mask_dilation_radius < 0:
            raise ValueError("mask_dilation_radius must be non-negative")
        if self.mask_blur_radius < 0:
            raise ValueError("mask_blur_radius must be non-negative")
        if self.outside_weight < 0:
            raise ValueError("outside_weight must be non-negative")
        if self.cache_size < 1:
            raise ValueError("cache_size must be positive")


@dataclass(frozen=True)
class RewardBreakdown:
    sample_id: str
    status: RenderStatus
    reward: float
    masked_mae: float | None
    outside_mae: float | None
    background_masked_mae: float
    restoration_delta: float | None
    error: str | None = None


@dataclass(frozen=True)
class _RewardAssets:
    original: np.ndarray
    background_image: Image.Image
    background: np.ndarray
    mask: np.ndarray
    background_masked_mae: float


def prepare_text_mask(
    mask: Image.Image,
    *,
    threshold: float,
    dilation_radius: int,
    blur_radius: float,
) -> np.ndarray:
    """Convert an arbitrary mask to a dilated, soft float mask in ``[0, 1]``."""
    grayscale = mask.convert("L")
    cutoff = round(float(threshold) * 255)
    binary = grayscale.point(lambda value: 255 if value >= cutoff else 0)
    if dilation_radius:
        binary = binary.filter(ImageFilter.MaxFilter(2 * int(dilation_radius) + 1))
    if blur_radius:
        binary = binary.filter(ImageFilter.GaussianBlur(float(blur_radius)))
    return np.asarray(binary, dtype=np.float32) / 255.0


def masked_rgb_mae(candidate: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Mean absolute RGB error weighted by a 2-D soft mask."""
    if candidate.shape != target.shape or candidate.ndim != 3 or candidate.shape[2] != 3:
        raise ValueError(
            f"candidate and target must be equal HxWx3 arrays, got {candidate.shape} and "
            f"{target.shape}"
        )
    if mask.shape != candidate.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} does not match image {candidate.shape[:2]}")
    weight = float(mask.sum())
    if weight <= 0:
        return 0.0
    per_pixel = np.abs(candidate - target).mean(axis=2)
    return float((per_pixel * mask).sum() / weight)


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if not isinstance(completion, Sequence) or isinstance(completion, (bytes, bytearray)):
        raise TypeError(f"unsupported completion type: {type(completion).__name__}")
    parts: list[str] = []
    for message in completion:
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, Sequence):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
    if not parts:
        raise TypeError("completion contains no assistant text")
    return "".join(parts)


class PixelMAEReward:
    """TRL-compatible callable that renders completions and scores reconstruction."""

    def __init__(self, renderer: Any, config: PixelMAERewardConfig | None = None) -> None:
        # TRL names Python reward callables through ``__name__`` for metrics.
        self.__name__ = "pixel_mae"
        self.renderer = renderer
        self.config = config or PixelMAERewardConfig()
        self._asset_cache: OrderedDict[tuple[str, str, str], _RewardAssets] = OrderedDict()
        self.last_breakdowns: list[RewardBreakdown] = []
        self._failure_rewards = {
            RenderStatus.INVALID_JSON: self.config.invalid_json_reward,
            RenderStatus.INVALID_SCHEMA: self.config.invalid_schema_reward,
            RenderStatus.INVALID_SEMANTICS: self.config.invalid_semantics_reward,
            RenderStatus.UNKNOWN_FONT: self.config.unknown_font_reward,
            RenderStatus.RENDERER_FAILURE: self.config.renderer_failure_reward,
        }

    def _load_assets(
        self,
        original_path: str | Path,
        background_path: str | Path,
        text_mask_path: str | Path,
    ) -> _RewardAssets:
        key = (str(original_path), str(background_path), str(text_mask_path))
        cached = self._asset_cache.get(key)
        if cached is not None:
            self._asset_cache.move_to_end(key)
            return cached

        with Image.open(original_path) as image:
            original_image = image.convert("RGB")
            original = np.asarray(original_image, dtype=np.float32) / 255.0
        with Image.open(background_path) as image:
            background_image = image.convert("RGB")
            background = np.asarray(background_image, dtype=np.float32) / 255.0
        with Image.open(text_mask_path) as image:
            mask = prepare_text_mask(
                image,
                threshold=self.config.mask_threshold,
                dilation_radius=self.config.mask_dilation_radius,
                blur_radius=self.config.mask_blur_radius,
            )
        if original.shape != background.shape or mask.shape != original.shape[:2]:
            raise ValueError(
                "reward assets have inconsistent dimensions: "
                f"original={original.shape}, background={background.shape}, mask={mask.shape}"
            )
        assets = _RewardAssets(
            original=original,
            background_image=background_image,
            background=background,
            mask=mask,
            background_masked_mae=masked_rgb_mae(background, original, mask),
        )
        self._asset_cache[key] = assets
        self._asset_cache.move_to_end(key)
        while len(self._asset_cache) > self.config.cache_size:
            self._asset_cache.popitem(last=False)
        return assets

    def score(
        self,
        completion: Any,
        *,
        sample_id: str,
        original_path: str | Path,
        background_path: str | Path,
        text_mask_path: str | Path,
    ) -> RewardBreakdown:
        assets = self._load_assets(original_path, background_path, text_mask_path)
        try:
            completion_text = _completion_text(completion)
        except (TypeError, ValueError) as exc:
            return RewardBreakdown(
                sample_id=sample_id,
                status=RenderStatus.INVALID_JSON,
                reward=self.config.invalid_json_reward,
                masked_mae=None,
                outside_mae=None,
                background_masked_mae=assets.background_masked_mae,
                restoration_delta=None,
                error=str(exc),
            )

        outcome = self.renderer.render_prediction(
            completion_text,
            assets.background_image.copy(),
            sample_id=sample_id,
        )
        if outcome.status is not RenderStatus.OK or outcome.image is None:
            return RewardBreakdown(
                sample_id=sample_id,
                status=outcome.status,
                reward=float(self._failure_rewards[outcome.status]),
                masked_mae=None,
                outside_mae=None,
                background_masked_mae=assets.background_masked_mae,
                restoration_delta=None,
                error=outcome.error,
            )

        candidate = np.asarray(outcome.image.convert("RGB"), dtype=np.float32) / 255.0
        masked_mae = masked_rgb_mae(candidate, assets.original, assets.mask)
        outside_mae = masked_rgb_mae(candidate, assets.background, 1.0 - assets.mask)
        reward = 1.0 - masked_mae - self.config.outside_weight * outside_mae
        return RewardBreakdown(
            sample_id=sample_id,
            status=RenderStatus.OK,
            reward=float(reward),
            masked_mae=masked_mae,
            outside_mae=outside_mae,
            background_masked_mae=assets.background_masked_mae,
            restoration_delta=assets.background_masked_mae - masked_mae,
        )

    def __call__(self, completions: Sequence[Any], **kwargs: Any) -> list[float]:
        required = ("sample_id", "original_path", "background_path", "text_mask_path")
        missing = [name for name in required if name not in kwargs]
        if missing:
            raise ValueError(f"reward call is missing dataset columns: {', '.join(missing)}")

        breakdowns: list[RewardBreakdown] = []
        memo: dict[tuple[str, str, str, str, str], RewardBreakdown] = {}
        for index, completion in enumerate(completions):
            values = {name: kwargs[name][index] for name in required}
            try:
                completion_key = _completion_text(completion)
            except TypeError:
                completion_key = repr(completion)
            key = (
                str(values["sample_id"]),
                str(values["original_path"]),
                str(values["background_path"]),
                str(values["text_mask_path"]),
                completion_key,
            )
            breakdown = memo.get(key)
            if breakdown is None:
                breakdown = self.score(completion, **values)
                memo[key] = breakdown
            breakdowns.append(breakdown)
        self.last_breakdowns = breakdowns
        return [item.reward for item in breakdowns]
