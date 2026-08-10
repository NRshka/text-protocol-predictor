"""Contextual mask-query decoder attached to a Qwen3-VL causal model."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


GROUNDING_CONFIG_FILE = "grounding_config.json"
GROUNDING_WEIGHTS_FILE = "grounding_model.safetensors"


@dataclass(frozen=True)
class GroundedDecoderConfig:
    format_version: str = "1.0.0"
    mask_token: str = "<MASK>"
    feature_layers: tuple[int, ...] = (8, 16, 24, 26)
    feature_projection_dim: int = 128
    pixel_embedding_dim: int = 256
    query_hidden_dim: int = 1024
    output_stride: int = 4
    bezier_samples: int = 32
    mask_bce_weight: float = 1.0
    mask_dice_weight: float = 1.0
    geometry_weight: float = 0.5
    consistency_weight: float = 0.25
    quality_weight: float = 0.1
    positive_weight_max: float = 20.0

    @classmethod
    def from_mapping(cls, value: Any) -> "GroundedDecoderConfig":
        if value is None:
            return cls()
        known = {field.name for field in fields(cls)}
        values = {
            name: value[name] if isinstance(value, dict) else getattr(value, name)
            for name in known
            if (name in value if isinstance(value, dict) else hasattr(value, name))
        }
        if "feature_layers" in values:
            values["feature_layers"] = tuple(int(item) for item in values["feature_layers"])
        return cls(**values)


def unshuffle_qwen_patches(
    features: torch.Tensor,
    grid_thw: torch.Tensor | tuple[int, int, int],
    *,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Restore merge-block-major Qwen patches to ``C x H x W`` order."""
    temporal, height, width = (int(item) for item in grid_thw)
    if temporal != 1:
        raise ValueError("grounded decoder supports static images only")
    merge = int(spatial_merge_size)
    if height % merge or width % merge:
        raise ValueError(
            f"vision grid {(height, width)} is not divisible by merge size {merge}"
        )
    if features.ndim != 2 or features.shape[0] != temporal * height * width:
        raise ValueError(
            f"vision features {tuple(features.shape)} do not match grid "
            f"{(temporal, height, width)}"
        )
    channels = features.shape[-1]
    row_major = (
        features.reshape(
            temporal,
            height // merge,
            width // merge,
            merge,
            merge,
            channels,
        )
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(temporal, height, width, channels)
    )
    return row_major[0].permute(2, 0, 1).contiguous()


def _soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probabilities = logits.sigmoid()
    intersection = (probabilities * target).sum(dim=(-2, -1))
    denominator = probabilities.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def _soft_iou(probabilities: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    intersection = (probabilities * target).sum(dim=(-2, -1))
    union = (probabilities + target - probabilities * target).sum(dim=(-2, -1))
    return (intersection + 1.0) / (union + 1.0)


def _pixel_grid(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    ys = torch.arange(height, device=device, dtype=dtype) + 0.5
    xs = torch.arange(width, device=device, dtype=dtype) + 0.5
    return torch.meshgrid(ys, xs, indexing="ij")


def soft_rasterize_geometry(
    *,
    mode_logits: torch.Tensor,
    boxes: torch.Tensor,
    rotations: torch.Tensor,
    beziers: torch.Tensor,
    size: tuple[int, int],
    bezier_samples: int = 32,
) -> torch.Tensor:
    """Differentiably rasterize mixed oriented-box/Bézier predictions."""
    count = boxes.shape[0]
    height, width = size
    if count == 0:
        return boxes.new_empty((0, height, width))
    grid_y, grid_x = _pixel_grid(
        height,
        width,
        device=boxes.device,
        dtype=boxes.dtype,
    )
    results: list[torch.Tensor] = []
    ts = torch.linspace(0.0, 1.0, bezier_samples, device=boxes.device, dtype=boxes.dtype)
    coefficients = torch.stack(
        (
            (1.0 - ts) ** 3,
            3.0 * (1.0 - ts) ** 2 * ts,
            3.0 * (1.0 - ts) * ts**2,
            ts**3,
        ),
        dim=-1,
    )
    probabilities = mode_logits.softmax(dim=-1)
    for index in range(count):
        center_x, center_y, box_width, box_height = boxes[index]
        dx = grid_x - center_x * width
        dy = grid_y - center_y * height
        sine, cosine = rotations[index]
        local_x = dx * cosine + dy * sine
        local_y = -dx * sine + dy * cosine
        qx = local_x.abs() - box_width * width / 2.0
        qy = local_y.abs() - box_height * height / 2.0
        outside = torch.sqrt(F.relu(qx).square() + F.relu(qy).square() + 1.0e-6)
        inside = torch.minimum(torch.maximum(qx, qy), qx.new_zeros(()))
        straight = torch.sigmoid(-(outside + inside))

        controls = beziers[index].reshape(4, 2)
        curve = coefficients @ controls
        curve_x = curve[:, 0] * width
        curve_y = curve[:, 1] * height
        squared_distance = (
            (grid_x[..., None] - curve_x).square()
            + (grid_y[..., None] - curve_y).square()
        )
        sampled_distance = torch.sqrt(squared_distance + 1.0e-6)
        # A temperature-weighted average is a smooth, non-negative
        # approximation to the nearest sampled centerline point.
        nearest_weights = torch.softmax(-sampled_distance, dim=-1)
        distance = (nearest_weights * sampled_distance).sum(dim=-1)
        bezier = torch.sigmoid(-(distance - box_height * height / 2.0))
        results.append(
            probabilities[index, 0] * straight
            + probabilities[index, 1] * bezier
        )
    return torch.stack(results)


def _locate_qwen_modules(backbone: nn.Module) -> tuple[Any, nn.Module, nn.Module]:
    base = backbone.get_base_model() if hasattr(backbone, "get_base_model") else backbone
    qwen_model = getattr(base, "model", None)
    if qwen_model is None or not hasattr(qwen_model, "visual"):
        raise ValueError("could not locate Qwen3-VL visual module")
    if not hasattr(qwen_model, "language_model"):
        raise ValueError("could not locate Qwen3-VL language model")
    return base, qwen_model.visual, qwen_model.language_model


class _UpsampleBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(32, channels)
        self.activation = nn.GELU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(value, scale_factor=2, mode="bilinear", align_corners=False)
        return self.activation(self.norm(self.conv(value)))


class GroundedQwen3VL(nn.Module):
    """Qwen3-VL plus contextual instance-mask and geometry heads."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        mask_token_id: int,
        decoder_config: GroundedDecoderConfig | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.mask_token_id = int(mask_token_id)
        self.decoder_config = decoder_config or GroundedDecoderConfig()
        base, visual_module, language_module = _locate_qwen_modules(backbone)
        # These are hook targets already owned by ``backbone``. Keep
        # non-owning references so state_dict does not serialize Qwen twice or
        # mistake backbone tensors for grounding-head weights.
        object.__setattr__(self, "_visual_module", visual_module)
        object.__setattr__(self, "_language_module", language_module)
        model_config = base.config
        vision_config = model_config.vision_config
        text_config = model_config.text_config
        self.vision_patch_size = int(vision_config.patch_size)
        self.spatial_merge_size = int(vision_config.spatial_merge_size)
        self.vision_hidden_size = int(vision_config.hidden_size)
        self.text_hidden_size = int(text_config.hidden_size)
        if self.decoder_config.output_stride != 4:
            raise ValueError("the initial grounded decoder requires output_stride=4")
        depth = len(self._visual_module.blocks)
        invalid = [layer for layer in self.decoder_config.feature_layers if not 0 <= layer < depth]
        if invalid:
            raise ValueError(f"vision feature layers outside depth {depth}: {invalid}")

        projection_dim = self.decoder_config.feature_projection_dim
        pixel_dim = self.decoder_config.pixel_embedding_dim
        if pixel_dim <= 0 or pixel_dim % 32:
            raise ValueError("pixel_embedding_dim must be a positive multiple of 32")
        self.feature_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.vision_hidden_size),
                    nn.Linear(self.vision_hidden_size, projection_dim),
                )
                for _ in self.decoder_config.feature_layers
            ]
        )
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(
                projection_dim * len(self.decoder_config.feature_layers),
                pixel_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(32, pixel_dim),
            nn.GELU(),
        )
        self.upsample = nn.Sequential(_UpsampleBlock(pixel_dim), _UpsampleBlock(pixel_dim))
        self.pixel_projection = nn.Conv2d(pixel_dim, pixel_dim, kernel_size=1)
        self.query_projection = nn.Sequential(
            nn.LayerNorm(self.text_hidden_size),
            nn.Linear(self.text_hidden_size, self.decoder_config.query_hidden_dim),
            nn.GELU(),
            nn.Linear(self.decoder_config.query_hidden_dim, pixel_dim),
        )
        self.query_bias = nn.Linear(pixel_dim, 1)
        self.geometry_trunk = nn.Sequential(
            nn.LayerNorm(pixel_dim),
            nn.Linear(pixel_dim, pixel_dim),
            nn.GELU(),
        )
        self.mode_head = nn.Linear(pixel_dim, 2)
        self.box_head = nn.Linear(pixel_dim, 4)
        self.rotation_head = nn.Linear(pixel_dim, 2)
        self.bezier_head = nn.Linear(pixel_dim, 8)
        self.quality_head = nn.Linear(pixel_dim, 1)

        self._capture_enabled = False
        self._vision_features: dict[int, torch.Tensor] = {}
        self._language_hidden: torch.Tensor | None = None
        self._hook_handles = []
        for layer_index in self.decoder_config.feature_layers:
            self._hook_handles.append(
                self._visual_module.blocks[layer_index].register_forward_hook(
                    self._vision_hook(layer_index)
                )
            )
        self._hook_handles.append(
            self._language_module.register_forward_hook(self._language_hook)
        )
        self._joint_trainable_names = {
            name for name, parameter in self.backbone.named_parameters() if parameter.requires_grad
        }
        self.consistency_weight = self.decoder_config.consistency_weight

    @property
    def config(self) -> Any:
        return self.backbone.config

    @property
    def peft_config(self) -> Any:
        return getattr(self.backbone, "peft_config", {})

    def _vision_hook(self, layer_index: int):
        def capture(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if self._capture_enabled:
                self._vision_features[layer_index] = output[0] if isinstance(output, tuple) else output

        return capture

    def _language_hook(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        if self._capture_enabled:
            self._language_hidden = output[0]

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        self._capture_enabled = False
        self._vision_features.clear()
        self._language_hidden = None
        return self.backbone.generate(*args, **kwargs)

    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.get_input_embeddings()

    def get_output_embeddings(self) -> nn.Module:
        return self.backbone.get_output_embeddings()

    def save_pretrained(self, output_dir: str | Path, **kwargs: Any) -> None:
        self.backbone.save_pretrained(output_dir, **kwargs)
        self.save_grounding_pretrained(output_dir)

    def set_training_phase(self, phase: str) -> None:
        """Select the documented heads-only or joint trainable parameter set."""
        if phase not in {"heads_only", "joint"}:
            raise ValueError("grounded training phase must be heads_only or joint")
        for name, parameter in self.backbone.named_parameters():
            if phase == "joint":
                parameter.requires_grad_(name in self._joint_trainable_names)
            else:
                parameter.requires_grad_("trainable_token" in name.lower())
        for name, parameter in self.named_parameters():
            if not name.startswith("backbone."):
                parameter.requires_grad_(True)

    def _split_feature_maps(self, image_grid_thw: torch.Tensor) -> list[torch.Tensor]:
        grids = [tuple(int(item) for item in grid.tolist()) for grid in image_grid_thw]
        split_sizes = [temporal * height * width for temporal, height, width in grids]
        per_level: list[list[torch.Tensor]] = []
        for projection, layer_index in zip(
            self.feature_projections,
            self.decoder_config.feature_layers,
            strict=True,
        ):
            captured = self._vision_features.get(layer_index)
            if captured is None:
                raise RuntimeError(f"vision layer {layer_index} did not produce captured features")
            splits = captured.split(split_sizes, dim=0)
            per_level.append(
                [
                    projection(
                        unshuffle_qwen_patches(
                            split,
                            grid,
                            spatial_merge_size=self.spatial_merge_size,
                        )
                        .permute(1, 2, 0)
                        .to(dtype=projection[1].weight.dtype)
                    ).permute(2, 0, 1)
                    for split, grid in zip(splits, grids, strict=True)
                ]
            )
        maps: list[torch.Tensor] = []
        for sample_index in range(len(grids)):
            fused = self.feature_fusion(
                torch.cat(
                    [level[sample_index] for level in per_level], dim=0
                ).unsqueeze(0)
            )
            maps.append(self.pixel_projection(self.upsample(fused)).squeeze(0))
        return maps

    def _predict_instances(
        self,
        *,
        pixel_maps: list[torch.Tensor],
        mask_token_positions: torch.Tensor,
        instance_valid: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
        if self._language_hidden is None:
            raise RuntimeError("language model did not produce captured hidden states")
        logits: list[torch.Tensor] = []
        geometry: list[dict[str, torch.Tensor]] = []
        for batch_index, pixel_map in enumerate(pixel_maps):
            count = int(instance_valid[batch_index].sum().item())
            positions = mask_token_positions[batch_index, :count]
            hidden = self._language_hidden[batch_index].index_select(0, positions)
            hidden = hidden.to(dtype=self.query_projection[1].weight.dtype)
            queries = self.query_projection(hidden)
            instance_logits = torch.einsum("nc,chw->nhw", queries, pixel_map)
            instance_logits = instance_logits / math.sqrt(pixel_map.shape[0])
            instance_logits = instance_logits + self.query_bias(queries).view(-1, 1, 1)
            trunk = self.geometry_trunk(queries)
            rotation = F.normalize(self.rotation_head(trunk), dim=-1, eps=1.0e-6)
            logits.append(instance_logits)
            geometry.append(
                {
                    "mode_logits": self.mode_head(trunk),
                    "boxes": self.box_head(trunk).sigmoid(),
                    "rotations": rotation,
                    "beziers": self.bezier_head(trunk).sigmoid(),
                    "quality_logits": self.quality_head(trunk).squeeze(-1),
                }
            )
        return logits, geometry

    def _auxiliary_losses(
        self,
        *,
        mask_logits: list[torch.Tensor],
        geometry: list[dict[str, torch.Tensor]],
        instance_masks: list[torch.Tensor],
        mask_supervision: list[torch.Tensor],
        geometry_modes: list[torch.Tensor],
        geometry_boxes: list[torch.Tensor],
        geometry_rotations: list[torch.Tensor],
        geometry_beziers: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        # Keep every auxiliary parameter in the graph even for an entirely
        # unmasked batch. This avoids DDP unused-parameter failures while still
        # producing exactly zero gradients for heads without supervision.
        zero = torch.stack(
            [
                parameter.reshape(-1)[0]
                for name, parameter in self.named_parameters()
                if not name.startswith("backbone.")
            ]
        ).sum() * 0.0
        bce_losses: list[torch.Tensor] = []
        dice_losses: list[torch.Tensor] = []
        quality_losses: list[torch.Tensor] = []
        geometry_losses: list[torch.Tensor] = []
        consistency_losses: list[torch.Tensor] = []
        for logits, prediction, target_masks, supervised, target_mode, target_box, target_rotation, target_bezier in zip(
            mask_logits,
            geometry,
            instance_masks,
            mask_supervision,
            geometry_modes,
            geometry_boxes,
            geometry_rotations,
            geometry_beziers,
            strict=True,
        ):
            device = logits.device
            target_masks = target_masks.to(device=device, dtype=logits.dtype)
            supervised = supervised.to(device=device)
            target_mode = target_mode.to(device=device)
            target_box = target_box.to(device=device, dtype=logits.dtype)
            target_rotation = target_rotation.to(device=device, dtype=logits.dtype)
            target_bezier = target_bezier.to(device=device, dtype=logits.dtype)
            if logits.shape != target_masks.shape:
                raise ValueError(
                    f"predicted masks {tuple(logits.shape)} do not match targets "
                    f"{tuple(target_masks.shape)}"
                )

            count = logits.shape[0]
            if count:
                mode_loss = F.cross_entropy(prediction["mode_logits"], target_mode)
                box_loss = F.smooth_l1_loss(prediction["boxes"], target_box)
                curved = target_mode.eq(1)
                rotation_loss = (
                    1.0
                    - (prediction["rotations"] * target_rotation).sum(-1)
                ).mean()
                bezier_loss = (
                    F.smooth_l1_loss(
                        prediction["beziers"][curved], target_bezier[curved]
                    )
                    if curved.any()
                    else zero
                )
                geometry_losses.append(
                    mode_loss + box_loss + rotation_loss + bezier_loss
                )

            for instance_index in torch.nonzero(supervised, as_tuple=False).flatten():
                index = int(instance_index.item())
                target = target_masks[index : index + 1]
                predicted = logits[index : index + 1]
                positive = target.sum().clamp_min(1.0)
                negative = target.numel() - positive
                positive_weight = (negative / positive).clamp(
                    1.0, self.decoder_config.positive_weight_max
                )
                bce_losses.append(
                    F.binary_cross_entropy_with_logits(
                        predicted,
                        target,
                        pos_weight=positive_weight,
                    )
                )
                dice_losses.append(_soft_dice_loss(predicted, target))
                quality_target = _soft_iou(predicted.sigmoid().detach(), target)
                quality_losses.append(
                    F.binary_cross_entropy_with_logits(
                        prediction["quality_logits"][index : index + 1],
                        quality_target,
                    )
                )

            if supervised.any() and self.consistency_weight > 0:
                target_size = (target_masks.shape[-2] // 2, target_masks.shape[-1] // 2)
                rasterized = soft_rasterize_geometry(
                    mode_logits=prediction["mode_logits"][supervised],
                    boxes=prediction["boxes"][supervised],
                    rotations=prediction["rotations"][supervised],
                    beziers=prediction["beziers"][supervised],
                    size=target_size,
                    bezier_samples=self.decoder_config.bezier_samples,
                )
                target_small = F.interpolate(
                    target_masks[supervised].unsqueeze(1),
                    size=target_size,
                    mode="area",
                ).squeeze(1)
                intersection = (rasterized * target_small).sum(dim=(-2, -1))
                denominator = rasterized.sum(dim=(-2, -1)) + target_small.sum(dim=(-2, -1))
                consistency_losses.append(
                    (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
                )

        def mean_or_zero(values: list[torch.Tensor]) -> torch.Tensor:
            return torch.stack(values).mean() if values else zero

        return {
            "mask_bce": mean_or_zero(bce_losses),
            "mask_dice": mean_or_zero(dice_losses),
            "geometry": mean_or_zero(geometry_losses),
            "consistency": mean_or_zero(consistency_losses),
            "quality": mean_or_zero(quality_losses),
        }

    def forward(
        self,
        *,
        mask_token_positions: torch.Tensor | None = None,
        instance_valid: torch.Tensor | None = None,
        instance_masks: list[torch.Tensor] | None = None,
        mask_supervision: list[torch.Tensor] | None = None,
        geometry_modes: list[torch.Tensor] | None = None,
        geometry_boxes: list[torch.Tensor] | None = None,
        geometry_rotations: list[torch.Tensor] | None = None,
        geometry_beziers: list[torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> Any:
        if mask_token_positions is None or instance_valid is None:
            return self.backbone(**kwargs)
        image_grid_thw = kwargs.get("image_grid_thw")
        if image_grid_thw is None:
            raise ValueError("grounded forward requires image_grid_thw")
        self._vision_features.clear()
        self._language_hidden = None
        self._capture_enabled = True
        try:
            output = self.backbone(**kwargs)
        finally:
            self._capture_enabled = False
        pixel_maps = self._split_feature_maps(image_grid_thw)
        mask_logits, geometry = self._predict_instances(
            pixel_maps=pixel_maps,
            mask_token_positions=mask_token_positions,
            instance_valid=instance_valid,
        )
        output.mask_logits = mask_logits
        output.geometry_predictions = geometry
        if instance_masks is None:
            return output
        target_arguments = (
            mask_supervision,
            geometry_modes,
            geometry_boxes,
            geometry_rotations,
            geometry_beziers,
        )
        if any(argument is None for argument in target_arguments):
            raise ValueError("grounded training targets are incomplete")
        losses = self._auxiliary_losses(
            mask_logits=mask_logits,
            geometry=geometry,
            instance_masks=instance_masks,
            mask_supervision=mask_supervision,
            geometry_modes=geometry_modes,
            geometry_boxes=geometry_boxes,
            geometry_rotations=geometry_rotations,
            geometry_beziers=geometry_beziers,
        )
        language_loss = output.loss
        if language_loss is None:
            language_loss = losses["mask_bce"] * 0.0
        total = (
            language_loss
            + self.decoder_config.mask_bce_weight * losses["mask_bce"]
            + self.decoder_config.mask_dice_weight * losses["mask_dice"]
            + self.decoder_config.geometry_weight * losses["geometry"]
            + self.consistency_weight * losses["consistency"]
            + self.decoder_config.quality_weight * losses["quality"]
        )
        output.language_loss = language_loss
        output.loss_components = losses
        output.loss = total
        return output

    def grounding_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu().contiguous()
            for name, value in self.state_dict().items()
            if not name.startswith("backbone.")
        }

    def save_grounding_pretrained(self, output_dir: str | Path) -> None:
        from safetensors.torch import save_file

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / GROUNDING_CONFIG_FILE).write_text(
            json.dumps(
                {
                    **asdict(self.decoder_config),
                    "feature_layers": list(self.decoder_config.feature_layers),
                    "mask_token_id": self.mask_token_id,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        save_file(self.grounding_state_dict(), output_dir / GROUNDING_WEIGHTS_FILE)

    def load_grounding_pretrained(self, weights_dir: str | Path) -> None:
        from safetensors.torch import load_file

        state = load_file(str(Path(weights_dir) / GROUNDING_WEIGHTS_FILE))
        missing, unexpected = self.load_state_dict(state, strict=False)
        missing_aux = [name for name in missing if not name.startswith("backbone.")]
        if missing_aux or unexpected:
            raise ValueError(
                f"invalid grounding weights; missing={missing_aux}, unexpected={unexpected}"
            )


def load_grounding_config(path: str | Path) -> tuple[GroundedDecoderConfig, int]:
    value = json.loads((Path(path) / GROUNDING_CONFIG_FILE).read_text(encoding="utf-8"))
    mask_token_id = int(value.pop("mask_token_id"))
    return GroundedDecoderConfig.from_mapping(value), mask_token_id
