from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from text_render_protocol_predictor.models import (
    GroundedDecoderConfig,
    GroundedQwen3VL,
    soft_rasterize_geometry,
    unshuffle_qwen_patches,
)
from text_render_protocol_predictor.models.grounded_qwen3_vl import load_grounding_config


def test_qwen_patch_unshuffle_restores_row_major_grid() -> None:
    block_major = torch.tensor(
        [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
    ).float().unsqueeze(1)

    restored = unshuffle_qwen_patches(
        block_major,
        (1, 4, 4),
        spatial_merge_size=2,
    )

    assert restored.shape == (1, 4, 4)
    assert torch.equal(restored[0], torch.arange(16).reshape(4, 4))


def test_soft_geometry_rasterizer_is_differentiable() -> None:
    mode = torch.tensor([[3.0, -3.0]], requires_grad=True)
    boxes = torch.tensor([[0.5, 0.5, 0.5, 0.25]], requires_grad=True)
    rotations = torch.tensor([[0.0, 1.0]], requires_grad=True)
    beziers = torch.tensor([[0.2, 0.5, 0.4, 0.5, 0.6, 0.5, 0.8, 0.5]], requires_grad=True)

    mask = soft_rasterize_geometry(
        mode_logits=mode,
        boxes=boxes,
        rotations=rotations,
        beziers=beziers,
        size=(16, 24),
    )
    mask.mean().backward()

    assert mask.shape == (1, 16, 24)
    assert boxes.grad is not None
    assert mode.grad is not None


class _IdentityBlock(torch.nn.Module):
    def forward(self, value, **kwargs):
        del kwargs
        return value + 0.01


class _FakeVisual(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_IdentityBlock() for _ in range(4)])
        self.spatial_merge_size = 2

    def forward(self, value):
        for block in self.blocks:
            value = block(value)
        return value


class _FakeLanguage(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(16)

    def forward(self, value):
        return (self.norm(value),)


class _FakeOutput:
    def __init__(self, loss, logits) -> None:
        self.loss = loss
        self.logits = logits


class _FakeBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = SimpleNamespace()
        self.model.visual = _FakeVisual()
        self.model.language_model = _FakeLanguage()
        self.add_module("visual", self.model.visual)
        self.add_module("language", self.model.language_model)
        self.embeddings = torch.nn.Embedding(32, 16)
        self.lm_head = torch.nn.Linear(16, 32)
        self.config = SimpleNamespace(
            vision_config=SimpleNamespace(
                patch_size=16,
                spatial_merge_size=2,
                hidden_size=8,
            ),
            text_config=SimpleNamespace(hidden_size=16),
        )

    def get_input_embeddings(self):
        return self.embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def save_pretrained(self, output_dir, **kwargs):
        del kwargs
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "backbone.marker").write_text("ok", encoding="utf-8")

    def forward(self, input_ids, pixel_values, **kwargs):
        del kwargs
        self.model.visual(pixel_values)
        hidden = self.model.language_model(self.embeddings(input_ids))[0]
        logits = self.lm_head(hidden)
        return _FakeOutput(hidden.square().mean(), logits)


def test_grounded_model_composes_losses_and_backpropagates() -> None:
    model = GroundedQwen3VL(
        _FakeBackbone(),
        mask_token_id=7,
        decoder_config=GroundedDecoderConfig(
            feature_layers=(0, 1, 2, 3),
            feature_projection_dim=8,
            pixel_embedding_dim=32,
            query_hidden_dim=32,
            consistency_weight=0.25,
        ),
    )
    input_ids = torch.tensor([[1, 2, 7, 3, 4]])
    result = model(
        input_ids=input_ids,
        pixel_values=torch.randn(4, 8),
        image_grid_thw=torch.tensor([[1, 2, 2]]),
        mask_token_positions=torch.tensor([[2]]),
        instance_valid=torch.tensor([[True]]),
        instance_masks=[torch.ones(1, 8, 8)],
        mask_supervision=[torch.tensor([True])],
        geometry_modes=[torch.tensor([0])],
        geometry_boxes=[torch.tensor([[0.5, 0.5, 0.5, 0.25]])],
        geometry_rotations=[torch.tensor([[0.0, 1.0]])],
        geometry_beziers=[torch.zeros(1, 8)],
    )
    result.loss.backward()

    assert result.mask_logits[0].shape == (1, 8, 8)
    assert torch.isfinite(result.loss)
    assert model.query_projection[-1].weight.grad is not None


def _tiny_grounded_model() -> GroundedQwen3VL:
    return GroundedQwen3VL(
        _FakeBackbone(),
        mask_token_id=7,
        decoder_config=GroundedDecoderConfig(
            feature_layers=(0, 1, 2, 3),
            feature_projection_dim=8,
            pixel_embedding_dim=32,
            query_hidden_dim=32,
            consistency_weight=0.0,
        ),
    )


def _tiny_batch() -> dict:
    return {
        "input_ids": torch.tensor([[1, 2, 7, 3, 4]]),
        "pixel_values": torch.arange(32, dtype=torch.float32).reshape(4, 8) / 32,
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
        "mask_token_positions": torch.tensor([[2]]),
        "instance_valid": torch.tensor([[True]]),
        "instance_masks": [torch.ones(1, 8, 8)],
        "mask_supervision": [torch.tensor([True])],
        "geometry_modes": [torch.tensor([0])],
        "geometry_boxes": [torch.tensor([[0.5, 0.5, 0.5, 0.25]])],
        "geometry_rotations": [torch.tensor([[0.0, 1.0]])],
        "geometry_beziers": [torch.zeros(1, 8)],
    }


def test_grounding_checkpoint_round_trip(tmp_path) -> None:
    model = _tiny_grounded_model()
    state = model.grounding_state_dict()
    assert state
    assert not any("backbone" in name or "_visual" in name for name in state)
    model.save_pretrained(tmp_path)
    config, token_id = load_grounding_config(tmp_path)
    restored = GroundedQwen3VL(
        _FakeBackbone(),
        mask_token_id=token_id,
        decoder_config=config,
    )
    restored.load_grounding_pretrained(tmp_path)

    assert token_id == 7
    assert (tmp_path / "backbone.marker").is_file()
    assert config.feature_layers == (0, 1, 2, 3)
    assert torch.equal(
        restored.query_projection[-1].weight,
        model.query_projection[-1].weight,
    )


def test_single_sample_grounding_heads_overfit() -> None:
    torch.manual_seed(4)
    model = _tiny_grounded_model()
    model.backbone.requires_grad_(False)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.02,
    )
    batch = _tiny_batch()
    losses = []
    for _ in range(35):
        result = model(**batch)
        auxiliary = sum(result.loss_components.values())
        losses.append(float(auxiliary.detach()))
        auxiliary.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    assert losses[-1] < losses[0] * 0.35


def test_zero_mask_supervision_keeps_geometry_loss_finite() -> None:
    model = _tiny_grounded_model()
    batch = _tiny_batch()
    batch["mask_supervision"] = [torch.tensor([False])]

    result = model(**batch)
    result.loss.backward()

    assert result.loss_components["mask_bce"].item() == 0.0
    assert result.loss_components["mask_dice"].item() == 0.0
    assert torch.isfinite(result.loss_components["geometry"])
    assert all(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.") and parameter.requires_grad
    )


def test_variable_image_and_instance_counts_produce_per_sample_shapes() -> None:
    model = _tiny_grounded_model()
    result = model(
        input_ids=torch.tensor([[1, 7, 2, 3], [7, 2, 7, 3]]),
        pixel_values=torch.randn(12, 8),
        image_grid_thw=torch.tensor([[1, 2, 2], [1, 2, 4]]),
        mask_token_positions=torch.tensor([[1, -1], [0, 2]]),
        instance_valid=torch.tensor([[True, False], [True, True]]),
        instance_masks=[torch.ones(1, 8, 8), torch.ones(2, 8, 16)],
        mask_supervision=[torch.tensor([False]), torch.tensor([False, False])],
        geometry_modes=[torch.tensor([0]), torch.tensor([0, 1])],
        geometry_boxes=[
            torch.tensor([[0.5, 0.5, 0.5, 0.25]]),
            torch.tensor([[0.4, 0.4, 0.2, 0.2], [0.6, 0.6, 0.3, 0.2]]),
        ],
        geometry_rotations=[
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        ],
        geometry_beziers=[torch.zeros(1, 8), torch.zeros(2, 8)],
    )

    assert [tuple(value.shape) for value in result.mask_logits] == [
        (1, 8, 8),
        (2, 8, 16),
    ]
    assert torch.isfinite(result.loss)


def test_grounded_training_phase_freezes_and_restores_backbone_trainables() -> None:
    model = _tiny_grounded_model()

    model.set_training_phase("heads_only")
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.")
    )

    model.set_training_phase("joint")
    assert all(parameter.requires_grad for parameter in model.backbone.parameters())
