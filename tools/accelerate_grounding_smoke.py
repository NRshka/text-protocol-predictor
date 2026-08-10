#!/usr/bin/env python3
"""Two-process CPU smoke test for the composite grounded model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from text_render_protocol_predictor.models import (  # noqa: E402
    GroundedDecoderConfig,
    GroundedQwen3VL,
)


class _Block(torch.nn.Module):
    def forward(self, value: torch.Tensor, **_kwargs) -> torch.Tensor:
        return value + 0.01


class _Visual(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block() for _ in range(4)])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            value = block(value)
        return value


class _Language(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(16)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor]:
        return (self.norm(value),)


class _Backbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = _Visual()
        self.language = _Language()
        self.model = SimpleNamespace(
            visual=self.visual,
            language_model=self.language,
        )
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

    def forward(self, input_ids, pixel_values, **_kwargs):
        self.visual(pixel_values)
        hidden = self.language(self.embeddings(input_ids))[0]
        return SimpleNamespace(
            loss=hidden.square().mean(),
            logits=self.lm_head(hidden),
        )


def main() -> None:
    from accelerate import Accelerator

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-processes", type=int, default=2)
    args = parser.parse_args()
    accelerator = Accelerator(cpu=True)
    if accelerator.num_processes != args.expected_processes:
        raise RuntimeError(
            f"expected {args.expected_processes} processes, got "
            f"{accelerator.num_processes}"
        )
    model = GroundedQwen3VL(
        _Backbone(),
        mask_token_id=7,
        decoder_config=GroundedDecoderConfig(
            feature_layers=(0, 1, 2, 3),
            feature_projection_dim=8,
            pixel_embedding_dim=32,
            query_hidden_dim=32,
        ),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    model, optimizer = accelerator.prepare(model, optimizer)
    device = accelerator.device
    output = model(
        input_ids=torch.tensor([[1, 7, 2]], device=device),
        pixel_values=torch.randn(4, 8, device=device),
        image_grid_thw=torch.tensor([[1, 2, 2]], device=device),
        mask_token_positions=torch.tensor([[1]], device=device),
        instance_valid=torch.tensor([[True]], device=device),
        instance_masks=[torch.ones(1, 8, 8, device=device)],
        mask_supervision=[torch.tensor([True], device=device)],
        geometry_modes=[torch.tensor([0], device=device)],
        geometry_boxes=[torch.tensor([[0.5, 0.5, 0.5, 0.25]], device=device)],
        geometry_rotations=[torch.tensor([[0.0, 1.0]], device=device)],
        geometry_beziers=[torch.zeros(1, 8, device=device)],
    )
    accelerator.backward(output.loss)
    optimizer.step()
    gathered = accelerator.gather(output.loss.detach().reshape(1))
    if accelerator.is_main_process:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "processes": accelerator.num_processes,
                    "finite_losses": int(torch.isfinite(gathered).sum().item()),
                }
            )
        )


if __name__ == "__main__":
    main()
