"""Signal head: frozen base + LoRA adapter (synthetic spine example).

In the manuscript, the learnable object that flows into the optimizer is the alpha
vector μ_t. Production implementations typically compute μ_t from:

  - a frozen Transformer/LLM backbone that turns tokenized text into a stable context
    representation z_t, and
  - a small, modular head on top of (x_t, z_t) that can be updated and rolled back
    cheaply.

The validation experiment in this repo follows that pattern:
  - the tiny Transformer LM that produces z_t lives in `tiny_llm.py` and is frozen
    during trading/continual updates,
  - this module implements the small regression head whose *only* trainable
    parameters during continual learning are low-rank LoRA factors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the synthetic signal model."""

    input_dim: int
    hidden_dim: int = 64
    rank: int = 8
    lora_alpha: float = 8.0


class MLPBackbone(nn.Module):
    """Small projection network used by the signal head.

    In a production LLM-driven stack, the "backbone" would be the frozen
    Transformer. Here, the Transformer text backbone is handled separately
    (see `tiny_llm.py`) and its embedding is appended to numeric features
    before reaching this head.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LoRALinear(nn.Module):
    """Linear layer with a frozen base weight + trainable low-rank adapter.

    Effective weight:
        W = W0 + (alpha/r) * (B @ A)

    where W0 is frozen, and A, B are trainable.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        lora_alpha: float,
        base_weight: torch.Tensor,
        base_bias: torch.Tensor,
    ):
        super().__init__()
        if base_weight.shape != (out_features, in_features):
            raise ValueError("base_weight shape mismatch")
        if base_bias.shape != (out_features,):
            raise ValueError("base_bias shape mismatch")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = float(lora_alpha) / float(rank)

        # Frozen base parameters
        self.weight_base = nn.Parameter(base_weight.clone().detach(), requires_grad=False)
        self.bias_base = nn.Parameter(base_bias.clone().detach(), requires_grad=False)

        # Trainable low-rank factors (initialized near zero so behavior starts unchanged)
        self.A = nn.Parameter(torch.zeros(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.normal_(self.A, mean=0.0, std=0.02)
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = x @ self.weight_base.t() + self.bias_base
        delta = (x @ self.A.t()) @ self.B.t() * self.scaling
        return base + delta


class BaseRegressor(nn.Module):
    """Backbone + standard linear head (used for initial pretraining)."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.backbone = MLPBackbone(input_dim=input_dim, hidden_dim=hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.head(h).squeeze(-1)


class LoRARegressor(nn.Module):
    """Frozen backbone + frozen base head + trainable LoRA adapter."""

    def __init__(self, backbone: nn.Module, lora_head: LoRALinear):
        super().__init__()
        self.backbone = backbone
        self.lora_head = lora_head

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

    @classmethod
    def from_pretrained(cls, base: BaseRegressor, cfg: ModelConfig) -> "LoRARegressor":
        """Create a LoRARegressor from a pretrained BaseRegressor."""

        backbone = base.backbone
        with torch.no_grad():
            base_w = base.head.weight.data.clone()
            base_b = base.head.bias.data.clone()

        lora_head = LoRALinear(
            in_features=base_w.shape[1],
            out_features=base_w.shape[0],
            rank=cfg.rank,
            lora_alpha=cfg.lora_alpha,
            base_weight=base_w,
            base_bias=base_b,
        )
        return cls(backbone=backbone, lora_head=lora_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.lora_head(h).squeeze(-1)

    def lora_parameters(self) -> Tuple[nn.Parameter, ...]:
        """Return only the trainable parameters (LoRA A and B)."""

        return (self.lora_head.A, self.lora_head.B)
