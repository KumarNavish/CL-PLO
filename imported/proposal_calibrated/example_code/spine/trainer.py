"""Training utilities for continual adaptation (synthetic spine example).

Implements operators:

- (C) LoRA adapters: only adapter params are trainable.
- (B) Replay distillation: penalize deviation from cached predictions on memory.
- (A) A-GEM projection: if the update would increase stress loss, project the gradient.

The goal is to provide a clean, testable implementation that matches the document's notation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from .replay import BufferKind, ReplayBuffer


@dataclass(frozen=True)
class TrainConfig:
    """Config for pretraining and continual updates."""

    pretrain_epochs: int = 15
    pretrain_lr: float = 1e-3

    online_lr: float = 2e-3
    online_steps_per_day: int = 1

    replay_batch_size: int = 1024
    stress_batch_size: int = 1024

    distill_lambda: float = 5.0
    use_agem: bool = True


def _flatten_grads(params: Tuple[nn.Parameter, ...]) -> torch.Tensor:
    vecs = []
    for p in params:
        if p.grad is None:
            vecs.append(torch.zeros_like(p).reshape(-1))
        else:
            vecs.append(p.grad.detach().reshape(-1))
    return torch.cat(vecs, dim=0)


def _assign_flat_grads(params: Tuple[nn.Parameter, ...], flat: torch.Tensor) -> None:
    offset = 0
    for p in params:
        n = p.numel()
        g = flat[offset : offset + n].reshape_as(p).detach()
        if p.grad is None:
            p.grad = g.clone()
        else:
            p.grad.copy_(g)
        offset += n


def agem_project(g_cur: torch.Tensor, g_ref: torch.Tensor) -> torch.Tensor:
    """A-GEM projection onto the half-space <g, g_ref> >= 0.

    Args:
        g_cur: flattened gradient for the current update.
        g_ref: flattened gradient computed on a reference (stress) batch.

    Returns:
        Projected gradient.
    """

    dot = torch.dot(g_cur, g_ref)
    if dot >= 0:
        return g_cur
    denom = torch.dot(g_ref, g_ref).clamp_min(1e-12)
    return g_cur - (dot / denom) * g_ref


def pretrain_base_model(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: TrainConfig,
) -> None:
    """Supervised pretraining on an initial regime.

    Args:
        model: BaseRegressor (backbone + head), trainable.
        x: (M, P) training features.
        y: (M,) labels.
        cfg: training config.
    """

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.pretrain_lr)
    loss_fn = nn.MSELoss()

    for _ in range(cfg.pretrain_epochs):
        opt.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()


def continual_update(
    model: nn.Module,
    x_cur: torch.Tensor,
    y_cur: torch.Tensor,
    buffer: ReplayBuffer,
    cfg: TrainConfig,
) -> None:
    """Perform an online update step on the LoRA adapter.

    The update implements:
    - current slice supervised loss
    - memory distillation loss
    - optional A-GEM projection against a stress reference batch

    Args:
        model: LoRARegressor
        x_cur: (B, P)
        y_cur: (B,)
        buffer: replay buffer
        cfg: training config
    """

    model.train()

    # Only LoRA params should be trainable.
    params = tuple(p for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(params, lr=cfg.online_lr)
    loss_fn = nn.MSELoss()

    for _ in range(cfg.online_steps_per_day):
        opt.zero_grad(set_to_none=True)

        pred_cur = model(x_cur)
        loss_cur = loss_fn(pred_cur, y_cur)

        loss = loss_cur

        # Distillation replay (DER++ style, using cached old outputs)
        if cfg.distill_lambda > 0.0 and cfg.replay_batch_size > 0:
            replay = buffer.sample(cfg.replay_batch_size, kind=None)
            if replay is not None and replay.x.shape[0] > 0:
                pred_mem = model(replay.x)
                loss_distill = loss_fn(pred_mem, replay.y_old)
                loss = loss + cfg.distill_lambda * loss_distill

        # Compute gradients for the combined objective
        loss.backward()
        g = _flatten_grads(params)

        if cfg.use_agem and cfg.stress_batch_size > 0 and buffer.stress_size() > 0:
            # Compute stress reference gradient
            opt.zero_grad(set_to_none=True)
            stress = buffer.sample(cfg.stress_batch_size, kind=BufferKind.STRESS)
            if stress is not None and stress.x.shape[0] > 0:
                pred_stress = model(stress.x)
                loss_stress = loss_fn(pred_stress, stress.y)  # supervised stress regression
                loss_stress.backward()
                g_ref = _flatten_grads(params)

                # Project g onto the feasible half-space.
                g_proj = agem_project(g, g_ref)
                _assign_flat_grads(params, g_proj)

        opt.step()
