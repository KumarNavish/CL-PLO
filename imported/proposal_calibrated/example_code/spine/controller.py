"""Update gating and rollback controller (synthetic spine example).

The document emphasizes that continual learning is only deployable when updates are:

- gated (offline regression tests),
- observable (metrics and monitoring),
- reversible (rollback / kill-switch).

This module implements a minimal version:

- After each candidate adapter update, evaluate stress regression loss on a fixed stress mini-suite.
- If stress loss increases beyond a tolerance, rollback the adapter parameters.

This is intentionally simple but matches the operational semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass(frozen=True)
class RollbackConfig:
    """Configuration for rollback gating."""

    stress_loss_tolerance: float = 0.02  # 2% relative degradation allowed


class RollbackController:
    """Tracks the last good model and rolls back on stress regression."""

    def __init__(self, cfg: RollbackConfig):
        self.cfg = cfg
        self._best_state: Optional[dict] = None
        self._best_stress_loss: Optional[float] = None
        self.n_rollbacks: int = 0

    @torch.no_grad()
    def snapshot(self, model: nn.Module) -> None:
        """Record current model as the latest 'known-good' state."""

        self._best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def evaluate_stress_loss(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute stress MSE loss."""

        model.eval()
        pred = model(x)
        return float(torch.mean((pred - y) ** 2).item())

    @torch.no_grad()
    def gate_or_rollback(self, model: nn.Module, stress_loss: float) -> bool:
        """Apply gate; rollback if regression exceeds tolerance.

        Args:
            model: current candidate model
            stress_loss: computed stress loss

        Returns:
            True if kept (passed gate), False if rolled back.
        """

        if self._best_state is None:
            # First snapshot is always accepted.
            self._best_stress_loss = stress_loss
            self.snapshot(model)
            return True

        assert self._best_stress_loss is not None
        rel = (stress_loss - self._best_stress_loss) / (abs(self._best_stress_loss) + 1e-12)
        if rel > self.cfg.stress_loss_tolerance:
            # Rollback
            model.load_state_dict(self._best_state)
            self.n_rollbacks += 1
            return False

        # Accept new state as best
        self._best_stress_loss = stress_loss
        self.snapshot(model)
        return True
