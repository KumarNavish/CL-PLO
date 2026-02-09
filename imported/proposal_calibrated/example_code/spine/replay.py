"""Replay memory for the spine example.

In the document, the replay memory is:

    M_t = M_recent U (U_j M_stress(j)) U M_edge

Here we implement a simple partitioned FIFO buffer. Each stored item contains:

- features x
- label y (residual return)
- cached old prediction y_hat_old (to implement distillation / trust region)

This keeps replay cheap and makes the stabilizer (operator B) deployment-friendly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import torch


class BufferKind(str, Enum):
    """Types of replay memory."""

    RECENT = "recent"
    STRESS = "stress"
    EDGE = "edge"


@dataclass
class ReplayBatch:
    """A sampled replay batch."""

    x: torch.Tensor
    y: torch.Tensor
    y_old: torch.Tensor


class ReplayBuffer:
    """Partitioned replay buffer with FIFO eviction."""

    def __init__(
        self,
        max_recent: int = 30_000,
        max_stress: int = 20_000,
        max_edge: int = 10_000,
        device: torch.device | str = "cpu",
    ):
        self.max_sizes: Dict[BufferKind, int] = {
            BufferKind.RECENT: int(max_recent),
            BufferKind.STRESS: int(max_stress),
            BufferKind.EDGE: int(max_edge),
        }
        self.device = torch.device(device)

        self._store: Dict[BufferKind, Dict[str, torch.Tensor]] = {
            k: {"x": torch.empty((0, 1)), "y": torch.empty((0,)), "y_old": torch.empty((0,))}
            for k in BufferKind
        }

    def __len__(self) -> int:
        return int(sum(self._store[k]["y"].numel() for k in BufferKind))

    def add(self, x: torch.Tensor, y: torch.Tensor, y_old: torch.Tensor, kind: BufferKind) -> None:
        """Add samples to the buffer (FIFO eviction).

        Args:
            x: (B, P)
            y: (B,)
            y_old: (B,)
            kind: buffer partition
        """

        x = x.detach().to(self.device)
        y = y.detach().to(self.device)
        y_old = y_old.detach().to(self.device)

        if x.ndim != 2:
            raise ValueError("x must be (B,P)")
        if y.ndim != 1 or y_old.ndim != 1:
            raise ValueError("y and y_old must be (B,)")
        if y.shape[0] != x.shape[0] or y_old.shape[0] != x.shape[0]:
            raise ValueError("batch size mismatch")

        cur = self._store[kind]
        if cur["x"].numel() == 0:
            cur["x"] = x
            cur["y"] = y
            cur["y_old"] = y_old
        else:
            cur["x"] = torch.cat([cur["x"], x], dim=0)
            cur["y"] = torch.cat([cur["y"], y], dim=0)
            cur["y_old"] = torch.cat([cur["y_old"], y_old], dim=0)

        # FIFO eviction
        max_size = self.max_sizes[kind]
        excess = cur["y"].shape[0] - max_size
        if excess > 0:
            cur["x"] = cur["x"][excess:]
            cur["y"] = cur["y"][excess:]
            cur["y_old"] = cur["y_old"][excess:]

    def sample(self, batch_size: int, kind: Optional[BufferKind] = None) -> Optional[ReplayBatch]:
        """Sample a batch uniformly.

        If kind is None, samples proportionally from all partitions.
        Returns None if insufficient data.
        """

        if len(self) == 0:
            return None

        if kind is not None:
            pool = self._store[kind]
            n = pool["y"].shape[0]
            if n == 0:
                return None
            idx = torch.randint(0, n, (batch_size,), device=self.device)
            return ReplayBatch(x=pool["x"][idx], y=pool["y"][idx], y_old=pool["y_old"][idx])

        # Mixed sampling
        # Build a concatenated view (cheap at these sizes)
        xs = []
        ys = []
        yolds = []
        for k in BufferKind:
            pool = self._store[k]
            if pool["y"].shape[0] > 0:
                xs.append(pool["x"])
                ys.append(pool["y"])
                yolds.append(pool["y_old"])

        X = torch.cat(xs, dim=0)
        Y = torch.cat(ys, dim=0)
        Yold = torch.cat(yolds, dim=0)

        n = Y.shape[0]
        idx = torch.randint(0, n, (batch_size,), device=self.device)
        return ReplayBatch(x=X[idx], y=Y[idx], y_old=Yold[idx])

    def stress_size(self) -> int:
        return int(self._store[BufferKind.STRESS]["y"].shape[0])
