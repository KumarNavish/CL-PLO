"""Deterministic constrained decision layer (synthetic spine example).

This module corresponds to the paper's key production principle:

> Keep hard constraints outside the model.

Given:
- predicted alphas mu_t (from the signal model)
- a risk model Sigma_t (factor risk model)
- constraint parameters (market neutrality, leverage, box limits, stress ES budget)

we produce feasible portfolio weights w_t.

To keep dependencies minimal (no cvxpy), we implement a simple solver:
1) solve a mean-variance quadratic program with equality constraints via KKT
2) enforce leverage/box constraints by alternating projection + rescaling
3) enforce a stress ES budget by scaling exposure if needed

This is sufficient to validate the end-to-end workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DecisionLayerConfig:
    """Configuration for the decision layer."""

    gamma: float = 50.0
    leverage_l1: float = 1.0
    w_max: float = 0.05
    tc_per_turnover: float = 5e-4  # 5 bps per unit turnover

    # Equality neutrality constraints
    enforce_dollar_neutral: bool = True
    enforce_market_neutral: bool = True

    # Stress control
    es_alpha: float = 0.2
    es_budget: float = 7e-4  # e.g., 7 bps daily ES budget under stress

    # Numerical stability
    sigma_ridge: float = 1e-8


def expected_shortfall(x: np.ndarray, alpha: float) -> float:
    """Compute Expected Shortfall (CVaR) at level alpha for 1D returns.

    Args:
        x: (S,) scenario returns (negative means loss).
        alpha: tail fraction (e.g., 0.2)

    Returns:
        ES_alpha: mean of the worst alpha fraction.
    """

    if x.ndim != 1:
        raise ValueError("x must be 1D")
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0,1]")

    S = x.shape[0]
    k = int(np.ceil(alpha * S))
    k = max(1, k)
    return float(np.mean(np.sort(x)[:k]))


def _project_nullspace(w: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Project w onto the nullspace of A (i.e., enforce A w = 0).

    Args:
        w: (N,)
        A: (m, N) full row rank

    Returns:
        w_proj with A w_proj = 0 (up to numerical tolerance).
    """

    if A.size == 0:
        return w

    # w_proj = w - A^T (A A^T)^{-1} A w
    At = A
    M = At @ At.T
    rhs = At @ w
    lam = np.linalg.solve(M, rhs)
    return w - At.T @ lam


class DecisionLayer:
    """Mean-variance decision layer with hard constraints."""

    def __init__(self, cfg: DecisionLayerConfig):
        self.cfg = cfg

    def solve(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        B: np.ndarray,
        w_prev: Optional[np.ndarray],
        stress_returns: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, float]:
        """Compute portfolio weights.

        Args:
            mu: (N,) predicted residual returns.
            Sigma: (N,N) covariance matrix.
            B: (N,K) factor exposures (only B[:,0] used for market neutrality).
            w_prev: (N,) previous weights (for turnover/cost accounting).
            stress_returns: Optional (S,N) matrix of scenario returns for ES budget.

        Returns:
            w: (N,) portfolio weights.
            turnover: L1 turnover vs w_prev (0 if w_prev is None).
            es: stress ES (nan if stress_returns is None).
        """

        N = mu.shape[0]
        if Sigma.shape != (N, N):
            raise ValueError("Sigma shape mismatch")
        if B.shape[0] != N:
            raise ValueError("B shape mismatch")

        # Build equality constraints A w = 0.
        A_rows = []
        if self.cfg.enforce_dollar_neutral:
            A_rows.append(np.ones(N, dtype=float))
        if self.cfg.enforce_market_neutral:
            A_rows.append(B[:, 0].astype(float))
        A = np.stack(A_rows, axis=0) if len(A_rows) > 0 else np.zeros((0, N), dtype=float)

        # KKT solve for equality-constrained mean-variance optimum
        Sigma_reg = Sigma + self.cfg.sigma_ridge * np.eye(N)
        m = A.shape[0]
        KKT = np.block(
            [
                [self.cfg.gamma * Sigma_reg, A.T],
                [A, np.zeros((m, m), dtype=float)],
            ]
        )
        rhs = np.concatenate([mu, np.zeros(m, dtype=float)], axis=0)
        sol = np.linalg.solve(KKT, rhs)
        w = sol[:N]

        # Enforce inequality constraints by alternating projection + clipping + scaling.
        for _ in range(6):
            if self.cfg.w_max is not None:
                w = np.clip(w, -self.cfg.w_max, self.cfg.w_max)
            w = _project_nullspace(w, A)
            l1 = float(np.sum(np.abs(w)))
            if l1 > self.cfg.leverage_l1 and l1 > 0:
                w = w * (self.cfg.leverage_l1 / l1)
            w = _project_nullspace(w, A)

        # Stress ES budget: scale down if violated.
        es = float("nan")
        if stress_returns is not None:
            p = stress_returns @ w
            es = expected_shortfall(p, alpha=self.cfg.es_alpha)
            if es < -self.cfg.es_budget and es < 0:
                scale = self.cfg.es_budget / abs(es)
                w = w * scale
                w = _project_nullspace(w, A)
                # Recompute ES after scaling
                p = stress_returns @ w
                es = expected_shortfall(p, alpha=self.cfg.es_alpha)

        turnover = 0.0
        if w_prev is not None:
            turnover = float(np.sum(np.abs(w - w_prev)))

        return w, turnover, es

    def transaction_cost(self, turnover: float) -> float:
        """Linear cost proxy."""

        return float(self.cfg.tc_per_turnover * turnover)
