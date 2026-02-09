"""Rolling factor risk model (synthetic spine example).

This implements the portfolio primitive \Sigma_t used by the decision layer:

    \Sigma_t = B_t Cov_t(f) B_t^T + D_t

where
- f_t are observed factor returns,
- B_t are per-asset factor exposures estimated by rolling OLS,
- D_t are idiosyncratic residual variances from the same regressions.

The implementation is vectorized across assets and uses a small ridge term for numerical stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class RiskModelConfig:
    """Configuration for the rolling factor risk model."""

    window: int = 252
    ridge: float = 1e-6


class RollingFactorRiskModel:
    """Estimate a factor risk model on a rolling window."""

    def __init__(self, cfg: RiskModelConfig):
        self.cfg = cfg

    def estimate(self, t_end_exclusive: int, returns: np.ndarray, factor_returns: np.ndarray) -> Dict[str, np.ndarray]:
        """Estimate B_t, factor covariance, D_t, and Sigma_t.

        Args:
            t_end_exclusive: Index t such that the window is [t-window, t).
            returns: (T, N) array of asset returns.
            factor_returns: (T, K) array of factor returns.

        Returns:
            Dict with keys:
                - B: (N, K)
                - F_cov: (K, K)
                - D: (N,) idiosyncratic variances
                - Sigma: (N, N)

        Raises:
            ValueError: if not enough history is available.
        """

        T, N = returns.shape
        Tf, K = factor_returns.shape
        if Tf != T:
            raise ValueError("returns and factor_returns must have the same length")

        H = self.cfg.window
        if t_end_exclusive < H:
            raise ValueError(f"Need at least {H} days of history, got t_end_exclusive={t_end_exclusive}")

        Rw = returns[t_end_exclusive - H : t_end_exclusive]  # (H, N)
        Fw = factor_returns[t_end_exclusive - H : t_end_exclusive]  # (H, K)

        # Ridge OLS: B_hat_T = (F^T F + ridge I)^{-1} F^T R
        XtX = Fw.T @ Fw + self.cfg.ridge * np.eye(K)
        XtY = Fw.T @ Rw  # (K, N)
        B_hat_T = np.linalg.solve(XtX, XtY)  # (K, N)
        B = B_hat_T.T  # (N, K)

        # Residuals and idiosyncratic variances
        Ew = Rw - Fw @ B_hat_T  # (H, N)
        D = Ew.var(axis=0, ddof=1)

        # Factor covariance
        F_cov = np.cov(Fw, rowvar=False, ddof=1)

        Sigma = B @ F_cov @ B.T + np.diag(D)

        return {"B": B, "F_cov": F_cov, "D": D, "Sigma": Sigma}


def residualize_next_day(
    B_t: np.ndarray,
    f_next: np.ndarray,
    r_next: np.ndarray,
) -> np.ndarray:
    """Compute the residual label y_t = r_{t+1} - B_t f_{t+1}.

    Args:
        B_t: (N, K) exposures estimated at time t.
        f_next: (K,) factor return at t+1.
        r_next: (N,) realized asset returns at t+1.

    Returns:
        y: (N,) residual returns.
    """

    return r_next - B_t @ f_next
