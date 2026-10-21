"""Feature construction for the synthetic spine example.

In the document, the streaming slice is (x_t, z_t, r_{t+1}).
Here we implement x_t for equities as a minimal, realistic set of numeric features
constructed from lagged returns:

- Momentum (mean of last 20 days)
- Reversal (negative mean of last 5 days)
- Realized volatility (std of last 20 days)
- Market beta (static exposure, included to emulate readily available risk features)
- Intercept

The output is a 3D array x[t, i, :] suitable for cross-sectional learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for lagged-return features."""

    win_mom: int = 20
    win_rev: int = 5
    win_vol: int = 20


class FeatureBuilder:
    """Builds per-asset daily features from lagged returns."""

    def __init__(self, cfg: FeatureConfig):
        self.cfg = cfg

    def build(self, returns: np.ndarray, betas: np.ndarray) -> Dict[str, np.ndarray]:
        """Build features.

        Args:
            returns: Array of shape (T, N) with daily asset returns.
            betas: Array of shape (N, K) with factor exposures; only betas[:,0] is used.

        Returns:
            Dict with:
                - x: (T, N, P) features with NaNs for t < max_window
                - valid_mask: (T,) bool, True if features for day t are valid
        """

        T, N = returns.shape
        assert betas.shape[0] == N

        w_m = self.cfg.win_mom
        w_r = self.cfg.win_rev
        w_v = self.cfg.win_vol
        t0 = max(w_m, w_r, w_v)

        # Feature dimension: mom, rev, vol, beta_mkt, intercept
        P = 5
        x = np.full((T, N, P), np.nan, dtype=float)

        for t in range(t0, T):
            past = returns[:t]
            mom = past[t - w_m : t].mean(axis=0)
            rev = -past[t - w_r : t].mean(axis=0)
            vol = past[t - w_v : t].std(axis=0, ddof=1)
            beta_mkt = betas[:, 0]
            intercept = np.ones(N, dtype=float)

            x[t, :, 0] = mom
            x[t, :, 1] = rev
            x[t, :, 2] = vol
            x[t, :, 3] = beta_mkt
            x[t, :, 4] = intercept

        valid_mask = ~np.isnan(x[:, 0, 0])
        return {"x": x, "valid_mask": valid_mask}
