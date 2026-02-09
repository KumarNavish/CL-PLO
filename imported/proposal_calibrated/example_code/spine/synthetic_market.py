"""Synthetic regime-switching market for the spine example.

The goal is to create a *faithful* toy version of the real setting:

- A multi-factor equity universe with time-varying factor covariance (regimes).
- Regime-dependent alpha structure (the prediction task drifts).
- Explicit stress episodes (rare-but-critical) that must be preserved via replay.

All outputs are numpy arrays to keep the simulator independent of PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SyntheticMarketConfig:
    """Configuration for the regime-switching synthetic market."""

    seed: int = 7
    n_assets: int = 200
    n_factors: int = 3
    n_regimes: int = 3  # 0=low vol, 1=high vol, 2=crisis
    t_total: int = 1250

    # Feature windows (days)
    win_mom: int = 20
    win_rev: int = 5
    win_vol: int = 20

    # Factor covariance scale per regime (daily)
    factor_vol_scale: Tuple[float, float, float] = (0.008, 0.014, 0.030)

    # Idiosyncratic residual vol per regime (daily)
    idio_vol: Tuple[float, float, float] = (0.012, 0.020, 0.040)

    # Regime transition probabilities (Markov)
    # Rows sum to 1.
    transition: Tuple[Tuple[float, float, float], ...] = (
        (0.97, 0.03, 0.00),
        (0.04, 0.95, 0.01),
        (0.05, 0.10, 0.85),
    )

    # Forced stress windows: list of (start, length) indices.
    # These are set to regime=2 (crisis) deterministically.
    forced_crisis: Tuple[Tuple[int, int], ...] = ((520, 25), (900, 20))

    # --- Synthetic text stream (LLM testbed) ---
    # The manuscript assumes the signal model consumes (x_t, z_t), where z_t
    # is derived from tokenized text (news, filings, internal logs). The
    # proof-of-concept therefore includes a minimal token stream whose
    # distribution shifts with regimes. A tiny Transformer LM is pretrained on
    # these tokens and frozen; its embedding is the z_t used by the signal head.
    vocab_size: int = 64
    text_seq_len: int = 24


def _make_regime_factor_cov(cfg: SyntheticMarketConfig) -> np.ndarray:
    """Create regime-dependent factor covariance matrices.

    We keep a simple structure: market factor is correlated with others, and
    correlations increase in high-vol/crisis regimes.

    Returns:
        covs: array of shape (n_regimes, n_factors, n_factors)
    """

    K = cfg.n_factors
    assert K >= 1

    base_corr = np.eye(K)
    if K >= 2:
        # Mild correlation structure
        base_corr[0, 1] = base_corr[1, 0] = 0.25
    if K >= 3:
        base_corr[0, 2] = base_corr[2, 0] = -0.10
        base_corr[1, 2] = base_corr[2, 1] = 0.15

    covs = np.zeros((cfg.n_regimes, K, K), dtype=float)
    for s in range(cfg.n_regimes):
        # Correlation strength increases with s.
        corr = base_corr.copy()
        if s == 1 and K >= 2:
            corr[0, 1] = corr[1, 0] = 0.45
        if s == 2 and K >= 2:
            corr[0, 1] = corr[1, 0] = 0.70
        if s == 2 and K >= 3:
            corr[0, 2] = corr[2, 0] = -0.25
            corr[1, 2] = corr[2, 1] = 0.35

        vols = np.full(K, cfg.factor_vol_scale[s], dtype=float)
        cov = np.outer(vols, vols) * corr
        covs[s] = cov

    return covs


def _simulate_regimes(cfg: SyntheticMarketConfig, rng: np.random.Generator) -> np.ndarray:
    """Simulate regime path with optional forced crisis windows."""

    T = cfg.t_total
    P = np.array(cfg.transition, dtype=float)
    assert P.shape == (cfg.n_regimes, cfg.n_regimes)

    regimes = np.zeros(T, dtype=int)
    regimes[0] = 0
    for t in range(1, T):
        regimes[t] = rng.choice(cfg.n_regimes, p=P[regimes[t - 1]])

    # Force deterministic crisis windows (rare-but-critical stress episodes).
    for start, length in cfg.forced_crisis:
        end = min(T, start + length)
        regimes[start:end] = 2

    return regimes


def _simulate_text_tokens(
    cfg: SyntheticMarketConfig, regimes: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Simulate a per-day token sequence whose distribution shifts with regimes.

    This is not meant to be a realistic language model dataset. It is a minimal,
    controllable stream that (i) is genuinely token-based, and (ii) carries a
    regime-dependent signature that a tiny Transformer LM can learn.

    Returns:
        tokens: (T, L) int32, token ids in [0, vocab_size)
    """

    T = regimes.shape[0]
    V = int(cfg.vocab_size)
    L = int(cfg.text_seq_len)
    if V < 16:
        raise ValueError("vocab_size too small")
    if L < 8:
        raise ValueError("text_seq_len too small")

    # Carve the vocabulary into regime-specific bands plus a small shared band.
    #  - shared tokens: common boilerplate / background
    #  - regime band: regime-specific "topic" vocabulary
    shared = np.arange(max(0, V - 8), V, dtype=int)
    bands = {
        0: np.arange(0, V // 3, dtype=int),
        1: np.arange(V // 3, 2 * V // 3, dtype=int),
        2: np.arange(2 * V // 3, max(2 * V // 3 + 1, V - 8), dtype=int),
    }

    tokens = np.zeros((T, L), dtype=np.int32)
    for t in range(T):
        s = int(regimes[t])
        band = bands[s]

        # Start each sequence with a small set of shared tokens.
        seq = []
        seq.extend(rng.choice(shared, size=3, replace=True).tolist())

        # Then emit regime-skewed content.
        # Crisis has more repetition (headlines echo), so we skew toward
        # fewer unique tokens when s=2.
        if s == 2:
            core = rng.choice(band, size=L - len(seq), replace=True, p=None)
        else:
            core = rng.choice(band, size=L - len(seq), replace=True)
        seq.extend(core.tolist())

        tokens[t] = np.asarray(seq[:L], dtype=np.int32)

        # Add weak temporal dependence to make next-token prediction non-trivial:
        # copy a prefix from yesterday with small probability.
        if t > 0 and rng.uniform() < 0.25:
            k = rng.integers(low=2, high=min(8, L))
            tokens[t, :k] = tokens[t - 1, :k]

    return tokens


def simulate_synthetic_market(cfg: SyntheticMarketConfig) -> Dict[str, np.ndarray]:
    """Simulate a regime-switching multi-factor market.

    Output keys:
        - regimes: (T,) int
        - factor_returns: (T, K)
        - betas: (N, K) factor exposures
        - returns: (T, N) total returns
        - residuals: (T, N) idiosyncratic (alpha) residuals (what the model predicts)
        - text_tokens: (T, L) int token ids (synthetic "news" stream)

    Notes:
        - Residuals are generated from regime-dependent feature->alpha mappings.
        - Features themselves are computed from lagged returns; see `features.py`.
    """

    rng = np.random.default_rng(cfg.seed)

    N, K, T = cfg.n_assets, cfg.n_factors, cfg.t_total

    # Factor covariances by regime
    covs = _make_regime_factor_cov(cfg)

    # Regime path
    regimes = _simulate_regimes(cfg, rng)

    # Constant factor exposures (can be estimated later by rolling OLS)
    betas = rng.normal(loc=0.0, scale=0.8, size=(N, K)).astype(float)
    # Ensure market beta roughly positive (realistic)
    betas[:, 0] = np.abs(betas[:, 0]) + 0.2

    factor_returns = np.zeros((T, K), dtype=float)
    returns = np.zeros((T, N), dtype=float)
    residuals = np.zeros((T, N), dtype=float)

    # Synthetic tokenized text stream (shared across assets each day)
    text_tokens = _simulate_text_tokens(cfg, regimes, rng)

    # Regime-dependent alpha coefficients for features [mom, rev, vol]
    # Scaled to generate daily alpha on the order of tens of bps.
    alpha_coef = {
        0: np.array([+0.9, -0.3, -0.4], dtype=float),  # low-vol: momentum works
        1: np.array([-0.4, +0.9, -0.3], dtype=float),  # high-vol: reversal dominates
        2: np.array([+0.0, +0.2, -1.1], dtype=float),  # crisis: prefer low vol
    }

    # Bootstrap initial returns with small noise (no alpha yet)
    init = max(cfg.win_mom, cfg.win_vol, cfg.win_rev) + 1
    for t in range(min(init, T)):
        s = regimes[t]
        factor_returns[t] = rng.multivariate_normal(mean=np.zeros(K), cov=covs[s])
        eps = rng.normal(0.0, cfg.idio_vol[s], size=N)
        residuals[t] = eps
        returns[t] = betas @ factor_returns[t] + residuals[t]

    # Main simulation: sequentially generate alpha based on lagged returns.
    for t in range(init, T):
        s = regimes[t]

        # Factor returns (systematic)
        mean_f = np.zeros(K)
        if s == 2:
            # Crisis: negative market drift to create stress signature.
            mean_f[0] = -0.004
        factor_returns[t] = rng.multivariate_normal(mean=mean_f, cov=covs[s])

        # Lagged feature signals from past returns
        past = returns[:t]
        mom = past[t - cfg.win_mom : t].mean(axis=0)
        rev = -past[t - cfg.win_rev : t].mean(axis=0)
        vol = past[t - cfg.win_vol : t].std(axis=0, ddof=1)

        feats = np.stack([mom, rev, vol], axis=1)  # (N, 3)

        # Regime-dependent alpha mean
        alpha_mean = feats @ alpha_coef[int(s)]

        # Residual noise
        eps = rng.normal(0.0, cfg.idio_vol[s], size=N)

        residuals[t] = alpha_mean + eps
        returns[t] = betas @ factor_returns[t] + residuals[t]

    return {
        "regimes": regimes,
        "factor_returns": factor_returns,
        "betas": betas,
        "returns": returns,
        "residuals": residuals,
        "text_tokens": text_tokens,
    }
