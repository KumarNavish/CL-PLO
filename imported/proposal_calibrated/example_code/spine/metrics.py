"""Metrics and monitoring quantities for the spine example."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation without scipy.

    Args:
        x: 1D array
        y: 1D array

    Returns:
        Spearman correlation in [-1,1]. Returns nan if degenerate.
    """

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have same length")

    n = x.shape[0]
    if n < 2:
        return float("nan")

    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx = (rx - rx.mean()) / (rx.std(ddof=1) + 1e-12)
    ry = (ry - ry.mean()) / (ry.std(ddof=1) + 1e-12)
    return float(np.mean(rx * ry))


def sharpe(daily_returns: np.ndarray, ann_factor: float = 252.0) -> float:
    """Annualized Sharpe ratio."""

    mu = float(np.mean(daily_returns))
    sd = float(np.std(daily_returns, ddof=1))
    if sd <= 1e-12:
        return float("nan")
    return (mu / sd) * float(np.sqrt(ann_factor))


def max_drawdown(cum_curve: np.ndarray) -> float:
    """Max drawdown of a cumulative return curve (in log or simple space)."""

    peak = -np.inf
    mdd = 0.0
    for v in cum_curve:
        peak = max(peak, v)
        mdd = min(mdd, v - peak)
    return float(mdd)


@dataclass(frozen=True)
class RunSummary:
    """Aggregated run metrics."""

    mean_ic: float
    sharpe: float
    max_drawdown: float
    mean_turnover: float
    mean_stress_es: float
    n_rollbacks: int


def summarize_run(
    ics: np.ndarray,
    pnl: np.ndarray,
    turnovers: np.ndarray,
    stress_es: np.ndarray,
    n_rollbacks: int,
) -> RunSummary:
    """Summarize a backtest run."""

    cum = np.cumsum(pnl)
    return RunSummary(
        mean_ic=float(np.nanmean(ics)),
        sharpe=sharpe(pnl),
        max_drawdown=max_drawdown(cum),
        mean_turnover=float(np.nanmean(turnovers)),
        mean_stress_es=float(np.nanmean(stress_es)),
        n_rollbacks=int(n_rollbacks),
    )
