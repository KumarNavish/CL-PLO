"""Minimal validation experiment for constraint-aligned PEFT under non-stationarity.

This script implements the smallest end-to-end loop that is finance-relevant:

- A frozen linear predictor ("backbone") is augmented with LoRA (low-rank) parameters.
- The environment exhibits regime drift (new mapping) and rare stress events.
- Naive LoRA adaptation fits drift but can silently regress on stress contexts.
- Anchoring (distillation to a deployed reference on a fixed stress suite) and
  A-GEM-style gradient projection reduce stress regression.

Outputs (written to ./figs/):
- validation_wide.pdf : 2-panel figure (adaptation/retention + equity curve)
- validation_wide.png : PNG copy
- metrics.tex         : LaTeX macros with key metrics for embedding in the write-up

The experiment is synthetic by design (fast, reproducible) but operationally faithful:
stress contexts are treated as invariants/anchors; deployment quality is measured both
in prediction space and in a simple rebalancing simulator.
"""

from __future__ import annotations

import dataclasses
import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class Config:
    """Experiment configuration."""

    seed: int = 0

    # Problem dimensions
    d_signal: int = 8
    n_assets_risky: int = 4
    n_assets_cash: int = 1

    # Regime stream
    n_train_base: int = 4000
    n_train_drift: int = 2000
    n_anchor_stress: int = 512
    n_test_drift: int = 2000
    n_test_stress: int = 1024

    # LoRA / optimization
    lora_rank: int = 4
    lr: float = 3e-2
    steps: int = 1200
    batch_size: int = 128
    anchor_batch_size: int = 128
    anchor_beta: float = 0.05  # trust-region strength

    # Portfolio simulation
    sim_T: int = 300
    p_stress: float = 0.35
    w_max_risky: float = 0.35
    turnover_eta: float = 0.20

    # Noise
    noise_std: float = 0.10

    # Return scaling (keeps the simulator numerically stable and finance-plausible)
    return_scale: float = 0.02


def set_seed(seed: int) -> None:
    """Set all relevant RNG seeds."""

    torch.manual_seed(seed)
    np.random.seed(seed)


def _stack_features(
    x_signal: torch.Tensor,
    stress_flag: torch.Tensor,
    drift_flag: torch.Tensor,
) -> torch.Tensor:
    """Concatenate signal + regime flags into the model input."""

    return torch.cat([x_signal, stress_flag, drift_flag], dim=1)


def generate_regime_batch(
    *,
    n: int,
    d_signal: int,
    regime: str,
    device: torch.device,
) -> torch.Tensor:
    """Generate a batch of features for a given regime.

    Regimes:
      - "base": normal market regime used for initial deployment.
      - "stress": rare stress contexts; treated as invariants/anchors.
      - "drift": new regime requiring adaptation.

    The final two coordinates are (stress_flag, drift_flag).
    """

    x = torch.randn(n, d_signal, device=device)

    if regime == "base":
        stress = torch.zeros(n, 1, device=device)
        drift = torch.zeros(n, 1, device=device)
        # mild structure
        x = x + 0.15 * torch.randn_like(x)
    elif regime == "stress":
        stress = torch.ones(n, 1, device=device)
        drift = torch.zeros(n, 1, device=device)
        # heavier tails to emulate volatility shocks
        x = 1.8 * torch.randn(n, d_signal, device=device)
    elif regime == "drift":
        stress = torch.zeros(n, 1, device=device)
        drift = torch.ones(n, 1, device=device)
        # feature drift (mean shift)
        x = x + 0.40
    else:
        raise ValueError(f"Unknown regime: {regime}")

    return _stack_features(x, stress, drift)


def make_true_weights(cfg: Config, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct base and drift ground-truth linear maps.

    Returns:
        W_base_true: (n_assets, d_in)
        W_drift_true: (n_assets, d_in)
    """

    d_in = cfg.d_signal + 2
    n_assets = cfg.n_assets_cash + cfg.n_assets_risky

    # Base mapping: risky assets have signal exposures; stress flag induces a strong negative shift.
    W_base = torch.zeros(n_assets, d_in, device=device)

    # Cash asset: index 0 -> near zero return.
    # Risky assets: indices 1..n_assets_risky
    W_base[1:, : cfg.d_signal] = 0.35 * torch.randn(cfg.n_assets_risky, cfg.d_signal, device=device)

    # Stress flag column: strong negative for risky assets.
    stress_col = cfg.d_signal
    # Make stress events economically meaningful: risky assets go sharply negative.
    W_base[1:, stress_col] = -5.0

    # Drift flag column: base model does not respond (initially).
    drift_col = cfg.d_signal + 1
    W_base[1:, drift_col] = 0.0

    # Drift mapping: shift exposures in a low-rank way (alpha drift).
    # Implemented as W_drift = W_base + Delta, where Delta is rank-1/2.
    r_true = max(1, min(2, cfg.lora_rank))
    U = torch.randn(n_assets, r_true, device=device)
    V = torch.randn(d_in, r_true, device=device)
    Delta = (U @ V.T) * 0.25

    # Make drift respond positively to drift_flag for some assets.
    Delta[1:, drift_col] += 2.0

    W_drift = W_base + Delta

    return W_base, W_drift


def generate_returns(
    X: torch.Tensor,
    W_true: torch.Tensor,
    noise_std: float,
    return_scale: float,
) -> torch.Tensor:
    """Generate linear returns with additive Gaussian noise."""

    # Raw synthetic "returns" are scaled down to be numerically stable and
    # finance-plausible (think daily returns).
    Y = (X @ W_true.T) * return_scale
    if noise_std > 0:
        Y = Y + (noise_std * return_scale) * torch.randn_like(Y)

    # Cash asset return is always 0 (deterministic), to make stress behavior interpretable.
    Y[:, 0] = 0.0
    return Y


class LoRALinear(nn.Module):
    """Linear model with frozen base weight plus trainable LoRA delta.

    Forward: y = x W0^T + scale * x (B A)^T

    Shapes:
      - W0: (n_assets, d_in) frozen
      - A:  (r, d_in) trainable
      - B:  (n_assets, r) trainable
    """

    def __init__(self, W0: torch.Tensor, rank: int, alpha: float = 1.0):
        super().__init__()
        assert W0.ndim == 2
        self.n_assets, self.d_in = W0.shape
        self.rank = int(rank)
        self.alpha = float(alpha)

        self.register_buffer("W0", W0.detach().clone())

        # LoRA factors (trainable)
        self.A = nn.Parameter(torch.zeros(self.rank, self.d_in))
        self.B = nn.Parameter(torch.zeros(self.n_assets, self.rank))

        # Kaiming init on A; B zeros => starts at exactly the base model.
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    @property
    def scale(self) -> float:
        return self.alpha / max(1, self.rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = x @ self.W0.T
        lora = (x @ self.A.T) @ self.B.T
        return base + self.scale * lora


def mse(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean((y_hat - y) ** 2)


@torch.no_grad()
def eval_mse(model: nn.Module, X: torch.Tensor, Y: torch.Tensor, batch: int = 1024) -> float:
    model.eval()
    losses = []
    for i in range(0, X.shape[0], batch):
        xb = X[i : i + batch]
        yb = Y[i : i + batch]
        losses.append(mse(model(xb), yb).item())
    return float(np.mean(losses))


def project_agem(g_new: torch.Tensor, g_mem: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """A-GEM-style closed-form projection for a single constraint.

    Enforces <g, g_mem> >= -eps.
    """

    dot = torch.dot(g_new, g_mem)
    if dot >= -eps:
        return g_new
    denom = torch.dot(g_mem, g_mem) + 1e-12
    return g_new - ((dot + eps) / denom) * g_mem


def _flatten_grads(params: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.cat([p.grad.detach().flatten() for p in params])


def _apply_flat_grads(params: Tuple[torch.Tensor, ...], g_flat: torch.Tensor) -> None:
    """Write a flattened gradient back into parameter .grad buffers."""

    offset = 0
    for p in params:
        n = p.numel()
        p.grad = g_flat[offset : offset + n].view_as(p).clone()
        offset += n
    assert offset == g_flat.numel()


def train_lora(
    *,
    cfg: Config,
    W0: torch.Tensor,
    X_drift: torch.Tensor,
    Y_drift: torch.Tensor,
    X_anchor: torch.Tensor,
    Y_anchor_teacher: torch.Tensor,
    method: str,
) -> Tuple[LoRALinear, Dict[str, float]]:
    """Train LoRA parameters for a given method.

    Methods:
      - "naive": drift loss only.
      - "anchor": drift + beta * anchor distillation.
      - "anchor_proj": drift + beta * anchor distillation, with A-GEM projection of the
        drift gradient against the anchor gradient.

    Returns:
      model, logs
    """

    device = W0.device
    model = LoRALinear(W0=W0, rank=cfg.lora_rank, alpha=1.0).to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)

    params = (model.A, model.B)

    n = X_drift.shape[0]
    m = X_anchor.shape[0]

    rng = torch.Generator(device=device)
    rng.manual_seed(cfg.seed + 123)

    interference = 0
    distortion_sum = 0.0

    for step in range(cfg.steps):
        # minibatch indices
        idx = torch.randint(0, n, (cfg.batch_size,), generator=rng, device=device)
        jdx = torch.randint(0, m, (cfg.anchor_batch_size,), generator=rng, device=device)

        xb = X_drift[idx]
        yb = Y_drift[idx]
        xa = X_anchor[jdx]
        ya = Y_anchor_teacher[jdx]

        # Forward
        pred_b = model(xb)
        pred_a = model(xa)

        L_new = mse(pred_b, yb)
        L_anchor = mse(pred_a, ya)

        opt.zero_grad(set_to_none=True)

        if method == "naive":
            L = L_new
            L.backward()

        elif method == "anchor":
            L = L_new + cfg.anchor_beta * L_anchor
            L.backward()

        elif method == "anchor_proj":
            # Compute g_new
            L_new.backward(retain_graph=True)
            g_new = _flatten_grads(params)

            # Compute g_anchor
            opt.zero_grad(set_to_none=True)
            L_anchor.backward(retain_graph=False)
            g_anchor = _flatten_grads(params)

            # Project drift gradient against anchor gradient
            dot = torch.dot(g_new, g_anchor).item()
            if dot < 0.0:
                interference += 1
            g_proj = project_agem(g_new, g_anchor, eps=0.0)
            distortion = torch.norm(g_proj - g_new).item() / (torch.norm(g_new).item() + 1e-12)
            distortion_sum += float(distortion)

            # Combine: projected drift step + anchor regularization
            g = g_proj + cfg.anchor_beta * g_anchor
            _apply_flat_grads(params, g)

        else:
            raise ValueError(f"Unknown method: {method}")

        opt.step()

    logs: Dict[str, float] = {}
    if method == "anchor_proj":
        logs["interference_rate"] = interference / max(1, cfg.steps)
        logs["update_distortion"] = distortion_sum / max(1, cfg.steps)
    return model, logs


def portfolio_policy(mu: np.ndarray, w_prev: np.ndarray, cfg: Config) -> np.ndarray:
    """A minimal, deterministic rebalancing rule.

    Long-only with a cash asset at index 0:

    - allocate to risky assets proportional to positive predicted returns
    - cap risky weights; remaining weight goes to cash
    - apply a turnover-smoothing step (lazy update)

    This is not meant to be a production optimizer; it is a transparent proxy to translate
    prediction drift into portfolio-level outcomes.
    """

    assert mu.ndim == 1
    n_assets = mu.shape[0]
    assert n_assets == cfg.n_assets_cash + cfg.n_assets_risky

    cash = 0
    risky = np.arange(1, n_assets)

    scores = np.maximum(mu[risky], 0.0)
    if scores.sum() <= 1e-12:
        w_tgt = np.zeros(n_assets)
        w_tgt[cash] = 1.0
    else:
        w_risky = scores / scores.sum()
        # cap risky weights
        w_risky = np.minimum(w_risky, cfg.w_max_risky)
        # renormalize if caps bind
        if w_risky.sum() > 1.0:
            w_risky = w_risky / w_risky.sum()
        w_tgt = np.zeros(n_assets)
        w_tgt[risky] = w_risky
        w_tgt[cash] = max(0.0, 1.0 - w_risky.sum())

    # turnover smoothing (proxy for transaction costs)
    w = (1.0 - cfg.turnover_eta) * w_prev + cfg.turnover_eta * w_tgt
    # numerical cleanup
    w[w < 0] = 0
    w = w / w.sum()
    return w


def max_drawdown(equity: np.ndarray) -> float:
    """Compute max drawdown of an equity curve (equity > 0)."""

    peak = np.maximum.accumulate(equity)
    dd = (equity / peak) - 1.0
    return float(dd.min())


@torch.no_grad()
def simulate_stream(
    *,
    cfg: Config,
    model: nn.Module,
    W_base_true: torch.Tensor,
    W_drift_true: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """Run a minimal rebalancing simulation on a drift+stress stream."""

    model.eval()

    n_assets = cfg.n_assets_cash + cfg.n_assets_risky

    w = np.zeros(n_assets)
    w[0] = 1.0

    equity = [1.0]
    worst_stress_day = 0.0
    n_stress_days = 0

    for t in range(cfg.sim_T):
        is_stress = np.random.rand() < cfg.p_stress
        regime = "stress" if is_stress else "drift"

        X = generate_regime_batch(n=1, d_signal=cfg.d_signal, regime=regime, device=device)
        W_true = W_base_true if is_stress else W_drift_true
        y_true = generate_returns(
            X,
            W_true,
            noise_std=cfg.noise_std,
            return_scale=cfg.return_scale,
        ).cpu().numpy().reshape(-1)

        mu = model(X).cpu().numpy().reshape(-1)
        w = portfolio_policy(mu, w, cfg)

        r_p = float(np.dot(w, y_true))
        equity.append(equity[-1] * (1.0 + r_p))

        if is_stress:
            n_stress_days += 1
            worst_stress_day = min(worst_stress_day, r_p)

    equity_arr = np.array(equity)
    out = {
        "total_return": float(equity_arr[-1] - 1.0),
        "max_drawdown": float(max_drawdown(equity_arr)),
        "worst_stress_day": float(worst_stress_day),
        "n_stress_days": float(n_stress_days),
    }
    return out


def save_metrics_tex(path: str, metrics: Dict[str, float]) -> None:
    """Write metrics as LaTeX macros for embedding."""

    def fmt(x: float) -> str:
        if abs(x) >= 1e3 or (abs(x) > 0 and abs(x) < 1e-3):
            return f"{x:.2e}"
        return f"{x:.4f}"

    lines = ["% Auto-generated by validation_experiment.py"]
    for k, v in metrics.items():
        macro = "\\newcommand{\\" + k + "}{" + fmt(v) + "}"
        lines.append(macro)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    cfg = Config()
    set_seed(cfg.seed)

    device = torch.device("cpu")

    os.makedirs("figs", exist_ok=True)

    # Ground-truth (unscaled) linear maps.
    W_base_true, W_drift_true = make_true_weights(cfg, device)

    # Deployed reference mapping (scaled into "daily-return" units).
    W0 = W_base_true * cfg.return_scale

    # --- Construct datasets ---
    X_base = generate_regime_batch(n=cfg.n_train_base, d_signal=cfg.d_signal, regime="base", device=device)
    Y_base = generate_returns(
        X_base,
        W_base_true,
        noise_std=cfg.noise_std,
        return_scale=cfg.return_scale,
    )

    X_stress_anchor = generate_regime_batch(n=cfg.n_anchor_stress, d_signal=cfg.d_signal, regime="stress", device=device)
    # Teacher outputs from deployed reference (LoRA=0)
    Y_teacher = X_stress_anchor @ W0.T
    Y_teacher[:, 0] = 0.0

    X_drift = generate_regime_batch(n=cfg.n_train_drift, d_signal=cfg.d_signal, regime="drift", device=device)
    Y_drift = generate_returns(
        X_drift,
        W_drift_true,
        noise_std=cfg.noise_std,
        return_scale=cfg.return_scale,
    )

    X_drift_test = generate_regime_batch(n=cfg.n_test_drift, d_signal=cfg.d_signal, regime="drift", device=device)
    Y_drift_test = generate_returns(
        X_drift_test,
        W_drift_true,
        noise_std=cfg.noise_std,
        return_scale=cfg.return_scale,
    )

    X_stress_test = generate_regime_batch(n=cfg.n_test_stress, d_signal=cfg.d_signal, regime="stress", device=device)
    Y_stress_teacher_test = X_stress_test @ W0.T
    Y_stress_teacher_test[:, 0] = 0.0

    # Reference base model (no LoRA) for evaluation
    ref = LoRALinear(W0=W0, rank=cfg.lora_rank, alpha=1.0).to(device)
    with torch.no_grad():
        ref.B.zero_()  # ensure delta=0

    # --- Train 3 methods ---
    results = {}
    train_logs = {}
    trained_models: Dict[str, nn.Module] = {}
    for method in ["naive", "anchor", "anchor_proj"]:
        model, logs = train_lora(
            cfg=cfg,
            W0=W0,
            X_drift=X_drift,
            Y_drift=Y_drift,
            X_anchor=X_stress_anchor,
            Y_anchor_teacher=Y_teacher,
            method=method,
        )
        train_logs[method] = logs
        trained_models[method] = model

        drift_mse = eval_mse(model, X_drift_test, Y_drift_test)
        stress_mse = eval_mse(model, X_stress_test, Y_stress_teacher_test)

        # Portfolio simulation
        sim = simulate_stream(cfg=cfg, model=model, W_base_true=W_base_true, W_drift_true=W_drift_true, device=device)

        results[method] = {
            "drift_mse": drift_mse,
            "stress_anchor_mse": stress_mse,
            **sim,
        }

    # --- Create figure ---
    import matplotlib.pyplot as plt

    methods = [
        ("naive", "LoRA (naive)"),
        ("anchor", "LoRA + anchor"),
        ("anchor_proj", "LoRA + anchor + proj"),
    ]

    fig = plt.figure(figsize=(11.5, 3.2))

    ax1 = fig.add_subplot(1, 2, 1)
    for key, label in methods:
        ax1.scatter(results[key]["drift_mse"], results[key]["stress_anchor_mse"], s=55, label=label)
        ax1.annotate(label, (results[key]["drift_mse"], results[key]["stress_anchor_mse"]),
                     textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax1.set_xlabel("Drift-regime prediction MSE (lower = better)")
    ax1.set_ylabel("Stress-anchor regression (MSE to reference) (lower = better)")
    ax1.grid(True, alpha=0.25)

    # Equity curves
    ax2 = fig.add_subplot(1, 2, 2)

    # Re-run a single shared stream for comparability
    set_seed(cfg.seed + 999)
    device = torch.device("cpu")
    stream_regimes = ["stress" if np.random.rand() < cfg.p_stress else "drift" for _ in range(cfg.sim_T)]
    Xs = [generate_regime_batch(n=1, d_signal=cfg.d_signal, regime=r, device=device) for r in stream_regimes]
    Ys_true = [
        generate_returns(
            x,
            W_base_true if r == "stress" else W_drift_true,
            noise_std=cfg.noise_std,
            return_scale=cfg.return_scale,
        ).cpu().numpy().reshape(-1)
               for x, r in zip(Xs, stream_regimes)]

    def run_equity(model: nn.Module) -> np.ndarray:
        w = np.zeros(cfg.n_assets_cash + cfg.n_assets_risky)
        w[0] = 1.0
        eq = [1.0]
        for x, y_true, r in zip(Xs, Ys_true, stream_regimes):
            # The simulator is purely evaluative; detach to avoid autograd tracking.
            mu = model(x).detach().cpu().numpy().reshape(-1)
            w = portfolio_policy(mu, w, cfg)
            eq.append(eq[-1] * (1.0 + float(np.dot(w, y_true))))
        return np.array(eq)

    for key, label in methods:
        eq = run_equity(trained_models[key])
        ax2.plot(eq, label=label)

    # Mark stress days
    for t, r in enumerate(stream_regimes):
        if r == "stress":
            ax2.axvline(t, linewidth=0.6, alpha=0.10)

    ax2.set_title("Equity curve on a drift+stress stream")
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Equity")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=8, frameon=False, loc="best")

    fig.tight_layout()

    fig_path_pdf = os.path.join("figs", "validation_wide.pdf")
    fig_path_png = os.path.join("figs", "validation_wide.png")
    fig.savefig(fig_path_pdf, bbox_inches="tight")
    fig.savefig(fig_path_png, dpi=200, bbox_inches="tight")

    # --- Export LaTeX macros (key metrics for the write-up) ---
    # NOTE: LaTeX command names cannot contain underscores; keep macro keys letter-only.
    metrics_tex = {
        "DriftMSENaive": results["naive"]["drift_mse"],
        "StressMSENaive": results["naive"]["stress_anchor_mse"],
        "MDDNaive": results["naive"]["max_drawdown"],
        "TotRetNaive": results["naive"]["total_return"],
        "WorstStressNaive": results["naive"]["worst_stress_day"],

        "DriftMSEAnchor": results["anchor"]["drift_mse"],
        "StressMSEAnchor": results["anchor"]["stress_anchor_mse"],
        "MDDAnchor": results["anchor"]["max_drawdown"],
        "TotRetAnchor": results["anchor"]["total_return"],
        "WorstStressAnchor": results["anchor"]["worst_stress_day"],

        "DriftMSEProj": results["anchor_proj"]["drift_mse"],
        "StressMSEProj": results["anchor_proj"]["stress_anchor_mse"],
        "MDDProj": results["anchor_proj"]["max_drawdown"],
        "TotRetProj": results["anchor_proj"]["total_return"],
        "WorstStressProj": results["anchor_proj"]["worst_stress_day"],

        "InterferenceRate": float(train_logs["anchor_proj"].get("interference_rate", float("nan"))),
        "UpdateDistortion": float(train_logs["anchor_proj"].get("update_distortion", float("nan"))),
    }

    save_metrics_tex(os.path.join("figs", "metrics.tex"), metrics_tex)

    # Console summary for quick checks
    print("=== Validation summary ===")
    for k, label in methods:
        r = results[k]
        print(f"{label:>20s} | drift MSE={r['drift_mse']:.4f} | stress MSE={r['stress_anchor_mse']:.4f} | MDD={r['max_drawdown']:.3f} | TotRet={r['total_return']:.3f}")
    print("Figures written to:", fig_path_pdf)


if __name__ == "__main__":
    main()
