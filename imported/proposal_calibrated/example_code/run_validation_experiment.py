"""Minimal end-to-end validation experiment (proof-of-concept).

This script is deliberately small but complete: it runs an end-to-end loop

    synthetic market → (x_t + tokenized text z_t) → (frozen Transformer LM + LoRA head) →
    decision layer (hard constraints) → online updates (naive vs CL) → figures

and writes the figures that are embedded in the accompanying PDF.

The goal is not benchmarking. The goal is an executable, falsifiable check of
the proposal's *core mechanism*:

  (i)  feasibility is enforced by construction in the decision layer, and
  (ii) continual updates adapt to drift without regressing on a fixed stress fixture.

Run (from project root):

    python example_code/run_validation_experiment.py

Outputs (written to ./figures/):

    validation_stress_loss.pdf   stress-suite regression / rollback evidence
    validation_ic.pdf            adaptation signal (predictive IC)
    validation_stress_es.pdf     decision-layer tail budget is always enforced
    summary.json                 raw numbers + experiment configuration
"""

from __future__ import annotations

# Limit BLAS thread fan-out for determinism and predictable runtime.
import os


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from spine.controller import RollbackConfig, RollbackController
from spine.decision_layer import DecisionLayer, DecisionLayerConfig
from spine.features import FeatureBuilder, FeatureConfig
from spine.metrics import RunSummary, spearmanr, summarize_run
from spine.replay import BufferKind, ReplayBuffer
from spine.risk_model import RiskModelConfig, RollingFactorRiskModel, residualize_next_day
from spine.signal_model import BaseRegressor, LoRARegressor, ModelConfig
from spine.synthetic_market import SyntheticMarketConfig, simulate_synthetic_market
from spine.tiny_llm import TinyCausalTransformerLM, TinyLMConfig, freeze_module


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the toy validation experiment.

    Defaults are chosen to:
      - run quickly on CPU,
      - include two forced crisis windows (so the stress fixture matters),
      - make forgetting visible for the naive updater.
    """

    seed: int = 7
    device: str = "cpu"

    # Market size / horizon
    n_assets: int = 28
    t_total: int = 360

    # Risk model window (short to keep the toy light)
    risk_window: int = 60

    # Pretraining window (days)
    pretrain_days: int = 80

    # --- tiny LLM testbed (Transformer language model) ---
    # We keep the Transformer deliberately small so the script stays runnable on CPU.
    # The LLM is trained once on a synthetic token stream and then frozen.
    lm_d_model: int = 32
    lm_n_heads: int = 4
    lm_n_layers: int = 2
    lm_d_ff: int = 128
    lm_train_epochs: int = 20
    lm_train_lr: float = 3e-3
    lm_batch_size: int = 64

    # Model
    hidden_dim: int = 32
    lora_rank: int = 4
    lora_alpha: float = 4.0

    # Online update
    online_lr: float = 2e-3
    distill_lambda: float = 3.0
    replay_batch_size: int = 256
    stress_batch_size: int = 256

    # Gate
    stress_loss_tolerance: float = 0.02  # 2% relative degradation allowed

    # Decision layer
    gamma: float = 50.0
    leverage_l1: float = 1.0
    w_max: float = 0.07
    tc_per_turnover: float = 5e-4
    es_alpha: float = 0.2
    es_budget: float = 8e-4

    # Crisis windows (placed as fractions of t_total)
    crisis_len_1: int = 20
    crisis_len_2: int = 18
    crisis_start_frac_1: float = 0.36
    crisis_start_frac_2: float = 0.78


@dataclass
class Series:
    """Per-day time series collected from a run."""

    pnl: np.ndarray
    ic: np.ndarray
    turnover: np.ndarray
    stress_es: np.ndarray
    stress_loss: np.ndarray
    rollbacks: np.ndarray  # 0/1 mask
    days: np.ndarray  # absolute day index in the simulation


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _forced_crisis_windows(cfg: ExperimentConfig) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    s1 = int(round(cfg.crisis_start_frac_1 * cfg.t_total))
    s2 = int(round(cfg.crisis_start_frac_2 * cfg.t_total))
    s1 = max(0, min(cfg.t_total - cfg.crisis_len_1 - 1, s1))
    s2 = max(0, min(cfg.t_total - cfg.crisis_len_2 - 1, s2))
    if s2 <= s1 + cfg.crisis_len_1 + 5:
        s2 = min(cfg.t_total - cfg.crisis_len_2 - 1, s1 + cfg.crisis_len_1 + 35)
    return (s1, cfg.crisis_len_1), (s2, cfg.crisis_len_2)


def _build_stress_returns(returns: np.ndarray, crisis1: Tuple[int, int]) -> np.ndarray:
    start, length = crisis1
    end = min(returns.shape[0], start + length)
    return returns[start:end].copy()


def _pretrain_base(
    *,
    x: np.ndarray,
    returns: np.ndarray,
    factor_returns: np.ndarray,
    risk_model: RollingFactorRiskModel,
    t_min: int,
    pre_end: int,
    device: torch.device,
    hidden_dim: int,
    epochs: int = 5,
    lr: float = 1e-3,
) -> BaseRegressor:
    """Supervised pretrain on an initial window (before the first crisis)."""

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for t in range(t_min, pre_end):
        rm = risk_model.estimate(t_end_exclusive=t, returns=returns, factor_returns=factor_returns)
        y_t = residualize_next_day(B_t=rm["B"], f_next=factor_returns[t], r_next=returns[t])
        xs.append(x[t])
        ys.append(y_t)

    X = torch.tensor(np.concatenate(xs, axis=0), dtype=torch.float32, device=device)
    Y = torch.tensor(np.concatenate(ys, axis=0), dtype=torch.float32, device=device)

    base = BaseRegressor(input_dim=X.shape[1], hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(base.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    base.train()
    for _ in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(base(X), Y)
        loss.backward()
        opt.step()
    return base


@torch.no_grad()
def _stress_suite(
    *,
    x: np.ndarray,
    returns: np.ndarray,
    factor_returns: np.ndarray,
    risk_model: RollingFactorRiskModel,
    days: np.ndarray,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fixed stress suite used for gating + the stress-loss figure."""

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for t in days:
        rm = risk_model.estimate(t_end_exclusive=int(t), returns=returns, factor_returns=factor_returns)
        y_t = residualize_next_day(B_t=rm["B"], f_next=factor_returns[int(t)], r_next=returns[int(t)])
        xs.append(x[int(t)])
        ys.append(y_t)

    X = torch.tensor(np.concatenate(xs, axis=0), dtype=torch.float32, device=device)
    Y = torch.tensor(np.concatenate(ys, axis=0), dtype=torch.float32, device=device)
    return X, Y


def _flatten_grads(params: List[nn.Parameter]) -> torch.Tensor:
    vec = []
    for p in params:
        g = torch.zeros_like(p).reshape(-1) if p.grad is None else p.grad.detach().reshape(-1)
        vec.append(g)
    return torch.cat(vec, dim=0)


def _assign_flat_grads(params: List[nn.Parameter], flat: torch.Tensor) -> None:
    offset = 0
    for p in params:
        n = p.numel()
        g = flat[offset : offset + n].reshape_as(p).detach()
        if p.grad is None:
            p.grad = g.clone()
        else:
            p.grad.copy_(g)
        offset += n


def _agem_project(g_cur: torch.Tensor, g_ref: torch.Tensor) -> torch.Tensor:
    dot = torch.dot(g_cur, g_ref)
    if dot >= 0:
        return g_cur
    denom = torch.dot(g_ref, g_ref).clamp_min(1e-12)
    return g_cur - (dot / denom) * g_ref


@torch.no_grad()
def _eval_stress_mse(model: nn.Module, X: torch.Tensor, Y: torch.Tensor) -> float:
    model.eval()
    return float(torch.mean((model(X) - Y) ** 2).item())


def _run_policy(
    *,
    mode: str,
    cfg: ExperimentConfig,
    sim: Dict[str, np.ndarray],
    x: np.ndarray,
    valid_mask: np.ndarray,
    risk_model: RollingFactorRiskModel,
    pre_end: int,
    stress_returns: np.ndarray,
    X_stress: torch.Tensor,
    Y_stress: torch.Tensor,
    init_state: Dict[str, torch.Tensor],
) -> Tuple[Series, RunSummary]:
    """Run one update policy on the fixed simulation."""

    if mode not in {"naive", "cl"}:
        raise ValueError("mode must be 'naive' or 'cl'")

    device = torch.device(cfg.device)
    returns = sim["returns"]
    factor_returns = sim["factor_returns"]
    regimes = sim["regimes"]

    # Recreate model and load the same initial weights for a fair comparison.
    # (The only difference between runs should be the update policy.)
    dummy_base = BaseRegressor(input_dim=x.shape[-1], hidden_dim=cfg.hidden_dim).to(device)
    mcfg = ModelConfig(input_dim=x.shape[-1], hidden_dim=cfg.hidden_dim, rank=cfg.lora_rank, lora_alpha=cfg.lora_alpha)
    model = LoRARegressor.from_pretrained(dummy_base, cfg=mcfg).to(device)
    model.load_state_dict(init_state)

    # Decision layer (hard constraints)
    decision = DecisionLayer(
        DecisionLayerConfig(
            gamma=cfg.gamma,
            leverage_l1=cfg.leverage_l1,
            w_max=cfg.w_max,
            tc_per_turnover=cfg.tc_per_turnover,
            enforce_dollar_neutral=True,
            enforce_market_neutral=True,
            es_alpha=cfg.es_alpha,
            es_budget=cfg.es_budget,
        )
    )

    # Replay + gate (CL only)
    buffer = ReplayBuffer(max_recent=8_000, max_stress=8_000, max_edge=2_000, device=device)
    gate = RollbackController(RollbackConfig(stress_loss_tolerance=cfg.stress_loss_tolerance))
    gate.snapshot(model)
    gate._best_stress_loss = _eval_stress_mse(model, X_stress, Y_stress)

    lora_params = list(model.lora_parameters())
    opt = torch.optim.Adam(lora_params, lr=cfg.online_lr)
    loss_fn = nn.MSELoss()

    # Collectors
    pnl: List[float] = []
    ics: List[float] = []
    turns: List[float] = []
    es_list: List[float] = []
    stress_loss: List[float] = []
    rollbacks: List[int] = []
    days: List[int] = []

    w_prev: Optional[np.ndarray] = None
    rng = np.random.default_rng(cfg.seed + (0 if mode == "naive" else 10_000))

    for t in range(pre_end, returns.shape[0] - 1):
        if not bool(valid_mask[t]):
            continue

        rm = risk_model.estimate(t_end_exclusive=t, returns=returns, factor_returns=factor_returns)
        B_t = rm["B"]
        Sigma_t = rm["Sigma"]

        x_t = torch.tensor(x[t], dtype=torch.float32, device=device)
        with torch.no_grad():
            mu_t = model(x_t).detach().cpu().numpy()

        w_t, turn, es = decision.solve(mu=mu_t, Sigma=Sigma_t, B=B_t, w_prev=w_prev, stress_returns=stress_returns)
        daily = float(w_t @ returns[t + 1])
        daily_net = daily - decision.transaction_cost(turn)

        y_label = residualize_next_day(B_t=B_t, f_next=factor_returns[t + 1], r_next=returns[t + 1])
        ic_t = spearmanr(mu_t, y_label)

        pnl.append(daily_net)
        ics.append(ic_t)
        turns.append(turn)
        es_list.append(es)
        days.append(t)

        # ---- memory update (small, stable sample size) ----
        idx = rng.choice(cfg.n_assets, size=min(48, cfg.n_assets), replace=False)
        x_s = torch.tensor(x[t, idx], dtype=torch.float32, device=device)
        y_s = torch.tensor(y_label[idx], dtype=torch.float32, device=device)
        mu_s = torch.tensor(mu_t[idx], dtype=torch.float32, device=device)
        kind = BufferKind.STRESS if int(regimes[t]) == 2 else BufferKind.RECENT
        buffer.add(x_s, y_s, mu_s, kind=kind)

        # ---- online update ----
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x_t), torch.tensor(y_label, dtype=torch.float32, device=device))

        if mode == "cl" and cfg.distill_lambda > 0.0 and buffer.stress_size() > 0:
            rep = buffer.sample(cfg.replay_batch_size, kind=BufferKind.STRESS)
            if rep is not None and rep.x.shape[0] > 0:
                loss = loss + cfg.distill_lambda * loss_fn(model(rep.x), rep.y_old)

        loss.backward()
        g_cur = _flatten_grads(lora_params)

        if mode == "cl" and buffer.stress_size() > 0:
            stress = buffer.sample(cfg.stress_batch_size, kind=BufferKind.STRESS)
            if stress is not None and stress.x.shape[0] > 0:
                opt.zero_grad(set_to_none=True)
                loss_s = loss_fn(model(stress.x), stress.y)
                loss_s.backward()
                g_ref = _flatten_grads(lora_params)
                g_cur = _agem_project(g_cur, g_ref)

        _assign_flat_grads(lora_params, g_cur)
        opt.step()

        sl = _eval_stress_mse(model, X_stress, Y_stress)
        stress_loss.append(sl)
        kept = True
        if mode == "cl":
            kept = gate.gate_or_rollback(model, stress_loss=sl)
        rollbacks.append(0 if kept else 1)

        w_prev = w_t

    pnl_a = np.asarray(pnl, dtype=float)
    ic_a = np.asarray(ics, dtype=float)
    turn_a = np.asarray(turns, dtype=float)
    es_a = np.asarray(es_list, dtype=float)
    sl_a = np.asarray(stress_loss, dtype=float)
    rb_a = np.asarray(rollbacks, dtype=int)
    days_a = np.asarray(days, dtype=int)

    summary = summarize_run(ic_a, pnl_a, turn_a, es_a, n_rollbacks=gate.n_rollbacks)
    return (
        Series(pnl=pnl_a, ic=ic_a, turnover=turn_a, stress_es=es_a, stress_loss=sl_a, rollbacks=rb_a, days=days_a),
        summary,
    )


def _shade_crisis(ax: plt.Axes, windows: Tuple[Tuple[int, int], Tuple[int, int]]) -> None:
    for start, length in windows:
        ax.axvspan(start, start + length, alpha=0.12)


def _movavg(x: np.ndarray, w: int = 20) -> np.ndarray:
    if x.size == 0:
        return x
    w = max(1, int(w))
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def _pretrain_tiny_llm(
    *,
    tokens: np.ndarray,
    vocab_size: int,
    seq_len: int,
    cfg: ExperimentConfig,
    device: torch.device,
) -> TinyCausalTransformerLM:
    """Pretrain a tiny causal Transformer LM on the synthetic token stream.

    This is a stand-in for a production pretrained LLM. The important property
    for the proposal is operational: the Transformer is frozen during trading
    and continual updates, and only a small adapter on top is updated.
    """

    lm_cfg = TinyLMConfig(
        vocab_size=int(vocab_size),
        max_seq_len=int(seq_len),
        d_model=int(cfg.lm_d_model),
        n_heads=int(cfg.lm_n_heads),
        n_layers=int(cfg.lm_n_layers),
        d_ff=int(cfg.lm_d_ff),
        dropout=0.0,
    )
    lm = TinyCausalTransformerLM(lm_cfg).to(device)
    lm.train()

    # Dataset: each day is an independent "document".
    data = torch.tensor(tokens, dtype=torch.long, device=device)  # (T, L)
    if data.ndim != 2 or data.shape[1] != seq_len:
        raise ValueError("tokens must be (T, seq_len)")

    opt = torch.optim.AdamW(lm.parameters(), lr=float(cfg.lm_train_lr))
    loss_fn = nn.CrossEntropyLoss()

    T = int(data.shape[0])
    B = int(cfg.lm_batch_size)
    V = int(vocab_size)

    for _ in range(int(cfg.lm_train_epochs)):
        perm = torch.randperm(T, device=device)
        for s in range(0, T, B):
            idx = perm[s : s + B]
            batch = data[idx]  # (B, L)
            x_in = batch[:, :-1]
            y_tgt = batch[:, 1:]
            logits, _ = lm(x_in)
            loss = loss_fn(logits.reshape(-1, V), y_tgt.reshape(-1))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    freeze_module(lm)
    return lm


@torch.no_grad()
def _llm_day_embeddings(
    lm: TinyCausalTransformerLM, tokens: np.ndarray, device: torch.device, pool: str = "last"
) -> np.ndarray:
    """Compute a per-day embedding z_t from tokenized text."""

    data = torch.tensor(tokens, dtype=torch.long, device=device)
    z = lm.embed(data, pool=pool)  # (T, d_model)
    return z.detach().cpu().numpy()


def main() -> None:
    cfg = ExperimentConfig()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(cfg.device)

    crisis1, crisis2 = _forced_crisis_windows(cfg)

    # --- simulate market once ---
    sim_cfg = SyntheticMarketConfig(
        seed=cfg.seed,
        n_assets=cfg.n_assets,
        t_total=cfg.t_total,
        forced_crisis=(crisis1, crisis2),
    )
    sim = simulate_synthetic_market(sim_cfg)

    # --- features (numeric) ---
    feat = FeatureBuilder(
        FeatureConfig(win_mom=sim_cfg.win_mom, win_rev=sim_cfg.win_rev, win_vol=sim_cfg.win_vol)
    ).build(returns=sim["returns"], betas=sim["betas"])
    x_num = feat["x"]
    valid_mask = feat["valid_mask"]

    # --- LLM text context (token stream -> frozen Transformer -> z_t embeddings) ---
    lm = _pretrain_tiny_llm(
        tokens=sim["text_tokens"],
        vocab_size=sim_cfg.vocab_size,
        seq_len=sim_cfg.text_seq_len,
        cfg=cfg,
        device=device,
    )
    z = _llm_day_embeddings(lm, tokens=sim["text_tokens"], device=device, pool="last")  # (T, d_model)
    z_rep = np.repeat(z[:, None, :], repeats=cfg.n_assets, axis=1)
    x = np.concatenate([x_num, z_rep], axis=2)  # (T, N, P + d_model)

    # --- risk model ---
    risk_model = RollingFactorRiskModel(RiskModelConfig(window=cfg.risk_window, ridge=1e-6))
    t_feat0 = int(np.where(valid_mask)[0][0])
    t_min = int(max(t_feat0, cfg.risk_window))
    pre_end = t_min + int(cfg.pretrain_days)

    # Stress scenario library used by the decision layer's ES constraint.
    stress_returns = _build_stress_returns(sim["returns"], crisis1)

    # Fixed stress suite used for gating and for the stress-loss plot.
    stress_days = np.arange(crisis1[0], crisis1[0] + max(10, crisis1[1] // 2), dtype=int)
    stress_days = stress_days[stress_days >= t_min]
    X_stress, Y_stress = _stress_suite(
        x=x,
        returns=sim["returns"],
        factor_returns=sim["factor_returns"],
        risk_model=risk_model,
        days=stress_days,
        device=device,
    )

    # --- pretrain once, then clone initial state for both policies ---
    base = _pretrain_base(
        x=x,
        returns=sim["returns"],
        factor_returns=sim["factor_returns"],
        risk_model=risk_model,
        t_min=t_min,
        pre_end=pre_end,
        device=device,
        hidden_dim=cfg.hidden_dim,
        epochs=5,
        lr=1e-3,
    )
    init_model = LoRARegressor.from_pretrained(
        base=base,
        cfg=ModelConfig(input_dim=x.shape[-1], hidden_dim=cfg.hidden_dim, rank=cfg.lora_rank, lora_alpha=cfg.lora_alpha),
    ).to(device)
    init_state = {k: v.detach().clone() for k, v in init_model.state_dict().items()}

    series_naive, summ_naive = _run_policy(
        mode="naive",
        cfg=cfg,
        sim=sim,
        x=x,
        valid_mask=valid_mask,
        risk_model=risk_model,
        pre_end=pre_end,
        stress_returns=stress_returns,
        X_stress=X_stress,
        Y_stress=Y_stress,
        init_state=init_state,
    )
    series_cl, summ_cl = _run_policy(
        mode="cl",
        cfg=cfg,
        sim=sim,
        x=x,
        valid_mask=valid_mask,
        risk_model=risk_model,
        pre_end=pre_end,
        stress_returns=stress_returns,
        X_stress=X_stress,
        Y_stress=Y_stress,
        init_state=init_state,
    )

    # --- write figures ---
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
    _ensure_dir(out_dir)

    windows = (crisis1, crisis2)

    # 1) Stress loss (the gate's regression metric)
    fig, ax = plt.subplots(figsize=(6.7, 2.6))
    ax.plot(series_naive.days, series_naive.stress_loss, label="naive")
    ax.plot(series_cl.days, series_cl.stress_loss, label="cl (distill+proj+gate)")
    rb_days = series_cl.days[series_cl.rollbacks.astype(bool)]
    if rb_days.size > 0:
        ax.scatter(rb_days, series_cl.stress_loss[series_cl.rollbacks.astype(bool)], marker="x", s=18, label="rollback")
    _shade_crisis(ax, windows)
    ax.set_title("Stress-suite regression (lower is better)")
    ax.set_xlabel("day")
    ax.set_ylabel("MSE on fixed stress suite")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "validation_stress_loss.pdf"))
    plt.close(fig)

    # 2) IC (predictive adaptation signal)
    fig, ax = plt.subplots(figsize=(6.7, 2.6))
    ax.plot(series_naive.days, _movavg(series_naive.ic, w=15), label="naive")
    ax.plot(series_cl.days, _movavg(series_cl.ic, w=15), label="cl (distill+proj+gate)")
    _shade_crisis(ax, windows)
    ax.set_title("Predictive adaptation: cross-sectional IC on next-day residuals")
    ax.set_xlabel("day")
    ax.set_ylabel("Spearman IC (15d MA)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "validation_ic.pdf"))
    plt.close(fig)
    # 3) Decision-layer constraint: stress ES budget satisfaction
    # Plot in basis points to avoid matplotlib's offset notation when the constraint binds tightly.
    fig, ax = plt.subplots(figsize=(6.7, 2.6))
    scale_bp = 1e4  # 1 return unit = 100%; 1 bp = 1e-4
    y_naive = scale_bp * series_naive.stress_es
    y_cl = scale_bp * series_cl.stress_es
    y_budget = scale_bp * (-cfg.es_budget)

    ax.plot(series_naive.days, y_naive, label="naive")
    ax.plot(series_cl.days, y_cl, label="cl (distill+proj+gate)")
    ax.axhline(y_budget, linestyle="--", linewidth=1.0, label=f"ES budget ({y_budget:.1f} bp)")
    _shade_crisis(ax, windows)
    ax.set_title("Feasibility check: stress ES is enforced by construction")
    ax.set_xlabel("day")
    ax.set_ylabel(f"Stress ES$_{{{cfg.es_alpha:.1f}}}$ (bp)")

    # Tighten y-limits around the budget (the constraint often binds in this toy).
    y_all = np.concatenate([y_naive, y_cl, np.asarray([y_budget])], axis=0)
    y_min = float(np.nanmin(y_all))
    y_max = float(np.nanmax(y_all))
    pad = max(0.6, 0.2 * (y_max - y_min + 1e-12))
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "validation_stress_es.pdf"))
    plt.close(fig)
    # Summary artifacts used by the PDF
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": asdict(cfg),
                "crisis_windows": {"crisis1": crisis1, "crisis2": crisis2},
                "naive": asdict(summ_naive),
                "cl": asdict(summ_cl),
            },
            f,
            indent=2,
        )

    print("\n=== Minimal validation experiment ===")
    print(f"seed={cfg.seed}, n_assets={cfg.n_assets}, t_total={cfg.t_total}, risk_window={cfg.risk_window}")
    print(f"forced crisis windows: {crisis1}, {crisis2}")
    print("naive:", summ_naive)
    print("cl:   ", summ_cl)


if __name__ == "__main__":
    main()
