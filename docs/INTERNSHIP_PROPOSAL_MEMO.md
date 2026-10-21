# Internship Proposal Memo
## Constraint-Aligned Continual PEFT for Portfolio Signal Models

## Thesis
I want to test one concrete idea: adapt a signal model to regime drift, but reject updates that regress on predefined stress contexts.

## Problem
Continual tuning usually optimizes short-horizon fit. In production, that can create silent failures on low-frequency, high-impact regimes. The cost is asymmetric: one bad stress update can erase months of incremental fit gains.

## Proposed mechanism
- Keep the backbone frozen.
- Train only PEFT adapters.
- Use two objectives per update cycle:
  - `g_new`: fit current drift window.
  - `g_anchor`: preserve stress-anchor behavior from deployed checkpoint outputs.
- Project the update direction to enforce compatibility with anchor behavior.
- Promote only if gate metrics pass.

## What already exists in this repo
- End-to-end runnable mechanism with three methods:
  - `LoRA (naive)`
  - `LoRA + anchor`
  - `LoRA + anchor + projection`
- Portfolio-level simulation to observe downstream effects.
- Interactive website with adjustable knobs and exported run reports.

## Current evidence (from provided implementation)
From `figs/metrics.tex`:
- `Stress MSE`: `0.0024` -> `0.0000656` (naive -> projection)
- `Max Drawdown`: `-0.4318` -> `-0.2348` (naive -> projection)

Interpretation: in this controlled setup, projected PEFT materially improves retention and drawdown behavior relative to unconstrained adaptation.

## Explicit non-claims
- This synthetic environment is not a substitute for production replay/backtesting.
- This is not a claim of portable alpha.
- This does not replace portfolio-level risk controls.

## Why this is practical in an internship
This can sit inside an existing stack with low disruption:
- no backbone retraining,
- small adapter deltas,
- explicit promotion/rollback policies,
- auditable decision logs.

## 10-week execution plan
1. Weeks 1-3: connect to internal replay data and establish baseline metrics.
2. Weeks 4-7: implement promotion gate and stress-window validation.
3. Weeks 8-10: run shadow deployment and write rollout recommendation.

## Deliverables
- Adapter update module with anchor-projection logic.
- Gate evaluator with pass/fail audit record.
- Monitoring for retention, tail risk proxies, and rollback triggers.
- Final recommendation memo: pilot scope, threshold policy, and open risks.

## How to review quickly
- Run the interactive artifact (`index.html`).
- Read deployment details (`docs/DEPLOYMENT_PLAYBOOK.md`).
- Reproduce baseline figure/metrics (`./scripts/run_poc.sh`).
