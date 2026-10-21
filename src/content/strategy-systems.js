export const VERSION_LABEL = "Signal Gate V1";
export const VERSION_TAG = "signal-gate-v1";

export const STRATEGY_SYSTEMS = {
  naive: {
    id: "naive",
    label: "Naive Continual LoRA",
    shortLabel: "Naive",
    color: "#cc5539",
    protects: "Fast fit to current drift regime.",
    reliesOn: "Only current drift minibatches (D_t).",
    sacrifices: "No explicit retention guard for stress behavior.",
    deployWhen: "Exploration stage where adaptation speed matters more than stability.",
    failureMode: "Catastrophic stress forgetting after long drift periods.",
    symbols: ["D_t", "g_new", "theta_t+1"],
    pipeline: [
      { kind: "data", text: "Drift data D_t" },
      { kind: "update", text: "g_new = grad(D_t)" },
      { kind: "rule", text: "theta_t+1 = theta_t - eta g_new" },
      { kind: "opt", text: "mu_t -> optimizer" },
    ],
  },
  anchor: {
    id: "anchor",
    label: "Anchor-Regularized LoRA",
    shortLabel: "Anchor",
    color: "#1a7b64",
    protects: "Partial retention on stress anchor behavior.",
    reliesOn: "Drift minibatches (D_t) plus anchor set (A_t).",
    sacrifices: "Can underfit new drift if anchor weight is too large.",
    deployWhen: "Need soft retention without hard constraints.",
    failureMode: "Anchor term may be too weak under persistent drift pressure.",
    symbols: ["D_t", "A_t", "g_new + beta g_anc"],
    pipeline: [
      { kind: "data", text: "Drift D_t + Anchor A_t" },
      { kind: "update", text: "g_new , g_anc" },
      { kind: "rule", text: "g = g_new + beta g_anc" },
      { kind: "opt", text: "mu_t -> optimizer" },
    ],
  },
  anchor_proj: {
    id: "anchor_proj",
    label: "Anchor + Projection Constrained LoRA",
    shortLabel: "Anchor + Projection",
    color: "#2b57aa",
    protects: "Stress retention as a hard first-order constraint.",
    reliesOn: "Drift minibatches, anchor set, and projected update geometry.",
    sacrifices: "Accepts controlled drift-fit penalty to enforce retention.",
    deployWhen: "Stress preservation is non-negotiable for production gating.",
    failureMode: "Excess projection distortion can throttle adaptation.",
    symbols: ["D_t", "A_t", "Pi(g_new | g_anc)"],
    pipeline: [
      { kind: "data", text: "Drift D_t + Anchor A_t" },
      { kind: "update", text: "g_new , g_anc" },
      { kind: "rule", text: "g = Pi(g_new|g_anc) + beta g_anc" },
      { kind: "opt", text: "mu_t -> optimizer" },
    ],
  },
};

export const PRODUCTION_LENSES = [
  {
    id: "stress_retention",
    title: "Stress Retention",
    question: "Does the strategy retain stress behavior while adapting to drift?",
    why: "Prevents capital impairment when market regime shifts abruptly.",
  },
  {
    id: "drawdown_control",
    title: "Drawdown Control",
    question: "How deep does the equity trough get under shared market paths?",
    why: "Institutional risk teams optimize survival first, then upside.",
  },
  {
    id: "adaptation_quality",
    title: "Adaptation Quality",
    question: "How well does the model fit current drift information?",
    why: "Without drift learning, edge decays and strategy relevance fades.",
  },
  {
    id: "implementation_friction",
    title: "Implementation Friction",
    question: "How much turnover and exposure churn does it induce?",
    why: "Execution costs and operational complexity can erase paper alpha.",
  },
  {
    id: "regime_robustness",
    title: "Regime Robustness",
    question: "Are outcomes stable across drift and stress segments?",
    why: "A strategy that only works in one regime is not production reliable.",
  },
];

export const PRACTITIONER_REFERENCES = [
  {
    title: "Multi-Period Trading via Convex Optimization",
    link: "https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf",
    why: "Production framing for constraints, transaction costs, and turnover-aware optimization.",
  },
  {
    title: "Transaction Costs and Portfolio Strategies",
    link: "https://www.nber.org/papers/w22355",
    why: "Quantifies how turnover and costs change optimal strategy choices.",
  },
  {
    title: "The Deflated Sharpe Ratio",
    link: "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551",
    why: "Adjusts conviction for multiple testing and backtest overfitting risk.",
  },
  {
    title: "The Probability of Backtest Overfitting",
    link: "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253",
    why: "Defines practical failure modes in strategy selection pipelines.",
  },
  {
    title: "A Backtesting Protocol in the Era of Machine Learning",
    link: "https://arxiv.org/abs/1907.12665",
    why: "Recommended validation protocol for model-based trading systems.",
  },
];
