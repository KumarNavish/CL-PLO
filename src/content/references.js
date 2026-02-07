export const REFERENCES = {
  papers: [
    {
      title: "Online Portfolio Selection: A Survey",
      authors: "Li, Hoi (2014)",
      link: "https://dl.acm.org/doi/10.1145/2638548",
      why: "Background on online portfolio adaptation and drift handling.",
    },
    {
      title: "Online Optimization with Long Term Constraints",
      authors: "Mahdavi, Jin, Yang (NeurIPS 2012)",
      link: "https://proceedings.neurips.cc/paper/2012/hash/5d44ee6f2c3f71b73125876103c8f6c4-Abstract.html",
      why: "Foundation for constraint-aware online updates.",
    },
    {
      title: "Multi-Period Trading via Convex Optimization",
      authors: "Boyd et al. (2017)",
      link: "https://web.stanford.edu/~boyd/papers/pdf/cvx_portfolio.pdf",
      why: "Deterministic optimization layer for risk and turnover guardrails.",
    },
    {
      title: "Optimization of Conditional Value-at-Risk",
      authors: "Rockafellar, Uryasev (2000)",
      link: "https://www.sciencedirect.com/science/article/pii/S0927539899000525",
      why: "CVaR as a tail-risk control in constrained portfolio optimization.",
    },
    {
      title: "LoRA: Low-Rank Adaptation of Large Language Models",
      authors: "Hu et al. (ICLR 2022)",
      link: "https://arxiv.org/abs/2106.09685",
      why: "PEFT mechanism used for frozen backbone + small adaptive subspace.",
    },
    {
      title: "Efficient Lifelong Learning with A-GEM",
      authors: "Chaudhry et al. (ICLR 2019)",
      link: "https://openreview.net/forum?id=Hkf2_sC5FX",
      why: "Closed-form gradient projection to avoid catastrophic interference.",
    },
    {
      title: "Gradient Projection for Continual Parameter-Efficient Tuning",
      authors: "Qiao et al. (2024)",
      link: "https://arxiv.org/abs/2405.13383",
      why: "Projection in PEFT subspace for retention under streaming updates.",
    },
  ],
  datasets: [
    {
      title: "Fama/French Data Library",
      link: "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html",
      why: "Factors and benchmark returns for stress and regime studies.",
    },
    {
      title: "FRED (Federal Reserve Economic Data)",
      link: "https://fred.stlouisfed.org/",
      why: "Macro state variables for regime labeling and context features.",
    },
    {
      title: "SEC EDGAR Filings",
      link: "https://www.sec.gov/edgar/search/",
      why: "Textual disclosures for market+language multimodal signals.",
    },
  ],
  repositories: [
    {
      title: "microsoft/LoRA",
      link: "https://github.com/microsoft/LoRA",
      why: "Reference implementation of low-rank adaptation.",
    },
    {
      title: "ContinualAI/avalanche",
      link: "https://github.com/ContinualAI/avalanche",
      why: "Continual learning baselines and protocols.",
    },
    {
      title: "cvxgrp/cvxportfolio",
      link: "https://github.com/cvxgrp/cvxportfolio",
      why: "Constraint-aware portfolio optimization framework.",
    },
  ],
};
