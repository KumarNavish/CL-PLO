const REPOSITORIES = [
  {
    name: "youngandbin/LLM-BLM",
    role: "Optimization Core",
    description: "Official implementation of LLM-driven Black-Litterman portfolio construction.",
    objective:
      "Convert LLM-generated market views into structured Black-Litterman priors/posteriors for portfolio optimization.",
    repo: "https://github.com/youngandbin/LLM-BLM",
    docs: "https://github.com/youngandbin/LLM-BLM#readme",
  },
  {
    name: "franjgs/llm-rl-finance-trader",
    role: "Hybrid Decision System",
    description: "FinBERT sentiment + PPO reinforcement learning for trading and allocation policies.",
    objective:
      "Test whether language sentiment signals improve RL trading outcomes relative to non-LLM baselines.",
    repo: "https://github.com/franjgs/llm-rl-finance-trader",
    docs: "https://github.com/franjgs/llm-rl-finance-trader#readme",
  },
  {
    name: "pipiku915/FinMem-LLM-StockTrading",
    role: "Signal + Memory Engine",
    description: "FinMem: memory-augmented LLM framework for stock trading decisions.",
    objective:
      "Model market context with layered memory and LLM reasoning to support higher-quality trading signals.",
    repo: "https://github.com/pipiku915/FinMem-LLM-StockTrading",
    docs: "https://github.com/pipiku915/FinMem-LLM-StockTrading#readme",
  },
  {
    name: "AI4Finance-Foundation/FinGPT",
    role: "Signal Generation",
    description: "Open-source financial LLM ecosystem for data engineering, tuning, and task-specific finance signals.",
    objective:
      "Produce robust financial NLP outputs (sentiment, classification, analysis) that can feed optimization layers.",
    repo: "https://github.com/AI4Finance-Foundation/FinGPT",
    docs: "https://github.com/AI4Finance-Foundation/FinGPT#readme",
  },
  {
    name: "AI4Finance-Foundation/FinRobot",
    role: "Agentic Research Stack",
    description: "LLM-agent framework for financial analysis and decision workflows.",
    objective:
      "Orchestrate analyst-style financial workflows where LLM outputs can be converted to actionable portfolio views.",
    repo: "https://github.com/AI4Finance-Foundation/FinRobot",
    docs: "https://github.com/AI4Finance-Foundation/FinRobot#readme",
  },
  {
    name: "TradingGoose/TradingGoose.github.io",
    role: "Multi-Agent Pipeline",
    description: "Open-source multi-agent LLM stock analysis and portfolio management system.",
    objective:
      "Combine multiple specialized LLM agents to generate, validate, and execute portfolio decisions.",
    repo: "https://github.com/TradingGoose/TradingGoose.github.io",
    docs: "https://tradinggoose.github.io/",
  },
  {
    name: "OnePunchMonk/AgentQuant",
    role: "Strategy Discovery",
    description: "Agentic quant research workflow using LLM planning and backtest loops.",
    objective:
      "Generate and evaluate strategy ideas from structured prompts, then feed selected signals to portfolio construction.",
    repo: "https://github.com/OnePunchMonk/AgentQuant",
    docs: "https://github.com/OnePunchMonk/AgentQuant#readme",
  },
  {
    name: "The-FinAI/PIXIU",
    role: "Benchmark + Testbed",
    description: "Financial benchmark ecosystem with LLM-centered evaluation and related trading research assets.",
    objective:
      "Provide evaluation protocols and public testbeds for finance-focused LLM decision systems.",
    repo: "https://github.com/The-FinAI/PIXIU",
    docs: "https://github.com/The-FinAI/PIXIU#readme",
  },
  {
    name: "Finance-LLMs/FinMCP",
    role: "Data/Execution Infrastructure",
    description: "Model Context Protocol setup for integrating LLMs with financial data and broker/trading endpoints.",
    objective:
      "Operationalize LLM-powered workflows by standardizing tool/data access for decision and execution pipelines.",
    repo: "https://github.com/Finance-LLMs/FinMCP",
    docs: "https://github.com/Finance-LLMs/FinMCP#readme",
  },
];

const PAPER_REFS = [
  {
    title: "Language Models Meet Portfolio Optimization: LLM-Enhanced Black-Litterman",
    link: "https://arxiv.org/abs/2411.18565",
    note: "Shows direct integration of LLM views into a portfolio optimizer.",
  },
  {
    title: "FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory",
    link: "https://arxiv.org/abs/2311.13743",
    note: "Memory-augmented LLM design for financial decision consistency.",
  },
  {
    title: "FinGPT: Open-Source Financial Large Language Models",
    link: "https://arxiv.org/abs/2306.06031",
    note: "Open finance-LLM stack for generating domain-specific signals.",
  },
  {
    title: "A Backtesting Protocol in the Era of Machine Learning",
    link: "https://arxiv.org/abs/1907.12665",
    note: "Validation standard for avoiding false strategy conviction.",
  },
];

const EVIDENCE_REFS = [
  {
    label: "LLM-BLM experiments",
    link: "https://github.com/youngandbin/LLM-BLM",
    note: "Ablations on LLM-generated views and portfolio outcomes.",
  },
  {
    label: "FinMem paper + code",
    link: "https://github.com/pipiku915/FinMem-LLM-StockTrading",
    note: "Demonstrates memory effects on decision quality and trading performance.",
  },
  {
    label: "FinGPT tasks and benchmarks",
    link: "https://github.com/AI4Finance-Foundation/FinGPT",
    note: "Evidence of financial language adaptation capabilities.",
  },
  {
    label: "Agentic workflow demos",
    link: "https://github.com/OnePunchMonk/AgentQuant",
    note: "Shows practical strategy-iteration loops around LLM planning.",
  },
];

const ROLE_DESCRIPTIONS = {
  "Signal Generation": "Transformer models convert text to structured alpha signals.",
  "Signal + Memory Engine": "LLM memory systems improve temporal consistency of decisions.",
  "Optimization Core": "LLM outputs are fused directly into formal portfolio optimization.",
  "Hybrid Decision System": "Language signals are combined with RL/control-style decision policies.",
  "Agentic Research Stack": "LLM agents orchestrate multi-step investment analysis workflows.",
  "Multi-Agent Pipeline": "Specialized agents collaborate to produce portfolio-level actions.",
  "Strategy Discovery": "LLMs generate and test strategy hypotheses rapidly.",
  "Benchmark + Testbed": "Evaluation frameworks and tasks for LLM finance systems.",
  "Data/Execution Infrastructure": "Connector layer between LLM reasoning and market/execution tools.",
};

const IC_BRIEF = [
  {
    title: "Decision Use",
    body: "Use LLM outputs as views/scenarios feeding a constrained optimizer, not as direct execution orders.",
  },
  {
    title: "Expected Benefit",
    body: "Earlier detection of narrative and regime inflections that market-only features can miss.",
  },
  {
    title: "Key Constraint",
    body: "Signal quality must be continuously stress-tested to avoid hidden turnover and false confidence.",
  },
];

const DEPLOYMENT_SNAPSHOT = [
  {
    title: "Data Inputs",
    body: "Daily prices/volumes, benchmark/factor data, and timestamped text streams (news, filings, call transcripts).",
  },
  {
    title: "Integration Point",
    body: "LLM views enter as expected-return/scenario adjustments before constrained portfolio optimization.",
  },
  {
    title: "Monitoring",
    body: "Track signal hit-rate, drawdown pressure, turnover budget, and confidence drift each rebalance cycle.",
  },
  {
    title: "Success Metric",
    body: "Sustained uplift in risk-adjusted return after costs, with stable drawdown and policy-compliant turnover.",
  },
];

const REPO_SCORES = {
  "youngandbin/LLM-BLM": { llmDepth: 5, optFit: 5, repro: 4, deploy: 3 },
  "franjgs/llm-rl-finance-trader": { llmDepth: 4, optFit: 3, repro: 3, deploy: 2 },
  "pipiku915/FinMem-LLM-StockTrading": { llmDepth: 5, optFit: 3, repro: 3, deploy: 2 },
  "AI4Finance-Foundation/FinGPT": { llmDepth: 5, optFit: 3, repro: 4, deploy: 4 },
  "AI4Finance-Foundation/FinRobot": { llmDepth: 4, optFit: 3, repro: 4, deploy: 3 },
  "TradingGoose/TradingGoose.github.io": { llmDepth: 4, optFit: 4, repro: 3, deploy: 3 },
  "OnePunchMonk/AgentQuant": { llmDepth: 4, optFit: 3, repro: 3, deploy: 3 },
  "The-FinAI/PIXIU": { llmDepth: 4, optFit: 2, repro: 4, deploy: 2 },
  "Finance-LLMs/FinMCP": { llmDepth: 3, optFit: 2, repro: 3, deploy: 5 },
};

const FAILURE_MODES = [
  {
    item: "Narrative overreaction",
    note: "LLM overweights transient headlines, causing unstable allocations.",
  },
  {
    item: "Silent turnover drift",
    note: "Signal updates appear profitable but implementation costs erase edge.",
  },
  {
    item: "Regime misclassification",
    note: "Text tone differs from realized market state, degrading portfolio positioning.",
  },
  {
    item: "Prompt/config fragility",
    note: "Performance depends on prompt settings not robust across time windows.",
  },
];

const VALIDATION_PROTOCOL = [
  {
    item: "Walk-forward backtests",
    note: "No global tuning leaks. Refit and evaluate by realistic temporal blocks.",
  },
  {
    item: "Ablation tests",
    note: "Compare optimizer-only vs optimizer+LLM to isolate true incremental value.",
  },
  {
    item: "Stress segment evaluation",
    note: "Report drawdown and tail losses specifically during high-volatility regimes.",
  },
  {
    item: "Cost-aware simulation",
    note: "Model fees, slippage, and capacity before claiming production readiness.",
  },
];

const MONITORING_GATES = [
  {
    item: "Signal-health gate",
    note: "If directional hit rate drops below threshold, reduce LLM influence alpha.",
  },
  {
    item: "Risk gate",
    note: "If drawdown or VaR breaches policy, enforce conservative fallback weights.",
  },
  {
    item: "Execution gate",
    note: "If turnover exceeds budget, freeze rebalance frequency and recalc.",
  },
  {
    item: "Rollback gate",
    note: "Automatic reversion to baseline optimizer if two consecutive gates fail.",
  },
];

const ASSETS = ["US Equities", "Rates", "Commodities", "Cash"];

const REGIME_CONFIG = {
  bull: {
    mu: [0.12, 0.04, 0.07, 0.02],
    vol: [0.2, 0.07, 0.16, 0.01],
    signal: [0.8, -0.2, 0.5, 0],
  },
  neutral: {
    mu: [0.07, 0.03, 0.05, 0.02],
    vol: [0.16, 0.06, 0.13, 0.01],
    signal: [0.5, 0.1, 0.35, 0],
  },
  bear: {
    mu: [-0.02, 0.04, 0.02, 0.02],
    vol: [0.22, 0.08, 0.17, 0.01],
    signal: [-0.6, 0.4, -0.2, 0],
  },
  stress: {
    mu: [-0.08, 0.05, -0.03, 0.02],
    vol: [0.29, 0.09, 0.22, 0.01],
    signal: [-0.9, 0.55, -0.45, 0],
  },
};

const REBAL_EVERY = 20;
const STEPS = 220;

function init() {
  wireModeToggle();
  renderIcBrief();
  renderDeploymentSnapshot();
  renderTaxonomy();
  renderRepositories();
  renderRepoMatrix();
  renderReferences();
  renderDiligence();
  wireControls();
  runDemo();
}

function wireModeToggle() {
  const focusBtn = document.getElementById("mode-focus");
  const researchBtn = document.getElementById("mode-research");
  const hint = document.getElementById("mode-hint");
  const deepLinks = document.querySelectorAll('a[href="#repos"], a[href="#math"], a[href="#comparison"], a[href="#diligence"]');

  function setMode(mode) {
    const isFocus = mode === "focus";
    document.body.classList.toggle("mode-focus", isFocus);
    focusBtn.classList.toggle("active", isFocus);
    researchBtn.classList.toggle("active", !isFocus);
    hint.textContent = isFocus
      ? "First-Read Mode shows only decision-critical content. Switch to Research Mode for full repository, math, and production due-diligence depth."
      : "Research Mode enabled. Full repository map, equations, comparative analysis, and due-diligence layers are visible.";
  }

  focusBtn.addEventListener("click", () => setMode("focus"));
  researchBtn.addEventListener("click", () => setMode("research"));
  for (const link of deepLinks) {
    link.addEventListener("click", () => setMode("research"));
  }
  setMode("focus");
}

function renderIcBrief() {
  const host = document.getElementById("ic-cards");
  host.innerHTML = IC_BRIEF.map(
    (x) => `
      <article>
        <h3>${x.title}</h3>
        <p>${x.body}</p>
      </article>
    `,
  ).join("");
}

function renderDeploymentSnapshot() {
  const host = document.getElementById("deploy-cards");
  host.innerHTML = DEPLOYMENT_SNAPSHOT.map(
    (x) => `
      <article>
        <h3>${x.title}</h3>
        <p>${x.body}</p>
      </article>
    `,
  ).join("");
}

function renderTaxonomy() {
  const host = document.getElementById("taxonomy");

  const counts = {};
  for (const repo of REPOSITORIES) {
    counts[repo.role] = (counts[repo.role] || 0) + 1;
  }

  host.innerHTML = Object.keys(counts)
    .map(
      (role) => `
      <article>
        <h3>${role}</h3>
        <p>${ROLE_DESCRIPTIONS[role]}</p>
        <p><strong>${counts[role]} repository${counts[role] > 1 ? "ies" : ""}</strong></p>
      </article>
    `,
    )
    .join("");
}

function renderRepositories() {
  const host = document.getElementById("repo-list");
  host.innerHTML = REPOSITORIES.map(
    (repo) => `
      <article class="repo-card">
        <header>
          <h3>${repo.name}</h3>
          <span class="repo-role">${repo.role}</span>
        </header>
        <p class="repo-desc">${repo.description}</p>
        <p class="repo-obj"><strong>Objective:</strong> ${repo.objective}</p>
        <div class="repo-links">
          <a href="${repo.repo}" target="_blank" rel="noreferrer">Repository</a>
          <a href="${repo.docs}" target="_blank" rel="noreferrer">Documentation</a>
        </div>
      </article>
    `,
  ).join("");
}

function renderRepoMatrix() {
  const host = document.getElementById("repo-matrix");
  const ranked = REPOSITORIES.map((r) => {
    const s = REPO_SCORES[r.name];
    const score = 0.3 * s.llmDepth + 0.35 * s.optFit + 0.15 * s.repro + 0.2 * s.deploy;
    return { repo: r, scores: s, score };
  }).sort((a, b) => b.score - a.score);

  host.innerHTML = `
    <table class="comparison-table">
      <thead>
        <tr>
          <th>Repository</th>
          <th>LLM Depth</th>
          <th>Optimization Fit</th>
          <th>Reproducibility</th>
          <th>Deployment Readiness</th>
          <th>Conviction Score</th>
        </tr>
      </thead>
      <tbody>
        ${ranked.map(({ repo, scores, score }) => {
          return `
            <tr>
              <th>${repo.name}</th>
              <td>${scores.llmDepth}/5</td>
              <td>${scores.optFit}/5</td>
              <td>${scores.repro}/5</td>
              <td>${scores.deploy}/5</td>
              <td><strong>${score.toFixed(2)}/5</strong></td>
            </tr>
          `;
        }).join("")}
      </tbody>
    </table>
  `;
}

function renderReferences() {
  const paperHost = document.getElementById("paper-refs");
  const codeHost = document.getElementById("codebase-refs");
  const evidenceHost = document.getElementById("evidence-refs");

  paperHost.innerHTML = PAPER_REFS.map(
    (r) => `<li><a href="${r.link}" target="_blank" rel="noreferrer">${r.title}</a><span>${r.note}</span></li>`,
  ).join("");

  codeHost.innerHTML = REPOSITORIES.map(
    (r) => `<li><a href="${r.repo}" target="_blank" rel="noreferrer">${r.name}</a><span>${r.role}</span></li>`,
  ).join("");

  evidenceHost.innerHTML = EVIDENCE_REFS.map(
    (r) => `<li><a href="${r.link}" target="_blank" rel="noreferrer">${r.label}</a><span>${r.note}</span></li>`,
  ).join("");
}

function renderDiligence() {
  const failureHost = document.getElementById("failure-modes");
  const protocolHost = document.getElementById("validation-protocol");
  const monitoringHost = document.getElementById("monitoring-gates");

  failureHost.innerHTML = FAILURE_MODES.map(
    (x) => `<li><strong>${x.item}</strong><span>${x.note}</span></li>`,
  ).join("");
  protocolHost.innerHTML = VALIDATION_PROTOCOL.map(
    (x) => `<li><strong>${x.item}</strong><span>${x.note}</span></li>`,
  ).join("");
  monitoringHost.innerHTML = MONITORING_GATES.map(
    (x) => `<li><strong>${x.item}</strong><span>${x.note}</span></li>`,
  ).join("");
}

function wireControls() {
  const risk = document.getElementById("risk");
  const signal = document.getElementById("signal");

  risk.addEventListener("input", () => {
    document.getElementById("risk-val").textContent = risk.value;
  });

  signal.addEventListener("input", () => {
    document.getElementById("signal-val").textContent = Number(signal.value).toFixed(2);
  });

  document.getElementById("run-demo").addEventListener("click", runDemo);
}

function runDemo() {
  const regime = document.getElementById("regime").value;
  const risk = Number(document.getElementById("risk").value);
  const signalStrength = Number(document.getElementById("signal").value);

  const result = simulateStrategies({ regime, risk, signalStrength });

  renderDecision(result, regime, risk, signalStrength);
  renderMetrics(result);
  renderRunInterpretation(result);
  drawEquityChart(document.getElementById("equity-chart"), result.paths);
  drawWeightsChart(document.getElementById("weights-chart"), result.finalWeights);
}

function renderRunInterpretation(result) {
  const host = document.getElementById("run-interpretation");
  const b = result.metrics.baseline;
  const l = result.metrics.llm;

  const retDelta = l.annReturn - b.annReturn;
  const sharpeDelta = l.sharpe - b.sharpe;
  const ddDelta = l.maxDrawdown - b.maxDrawdown;
  const turnDelta = l.turnover - b.turnover;
  const gates = evaluateRunGates({ retDelta, sharpeDelta, ddDelta, turnDelta });
  const passCount = gates.filter((g) => g.status === "pass").length;
  let call = "Hold: investigate signal quality and cost controls before promotion.";
  if (passCount >= 4) {
    call = "Promote to larger paper-trading allocation with monitoring gates active.";
  } else if (passCount <= 1) {
    call = "Reject for deployment in current form; keep as research candidate only.";
  }

  host.innerHTML = `
    <h3>Run Interpretation</h3>
    <ul>
      <li>Return spread: <strong>${pp(retDelta)}</strong> annualized vs baseline.</li>
      <li>Risk-adjusted spread: Sharpe <strong>${delta(sharpeDelta)}</strong>.</li>
      <li>Tail impact: drawdown change <strong>${pp(ddDelta)}</strong> (higher is better).</li>
      <li>Implementation burden: turnover change <strong>${delta(turnDelta)}</strong> per rebalance cycle.</li>
    </ul>
    <div class="gate-grid">
      ${gates
        .map(
          (g) => `
            <article class="gate ${g.status}">
              <h4>${g.name}</h4>
              <p>${g.value}</p>
              <span>${g.label}</span>
            </article>
          `,
        )
        .join("")}
    </div>
    <p class="gate-call"><strong>Deployment call:</strong> ${call}</p>
  `;
}

function simulateStrategies({ regime, risk, signalStrength }) {
  const cfg = REGIME_CONFIG[regime];
  const seed = hashSeed(`${regime}-${risk}-${signalStrength.toFixed(2)}`);
  const rng = mulberry32(seed);

  const baseCov = buildCov(cfg.vol);

  let wBaseline = [0.25, 0.25, 0.25, 0.25];
  let wLlm = [0.25, 0.25, 0.25, 0.25];

  const eqBaseline = [1];
  const eqLlm = [1];

  let turnBase = 0;
  let turnLlm = 0;
  const edgeDaily = cfg.signal.map((x) => 0.00022 * signalStrength * x);

  const retBase = [];
  const retLlm = [];

  for (let t = 0; t < STEPS; t += 1) {
    if (t % REBAL_EVERY === 0) {
      const muAnnual = cfg.mu;
      const llmShift = cfg.signal.map((x) => 0.08 * signalStrength * x);

      const muBase = muAnnual;
      const muLlm = muAnnual.map((m, i) => m + llmShift[i]);

      const newBase = optimizeWeights(muBase, baseCov, risk, wBaseline);
      const newLlm = optimizeWeights(muLlm, baseCov, risk, wLlm);

      turnBase += l1Dist(newBase, wBaseline);
      turnLlm += l1Dist(newLlm, wLlm);

      wBaseline = newBase;
      wLlm = newLlm;
    }

    const draw = sampleReturns(cfg.mu, cfg.vol, edgeDaily, rng);

    const rb = dot(wBaseline, draw);
    const rl = dot(wLlm, draw);

    retBase.push(rb);
    retLlm.push(rl);

    eqBaseline.push(eqBaseline[eqBaseline.length - 1] * (1 + rb));
    eqLlm.push(eqLlm[eqLlm.length - 1] * (1 + rl));
  }

  const metricsBase = computeMetrics(retBase, eqBaseline, turnBase / (STEPS / REBAL_EVERY));
  const metricsLlm = computeMetrics(retLlm, eqLlm, turnLlm / (STEPS / REBAL_EVERY));

  return {
    metrics: {
      baseline: metricsBase,
      llm: metricsLlm,
    },
    paths: {
      baseline: eqBaseline,
      llm: eqLlm,
    },
    finalWeights: wLlm,
  };
}

function optimizeWeights(muAnnual, covAnnual, risk, prevW) {
  const n = muAnnual.length;
  const mu = muAnnual.map((x) => x / 12);
  const cov = covAnnual.map((row) => row.map((x) => x / 12));

  const gamma = 0.15 + 0.12 * risk;
  const tc = 0.04;

  let w = prevW.slice();

  for (let iter = 0; iter < 220; iter += 1) {
    const grad = new Array(n).fill(0);

    for (let i = 0; i < n; i += 1) {
      let riskTerm = 0;
      for (let j = 0; j < n; j += 1) {
        riskTerm += cov[i][j] * w[j];
      }

      const tcTerm = Math.sign(w[i] - prevW[i]);
      grad[i] = mu[i] - 2 * gamma * riskTerm - tc * tcTerm;
    }

    for (let i = 0; i < n; i += 1) {
      w[i] += 0.22 * grad[i];
    }

    w = projectSimplex(w);
  }

  return w;
}

function evaluateRunGates({ retDelta, sharpeDelta, ddDelta, turnDelta }) {
  return [
    {
      name: "Return Spread",
      value: `${pp(retDelta)}`,
      status: retDelta >= 0.01 ? "pass" : retDelta >= 0 ? "warn" : "fail",
      label: retDelta >= 0.01 ? "Pass" : retDelta >= 0 ? "Watch" : "Fail",
    },
    {
      name: "Sharpe Spread",
      value: `${delta(sharpeDelta)}`,
      status: sharpeDelta >= 0.05 ? "pass" : sharpeDelta >= 0 ? "warn" : "fail",
      label: sharpeDelta >= 0.05 ? "Pass" : sharpeDelta >= 0 ? "Watch" : "Fail",
    },
    {
      name: "Drawdown Impact",
      value: `${pp(ddDelta)}`,
      status: ddDelta >= -0.01 ? "pass" : ddDelta >= -0.02 ? "warn" : "fail",
      label: ddDelta >= -0.01 ? "Pass" : ddDelta >= -0.02 ? "Watch" : "Fail",
    },
    {
      name: "Turnover Impact",
      value: `${delta(turnDelta)}`,
      status: turnDelta <= 0.04 ? "pass" : turnDelta <= 0.08 ? "warn" : "fail",
      label: turnDelta <= 0.04 ? "Pass" : turnDelta <= 0.08 ? "Watch" : "Fail",
    },
  ];
}

function projectSimplex(v) {
  const n = v.length;
  const u = v.slice().sort((a, b) => b - a);
  let rho = 0;
  let cssv = 0;

  for (let i = 0; i < n; i += 1) {
    cssv += u[i];
    const t = (cssv - 1) / (i + 1);
    if (u[i] - t > 0) {
      rho = i + 1;
    }
  }

  let sumTop = 0;
  for (let i = 0; i < rho; i += 1) {
    sumTop += u[i];
  }
  const theta = (sumTop - 1) / Math.max(1, rho);

  const out = v.map((x) => Math.max(0, x - theta));
  const s = out.reduce((a, b) => a + b, 0);
  if (s <= 1e-12) {
    return new Array(n).fill(1 / n);
  }
  return out.map((x) => x / s);
}

function sampleReturns(muAnnual, volAnnual, edgeDaily, rng) {
  const dt = 1 / 252;
  const out = [];

  for (let i = 0; i < muAnnual.length; i += 1) {
    const z = boxMuller(rng);
    const r = muAnnual[i] * dt + volAnnual[i] * Math.sqrt(dt) * z + edgeDaily[i];
    out.push(r);
  }

  return out;
}

function buildCov(volAnnual) {
  const corr = [
    [1, -0.2, 0.25, 0],
    [-0.2, 1, -0.1, 0],
    [0.25, -0.1, 1, 0],
    [0, 0, 0, 1],
  ];

  const n = volAnnual.length;
  const cov = Array.from({ length: n }, () => new Array(n).fill(0));

  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      cov[i][j] = corr[i][j] * volAnnual[i] * volAnnual[j];
    }
  }

  return cov;
}

function computeMetrics(returns, equity, turnover) {
  const meanDaily = mean(returns);
  const volDaily = std(returns);

  const annReturn = meanDaily * 252;
  const annVol = volDaily * Math.sqrt(252);
  const sharpe = annReturn / Math.max(1e-12, annVol);

  return {
    annReturn,
    annVol,
    sharpe,
    maxDrawdown: maxDrawdown(equity),
    turnover,
  };
}

function renderDecision(result, regime, risk, signalStrength) {
  const host = document.getElementById("decision-summary");
  const b = result.metrics.baseline;
  const l = result.metrics.llm;

  const retDelta = l.annReturn - b.annReturn;
  const ddDelta = l.maxDrawdown - b.maxDrawdown;
  const sharpeDelta = l.sharpe - b.sharpe;

  let verdict = "caution";
  let title = "Decision: Validate Further";

  if (retDelta > 0.01 && sharpeDelta > 0.05 && ddDelta >= -0.01) {
    verdict = "good";
    title = "Decision: LLM-Augmented Variant Is Candidate";
  }

  host.className = `decision-summary ${verdict}`;
  host.innerHTML = `
    <h3>${title}</h3>
    <p>
      Regime <strong>${regime}</strong>, risk tolerance <strong>${risk}</strong>, signal strength
      <strong>${signalStrength.toFixed(2)}</strong>.
    </p>
    <p>
      LLM vs baseline: annual return <strong>${pp(retDelta)}</strong>, Sharpe <strong>${delta(sharpeDelta)}</strong>,
      max drawdown <strong>${pp(ddDelta)}</strong>.
    </p>
  `;
}

function renderMetrics(result) {
  const host = document.getElementById("metric-cards");
  const b = result.metrics.baseline;
  const l = result.metrics.llm;

  host.innerHTML = `
    <article>
      <div class="k">Ann Return</div>
      <div class="v">${pct(l.annReturn)}</div>
      <div class="n">Baseline ${pct(b.annReturn)}</div>
    </article>
    <article>
      <div class="k">Ann Vol</div>
      <div class="v">${pct(l.annVol)}</div>
      <div class="n">Baseline ${pct(b.annVol)}</div>
    </article>
    <article>
      <div class="k">Sharpe</div>
      <div class="v">${l.sharpe.toFixed(2)}</div>
      <div class="n">Baseline ${b.sharpe.toFixed(2)}</div>
    </article>
    <article>
      <div class="k">Max Drawdown</div>
      <div class="v">${pct(l.maxDrawdown)}</div>
      <div class="n">Baseline ${pct(b.maxDrawdown)}</div>
    </article>
    <article>
      <div class="k">Turnover / Rebalance</div>
      <div class="v">${l.turnover.toFixed(2)}</div>
      <div class="n">Baseline ${b.turnover.toFixed(2)}</div>
    </article>
  `;
}

function drawEquityChart(canvas, paths) {
  const series = [
    { label: "Traditional baseline", color: "#cf5a3e", values: paths.baseline },
    { label: "LLM-augmented", color: "#1f6eb2", values: paths.llm },
  ];

  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);

  const dims = { left: 52, right: rect.width - 14, top: 20, bottom: rect.height - 38 };

  const all = series.flatMap((s) => s.values);
  const yMin = Math.min(...all) * 0.98;
  const yMax = Math.max(...all) * 1.02;

  drawGrid(ctx, dims, 4);
  drawAxes(ctx, dims, "Time", "Portfolio value");

  const maxT = Math.max(...series.map((s) => s.values.length - 1));
  const xToPx = (t) => dims.left + (t / Math.max(1, maxT)) * (dims.right - dims.left);
  const yToPx = (y) => dims.bottom - ((y - yMin) / Math.max(1e-12, yMax - yMin)) * (dims.bottom - dims.top);

  for (const s of series) {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2;
    ctx.beginPath();

    for (let t = 0; t < s.values.length; t += 1) {
      const x = xToPx(t);
      const y = yToPx(s.values[t]);
      if (t === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }

    ctx.stroke();
  }

  drawLegend(ctx, series, rect.width, 10);
}

function drawWeightsChart(canvas, weights) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);

  const dims = { left: 52, right: rect.width - 14, top: 18, bottom: rect.height - 42 };

  drawGrid(ctx, dims, 3);
  drawAxes(ctx, dims, "Assets", "Weight");

  const n = weights.length;
  const w = (dims.right - dims.left) / n;
  const barW = Math.min(36, w * 0.48);

  for (let i = 0; i < n; i += 1) {
    const x = dims.left + i * w + w / 2;
    const h = (weights[i] / 1.0) * (dims.bottom - dims.top);
    const y = dims.bottom - h;

    ctx.fillStyle = "#1f6eb2";
    ctx.fillRect(x - barW / 2, y, barW, h);

    ctx.fillStyle = "#26415f";
    ctx.font = "11px 'IBM Plex Sans', 'Avenir Next', sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(ASSETS[i], x, dims.bottom + 14);
    ctx.fillText(pct(weights[i]), x, y - 6);
  }
}

function drawGrid(ctx, dims, ticks) {
  ctx.strokeStyle = "rgba(83, 106, 137, 0.18)";
  ctx.lineWidth = 1;

  for (let i = 1; i <= ticks; i += 1) {
    const x = dims.left + ((dims.right - dims.left) * i) / (ticks + 1);
    const y = dims.top + ((dims.bottom - dims.top) * i) / (ticks + 1);

    ctx.beginPath();
    ctx.moveTo(x, dims.top);
    ctx.lineTo(x, dims.bottom);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(dims.left, y);
    ctx.lineTo(dims.right, y);
    ctx.stroke();
  }
}

function drawAxes(ctx, dims, xLabel, yLabel) {
  ctx.strokeStyle = "#8a99b1";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(dims.left, dims.top);
  ctx.lineTo(dims.left, dims.bottom);
  ctx.lineTo(dims.right, dims.bottom);
  ctx.stroke();

  ctx.fillStyle = "#4a5e79";
  ctx.font = "12px 'IBM Plex Sans', 'Avenir Next', sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(xLabel, (dims.left + dims.right) / 2, dims.bottom + 28);

  ctx.save();
  ctx.translate(dims.left - 34, (dims.top + dims.bottom) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
}

function drawLegend(ctx, series, width, y) {
  let x = width - 260;
  ctx.font = "12px 'IBM Plex Sans', 'Avenir Next', sans-serif";

  for (const s of series) {
    ctx.fillStyle = s.color;
    ctx.fillRect(x, y, 14, 3);

    ctx.fillStyle = "#22374f";
    ctx.textAlign = "left";
    ctx.fillText(s.label, x + 18, y + 4);

    x += ctx.measureText(s.label).width + 42;
  }
}

function maxDrawdown(eq) {
  let peak = eq[0];
  let worst = 0;

  for (let i = 0; i < eq.length; i += 1) {
    if (eq[i] > peak) {
      peak = eq[i];
    }
    const dd = eq[i] / peak - 1;
    if (dd < worst) {
      worst = dd;
    }
  }

  return worst;
}

function l1Dist(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i += 1) {
    s += Math.abs(a[i] - b[i]);
  }
  return s;
}

function boxMuller(rng) {
  let u = 0;
  let v = 0;
  while (u === 0) {
    u = rng();
  }
  while (v === 0) {
    v = rng();
  }
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function mulberry32(seed) {
  let t = seed >>> 0;
  return function next() {
    t += 0x6d2b79f5;
    let z = t;
    z = Math.imul(z ^ (z >>> 15), z | 1);
    z ^= z + Math.imul(z ^ (z >>> 7), z | 61);
    return ((z ^ (z >>> 14)) >>> 0) / 4294967296;
  };
}

function hashSeed(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i += 1) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i += 1) {
    s += a[i] * b[i];
  }
  return s;
}

function mean(values) {
  if (values.length === 0) {
    return 0;
  }
  let s = 0;
  for (let i = 0; i < values.length; i += 1) {
    s += values[i];
  }
  return s / values.length;
}

function std(values) {
  if (values.length < 2) {
    return 0;
  }
  const m = mean(values);
  let s = 0;
  for (let i = 0; i < values.length; i += 1) {
    const d = values[i] - m;
    s += d * d;
  }
  return Math.sqrt(s / (values.length - 1));
}

function pct(x) {
  return `${(x * 100).toFixed(2)}%`;
}

function pp(x) {
  const v = x * 100;
  return `${v >= 0 ? "+" : ""}${v.toFixed(2)} pp`;
}

function delta(x) {
  return `${x >= 0 ? "+" : ""}${x.toFixed(2)}`;
}

init();
