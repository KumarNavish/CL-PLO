import { clampConfig, DEFAULT_CONFIG, METHOD_SPECS, PRESETS } from "../config.js";
import {
  PRACTITIONER_REFERENCES,
  PRODUCTION_LENSES,
  STRATEGY_SYSTEMS,
  VERSION_LABEL,
  VERSION_TAG,
} from "../content/strategy-systems.js";
import { drawEquity } from "./charts.js";

const RUNNABLE_KEYS = ["seed", "steps", "anchorBeta", "pStress"];

let worker = null;
let activeConfig = { ...DEFAULT_CONFIG };
let latestResult = null;

export function initComparisonPage() {
  renderVersionChip();
  renderProductionLenses();
  renderSystemCards();
  renderPractitionerRefs();
  syncControls(activeConfig);
  bindEvents();
  runScorecard();
}

function bindEvents() {
  document.getElementById("compare-run").addEventListener("click", () => runScorecard());

  document.getElementById("compare-quick").addEventListener("click", () => applyPreset("quick_check"));
  document.getElementById("compare-default").addEventListener("click", () => applyPreset("proposal_like"));
  document.getElementById("compare-stress").addEventListener("click", () => applyPreset("stress_heavy"));

  for (const key of RUNNABLE_KEYS) {
    const input = document.getElementById(controlId(key));
    if (!input) {
      continue;
    }

    input.addEventListener("change", () => {
      const merged = clampConfig({ ...activeConfig, ...readControls() });
      activeConfig = merged;
      syncControls(activeConfig);
      setStatus("Knobs updated. Run scorecard to refresh decisions.");
    });
  }

  window.addEventListener("resize", () => {
    if (latestResult) {
      renderCharts(latestResult, summarizeAllMethods(latestResult));
    }
  });
}

function renderVersionChip() {
  const host = document.getElementById("version-chip");
  host.textContent = `Locked baseline: ${VERSION_LABEL} (${VERSION_TAG})`;
}

function renderProductionLenses() {
  const host = document.getElementById("production-lenses");
  host.innerHTML = PRODUCTION_LENSES.map(
    (lens) => `
      <article class="lens-card">
        <h3>${lens.title}</h3>
        <p class="question">${lens.question}</p>
        <p class="why">${lens.why}</p>
      </article>
    `,
  ).join("");
}

function renderSystemCards() {
  const host = document.getElementById("system-cards");

  host.innerHTML = METHOD_SPECS.map((method) => {
    const meta = STRATEGY_SYSTEMS[method.id];

    return `
      <article class="system-card">
        <header>
          <span class="dot" style="background:${method.color}"></span>
          <h3>${meta.label}</h3>
        </header>

        <div class="system-lines">
          <p><strong>Protects:</strong> ${meta.protects}</p>
          <p><strong>Relies on:</strong> ${meta.reliesOn}</p>
          <p><strong>Sacrifices:</strong> ${meta.sacrifices}</p>
          <p><strong>Deploy when:</strong> ${meta.deployWhen}</p>
        </div>

        <div class="symbol-row">
          ${meta.symbols.map((s) => `<code>${s}</code>`).join("")}
        </div>

        <div class="pipeline-strip">
          ${meta.pipeline
            .map(
              (step, idx) => `
                <div class="pipeline-step kind-${step.kind}">${step.text}</div>
                ${idx < meta.pipeline.length - 1 ? '<span class="pipeline-arrow">-></span>' : ""}
              `,
            )
            .join("")}
        </div>

        <p class="failure"><strong>Failure mode:</strong> ${meta.failureMode}</p>
      </article>
    `;
  }).join("");
}

function renderPractitionerRefs() {
  const host = document.getElementById("practitioner-refs");
  host.innerHTML = PRACTITIONER_REFERENCES.map(
    (ref) => `
      <li>
        <a href="${ref.link}" target="_blank" rel="noreferrer">${ref.title}</a>
        <span>${ref.why}</span>
      </li>
    `,
  ).join("");
}

function applyPreset(name) {
  const preset = PRESETS[name];
  if (!preset) {
    return;
  }

  activeConfig = clampConfig({ ...activeConfig, ...preset.values });
  syncControls(activeConfig);
  setStatus(`Preset applied: ${preset.label}`);
}

function runScorecard() {
  const merged = clampConfig({ ...activeConfig, ...readControls() });
  activeConfig = merged;
  syncControls(activeConfig);

  setRunning(true);
  setStatus("Running shared-path evaluation across all strategy systems...");
  setProgress(0);

  ensureWorker().postMessage({
    type: "run",
    payload: { config: activeConfig },
  });
}

function ensureWorker() {
  if (worker) {
    return worker;
  }

  worker = new Worker(new URL("../workers/experiment-worker.js", import.meta.url), {
    type: "module",
  });

  worker.addEventListener("message", (event) => {
    const { type, payload } = event.data;

    if (type === "progress") {
      handleProgress(payload);
      return;
    }

    if (type === "result") {
      setRunning(false);
      latestResult = payload;
      renderScorecardResult(payload);
      return;
    }

    if (type === "error") {
      setRunning(false);
      setStatus(`Run failed: ${payload.message}`, true);
    }
  });

  return worker;
}

function handleProgress(payload) {
  if (payload.kind === "training") {
    const numerator = payload.methodIndex * payload.totalSteps + payload.step;
    const denominator = payload.totalMethods * payload.totalSteps;

    setProgress((100 * numerator) / Math.max(1, denominator));
    setStatus(`Training ${payload.methodLabel}: step ${payload.step}/${payload.totalSteps}`);
    return;
  }

  if (payload.kind === "building_charts") {
    setProgress(100);
    setStatus("Finalizing scorecard...");
  }
}

function renderScorecardResult(result) {
  setStatus("Run complete. Compare systems and choose a deployment candidate.");
  setProgress(100);

  const summaryByMethod = summarizeAllMethods(result);
  const rows = buildDecisionRows(summaryByMethod);
  const totals = buildOverallTotals(rows);

  renderOverallRanking(totals, summaryByMethod);
  renderDecisionMatrix(rows);
  renderDeploymentCall(totals, summaryByMethod);
  renderCharts(result, summaryByMethod);
}

function summarizeAllMethods(result) {
  const summaries = {};
  const regimes = result.streamRegimes || [];

  for (const method of METHOD_SPECS) {
    const id = method.id;
    const metrics = result.metrics[id];
    const diag = result.sharedDiagnostics[id];

    const returns = diag.returns || [];
    const turnovers = diag.turnovers || [];

    const stressReturns = [];
    const driftReturns = [];
    for (let t = 0; t < returns.length; t += 1) {
      const regime = regimes[t] || "drift";
      if (regime === "stress") {
        stressReturns.push(returns[t]);
      } else {
        driftReturns.push(returns[t]);
      }
    }

    const annRet = mean(returns) * 252;
    const annVol = std(returns) * Math.sqrt(252);
    const sharpe = annRet / Math.max(1e-12, annVol);

    const downside = std(returns.filter((x) => x < 0)) * Math.sqrt(252);
    const sortino = annRet / Math.max(1e-12, downside);

    const stressHitRate = fraction(stressReturns, (x) => x > -0.05);
    const regimeGap = Math.abs(mean(stressReturns) - mean(driftReturns));

    summaries[id] = {
      ...metrics,
      annRet,
      annVol,
      sharpe,
      sortino,
      turnover: mean(turnovers),
      stressHitRate,
      regimeGap,
    };
  }

  return summaries;
}

function buildDecisionRows(summaryByMethod) {
  const ids = METHOD_SPECS.map((m) => m.id);

  const driftVals = ids.map((id) => summaryByMethod[id].driftMse);
  const stressVals = ids.map((id) => summaryByMethod[id].stressMse);
  const drawVals = ids.map((id) => summaryByMethod[id].maxDrawdown);
  const worstStressVals = ids.map((id) => summaryByMethod[id].worstStressDay);
  const turnoverVals = ids.map((id) => summaryByMethod[id].turnover);
  const stressHitVals = ids.map((id) => summaryByMethod[id].stressHitRate);
  const regimeGapVals = ids.map((id) => summaryByMethod[id].regimeGap);
  const returnVals = ids.map((id) => summaryByMethod[id].totalReturn);
  const sharpeVals = ids.map((id) => summaryByMethod[id].sharpe);

  const scores = {
    adaptation: normalizeLowerBetter(driftVals),
    stress: normalizeLowerBetter(stressVals),
    drawdown: normalizeHigherBetter(drawVals),
    worstStress: normalizeHigherBetter(worstStressVals),
    turnover: normalizeLowerBetter(turnoverVals),
    stressHit: normalizeHigherBetter(stressHitVals),
    regimeGap: normalizeLowerBetter(regimeGapVals),
    ret: normalizeHigherBetter(returnVals),
    sharpe: normalizeHigherBetter(sharpeVals),
  };

  return [
    {
      id: "stress_retention",
      title: "Stress Retention",
      gate: "Preserve anchor behavior in stress regime.",
      weight: 0.24,
      values: mapById(ids, stressVals),
      score: mapById(ids, scores.stress),
      format: (x) => fmtSci(x),
    },
    {
      id: "drawdown_control",
      title: "Tail + Drawdown Control",
      gate: "Avoid deep troughs and severe stress-day losses.",
      weight: 0.22,
      values: mapById(ids, ids.map((id, i) => 0.65 * scores.drawdown[i] + 0.35 * scores.worstStress[i])),
      score: mapById(ids, ids.map((id, i) => 0.65 * scores.drawdown[i] + 0.35 * scores.worstStress[i])),
      format: (x) => `${x.toFixed(1)} score`,
    },
    {
      id: "adaptation_quality",
      title: "Drift Adaptation Quality",
      gate: "Track changing drift signal without collapse.",
      weight: 0.15,
      values: mapById(ids, driftVals),
      score: mapById(ids, scores.adaptation),
      format: (x) => fmtSci(x),
    },
    {
      id: "implementation_friction",
      title: "Implementation Friction",
      gate: "Limit turnover-induced execution drag.",
      weight: 0.11,
      values: mapById(ids, turnoverVals),
      score: mapById(ids, scores.turnover),
      format: (x) => x.toFixed(3),
    },
    {
      id: "regime_robustness",
      title: "Regime Robustness",
      gate: "Stable outcomes across drift and stress segments.",
      weight: 0.17,
      values: mapById(ids, ids.map((id, i) => 0.5 * scores.stressHit[i] + 0.5 * scores.regimeGap[i])),
      score: mapById(ids, ids.map((id, i) => 0.5 * scores.stressHit[i] + 0.5 * scores.regimeGap[i])),
      format: (x) => `${x.toFixed(1)} score`,
    },
    {
      id: "outcome_quality",
      title: "Outcome Quality",
      gate: "Return quality after risk filters.",
      weight: 0.11,
      values: mapById(ids, ids.map((id, i) => 0.55 * scores.ret[i] + 0.45 * scores.sharpe[i])),
      score: mapById(ids, ids.map((id, i) => 0.55 * scores.ret[i] + 0.45 * scores.sharpe[i])),
      format: (x) => `${x.toFixed(1)} score`,
    },
  ];
}

function buildOverallTotals(rows) {
  const totals = {};

  for (const method of METHOD_SPECS) {
    let s = 0;
    for (const row of rows) {
      s += row.weight * row.score[method.id];
    }
    totals[method.id] = s;
  }

  return totals;
}

function renderOverallRanking(totals, summaryByMethod) {
  const host = document.getElementById("overall-ranking");
  const ordered = [...METHOD_SPECS].sort((a, b) => totals[b.id] - totals[a.id]);

  host.innerHTML = ordered
    .map((method, idx) => {
      const m = summaryByMethod[method.id];
      return `
        <article class="rank-card ${idx === 0 ? "winner" : ""}">
          <div class="rank-head">
            <span class="rank-pos">#${idx + 1}</span>
            <h3><span class="dot" style="background:${method.color}"></span>${STRATEGY_SYSTEMS[method.id].shortLabel}</h3>
          </div>
          <p class="rank-score">Composite score: <strong>${totals[method.id].toFixed(1)}</strong></p>
          <p class="rank-detail">Stress MSE ${fmtSci(m.stressMse)} | Max DD ${pct(m.maxDrawdown)} | Return ${pct(m.totalReturn)}</p>
        </article>
      `;
    })
    .join("");
}

function renderDecisionMatrix(rows) {
  const host = document.getElementById("decision-matrix");

  host.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Decision Gate</th>
          ${METHOD_SPECS.map((m) => `<th>${STRATEGY_SYSTEMS[m.id].shortLabel}</th>`).join("")}
          <th>Preferred</th>
        </tr>
      </thead>
      <tbody>
        ${rows
          .map((row) => {
            const winner = winnerId(row.score);
            return `
              <tr>
                <th>
                  <div class="gate-title">${row.title}</div>
                  <div class="gate-note">${row.gate}</div>
                </th>
                ${METHOD_SPECS.map((m) => renderScoreCell(row, m.id, winner)).join("")}
                <td class="winner-cell">${STRATEGY_SYSTEMS[winner].shortLabel}</td>
              </tr>
            `;
          })
          .join("")}
      </tbody>
    </table>
  `;
}

function renderScoreCell(row, methodId, winner) {
  const score = row.score[methodId];
  const raw = row.values[methodId];
  const best = methodId === winner;

  return `
    <td class="score-cell ${best ? "best" : ""}">
      <div class="score-bar"><span style="width:${Math.max(0, Math.min(100, score)).toFixed(1)}%"></span></div>
      <div class="score-meta">
        <strong>${score.toFixed(1)}</strong>
        <small>${row.format(raw)}</small>
      </div>
    </td>
  `;
}

function renderDeploymentCall(totals, summaryByMethod) {
  const host = document.getElementById("deployment-call");

  const ordered = [...METHOD_SPECS].sort((a, b) => totals[b.id] - totals[a.id]);
  const n = summaryByMethod.naive;

  const gated = ordered
    .map((m) => m.id)
    .filter((id) => passesHardGate(summaryByMethod[id], n));

  const winner = gated.length > 0 ? gated[0] : ordered[0].id;
  const second = ordered.map((m) => m.id).find((id) => id !== winner) || winner;

  const w = summaryByMethod[winner];

  const stressGain = improvement(n.stressMse, w.stressMse);
  const ddGain = w.maxDrawdown - n.maxDrawdown;

  const hardGatePass = passesHardGate(w, n);
  const compositeWinner = ordered[0].id;

  host.className = `takeaway ${hardGatePass ? "good" : "caution"}`;
  host.innerHTML = `
    <h4>Deployment Recommendation</h4>
    <ul>
      <li><strong>Primary candidate:</strong> ${STRATEGY_SYSTEMS[winner].label} (composite ${totals[winner].toFixed(1)}).</li>
      <li><strong>Primary challenger:</strong> ${STRATEGY_SYSTEMS[second].label} (composite ${totals[second].toFixed(1)}).</li>
      <li><strong>Composite winner:</strong> ${STRATEGY_SYSTEMS[compositeWinner].shortLabel}. Gate override is applied when hard stress retention thresholds are missed.</li>
      <li>Winner vs naive: stress retention gain <strong>${pct(stressGain)}</strong>, drawdown improvement <strong>${pp(ddGain)}</strong>.</li>
      <li>Pilot gate: ${hardGatePass ? "pass" : "conditional"} under current synthetic stress frequency ${pct(activeConfig.pStress)}.</li>
      <li>Monitor before promotion: stress-hit-rate floor, turnover drift, and projection distortion trends.</li>
    </ul>
  `;
}

function passesHardGate(candidate, naive) {
  const stressGain = improvement(naive.stressMse, candidate.stressMse);
  const ddGain = candidate.maxDrawdown - naive.maxDrawdown;
  return stressGain > 0.5 && ddGain > 0.04;
}

function renderCharts(result, summaryByMethod) {
  const eqSeries = METHOD_SPECS.map((method) => ({
    label: STRATEGY_SYSTEMS[method.id].shortLabel,
    color: method.color,
    values: result.equityCurves[method.id],
  }));

  drawEquity(document.getElementById("compare-equity"), eqSeries, result.stressMarkers || []);
  drawRegimeFrontier(document.getElementById("regime-frontier"), summaryByMethod);
}

function drawRegimeFrontier(canvas, summaryByMethod) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);

  const dims = { left: 64, right: rect.width - 16, top: 24, bottom: rect.height - 46 };

  const naive = summaryByMethod.naive;
  const points = METHOD_SPECS.map((method) => {
    const m = summaryByMethod[method.id];
    return {
      id: method.id,
      label: STRATEGY_SYSTEMS[method.id].shortLabel,
      color: method.color,
      x: 100 * improvement(naive.stressMse, m.stressMse),
      y: 100 * (m.maxDrawdown - naive.maxDrawdown),
      ret: 100 * (m.totalReturn - naive.totalReturn),
    };
  });

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);

  const xMin = Math.min(-5, ...xs) - 5;
  const xMax = Math.max(5, ...xs) + 5;
  const yMin = Math.min(-5, ...ys) - 5;
  const yMax = Math.max(5, ...ys) + 5;

  const xToPx = (x) => dims.left + ((x - xMin) / Math.max(1e-12, xMax - xMin)) * (dims.right - dims.left);
  const yToPx = (y) => dims.bottom - ((y - yMin) / Math.max(1e-12, yMax - yMin)) * (dims.bottom - dims.top);

  drawGrid(ctx, dims, 4);
  drawAxes(ctx, dims, "Stress retention gain vs naive (pp)", "Drawdown improvement vs naive (pp)");

  const zeroX = xToPx(0);
  const zeroY = yToPx(0);

  ctx.strokeStyle = "rgba(41, 59, 87, 0.55)";
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(zeroX, dims.top);
  ctx.lineTo(zeroX, dims.bottom);
  ctx.moveTo(dims.left, zeroY);
  ctx.lineTo(dims.right, zeroY);
  ctx.stroke();

  for (const p of points) {
    const x = xToPx(p.x);
    const y = yToPx(p.y);

    ctx.fillStyle = p.color;
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "#24374f";
    ctx.font = "12px 'IBM Plex Sans', 'Avenir Next', sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(`${p.label} (${p.ret >= 0 ? "+" : ""}${p.ret.toFixed(1)}pp ret)`, x + 9, y - 6);
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
  ctx.strokeStyle = "#8b99af";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(dims.left, dims.top);
  ctx.lineTo(dims.left, dims.bottom);
  ctx.lineTo(dims.right, dims.bottom);
  ctx.stroke();

  ctx.fillStyle = "#4d5f78";
  ctx.font = "12px 'IBM Plex Sans', 'Avenir Next', sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(xLabel, (dims.left + dims.right) / 2, dims.bottom + 30);

  ctx.save();
  ctx.translate(dims.left - 44, (dims.top + dims.bottom) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
}

function winnerId(scoreMap) {
  let best = METHOD_SPECS[0].id;
  for (const method of METHOD_SPECS) {
    if (scoreMap[method.id] > scoreMap[best]) {
      best = method.id;
    }
  }
  return best;
}

function mapById(ids, values) {
  const out = {};
  for (let i = 0; i < ids.length; i += 1) {
    out[ids[i]] = values[i];
  }
  return out;
}

function normalizeHigherBetter(values) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min;

  if (!Number.isFinite(span) || span < 1e-12) {
    return values.map(() => 50);
  }

  return values.map((v) => (100 * (v - min)) / span);
}

function normalizeLowerBetter(values) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min;

  if (!Number.isFinite(span) || span < 1e-12) {
    return values.map(() => 50);
  }

  return values.map((v) => (100 * (max - v)) / span);
}

function controlId(key) {
  if (key === "seed") {
    return "compare-seed";
  }
  if (key === "steps") {
    return "compare-steps";
  }
  if (key === "anchorBeta") {
    return "compare-anchor-beta";
  }
  if (key === "pStress") {
    return "compare-p-stress";
  }
  return "";
}

function readControls() {
  const cfg = {};
  for (const key of RUNNABLE_KEYS) {
    const input = document.getElementById(controlId(key));
    cfg[key] = Number(input.value);
  }
  return cfg;
}

function syncControls(cfg) {
  for (const key of RUNNABLE_KEYS) {
    const input = document.getElementById(controlId(key));
    if (!input) {
      continue;
    }
    input.value = String(cfg[key]);
  }
}

function setStatus(msg, isError = false) {
  const host = document.getElementById("compare-status");
  host.textContent = msg;
  host.classList.toggle("error", isError);
}

function setProgress(value) {
  const pctValue = Math.max(0, Math.min(100, value));
  const bar = document.getElementById("compare-progress-fill");
  bar.style.width = `${pctValue.toFixed(1)}%`;
}

function setRunning(isRunning) {
  const button = document.getElementById("compare-run");
  button.disabled = isRunning;
  button.textContent = isRunning ? "Running..." : "Run Production Scorecard";
}

function mean(values) {
  if (!values || values.length === 0) {
    return 0;
  }
  let s = 0;
  for (let i = 0; i < values.length; i += 1) {
    s += values[i];
  }
  return s / values.length;
}

function std(values) {
  if (!values || values.length < 2) {
    return 0;
  }
  const m = mean(values);
  let s2 = 0;
  for (let i = 0; i < values.length; i += 1) {
    const d = values[i] - m;
    s2 += d * d;
  }
  return Math.sqrt(s2 / (values.length - 1));
}

function fraction(values, predicate) {
  if (!values || values.length === 0) {
    return 0;
  }

  let c = 0;
  for (let i = 0; i < values.length; i += 1) {
    if (predicate(values[i])) {
      c += 1;
    }
  }
  return c / values.length;
}

function improvement(base, current) {
  if (Math.abs(base) < 1e-12) {
    return 0;
  }
  return (base - current) / Math.abs(base);
}

function fmtSci(x) {
  if (!Number.isFinite(x)) {
    return "n/a";
  }
  if (Math.abs(x) < 1e-4) {
    return x.toExponential(2);
  }
  return x.toFixed(6);
}

function pct(x) {
  if (!Number.isFinite(x)) {
    return "n/a";
  }
  return `${(x * 100).toFixed(2)}%`;
}

function pp(x) {
  if (!Number.isFinite(x)) {
    return "n/a";
  }
  const val = x * 100;
  return `${val >= 0 ? "+" : ""}${val.toFixed(2)} pp`;
}
