import { clampConfig, DEFAULT_CONFIG, METHOD_SPECS, PRESETS } from "../config.js";
import {
  PRACTITIONER_REFERENCES,
  STRATEGY_SYSTEMS,
  VERSION_LABEL,
  VERSION_TAG,
} from "../content/strategy-systems.js";
import { drawEquity } from "./charts.js";

const RUNNABLE_KEYS = ["seed", "steps", "anchorBeta", "pStress"];

let worker = null;
let activeConfig = { ...DEFAULT_CONFIG };
let latestResult = null;
let latestSummary = null;

export function initComparisonPage() {
  renderVersionChip();
  renderSystemRows();
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
      activeConfig = clampConfig({ ...activeConfig, ...readControls() });
      syncControls(activeConfig);
      setStatus("Knobs updated. Run scorecard to refresh the decision.");
    });
  }

  const deepDive = document.querySelector(".deep-dive");
  if (deepDive) {
    deepDive.addEventListener("toggle", () => {
      if (deepDive.open && latestResult && latestSummary) {
        renderCharts(latestResult, latestSummary);
      }
    });
  }

  window.addEventListener("resize", () => {
    if (latestResult && latestSummary) {
      renderCharts(latestResult, latestSummary);
    }
  });
}

function renderVersionChip() {
  const host = document.getElementById("version-chip");
  host.textContent = `Locked baseline: ${VERSION_LABEL} (${VERSION_TAG})`;
}

function renderSystemRows() {
  const host = document.getElementById("system-rows");

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
          <p><strong>Uses:</strong> ${meta.reliesOn}</p>
          <p><strong>Sacrifices:</strong> ${meta.sacrifices}</p>
          <p><strong>Deploy when:</strong> ${meta.deployWhen}</p>
        </div>
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
  activeConfig = clampConfig({ ...activeConfig, ...readControls() });
  syncControls(activeConfig);

  setRunning(true);
  setStatus("Running shared-path comparison...");
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
      renderProgress(payload);
      return;
    }

    if (type === "result") {
      setRunning(false);
      latestResult = payload;
      latestSummary = summarizeMethods(payload);
      renderAll(payload, latestSummary);
      return;
    }

    if (type === "error") {
      setRunning(false);
      setStatus(`Run failed: ${payload.message}`, true);
    }
  });

  return worker;
}

function renderProgress(payload) {
  if (payload.kind === "training") {
    const numerator = payload.methodIndex * payload.totalSteps + payload.step;
    const denominator = payload.totalMethods * payload.totalSteps;
    setProgress((100 * numerator) / Math.max(1, denominator));
    setStatus(`Training ${payload.methodLabel}: ${payload.step}/${payload.totalSteps}`);
    return;
  }

  if (payload.kind === "building_charts") {
    setProgress(100);
    setStatus("Finalizing decision outputs...");
  }
}

function renderAll(result, summary) {
  setStatus("Run complete. Read the decision card, then inspect the frontier chart.");
  setProgress(100);

  const rows = buildGateRows(summary);
  const totals = buildTotals(rows);

  renderWinnerCard(totals, summary);
  renderGateStrip(rows);
  renderDecisionMatrix(rows);
  renderCharts(result, summary);
}

function summarizeMethods(result) {
  const out = {};

  for (const method of METHOD_SPECS) {
    const id = method.id;
    const metrics = result.metrics[id];
    const diag = result.sharedDiagnostics[id];

    out[id] = {
      ...metrics,
      turnover: mean(diag.turnovers || []),
    };
  }

  return out;
}

function buildGateRows(summary) {
  const ids = METHOD_SPECS.map((m) => m.id);

  const stressVals = ids.map((id) => summary[id].stressMse);
  const drawdownVals = ids.map((id) => summary[id].maxDrawdown);
  const driftVals = ids.map((id) => summary[id].driftMse);
  const turnoverVals = ids.map((id) => summary[id].turnover);

  const stressScore = normalizeLowerBetter(stressVals);
  const drawdownScore = normalizeHigherBetter(drawdownVals);
  const driftScore = normalizeLowerBetter(driftVals);
  const turnoverScore = normalizeLowerBetter(turnoverVals);

  return [
    {
      id: "stress",
      title: "Stress Retention",
      subtitle: "Lower stress MSE is better",
      weight: 0.4,
      raw: mapById(ids, stressVals),
      score: mapById(ids, stressScore),
      format: (x) => fmtSci(x),
    },
    {
      id: "drawdown",
      title: "Drawdown Control",
      subtitle: "Less negative max drawdown is better",
      weight: 0.3,
      raw: mapById(ids, drawdownVals),
      score: mapById(ids, drawdownScore),
      format: (x) => pct(x),
    },
    {
      id: "adaptation",
      title: "Drift Adaptation",
      subtitle: "Lower drift MSE is better",
      weight: 0.2,
      raw: mapById(ids, driftVals),
      score: mapById(ids, driftScore),
      format: (x) => fmtSci(x),
    },
    {
      id: "turnover",
      title: "Implementation Friction",
      subtitle: "Lower turnover is better",
      weight: 0.1,
      raw: mapById(ids, turnoverVals),
      score: mapById(ids, turnoverScore),
      format: (x) => x.toFixed(3),
    },
  ];
}

function buildTotals(rows) {
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

function renderWinnerCard(totals, summary) {
  const host = document.getElementById("winner-card");
  const ordered = [...METHOD_SPECS].sort((a, b) => totals[b.id] - totals[a.id]);

  const naive = summary.naive;
  const gated = ordered.map((m) => m.id).filter((id) => passesHardGate(summary[id], naive));
  const winner = gated.length > 0 ? gated[0] : ordered[0].id;

  const w = summary[winner];
  const stressGain = improvement(naive.stressMse, w.stressMse);
  const drawdownGain = w.maxDrawdown - naive.maxDrawdown;
  const driftPenalty = ratioPenalty(naive.driftMse, w.driftMse);

  const hardPass = passesHardGate(w, naive);

  host.className = `winner-card ${hardPass ? "good" : "caution"}`;
  host.innerHTML = `
    <h3>Recommended for Pilot: ${STRATEGY_SYSTEMS[winner].label}</h3>
    <p>
      ${hardPass ? "Passes" : "Does not fully pass"} hard stress gate.
      Stress retention gain <strong>${pct(stressGain)}</strong>,
      drawdown improvement <strong>${pp(drawdownGain)}</strong>,
      drift penalty <strong>${pct(driftPenalty / 100)}</strong>.
    </p>
    <p class="next-action">
      Next action: ${hardPass ? "run pilot validation with live risk gates" : "tune anchor/projection settings before pilot"}.
    </p>
  `;
}

function renderGateStrip(rows) {
  const host = document.getElementById("gate-strip");

  host.innerHTML = rows
    .map((row) => {
      const winner = winnerId(row.score);
      return `
        <article class="gate-card">
          <h4>${row.title}</h4>
          <p>${row.subtitle}</p>
          <p><strong>Gate winner:</strong> ${STRATEGY_SYSTEMS[winner].shortLabel}</p>
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
          <th>Gate</th>
          ${METHOD_SPECS.map((m) => `<th>${STRATEGY_SYSTEMS[m.id].shortLabel}</th>`).join("")}
        </tr>
      </thead>
      <tbody>
        ${rows
          .map(
            (row) => `
              <tr>
                <th>
                  <div class="gate-title">${row.title}</div>
                  <div class="gate-note">Weight ${Math.round(100 * row.weight)}%</div>
                </th>
                ${METHOD_SPECS.map((m) => renderMatrixCell(row, m.id)).join("")}
              </tr>
            `,
          )
          .join("")}
      </tbody>
    </table>
  `;
}

function renderMatrixCell(row, methodId) {
  const score = row.score[methodId];
  const raw = row.raw[methodId];

  return `
    <td class="score-cell">
      <div class="score-bar"><span style="width:${Math.max(0, Math.min(100, score)).toFixed(1)}%"></span></div>
      <div class="score-meta">
        <strong>${score.toFixed(1)}</strong>
        <small>${row.format(raw)}</small>
      </div>
    </td>
  `;
}

function renderCharts(result, summary) {
  drawRegimeFrontier(document.getElementById("regime-frontier"), summary);

  const eqCanvas = document.getElementById("compare-equity");
  if (!eqCanvas) {
    return;
  }

  const rect = eqCanvas.getBoundingClientRect();
  if (rect.width < 40 || rect.height < 40) {
    return;
  }

  const eqSeries = METHOD_SPECS.map((method) => ({
    label: STRATEGY_SYSTEMS[method.id].shortLabel,
    color: method.color,
    values: result.equityCurves[method.id],
  }));

  drawEquity(eqCanvas, eqSeries, result.stressMarkers || []);
}

function drawRegimeFrontier(canvas, summary) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;

  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);

  const dims = { left: 64, right: rect.width - 16, top: 20, bottom: rect.height - 46 };
  const naive = summary.naive;

  const points = METHOD_SPECS.map((method) => {
    const m = summary[method.id];
    return {
      id: method.id,
      label: STRATEGY_SYSTEMS[method.id].shortLabel,
      color: method.color,
      x: 100 * improvement(naive.stressMse, m.stressMse),
      y: 100 * (m.maxDrawdown - naive.maxDrawdown),
    };
  });

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);

  const xMin = Math.min(-5, ...xs) - 4;
  const xMax = Math.max(5, ...xs) + 4;
  const yMin = Math.min(-5, ...ys) - 4;
  const yMax = Math.max(5, ...ys) + 4;

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
    ctx.fillText(p.label, x + 8, y - 6);
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

function passesHardGate(candidate, naive) {
  const stressGain = improvement(naive.stressMse, candidate.stressMse);
  const ddGain = candidate.maxDrawdown - naive.maxDrawdown;
  return stressGain > 0.5 && ddGain > 0.04;
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
  button.textContent = isRunning ? "Running..." : "Run Decision Scorecard";
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

function improvement(base, current) {
  if (Math.abs(base) < 1e-12) {
    return 0;
  }
  return (base - current) / Math.abs(base);
}

function ratioPenalty(base, current) {
  if (Math.abs(base) < 1e-12) {
    return 0;
  }
  return Math.max(0, (current / base - 1) * 100);
}

function fmtSci(x) {
  if (!Number.isFinite(x)) {
    return "n/a";
  }
  if (Math.abs(x) > 0 && Math.abs(x) < 1e-4) {
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
  const v = x * 100;
  return `${v >= 0 ? "+" : ""}${v.toFixed(2)} pp`;
}
