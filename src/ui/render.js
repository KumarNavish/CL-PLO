import { clampConfig, DEFAULT_CONFIG, PRESETS } from "../config.js";
import { drawEquity, drawImpactBars } from "./charts.js";

const FIELD_MAP = ["seed", "steps", "anchorBeta", "pStress", "loraRank"];
const METHOD_COLORS = {
  naive: "#767676",
  anchor: "#4b4b4b",
  anchor_proj: "#111111",
};

let worker = null;
let latestResult = null;

export function initApp() {
  fillForm(DEFAULT_CONFIG);

  document.getElementById("run-demo")?.addEventListener("click", () => runCurrentConfig());
  document.getElementById("apply-quick")?.addEventListener("click", () => applyPreset("quick_check"));
  document.getElementById("apply-proposal")?.addEventListener("click", () => applyPreset("proposal_like"));
  document.getElementById("apply-stress")?.addEventListener("click", () => applyPreset("stress_heavy"));
  document.getElementById("reset-form")?.addEventListener("click", () => {
    fillForm(DEFAULT_CONFIG);
    setStatus("Controls reset to default.");
  });
  document.getElementById("export-run")?.addEventListener("click", () => exportCurrentRun());

  window.addEventListener("resize", () => {
    if (latestResult) {
      renderCharts(latestResult);
    }
  });

  runCurrentConfig();
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
      renderAll(payload);
      return;
    }

    if (type === "error") {
      setRunning(false);
      setStatus(`Run failed: ${payload.message}`, true);
    }
  });

  return worker;
}

function runCurrentConfig() {
  const userCfg = readConfigFromForm();
  const safe = clampConfig({ ...DEFAULT_CONFIG, ...userCfg });
  fillForm(safe);

  setRunning(true);
  setStatus("Running validation...");
  setProgress(0);

  ensureWorker().postMessage({
    type: "run",
    payload: { config: safe },
  });
}

function readConfigFromForm() {
  const cfg = {};

  for (const key of FIELD_MAP) {
    const input = document.querySelector(`[name="${key}"]`);
    if (!input) {
      continue;
    }
    cfg[key] = Number(input.value);
  }

  return cfg;
}

function fillForm(cfg) {
  for (const key of FIELD_MAP) {
    const input = document.querySelector(`[name="${key}"]`);
    if (!input || cfg[key] === undefined) {
      continue;
    }
    input.value = String(cfg[key]);
  }
}

function applyPreset(name) {
  const preset = PRESETS[name];
  if (!preset) {
    return;
  }

  const merged = clampConfig({ ...DEFAULT_CONFIG, ...readConfigFromForm(), ...preset.values });
  fillForm(merged);
  setStatus(`Preset applied: ${preset.label}`);
}

function setRunning(isRunning) {
  const runButton = document.getElementById("run-demo");
  if (!runButton) {
    return;
  }

  runButton.disabled = isRunning;
  runButton.textContent = isRunning ? "Running..." : "Run Validation";
}

function setStatus(msg, isError = false) {
  const status = document.getElementById("status-line");
  if (!status) {
    return;
  }

  status.textContent = msg;
  status.classList.toggle("error", isError);
}

function setProgress(value) {
  const pct = Math.max(0, Math.min(100, value));
  const bar = document.getElementById("progress-fill");
  if (!bar) {
    return;
  }

  bar.style.width = `${pct.toFixed(1)}%`;
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
    setStatus("Compiling summary...");
  }
}

function renderAll(result) {
  setProgress(100);
  setStatus("Validation complete. Compare observed ordering with the pipeline expectations.");

  renderExpectationCheck(result);
  renderDecisionCard(result);
  renderKpis(result);
  renderCharts(result);
  renderMethodTable(result);
  renderTakeaway(result);
}

function renderExpectationCheck(result) {
  const host = document.getElementById("expectation-check");
  if (!host) {
    return;
  }

  const n = result.metrics.naive;
  const a = result.metrics.anchor;
  const p = result.metrics.anchor_proj;

  const stressOrder = n.stressMse > a.stressMse && a.stressMse > p.stressMse;
  const driftOrder = n.driftMse <= a.driftMse && a.driftMse <= p.driftMse;

  host.innerHTML = `
    <h3>What this run checks</h3>
    <ul>
      <li>
        Stress retention should improve from naive to anchor to projection (lower stress MSE).
        <span class="${stressOrder ? "pass" : "warn"}">${stressOrder ? "Observed" : "Not fully observed"}</span>
      </li>
      <li>
        Drift-fit error is expected to rise as constraints become stricter.
        <span class="${driftOrder ? "pass" : "warn"}">${driftOrder ? "Observed" : "Not fully observed"}</span>
      </li>
      <li>Use these checks before reading the decision card and score table.</li>
    </ul>
  `;
}

function renderDecisionCard(result) {
  const host = document.getElementById("decision-card");
  if (!host) {
    return;
  }

  const naive = result.metrics.naive;
  const proj = result.metrics.anchor_proj;

  const stressGain = improvement(naive.stressMse, proj.stressMse);
  const drawdownGain = proj.maxDrawdown - naive.maxDrawdown;
  const driftPenalty = ratioPenalty(naive.driftMse, proj.driftMse);

  let level = "caution";
  let title = "Provisional decision: keep in pilot review";

  if (stressGain > 0.8 && drawdownGain > 0.06 && driftPenalty < 70) {
    level = "good";
    title = "Provisional decision: promote to pilot";
  } else if (stressGain < 0.45 || drawdownGain < 0) {
    level = "bad";
    title = "Provisional decision: do not promote";
  }

  host.className = `decision-card ${level}`;
  host.innerHTML = `
    <h4>${title}</h4>
    <p>
      Relative to naive, projection changes stress retention by <strong>${pct(stressGain)}</strong>, drawdown by
      <strong>${pp(drawdownGain)}</strong>, and drift-fit error by <strong>${pct(driftPenalty / 100)}</strong>.
    </p>
  `;
}

function renderKpis(result) {
  const host = document.getElementById("impact-kpis");
  if (!host) {
    return;
  }

  const naive = result.metrics.naive;
  const anchor = result.metrics.anchor;
  const proj = result.metrics.anchor_proj;

  const stressGain = improvement(naive.stressMse, proj.stressMse);
  const drawdownGain = proj.maxDrawdown - naive.maxDrawdown;
  const anchorGap = proj.stressMse - anchor.stressMse;

  host.innerHTML = `
    <article class="${classBySign(stressGain)}">
      <div class="label">Stress retention</div>
      <div class="value">${pct(stressGain)}</div>
      <div class="note">projection vs naive</div>
    </article>
    <article class="${classBySign(drawdownGain)}">
      <div class="label">Drawdown change</div>
      <div class="value">${pp(drawdownGain)}</div>
      <div class="note">projection vs naive</div>
    </article>
    <article class="${classBySign(-anchorGap)}">
      <div class="label">Projection-anchor gap</div>
      <div class="value">${fmt(anchorGap, 6)}</div>
      <div class="note">stress MSE delta</div>
    </article>
  `;
}

function renderCharts(result) {
  const impactCanvas = document.getElementById("impact-chart");
  const equityCanvas = document.getElementById("equity-chart");

  if (!impactCanvas || !equityCanvas) {
    return;
  }

  const n = result.metrics.naive;
  const a = result.metrics.anchor;
  const p = result.metrics.anchor_proj;

  drawImpactBars(impactCanvas, [
    {
      label: "Stress retention gain",
      anchor: 100 * improvement(n.stressMse, a.stressMse),
      proj: 100 * improvement(n.stressMse, p.stressMse),
    },
    {
      label: "Drawdown change",
      anchor: 100 * (a.maxDrawdown - n.maxDrawdown),
      proj: 100 * (p.maxDrawdown - n.maxDrawdown),
    },
  ]);

  const lineSeries = result.methods.map((method) => ({
    label: method.label,
    color: METHOD_COLORS[method.id] || "#111111",
    values: result.equityCurves[method.id],
  }));

  drawEquity(equityCanvas, lineSeries, result.stressMarkers);
}

function renderMethodTable(result) {
  const host = document.getElementById("method-table");
  if (!host) {
    return;
  }

  const metrics = result.metrics;
  const ids = result.methods.map((m) => m.id);
  const stressVals = ids.map((id) => metrics[id].stressMse);
  const driftVals = ids.map((id) => metrics[id].driftMse);
  const drawVals = ids.map((id) => metrics[id].maxDrawdown);

  const stressScore = normalizeLowerBetter(stressVals);
  const driftScore = normalizeLowerBetter(driftVals);
  const drawScore = normalizeHigherBetter(drawVals);

  const rows = result.methods.map((method, idx) => {
    const m = metrics[method.id];
    const score = 100 * (0.55 * stressScore[idx] + 0.3 * drawScore[idx] + 0.15 * driftScore[idx]);

    return {
      method,
      stressMse: m.stressMse,
      driftMse: m.driftMse,
      maxDrawdown: m.maxDrawdown,
      totalReturn: m.totalReturn,
      score,
    };
  });

  rows.sort((a, b) => b.score - a.score);

  host.innerHTML = `
    <thead>
      <tr>
        <th>Method</th>
        <th>Stress MSE</th>
        <th>Drawdown</th>
        <th>Drift MSE</th>
        <th>Total Return</th>
        <th>Score</th>
      </tr>
    </thead>
    <tbody>
      ${rows
        .map(
          (row) => `
            <tr>
              <td>
                  <span class="method-name">
                  <span class="dot" style="background:${METHOD_COLORS[row.method.id] || "#111111"}"></span>
                  ${row.method.label}
                </span>
              </td>
              <td>${fmt(row.stressMse, 6)}</td>
              <td>${pct(row.maxDrawdown)}</td>
              <td>${fmt(row.driftMse, 6)}</td>
              <td>${pct(row.totalReturn)}</td>
              <td><strong>${row.score.toFixed(1)}</strong></td>
            </tr>
          `,
        )
        .join("")}
    </tbody>
  `;
}

function renderTakeaway(result) {
  const host = document.getElementById("takeaway");
  if (!host) {
    return;
  }

  const naive = result.metrics.naive;
  const anchor = result.metrics.anchor;
  const proj = result.metrics.anchor_proj;

  const stressGainProj = improvement(naive.stressMse, proj.stressMse);
  const stressGainAnchor = improvement(naive.stressMse, anchor.stressMse);
  const ddProj = proj.maxDrawdown - naive.maxDrawdown;

  host.innerHTML = `
    <h4>Reading this run</h4>
    <p>
      In this sample path, projection improves stress retention by <strong>${pct(stressGainProj)}</strong> relative to
      naive (anchor: ${pct(stressGainAnchor)}), with drawdown change <strong>${pp(ddProj)}</strong>. Treat this as
      one piece of evidence; promotion requires the same ordering to hold across repeated stress-heavy configurations.
    </p>
  `;
}

function exportCurrentRun() {
  if (!latestResult) {
    setStatus("Run validation before exporting.", true);
    return;
  }

  const report = {
    exportedAt: new Date().toISOString(),
    config: latestResult.config,
    keyResult: latestResult.keyResult,
    metrics: latestResult.metrics,
  };

  const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = `cl_plo_run_${Date.now()}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);

  setStatus("Run report exported.");
}

function classBySign(v) {
  if (v > 0.001) {
    return "good";
  }
  if (v < -0.001) {
    return "bad";
  }
  return "caution";
}

function normalizeLowerBetter(values) {
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = Math.max(1e-12, hi - lo);
  return values.map((v) => (hi - v) / span);
}

function normalizeHigherBetter(values) {
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = Math.max(1e-12, hi - lo);
  return values.map((v) => (v - lo) / span);
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

function fmt(x, digits = 4) {
  if (!Number.isFinite(x)) {
    return "n/a";
  }

  const ax = Math.abs(x);
  if (ax > 0 && (ax < 1e-4 || ax > 1e4)) {
    return x.toExponential(2);
  }
  return x.toFixed(digits);
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
