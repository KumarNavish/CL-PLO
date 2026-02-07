import { clampConfig, DEFAULT_CONFIG, PRESETS } from "../config.js";
import { REFERENCES } from "../content/references.js";
import { drawEquity, drawScatter } from "./charts.js";

const FIELD_MAP = [
  "seed",
  "nTrainDrift",
  "nAnchorStress",
  "nTestDrift",
  "nTestStress",
  "loraRank",
  "lr",
  "steps",
  "batchSize",
  "anchorBatchSize",
  "anchorBeta",
  "simT",
  "pStress",
  "wMaxRisky",
  "turnoverEta",
  "noiseStd",
  "returnScale",
];

let worker = null;
let latestResult = null;

export function initApp() {
  fillForm(DEFAULT_CONFIG);
  renderReferences();

  document.getElementById("run-demo").addEventListener("click", () => runCurrentConfig());
  document.getElementById("apply-quick").addEventListener("click", () => applyPreset("quick_check"));
  document.getElementById("apply-proposal").addEventListener("click", () => applyPreset("proposal_like"));
  document.getElementById("apply-stress").addEventListener("click", () => applyPreset("stress_heavy"));
  document.getElementById("reset-form").addEventListener("click", () => fillForm(DEFAULT_CONFIG));
  document.getElementById("export-run").addEventListener("click", () => exportCurrentRun());

  document.querySelector("#hero-run").addEventListener("click", () => {
    document.querySelector("#demo").scrollIntoView({ behavior: "smooth", block: "start" });
    runCurrentConfig();
  });

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
  const cfg = readConfigFromForm();
  const safe = clampConfig(cfg);
  fillForm(safe);

  setRunning(true);
  setStatus("Initializing experiment...");
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

  const merged = { ...readConfigFromForm(), ...preset.values };
  fillForm(clampConfig(merged));
  setStatus(`Preset applied: ${preset.label}`);
}

function setRunning(isRunning) {
  const runButton = document.getElementById("run-demo");
  runButton.disabled = isRunning;
  runButton.textContent = isRunning ? "Running..." : "Run Experiment";
}

function setStatus(msg, isError = false) {
  const status = document.getElementById("status-line");
  status.textContent = msg;
  status.classList.toggle("error", isError);
}

function setProgress(value) {
  const pct = Math.max(0, Math.min(100, value));
  const bar = document.getElementById("progress-fill");
  bar.style.width = `${pct.toFixed(1)}%`;
}

function renderProgress(payload) {
  if (payload.kind === "training") {
    const globalNumerator = payload.methodIndex * payload.totalSteps + payload.step;
    const globalDenominator = payload.totalMethods * payload.totalSteps;
    const pct = (100 * globalNumerator) / globalDenominator;
    setProgress(pct);

    setStatus(`Training ${payload.methodLabel}: step ${payload.step}/${payload.totalSteps}`);
    return;
  }

  if (payload.kind === "building_charts") {
    setProgress(100);
    setStatus("Finalizing diagnostics and evidence cards...");
  }
}

function renderAll(result) {
  setStatus("Run complete. Review decision signal, then evaluate deployment assumptions.");
  setProgress(100);

  renderDecisionCard(result);
  renderMetrics(result);
  renderCharts(result);
  renderQualitative(result);
  renderClaimChecks(result);
}

function renderDecisionCard(result) {
  const host = document.getElementById("decision-card");
  const naive = result.metrics.naive;
  const proj = result.metrics.anchor_proj;

  const stressImprovement = fracImprove(naive.stressMse, proj.stressMse);
  const drawdownImprovement = proj.maxDrawdown - naive.maxDrawdown;
  const driftRatio = proj.driftMse / Math.max(naive.driftMse, 1e-12);

  let level = "caution";
  let title = "Decision Signal: Candidate for Validation";

  if (stressImprovement > 0.75 && drawdownImprovement > 0.03) {
    level = "good";
    title = "Decision Signal: Strong Pilot Candidate";
  } else if (stressImprovement < 0.25 || drawdownImprovement < -0.01) {
    level = "bad";
    title = "Decision Signal: Tune Before Pilot";
  }

  host.className = `decision-card ${level}`;
  host.innerHTML = `
    <h4>${title}</h4>
    <p>
      Projection vs naive: stress regression improvement <strong>${pct(stressImprovement)}</strong>,
      drawdown change <strong>${pct(drawdownImprovement)}</strong>,
      drift-fit ratio <strong>${fmt(driftRatio, 2)}x</strong>.
      Readout: favor this method when stress retention and tail control are hard constraints.
    </p>
  `;
}

function renderMetrics(result) {
  const host = document.getElementById("metrics-grid");

  host.innerHTML = result.methods
    .map((method) => {
      const m = result.metrics[method.id];
      const log = result.trainLogs[method.id] || {};

      return `
      <article class="metric-card">
        <div class="metric-card-top">
          <span class="dot" style="background:${method.color}"></span>
          <h4>${method.label}</h4>
        </div>
        <dl>
          <div><dt>Drift MSE</dt><dd>${fmt(m.driftMse, 6)}</dd></div>
          <div><dt>Stress MSE</dt><dd>${fmt(m.stressMse, 6)}</dd></div>
          <div><dt>Total Return</dt><dd>${pct(m.totalReturn)}</dd></div>
          <div><dt>Max Drawdown</dt><dd>${pct(m.maxDrawdown)}</dd></div>
          <div><dt>Worst Stress Day</dt><dd>${pct(m.worstStressDay)}</dd></div>
          <div><dt>Avg Risky (Stress)</dt><dd>${pct(m.avgRiskyWeightStress)}</dd></div>
          <div><dt>Avg Risky (Drift)</dt><dd>${pct(m.avgRiskyWeightDrift)}</dd></div>
          ${
            method.id === "anchor_proj"
              ? `<div><dt>Interference Rate</dt><dd>${pct(log.interferenceRate || 0)}</dd></div>
                 <div><dt>Update Distortion</dt><dd>${fmt(log.updateDistortion || 0, 4)}</dd></div>`
              : ""
          }
        </dl>
      </article>
      `;
    })
    .join("");
}

function renderCharts(result) {
  const scatterCanvas = document.getElementById("scatter-chart");
  const equityCanvas = document.getElementById("equity-chart");

  const scatterPoints = result.methods.map((method) => ({
    label: method.label,
    x: result.metrics[method.id].driftMse,
    y: result.metrics[method.id].stressMse,
    color: method.color,
  }));

  const lineSeries = result.methods.map((method) => ({
    label: method.label,
    color: method.color,
    values: result.equityCurves[method.id],
  }));

  drawScatter(scatterCanvas, scatterPoints);
  drawEquity(equityCanvas, lineSeries, result.stressMarkers);
}

function renderQualitative(result) {
  const host = document.getElementById("takeaway");
  const naive = result.metrics.naive;
  const proj = result.metrics.anchor_proj;
  const anchor = result.metrics.anchor;

  const stressDelta = fracImprove(naive.stressMse, proj.stressMse);
  const drawdownDelta = proj.maxDrawdown - naive.maxDrawdown;
  const riskyStressDelta = naive.avgRiskyWeightStress - proj.avgRiskyWeightStress;

  host.innerHTML = `
    <h4>Run Takeaway</h4>
    <ul>
      <li>Stress retention gain: <strong>${pct(stressDelta)}</strong> for projection versus naive.</li>
      <li>Tail-risk impact: max drawdown change <strong>${pct(drawdownDelta)}</strong>.</li>
      <li>Risk behavior: stress-period risky allocation reduced by <strong>${pct(riskyStressDelta)}</strong>.</li>
      <li>Anchor-only improves retention (${fmt(anchor.stressMse, 6)}), projection improves it further (${fmt(proj.stressMse, 6)}).</li>
    </ul>
  `;
}

function renderClaimChecks(result) {
  const host = document.getElementById("claim-checks");
  const naive = result.metrics.naive;
  const anchor = result.metrics.anchor;
  const proj = result.metrics.anchor_proj;

  host.innerHTML = `
    <article>
      <h4>Claim A: Projection protects invariant behavior</h4>
      <p>Observed stress MSE: ${fmt(naive.stressMse, 6)} (naive) → ${fmt(proj.stressMse, 6)} (projection).</p>
    </article>
    <article>
      <h4>Claim B: Retention does not eliminate adaptation</h4>
      <p>Observed drift MSE: ${fmt(naive.driftMse, 6)} (naive), ${fmt(anchor.driftMse, 6)} (anchor), ${fmt(proj.driftMse, 6)} (projection).</p>
    </article>
    <article>
      <h4>Claim C: Model behavior changes portfolio outcomes</h4>
      <p>Observed max drawdown: ${pct(naive.maxDrawdown)} (naive) vs ${pct(proj.maxDrawdown)} (projection), with lower stress risk-taking.</p>
    </article>
  `;
}

function renderReferences() {
  renderRefGroup("paper-refs", REFERENCES.papers);
  renderRefGroup("dataset-refs", REFERENCES.datasets);
  renderRefGroup("repo-refs", REFERENCES.repositories);
}

function renderRefGroup(id, refs) {
  const host = document.getElementById(id);
  host.innerHTML = refs
    .map(
      (item) => `
      <li>
        <a href="${item.link}" target="_blank" rel="noreferrer">${item.title}</a>
        <span>${item.authors ? `${item.authors} · ` : ""}${item.why}</span>
      </li>
      `,
    )
    .join("");
}

function exportCurrentRun() {
  if (!latestResult) {
    setStatus("Run the demo before exporting a report.", true);
    return;
  }

  const report = {
    exportedAt: new Date().toISOString(),
    config: latestResult.config,
    keyResult: latestResult.keyResult,
    metrics: latestResult.metrics,
    trainLogs: latestResult.trainLogs,
  };

  const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = `constraint_aligned_peft_run_${Date.now()}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);

  setStatus("Run report exported for internal review.");
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

function fracImprove(base, current) {
  if (Math.abs(base) < 1e-12) {
    return 0;
  }
  return (base - current) / Math.abs(base);
}
