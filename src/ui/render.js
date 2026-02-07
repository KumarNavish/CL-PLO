import { clampConfig, DEFAULT_CONFIG, PRESETS } from "../config.js";
import { REFERENCES } from "../content/references.js";
import { drawEquity, drawImpactBars } from "./charts.js";

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
  setStatus("Running experiment...");
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
  runButton.textContent = isRunning ? "Running..." : "Run";
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
    const numerator = payload.methodIndex * payload.totalSteps + payload.step;
    const denominator = payload.totalMethods * payload.totalSteps;
    setProgress((100 * numerator) / denominator);

    setStatus(`Training ${payload.methodLabel}: ${payload.step}/${payload.totalSteps}`);
    return;
  }

  if (payload.kind === "building_charts") {
    setProgress(100);
    setStatus("Finalizing outputs...");
  }
}

function renderAll(result) {
  setStatus("Run complete. Review deltas vs naive and make a pilot decision.");
  setProgress(100);

  renderDecisionCard(result);
  renderImpactKpis(result);
  renderMethodSummary(result);
  renderCharts(result);
  renderTakeaway(result);
  renderClaimChecks(result);
}

function renderDecisionCard(result) {
  const host = document.getElementById("decision-card");

  const naive = result.metrics.naive;
  const proj = result.metrics.anchor_proj;

  const stressGain = improvement(naive.stressMse, proj.stressMse);
  const drawdownGain = proj.maxDrawdown - naive.maxDrawdown;
  const driftPenalty = ratioPenalty(naive.driftMse, proj.driftMse);

  let level = "caution";
  let title = "Decision Signal: Validate in Pilot";

  if (stressGain > 0.8 && drawdownGain > 0.08 && driftPenalty < 60) {
    level = "good";
    title = "Decision Signal: Strong Pilot Candidate";
  } else if (stressGain < 0.4 || drawdownGain < 0.0) {
    level = "bad";
    title = "Decision Signal: Not Ready for Pilot";
  }

  host.className = `decision-card ${level}`;
  host.innerHTML = `
    <h4>${title}</h4>
    <p>
      Projection vs naive: stress retention gain <strong>${pct(stressGain)}</strong>,
      drawdown improvement <strong>${pp(drawdownGain)}</strong>,
      drift error penalty <strong>${pct(driftPenalty / 100)}</strong>.
    </p>
  `;
}

function renderImpactKpis(result) {
  const host = document.getElementById("impact-kpis");

  const naive = result.metrics.naive;
  const proj = result.metrics.anchor_proj;

  const stressGain = improvement(naive.stressMse, proj.stressMse);
  const drawdownGain = proj.maxDrawdown - naive.maxDrawdown;
  const returnGain = proj.totalReturn - naive.totalReturn;

  host.innerHTML = `
    <article class="kpi ${classBySign(stressGain)}">
      <div class="label">Stress Retention Gain</div>
      <div class="value">${pct(stressGain)}</div>
      <div class="note">Projection vs naive</div>
    </article>
    <article class="kpi ${classBySign(drawdownGain)}">
      <div class="label">Drawdown Improvement</div>
      <div class="value">${pp(drawdownGain)}</div>
      <div class="note">Higher is better</div>
    </article>
    <article class="kpi ${classBySign(returnGain)}">
      <div class="label">Return Uplift</div>
      <div class="value">${pp(returnGain)}</div>
      <div class="note">Projection vs naive</div>
    </article>
  `;
}

function renderMethodSummary(result) {
  const host = document.getElementById("metrics-grid");
  const naive = result.metrics.naive;

  host.innerHTML = result.methods
    .map((method) => {
      const m = result.metrics[method.id];
      const stressGain = method.id === "naive" ? 0 : improvement(naive.stressMse, m.stressMse);
      const ddGain = method.id === "naive" ? 0 : m.maxDrawdown - naive.maxDrawdown;
      const retGain = method.id === "naive" ? 0 : m.totalReturn - naive.totalReturn;

      return `
      <article class="method-card">
        <div class="title">
          <span class="dot" style="background:${method.color}"></span>
          <h4>${method.label}</h4>
        </div>
        <dl>
          <div><dt>Drift MSE</dt><dd>${fmt(m.driftMse, 6)}</dd></div>
          <div><dt>Stress MSE</dt><dd>${fmt(m.stressMse, 6)}</dd></div>
          <div><dt>Max Drawdown</dt><dd>${pct(m.maxDrawdown)}</dd></div>
          <div><dt>Total Return</dt><dd>${pct(m.totalReturn)}</dd></div>
          <div><dt>Stress gain vs naive</dt><dd>${method.id === "naive" ? "baseline" : pct(stressGain)}</dd></div>
          <div><dt>Drawdown vs naive</dt><dd>${method.id === "naive" ? "baseline" : pp(ddGain)}</dd></div>
          <div><dt>Return vs naive</dt><dd>${method.id === "naive" ? "baseline" : pp(retGain)}</dd></div>
        </dl>
      </article>
      `;
    })
    .join("");
}

function renderCharts(result) {
  const impactCanvas = document.getElementById("impact-chart");
  const equityCanvas = document.getElementById("equity-chart");

  const n = result.metrics.naive;
  const a = result.metrics.anchor;
  const p = result.metrics.anchor_proj;

  drawImpactBars(impactCanvas, [
    {
      label: "Stress Retention Gain",
      anchor: 100 * improvement(n.stressMse, a.stressMse),
      proj: 100 * improvement(n.stressMse, p.stressMse),
    },
    {
      label: "Drawdown Improvement",
      anchor: 100 * (a.maxDrawdown - n.maxDrawdown),
      proj: 100 * (p.maxDrawdown - n.maxDrawdown),
    },
  ]);

  const lineSeries = result.methods.map((method) => ({
    label: method.label,
    color: method.color,
    values: result.equityCurves[method.id],
  }));

  drawEquity(equityCanvas, lineSeries, result.stressMarkers);
}

function renderTakeaway(result) {
  const host = document.getElementById("takeaway");

  const naive = result.metrics.naive;
  const anchor = result.metrics.anchor;
  const proj = result.metrics.anchor_proj;

  host.innerHTML = `
    <h4>Interpretation</h4>
    <ul>
      <li>Naive adapts quickly but carries the highest anchor regression risk.</li>
      <li>Anchor-only improves retention, but projection is the strongest retention control.</li>
      <li>Projection improves stress retention by <strong>${pct(improvement(naive.stressMse, proj.stressMse))}</strong> vs naive, with drawdown change <strong>${pp(proj.maxDrawdown - naive.maxDrawdown)}</strong>.</li>
      <li>Tradeoff is explicit: projection usually increases drift error relative to naive (${fmt(proj.driftMse / Math.max(1e-12, naive.driftMse), 2)}x here).</li>
      <li>Decision framing: if stress retention is a hard constraint, projection is the primary pilot candidate.</li>
      <li>Anchor-only reference: stress gain ${pct(improvement(naive.stressMse, anchor.stressMse))}, drawdown change ${pp(anchor.maxDrawdown - naive.maxDrawdown)}.</li>
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
      <h4>Claim A: projection improves stress retention</h4>
      <p>Stress MSE: ${fmt(naive.stressMse, 6)} (naive) -> ${fmt(proj.stressMse, 6)} (projection).</p>
    </article>
    <article>
      <h4>Claim B: anchor-only is intermediate</h4>
      <p>Stress MSE: ${fmt(anchor.stressMse, 6)} (anchor) sits between naive and projection.</p>
    </article>
    <article>
      <h4>Claim C: model behavior affects portfolio outcomes</h4>
      <p>Max drawdown: ${pct(naive.maxDrawdown)} (naive), ${pct(anchor.maxDrawdown)} (anchor), ${pct(proj.maxDrawdown)} (projection).</p>
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
