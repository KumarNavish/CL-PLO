import { clampConfig, DEFAULT_CONFIG, PRESETS } from "../config.js";
import { REFERENCES } from "../content/references.js";
import { drawEquity, drawImpactBars } from "./charts.js";

const FIELD_MAP = ["seed", "steps", "anchorBeta", "pStress", "loraRank"];

let worker = null;
let latestResult = null;

export function initApp() {
  fillForm(DEFAULT_CONFIG);
  renderReferences();

  document.getElementById("run-demo")?.addEventListener("click", () => runCurrentConfig());
  document.getElementById("apply-quick")?.addEventListener("click", () => applyPreset("quick_check"));
  document.getElementById("apply-proposal")?.addEventListener("click", () => applyPreset("proposal_like"));
  document.getElementById("apply-stress")?.addEventListener("click", () => applyPreset("stress_heavy"));
  document.getElementById("reset-form")?.addEventListener("click", () => {
    fillForm(DEFAULT_CONFIG);
    setStatus("Controls reset to default.");
  });
  document.getElementById("export-run")?.addEventListener("click", () => exportCurrentRun());

  document.getElementById("hero-run")?.addEventListener("click", () => {
    document.getElementById("demo")?.scrollIntoView({ behavior: "smooth", block: "start" });
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
  const userCfg = readConfigFromForm();
  const safe = clampConfig({ ...DEFAULT_CONFIG, ...userCfg });
  fillForm(safe);

  setRunning(true);
  setStatus("Running shared-path comparison...");
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
  runButton.textContent = isRunning ? "Running..." : "Run";
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
    setStatus("Finalizing decision outputs...");
  }
}

function renderAll(result) {
  setProgress(100);
  setStatus("Run complete. Review decision, gates, then method table.");

  renderDecisionCard(result);
  renderKpis(result);
  renderCharts(result);
  renderMethodTable(result);
  renderTakeaway(result);
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
  let title = "Decision: Validate in pilot before promotion";

  if (stressGain > 0.8 && drawdownGain > 0.06 && driftPenalty < 70) {
    level = "good";
    title = "Decision: Projection is pilot-ready";
  } else if (stressGain < 0.45 || drawdownGain < 0) {
    level = "bad";
    title = "Decision: Hold deployment";
  }

  host.className = `decision-card ${level}`;
  host.innerHTML = `
    <h4>${title}</h4>
    <p>
      Projection vs naive: stress retention <strong>${pct(stressGain)}</strong>, drawdown change
      <strong>${pp(drawdownGain)}</strong>, drift-fit penalty <strong>${pct(driftPenalty / 100)}</strong>.
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
      <div class="label">Stress Retention Gain</div>
      <div class="value">${pct(stressGain)}</div>
      <div class="note">Projection vs naive</div>
    </article>
    <article class="${classBySign(drawdownGain)}">
      <div class="label">Drawdown Change</div>
      <div class="value">${pp(drawdownGain)}</div>
      <div class="note">Higher is better</div>
    </article>
    <article class="${classBySign(-anchorGap)}">
      <div class="label">Projection vs Anchor</div>
      <div class="value">${fmt(anchorGap, 6)}</div>
      <div class="note">Stress MSE delta (lower is better)</div>
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
    color: method.color,
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
        <th>Deployment Score</th>
      </tr>
    </thead>
    <tbody>
      ${rows
        .map(
          (row) => `
            <tr>
              <td>
                <span class="method-name">
                  <span class="dot" style="background:${row.method.color}"></span>
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
    <h4>Practical Read</h4>
    <p>
      Naive is the speed baseline. Anchor improves retention, but projection is the strongest stress-preserving update
      in this setup: stress gain <strong>${pct(stressGainProj)}</strong> vs naive (anchor: ${pct(stressGainAnchor)}),
      with drawdown change <strong>${pp(ddProj)}</strong>. Promote only if this pattern holds under stress-heavy runs.
    </p>
  `;
}

function renderReferences() {
  const host = document.getElementById("ref-list");
  if (!host) {
    return;
  }

  const curated = [
    ...REFERENCES.papers.slice(0, 4),
    ...REFERENCES.repositories.slice(0, 2),
    ...REFERENCES.datasets.slice(0, 1),
  ];

  host.innerHTML = curated
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
    setStatus("Run the test before exporting.", true);
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
