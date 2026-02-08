import { clampConfig, DEFAULT_CONFIG, PRESETS } from "../config.js";
import { drawEquity, drawImpactBars, drawRegimeBars } from "./charts.js";

const FIELD_MAP = ["seed", "steps", "anchorBeta", "pStress", "loraRank"];
const METHOD_STYLES = {
  naive: { color: "#6e6e6e", dash: [10, 6] },
  anchor: { color: "#0f5fbf", dash: [2, 6] },
  anchor_proj: { color: "#111111", dash: [] },
};

const MODE_META = {
  quick_check: {
    label: "Quick",
    runLabel: "quick diagnostic",
    readout: "Quick mode is a coarse directional test. Use it to verify sign-level ordering before deeper runs.",
  },
  proposal_like: {
    label: "Default",
    runLabel: "deployment-like",
    readout: "Default mode approximates a realistic deployment balance of drift adaptation and stress exposure.",
  },
  stress_heavy: {
    label: "Stress+",
    runLabel: "stress-adversarial",
    readout: "Stress+ mode concentrates crisis regimes to expose hidden forgetting and fragile adaptation.",
  },
};

let worker = null;
let latestResult = null;
let activePreset = "proposal_like";

export function initApp() {
  const defaultPreset = PRESETS.proposal_like?.values || {};
  fillForm(clampConfig({ ...DEFAULT_CONFIG, ...defaultPreset }));
  setActiveMode("proposal_like");

  document.getElementById("run-demo")?.addEventListener("click", () => runCurrentConfig());
  document.getElementById("apply-quick")?.addEventListener("click", () => applyPreset("quick_check"));
  document.getElementById("apply-proposal")?.addEventListener("click", () => applyPreset("proposal_like"));
  document.getElementById("apply-stress")?.addEventListener("click", () => applyPreset("stress_heavy"));
  document.querySelectorAll("[data-mode-card]").forEach((card) => {
    card.addEventListener("click", () => {
      const mode = card.getAttribute("data-mode-card");
      if (mode) {
        applyPreset(mode);
      }
    });
  });
  document.getElementById("reset-form")?.addEventListener("click", () => {
    applyPreset("proposal_like", false);
    setStatus("Controls reset to Default mode.");
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
  const mode = MODE_META[activePreset] || MODE_META.proposal_like;
  setStatus(`Running ${mode.runLabel} validation...`);
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

function applyPreset(name, announce = true) {
  const preset = PRESETS[name];
  if (!preset) {
    return;
  }

  const merged = clampConfig({ ...DEFAULT_CONFIG, ...readConfigFromForm(), ...preset.values });
  fillForm(merged);
  setActiveMode(name);
  if (announce) {
    setStatus(`Mode set: ${preset.label}`);
  }
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
  const mode = MODE_META[activePreset] || MODE_META.proposal_like;
  setStatus(`Validation complete (${mode.label}). Compare observed ordering to the decision surface.`);

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

  const mode = MODE_META[activePreset] || MODE_META.proposal_like;
  host.innerHTML = `
    <h3>Bridge checks</h3>
    <ul>
      <li>
        Stress retention should improve from naive to anchor to projection (lower stress MSE).
        <span class="${stressOrder ? "pass" : "warn"}">${stressOrder ? "Observed" : "Not fully observed"}</span>
      </li>
      <li>
        Drift-fit error is expected to rise as constraints become stricter.
        <span class="${driftOrder ? "pass" : "warn"}">${driftOrder ? "Observed" : "Not fully observed"}</span>
      </li>
      <li>${mode.readout}</li>
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
  let title = "Decision gate: keep in pilot review";

  if (stressGain > 0.8 && drawdownGain > 0.06 && driftPenalty < 70) {
    level = "good";
    title = "Decision gate: promote to pilot";
  } else if (stressGain < 0.45 || drawdownGain < 0) {
    level = "bad";
    title = "Decision gate: do not promote";
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
  const regimeCanvas = document.getElementById("regime-chart");

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
    color: METHOD_STYLES[method.id]?.color || "#111111",
    dash: METHOD_STYLES[method.id]?.dash || [],
    values: result.equityCurves[method.id],
  }));

  drawEquity(equityCanvas, lineSeries, result.stressMarkers);

  if (regimeCanvas) {
    drawRegimeBars(regimeCanvas, buildRegimeReturnRows(result));
  }
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
                  <span class="dot" style="background:${METHOD_STYLES[row.method.id]?.color || "#111111"}"></span>
                  <span class="line-chip line-chip-${row.method.id}"></span>
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
  const mode = MODE_META[activePreset] || MODE_META.proposal_like;

  host.innerHTML = `
    <h4>Run interpretation</h4>
    <p>
      In this sample path, projection improves stress retention by <strong>${pct(stressGainProj)}</strong> relative to
      naive (anchor: ${pct(stressGainAnchor)}), with drawdown change <strong>${pp(ddProj)}</strong>. Treat this as
      one evidence point in <strong>${mode.label}</strong> mode; promotion requires this ordering across repeated seeds
      and especially under Stress+.
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

function setActiveMode(modeName) {
  activePreset = modeName;

  document.querySelectorAll("[data-preset]").forEach((btn) => {
    const target = btn.getAttribute("data-preset");
    btn.classList.toggle("active", target === modeName);
  });

  document.querySelectorAll("[data-mode-card]").forEach((card) => {
    const target = card.getAttribute("data-mode-card");
    card.classList.toggle("active", target === modeName);
  });

  const readout = document.getElementById("mode-readout");
  if (readout) {
    const meta = MODE_META[modeName] || MODE_META.proposal_like;
    readout.textContent = `Active mode: ${meta.label}. ${meta.readout}`;
  }
}

function buildRegimeReturnRows(result) {
  const regimes = result.streamRegimes || [];
  if (!Array.isArray(regimes) || regimes.length === 0) {
    return [];
  }

  return result.methods.map((method) => {
    const diag = result.sharedDiagnostics?.[method.id];
    const returns = diag?.returns || [];

    let driftSum = 0;
    let driftCount = 0;
    let stressSum = 0;
    let stressCount = 0;

    for (let i = 0; i < Math.min(regimes.length, returns.length); i += 1) {
      if (regimes[i] === "stress") {
        stressSum += returns[i];
        stressCount += 1;
      } else {
        driftSum += returns[i];
        driftCount += 1;
      }
    }

    const driftMean = driftCount > 0 ? driftSum / driftCount : 0;
    const stressMean = stressCount > 0 ? stressSum / stressCount : 0;

    return {
      label: method.label,
      color: METHOD_STYLES[method.id]?.color || "#111111",
      driftBp: driftMean * 10000,
      stressBp: stressMean * 10000,
    };
  });
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
