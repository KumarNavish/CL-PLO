import { clampConfig, DEFAULT_CONFIG, PRESETS } from "../config.js";
import { DEMO_RESULTS } from "../content/demo-results.js";
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
    readout: "Quick mode checks directional ordering only. If signs fail here, stop before heavier runs.",
  },
  proposal_like: {
    label: "Default",
    runLabel: "deployment-like validation",
    readout: "Default mode is the baseline deployment test: balance drift fit with stress retention.",
  },
  stress_heavy: {
    label: "Stress+",
    runLabel: "stress-adversarial validation",
    readout: "Stress+ mode is the release gate. Promotion requires stability under this regime concentration.",
  },
};

let latestResult = null;
let activePreset = "proposal_like";

export function initApp() {
  const defaultPreset = PRESETS.proposal_like?.values || {};
  fillForm(clampConfig({ ...DEFAULT_CONFIG, ...defaultPreset }));
  setActiveMode("proposal_like");

  bindControls();
  runCurrentConfig();

  window.addEventListener("resize", () => {
    if (latestResult) {
      renderCharts(latestResult);
    }
  });
}

function bindControls() {
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
    setStatus("Controls reset to Default mode. Click Run Demo.");
  });

  document.getElementById("export-run")?.addEventListener("click", () => exportCurrentRun());
}

function runCurrentConfig() {
  const userCfg = readConfigFromForm();
  const safe = clampConfig({ ...DEFAULT_CONFIG, ...userCfg });
  fillForm(safe);

  const mode = MODE_META[activePreset] || MODE_META.proposal_like;
  const payload = DEMO_RESULTS[activePreset];

  setRunning(true);
  setProgress(18);
  setStatus(`Loading ${mode.runLabel} dataset...`);

  if (!payload) {
    setStatus(`Run failed: missing dataset for ${mode.label}.`, true);
    setRunning(false);
    return;
  }

  latestResult = {
    ...JSON.parse(JSON.stringify(payload)),
    config: safe,
  };

  setProgress(86);
  renderAll(latestResult);
  setRunning(false);
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
    setStatus(`Mode set: ${preset.label}. Click "Run Demo" to execute.`);
  }
}

function setRunning(isRunning) {
  const runButton = document.getElementById("run-demo");
  if (!runButton) {
    return;
  }

  runButton.disabled = isRunning;
  runButton.textContent = isRunning ? "Running..." : "Run Demo";
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

function renderAll(result) {
  setProgress(100);
  const mode = MODE_META[activePreset] || MODE_META.proposal_like;
  setStatus(`Validation complete (${mode.label}). Read decision guidance before changing settings.`);

  renderExpectationCheck(result);
  renderDecisionCard(result);
  renderKpis(result);
  renderCharts(result);
  renderChartReadouts(result);
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
  const drawdownOrder = p.maxDrawdown > a.maxDrawdown && a.maxDrawdown > n.maxDrawdown;
  const driftCostControlled = relativeIncrease(n.driftMse, p.driftMse, 1e-4) < 2.0;

  const mode = MODE_META[activePreset] || MODE_META.proposal_like;

  const modeSpecific =
    activePreset === "quick_check"
      ? `Quick pass condition: stress ordering should hold (naive > anchor > constrained). ${stressOrder ? "It holds." : "It fails."}`
      : activePreset === "stress_heavy"
        ? `Stress+ pass condition: constrained method should show better drawdown control than naive. ${p.maxDrawdown > n.maxDrawdown ? "It holds." : "It fails."}`
        : `Default pass condition: stress protection improves while stabilized drift penalty stays bounded. ${stressOrder && driftCostControlled ? "It holds." : "It fails."}`;

  host.innerHTML = `
    <h3>Bridge Checks</h3>
    <ul>
      <li>Stress retention should improve from naive -> replay -> constrained.
        <span class="${stressOrder ? "pass" : "warn"}">${stressOrder ? "Observed" : "Not observed"}</span>
      </li>
      <li>Drawdown resilience should follow the same ordering.
        <span class="${drawdownOrder ? "pass" : "warn"}">${drawdownOrder ? "Observed" : "Not observed"}</span>
      </li>
      <li>Drift-fit penalty should remain bounded while protection improves.
        <span class="${driftCostControlled ? "pass" : "warn"}">${driftCostControlled ? "Bounded" : "High"}</span>
      </li>
      <li>${modeSpecific}</li>
      <li>${mode.readout}</li>
    </ul>
  `;
}

function renderDecisionCard(result) {
  const host = document.getElementById("decision-card");
  if (!host) {
    return;
  }

  const n = result.metrics.naive;
  const p = result.metrics.anchor_proj;

  const stressGain = improvement(n.stressMse, p.stressMse);
  const drawdownLift = p.maxDrawdown - n.maxDrawdown;
  const driftPenalty = relativeIncrease(n.driftMse, p.driftMse, 1e-4);
  const driftShift = p.driftMse - n.driftMse;

  let level = "caution";
  let title = "Decision gate: keep in pilot review";
  let next = "Run Stress+ with 3-5 seeds before promotion.";

  if (stressGain > 0.7 && drawdownLift > 0.02 && driftPenalty < 2.0) {
    level = "good";
    title = "Decision gate: eligible for promotion";
    next = "Confirm with repeated stress-heavy seeds, then ship as default updater.";
  } else if (stressGain < 0.35 || drawdownLift < -0.01) {
    level = "bad";
    title = "Decision gate: reject this configuration";
    next = "Increase replay/projection strength or reduce update aggressiveness.";
  }

  host.className = `decision-card ${level}`;
  host.innerHTML = `
    <h4>${title}</h4>
    <p>
      Constrained CL vs naive: stress retention <strong>${pct(stressGain)}</strong>, drawdown lift
      <strong>${pp(drawdownLift)}</strong>, drift MSE shift <strong>${fmt(driftShift, 6)}</strong>
      (stabilized increase <strong>${pct(driftPenalty)}</strong>).
      Next action: ${next}
    </p>
  `;
}

function renderKpis(result) {
  const host = document.getElementById("impact-kpis");
  if (!host) {
    return;
  }

  const n = result.metrics.naive;
  const p = result.metrics.anchor_proj;
  const logs = result.trainLogs?.anchor_proj || {};

  const stressGain = improvement(n.stressMse, p.stressMse);
  const drawdownLift = p.maxDrawdown - n.maxDrawdown;
  const interference = logs.interferenceRate || 0;
  const rollbacks = logs.nRollbacks || 0;

  host.innerHTML = `
    <article class="${classBySign(stressGain)}">
      <div class="label">Stress Retention Gain</div>
      <div class="value">${pct(stressGain)}</div>
      <div class="note">constrained vs naive</div>
    </article>
    <article class="${classBySign(drawdownLift)}">
      <div class="label">Drawdown Lift</div>
      <div class="value">${pp(drawdownLift)}</div>
      <div class="note">constrained vs naive</div>
    </article>
    <article class="${classByRollbacks(rollbacks)}">
      <div class="label">Gate Rollbacks</div>
      <div class="value">${rollbacks}</div>
      <div class="note">projection trigger rate: ${pct(interference)}</div>
    </article>
  `;
}

function renderCharts(result) {
  const impactCanvas = document.getElementById("impact-chart");
  const equityCanvas = document.getElementById("equity-chart");
  const regimeCanvas = document.getElementById("regime-chart");

  if (!impactCanvas || !equityCanvas || !regimeCanvas) {
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
      label: "Drawdown lift",
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
  drawEquity(equityCanvas, lineSeries, result.stressMarkers || []);

  drawRegimeBars(regimeCanvas, buildRegimeReturnRows(result));
}

function renderChartReadouts(result) {
  const impactHost = document.getElementById("impact-reading");
  const equityHost = document.getElementById("equity-reading");
  const regimeHost = document.getElementById("regime-reading");
  if (!impactHost || !equityHost || !regimeHost) {
    return;
  }

  const n = result.metrics.naive;
  const a = result.metrics.anchor;
  const p = result.metrics.anchor_proj;

  const stressA = improvement(n.stressMse, a.stressMse);
  const stressP = improvement(n.stressMse, p.stressMse);
  const ddA = a.maxDrawdown - n.maxDrawdown;
  const ddP = p.maxDrawdown - n.maxDrawdown;
  const retGap = p.totalReturn - n.totalReturn;

  impactHost.textContent =
    `Readout: replay gives ${pct(stressA)} stress retention gain; constrained update gives ${pct(stressP)}. ` +
    `Convincing signal is constrained > replay on both stress gain and drawdown lift (${pp(ddP)} vs ${pp(ddA)}).`;

  equityHost.textContent =
    `Readout: compare path shape, not only endpoints. Convincing signal is shallower stress troughs with no drift-phase collapse. ` +
    `Current constrained-vs-naive return gap: ${pp(retGap)}.`;

  const driftRows = buildRegimeReturnRows(result);
  const constrained = driftRows.find((r) => r.label.includes("Constrained"));
  const naive = driftRows.find((r) => r.label.includes("Naive"));
  const stressDelta = constrained && naive ? constrained.stressBp - naive.stressBp : 0;
  const driftDelta = constrained && naive ? constrained.driftBp - naive.driftBp : 0;

  regimeHost.textContent =
    `Readout: drift/stress decomposition should show explicit trade-off, not hidden averaging. ` +
    `Constrained minus naive: stress ${signedBp(stressDelta)}, drift ${signedBp(driftDelta)}.`;
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
  const retVals = ids.map((id) => metrics[id].totalReturn);

  const stressScore = normalizeLowerBetter(stressVals);
  const driftScore = normalizeLowerBetter(driftVals);
  const drawScore = normalizeHigherBetter(drawVals);
  const returnScore = normalizeHigherBetter(retVals);

  const rows = result.methods.map((method, idx) => {
    const m = metrics[method.id];
    const score = 100 * (0.5 * stressScore[idx] + 0.25 * drawScore[idx] + 0.15 * driftScore[idx] + 0.1 * returnScore[idx]);

    return {
      method,
      stressMse: m.stressMse,
      driftMse: m.driftMse,
      maxDrawdown: m.maxDrawdown,
      totalReturn: m.totalReturn,
      stressExposure: m.avgRiskyWeightStress,
      score,
    };
  });

  rows.sort((a, b) => b.score - a.score);

  host.innerHTML = `
    <thead>
      <tr>
        <th>Method</th>
        <th>Stress MSE</th>
        <th>Drift MSE</th>
        <th>Max DD</th>
        <th>Total Return</th>
        <th>Stress Risky Wt</th>
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
              <td>${fmt(row.driftMse, 6)}</td>
              <td>${pct(row.maxDrawdown)}</td>
              <td>${pct(row.totalReturn)}</td>
              <td>${pct(row.stressExposure)}</td>
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

  const n = result.metrics.naive;
  const a = result.metrics.anchor;
  const p = result.metrics.anchor_proj;
  const mode = MODE_META[activePreset] || MODE_META.proposal_like;

  const stressProj = improvement(n.stressMse, p.stressMse);
  const stressAbl = improvement(n.stressMse, a.stressMse);
  const driftPenalty = relativeIncrease(n.driftMse, p.driftMse, 1e-4);
  const driftShift = p.driftMse - n.driftMse;
  const rollbacks = result.trainLogs?.anchor_proj?.nRollbacks || 0;

  host.innerHTML = `
    <h4>Evidence-backed takeaway</h4>
    <p>
      In ${mode.label} mode, constrained CL delivers <strong>${pct(stressProj)}</strong> stress-retention gain
      (replay-only ablation: ${pct(stressAbl)}). Drift MSE shift is <strong>${fmt(driftShift, 6)}</strong>
      with stabilized increase <strong>${pct(driftPenalty)}</strong>.
      Gate rollbacks this run: <strong>${rollbacks}</strong>. If ordering holds under repeated Stress+ seeds, the
      policy is a promotion candidate.
    </p>
  `;
}

function exportCurrentRun() {
  if (!latestResult) {
    setStatus("Run the demo before exporting.", true);
    return;
  }

  const report = {
    exportedAt: new Date().toISOString(),
    mode: activePreset,
    config: latestResult.config,
    keyResult: latestResult.keyResult,
    metrics: latestResult.metrics,
    trainLogs: latestResult.trainLogs,
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

function relativeIncrease(base, current, floor = 0) {
  const denom = Math.max(Math.abs(base), floor, 1e-12);
  return Math.max(0, (current - base) / denom);
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

function classByRollbacks(n) {
  if (n === 0) {
    return "good";
  }
  if (n > 0 && n <= 12) {
    return "caution";
  }
  return "bad";
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

function signedBp(x) {
  if (!Number.isFinite(x)) {
    return "n/a";
  }
  return `${x >= 0 ? "+" : ""}${x.toFixed(1)} bp`;
}
