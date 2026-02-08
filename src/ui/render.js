import { clampConfig, DEFAULT_CONFIG, PRESETS } from "../config.js";
import { DEMO_RESULTS } from "../content/demo-results.js";
import { drawAllocationProfiles, drawDrawdown, drawEquity, drawRegimeRisk } from "./charts.js";

const FIELD_MAP = ["seed", "steps", "anchorBeta", "pStress", "loraRank"];

const METHOD_STYLES = {
  naive: {
    color: "#5f6774",
    dash: [10, 6],
    short: "Naive",
    mechanism: "Raw drift step only",
    protect: "Nothing except frozen backbone",
    tradeoff: "Fastest fit, brittle under stress",
  },
  anchor: {
    color: "#2f557f",
    dash: [2, 6],
    short: "Replay",
    mechanism: "Drift + anchor replay",
    protect: "Stored stress anchors",
    tradeoff: "Better memory, no hard feasibility gate",
  },
  anchor_proj: {
    color: "#7f4a1e",
    dash: [],
    short: "Hybrid",
    mechanism: "Replay + projected step",
    protect: "Anchors + update geometry",
    tradeoff: "Slight drift-cost increase for release safety",
  },
};

const MODE_META = {
  quick_check: {
    label: "Quick",
    runLabel: "quick diagnostic",
    readout: "Directional ordering check. Use this to catch obvious failures in minutes.",
    lens: "Look for clean separation, not final deploy score.",
  },
  proposal_like: {
    label: "Default",
    runLabel: "deployment-like validation",
    readout: "Primary portfolio validation with realistic stress frequency and horizon.",
    lens: "Promotion decision should be made on this mode first.",
  },
  stress_heavy: {
    label: "Stress+",
    runLabel: "stress-adversarial validation",
    readout: "Release gate under concentrated stress/regime-shift behavior.",
    lens: "Reject configs that lose drawdown control or recovery discipline here.",
  },
};

const FOCUS_META = {
  all: "Viewing all strategies side by side for direct deployment comparison.",
  naive: "Naive lens: unconstrained drift adaptation benchmark.",
  anchor: "Replay lens: anchor memory without hard projection control.",
  anchor_proj: "Hybrid lens: anchor memory plus projection gate for release safety.",
};

let latestResult = null;
let activePreset = "proposal_like";
let activeFocus = "all";

export function initApp() {
  const defaultPreset = PRESETS.proposal_like?.values || {};
  fillForm(clampConfig({ ...DEFAULT_CONFIG, ...defaultPreset }));

  bindControls();
  setActiveMode("proposal_like");
  setActiveFocus("all", false);
  runCurrentConfig();

  window.addEventListener("resize", () => {
    if (latestResult) {
      renderAll(latestResult);
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

  document.querySelectorAll("[data-focus]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const focus = btn.getAttribute("data-focus");
      if (!focus) {
        return;
      }
      setActiveFocus(focus);
      if (latestResult) {
        renderAll(latestResult);
      }
    });
  });

  document.getElementById("reset-form")?.addEventListener("click", () => {
    applyPreset("proposal_like", false);
    setActiveFocus("all", false);
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
  setProgress(16);
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

  setProgress(84);
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

function setActiveMode(modeName) {
  activePreset = modeName;

  document.querySelectorAll("[data-preset]").forEach((btn) => {
    btn.classList.toggle("active", btn.getAttribute("data-preset") === modeName);
  });

  document.querySelectorAll("[data-mode-card]").forEach((card) => {
    card.classList.toggle("active", card.getAttribute("data-mode-card") === modeName);
  });

  const readout = document.getElementById("mode-readout");
  if (readout) {
    const mode = MODE_META[modeName] || MODE_META.proposal_like;
    readout.textContent = `Active mode: ${mode.label}. ${mode.readout}`;
  }
}

function setActiveFocus(focusName, announce = true) {
  activeFocus = FOCUS_META[focusName] ? focusName : "all";

  document.querySelectorAll("[data-focus]").forEach((btn) => {
    btn.classList.toggle("active", btn.getAttribute("data-focus") === activeFocus);
  });

  const readout = document.getElementById("focus-readout");
  if (readout) {
    readout.textContent = FOCUS_META[activeFocus] || FOCUS_META.all;
  }

  if (announce) {
    setStatus(`Strategy lens switched to ${prettyFocus(activeFocus)}.`);
  }
}

function prettyFocus(id) {
  if (id === "anchor_proj") {
    return "Hybrid";
  }
  if (id === "anchor") {
    return "Replay";
  }
  if (id === "naive") {
    return "Naive";
  }
  return "All";
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
  const regimeInfo = buildRegimeInfo(result);
  const methodRows = buildMethodRows(result, regimeInfo);
  attachDeployScores(methodRows);

  setStatus(`Validation complete (${mode.label}). ${mode.lens}`);

  renderExpectationCheck(result, methodRows, regimeInfo);
  renderDecisionCard(result, methodRows, regimeInfo);
  renderKpis(methodRows);
  renderStrategySnapshots(methodRows);
  renderCharts(methodRows, regimeInfo);
  renderChartReadouts(methodRows, regimeInfo);
  renderMethodTable(methodRows);
  renderTakeaway(methodRows, regimeInfo);
}

function renderExpectationCheck(result, rows, regimeInfo) {
  const host = document.getElementById("expectation-check");
  if (!host) {
    return;
  }

  const naive = findRow(rows, "naive");
  const replay = findRow(rows, "anchor");
  const hybrid = findRow(rows, "anchor_proj");
  if (!naive || !replay || !hybrid) {
    return;
  }

  const stressOrder = naive.stressMse > replay.stressMse && replay.stressMse > hybrid.stressMse;
  const ddOrder = hybrid.maxDrawdown > replay.maxDrawdown && replay.maxDrawdown > naive.maxDrawdown;
  const stressSharpeOrder =
    (hybrid.sharpeByRegime.stress || 0) > (replay.sharpeByRegime.stress || 0) &&
    (replay.sharpeByRegime.stress || 0) > (naive.sharpeByRegime.stress || 0);

  const mode = MODE_META[activePreset] || MODE_META.proposal_like;

  host.innerHTML = `
    <h3>Decision Checks (${mode.label})</h3>
    <ul>
      <li>Stress retention ordering (naive > replay > hybrid loss)
        <span class="${stressOrder ? "pass" : "warn"}">${stressOrder ? "Pass" : "Fail"}</span>
      </li>
      <li>Drawdown ordering (hybrid should be shallowest)
        <span class="${ddOrder ? "pass" : "warn"}">${ddOrder ? "Pass" : "Review"}</span>
      </li>
      <li>Stress risk-adjusted return ordering
        <span class="${stressSharpeOrder ? "pass" : "warn"}">${stressSharpeOrder ? "Pass" : "Mixed"}</span>
      </li>
      <li>Regime mix: stress ${pct(regimeInfo.stressShare)} | shift events ${regimeInfo.shiftCount}</li>
      <li>${mode.readout}</li>
    </ul>
  `;
}

function renderDecisionCard(result, rows) {
  const host = document.getElementById("decision-card");
  if (!host) {
    return;
  }

  const sorted = [...rows].sort((a, b) => b.deployScore - a.deployScore);
  const winner = sorted[0];
  const runnerUp = sorted[1];

  if (!winner || !runnerUp) {
    return;
  }

  const naive = findRow(rows, "naive");
  const hybrid = findRow(rows, "anchor_proj");

  const lead = winner.deployScore - runnerUp.deployScore;
  const stressGain = naive && hybrid ? improvement(naive.stressMse, hybrid.stressMse) : 0;
  const drawLift = naive && hybrid ? hybrid.maxDrawdown - naive.maxDrawdown : 0;

  let level = "caution";
  let title = `Decision gate: keep in extended pilot (${winner.style.short} leads)`;
  let next = "Run additional seeds before promotion.";

  if (winner.id === "anchor_proj" && lead >= 4) {
    level = "good";
    title = "Decision gate: hybrid strategy is promotion-ready";
    next = "Freeze this updater for release candidate validation and monitor weekly drift diagnostics.";
  } else if (winner.id === "naive") {
    level = "bad";
    title = "Decision gate: reject naive updater for deployment";
    next = "Restore anchor and projection controls before any promotion review.";
  }

  host.className = `decision-card ${level}`;
  host.innerHTML = `
    <h4>${title}</h4>
    <p>
      Deployability score lead: <strong>${winner.deployScore.toFixed(1)}</strong> vs <strong>${runnerUp.deployScore.toFixed(1)}</strong>
      (${pp(lead / 100)}).
      Hybrid vs naive stress retention: <strong>${pct(stressGain)}</strong>.
      Drawdown lift: <strong>${pp(drawLift)}</strong>.
      Next action: ${next}
    </p>
  `;
}

function renderKpis(rows) {
  const host = document.getElementById("impact-kpis");
  if (!host) {
    return;
  }

  const naive = findRow(rows, "naive");
  const replay = findRow(rows, "anchor");
  const hybrid = findRow(rows, "anchor_proj");

  if (!naive || !replay || !hybrid) {
    return;
  }

  const stressGain = improvement(naive.stressMse, hybrid.stressMse);
  const ddLift = hybrid.maxDrawdown - naive.maxDrawdown;
  const stressSharpeLift = (hybrid.sharpeByRegime.stress || 0) - (naive.sharpeByRegime.stress || 0);
  const turnoverDelta = replay.turnover - hybrid.turnover;

  host.innerHTML = `
    <article class="${classBySign(stressGain)}">
      <div class="label">Stress Retention Gain</div>
      <div class="value">${pct(stressGain)}</div>
      <div class="note">hybrid vs naive</div>
    </article>
    <article class="${classBySign(ddLift)}">
      <div class="label">Drawdown Lift</div>
      <div class="value">${pp(ddLift)}</div>
      <div class="note">hybrid vs naive</div>
    </article>
    <article class="${classBySign(stressSharpeLift)}">
      <div class="label">Stress Sharpe Lift</div>
      <div class="value">${signed(stressSharpeLift, 2)}</div>
      <div class="note">annualized, stress bucket</div>
    </article>
    <article class="${classBySign(turnoverDelta)}">
      <div class="label">Turnover Discipline</div>
      <div class="value">${pp(turnoverDelta)}</div>
      <div class="note">replay minus hybrid turnover</div>
    </article>
  `;
}

function renderStrategySnapshots(rows) {
  const host = document.getElementById("strategy-snapshots");
  if (!host) {
    return;
  }

  host.innerHTML = rows
    .map((row) => {
      const focused = activeFocus === "all" || activeFocus === row.id;
      return `
        <article class="snapshot-card ${focused ? "focused" : "muted"}">
          <div class="snapshot-header">
            <div class="snapshot-title"><span class="dot" style="background:${row.style.color}"></span>${row.style.short}</div>
            <div class="snapshot-score">Deploy score ${row.deployScore.toFixed(1)}</div>
          </div>

          <div class="snapshot-grid">
            <div class="snapshot-metric">
              <div class="label">Mechanism</div>
              <div class="value">${row.style.mechanism}</div>
            </div>
            <div class="snapshot-metric">
              <div class="label">Protects</div>
              <div class="value">${row.style.protect}</div>
            </div>
            <div class="snapshot-metric">
              <div class="label">Stress MSE</div>
              <div class="value">${fmt(row.stressMse, 6)}</div>
            </div>
            <div class="snapshot-metric">
              <div class="label">Max Drawdown</div>
              <div class="value">${pct(row.maxDrawdown)}</div>
            </div>
            <div class="snapshot-metric">
              <div class="label">Stress Sharpe</div>
              <div class="value">${signed(row.sharpeByRegime.stress || 0, 2)}</div>
            </div>
            <div class="snapshot-metric">
              <div class="label">Trade-off</div>
              <div class="value">${row.style.tradeoff}</div>
            </div>
          </div>
        </article>
      `;
    })
    .join("");
}

function renderCharts(rows, regimeInfo) {
  const pnlCanvas = document.getElementById("pnl-chart");
  const drawdownCanvas = document.getElementById("drawdown-chart");
  const allocationCanvas = document.getElementById("allocation-chart");
  const regimeRiskCanvas = document.getElementById("regime-risk-chart");

  if (!pnlCanvas || !drawdownCanvas || !allocationCanvas || !regimeRiskCanvas) {
    return;
  }

  const lineSeries = rows.map((row) => ({
    id: row.id,
    label: row.label,
    color: row.style.color,
    dash: row.style.dash,
    values: row.equity,
    alpha: focusAlpha(row.id),
    lineWidth: focusLineWidth(row.id),
  }));

  const allocationSeries = rows.map((row) => ({
    id: row.id,
    label: row.label,
    color: row.style.color,
    dash: row.style.dash,
    values: row.riskyWeights,
    alpha: focusAlpha(row.id),
    lineWidth: focusLineWidth(row.id),
  }));

  drawEquity(pnlCanvas, lineSeries, regimeInfo.timelineStates);
  drawDrawdown(drawdownCanvas, lineSeries, regimeInfo.timelineStates);
  drawAllocationProfiles(allocationCanvas, allocationSeries, regimeInfo.timelineStates);

  drawRegimeRisk(
    regimeRiskCanvas,
    regimeInfo.regimes,
    rows.map((row) => ({
      id: row.id,
      label: row.label,
      color: row.style.color,
      alpha: focusAlpha(row.id),
      sharpe: row.sharpeByRegime,
    })),
  );
}

function renderChartReadouts(rows) {
  const pnlHost = document.getElementById("pnl-reading");
  const drawdownHost = document.getElementById("drawdown-reading");
  const allocationHost = document.getElementById("allocation-reading");
  const regimeHost = document.getElementById("regime-risk-reading");
  if (!pnlHost || !drawdownHost || !allocationHost || !regimeHost) {
    return;
  }

  const naive = findRow(rows, "naive");
  const replay = findRow(rows, "anchor");
  const hybrid = findRow(rows, "anchor_proj");
  if (!naive || !replay || !hybrid) {
    return;
  }

  const hybridVsNaiveRet = hybrid.totalReturn - naive.totalReturn;
  const hybridVsReplayRet = hybrid.totalReturn - replay.totalReturn;
  const hybridDdLift = hybrid.maxDrawdown - naive.maxDrawdown;

  pnlHost.textContent =
    `Signal: compare terminal value and stress trough depth. Hybrid vs naive return delta ${pp(hybridVsNaiveRet)}; ` +
    `hybrid vs replay ${pp(hybridVsReplayRet)}.`;

  drawdownHost.textContent =
    `Signal: drawdown curve should be shallower with faster recovery. Hybrid drawdown lift vs naive is ${pp(hybridDdLift)} ` +
    `with recovery in ${formatRecovery(hybrid.recoveryDays)}.`;

  const stressWeightGap = hybrid.stressWeight - naive.stressWeight;
  const turnoverGap = naive.turnover - hybrid.turnover;
  allocationHost.textContent =
    `Signal: robust strategy de-risks in stress without whipsaw turnover. Hybrid stress risky-weight gap vs naive ${pp(stressWeightGap)}; ` +
    `turnover improvement ${pp(turnoverGap)}.`;

  const stressSharpeLift = (hybrid.sharpeByRegime.stress || 0) - (naive.sharpeByRegime.stress || 0);
  const shiftSharpeLift = (hybrid.sharpeByRegime.shift || 0) - (naive.sharpeByRegime.shift || 0);
  regimeHost.textContent =
    `Signal: regime bars should prove robustness, not average it away. Hybrid minus naive Sharpe: ` +
    `stress ${signed(stressSharpeLift, 2)}, shift ${signed(shiftSharpeLift, 2)}.`;
}

function renderMethodTable(rows) {
  const host = document.getElementById("method-table");
  if (!host) {
    return;
  }

  const ordered = [...rows].sort((a, b) => b.deployScore - a.deployScore);

  host.innerHTML = `
    <thead>
      <tr>
        <th>Strategy</th>
        <th>Stress Retention</th>
        <th>Max DD</th>
        <th>Stress Sharpe</th>
        <th>Recovery</th>
        <th>Turnover</th>
        <th>Deploy Score</th>
      </tr>
    </thead>
    <tbody>
      ${ordered
        .map(
          (row, idx) => `
            <tr class="${idx === 0 ? "winner" : ""}">
              <td>
                <span class="method-name">
                  <span class="dot" style="background:${row.style.color}"></span>
                  <span class="line-chip line-chip-${row.id}"></span>
                  ${row.label}
                </span>
              </td>
              <td>${pct(row.stressRetention)}</td>
              <td>${pct(row.maxDrawdown)}</td>
              <td>${signed(row.sharpeByRegime.stress || 0, 2)}</td>
              <td>${formatRecovery(row.recoveryDays)}</td>
              <td>${pct(row.turnover)}</td>
              <td><strong>${row.deployScore.toFixed(1)}</strong></td>
            </tr>
          `,
        )
        .join("")}
    </tbody>
  `;
}

function renderTakeaway(rows) {
  const host = document.getElementById("takeaway");
  if (!host) {
    return;
  }

  const ordered = [...rows].sort((a, b) => b.deployScore - a.deployScore);
  const winner = ordered[0];
  const naive = findRow(rows, "naive");
  const hybrid = findRow(rows, "anchor_proj");

  if (!winner || !naive || !hybrid) {
    return;
  }

  host.innerHTML = `
    <h4>Evidence-backed deployment takeaway</h4>
    <p>
      In ${MODE_META[activePreset]?.label || "Default"} mode, <strong>${winner.style.short}</strong> is ranked first by
      stress retention, drawdown behavior, regime Sharpe, and turnover discipline.
      Hybrid vs naive stress retention is <strong>${pct(improvement(naive.stressMse, hybrid.stressMse))}</strong>,
      with drawdown lift <strong>${pp(hybrid.maxDrawdown - naive.maxDrawdown)}</strong>.
      Implementation next step: promote the winner into paper-trading with weekly stress-retention monitoring.
    </p>
  `;
}

function buildRegimeInfo(result) {
  const methods = result.methods || [];
  const priorityId = methods.find((m) => m.id === "anchor_proj")?.id || methods[0]?.id;
  const priorityDiag = getDiagnostics(result, priorityId);

  const rawRegimes = Array.isArray(result.streamRegimes) ? result.streamRegimes : [];
  const length = Math.max(rawRegimes.length, priorityDiag.returns.length, priorityDiag.equity.length, 1);

  const filledRegimes = Array.from({ length }, (_, i) => rawRegimes[i] || "drift");
  const baseReturns = Array.from({ length }, (_, i) => priorityDiag.returns[i] || 0);

  const absNonStress = [];
  for (let i = 0; i < length; i += 1) {
    if (filledRegimes[i] !== "stress") {
      absNonStress.push(Math.abs(baseReturns[i]));
    }
  }
  const volThreshold = quantile(absNonStress, 0.7);

  const baseStates = [];
  for (let i = 0; i < length; i += 1) {
    if (filledRegimes[i] === "stress") {
      baseStates.push("stress");
    } else if (Math.abs(baseReturns[i]) >= volThreshold && volThreshold > 0) {
      baseStates.push("volatile");
    } else {
      baseStates.push("calm");
    }
  }

  const timelineStates = [...baseStates];
  const shiftSet = new Set();
  for (let i = 1; i < baseStates.length; i += 1) {
    if (baseStates[i] !== baseStates[i - 1]) {
      timelineStates[i] = "shift";
      shiftSet.add(i);
    }
  }

  const indexByRegime = {
    calm: [],
    volatile: [],
    stress: [],
    shift: [],
  };

  for (let i = 0; i < length; i += 1) {
    if (shiftSet.has(i)) {
      indexByRegime.shift.push(i);
    } else {
      indexByRegime[baseStates[i]].push(i);
    }
  }

  return {
    timelineStates,
    indexByRegime,
    stressShare: indexByRegime.stress.length / Math.max(1, length),
    shiftCount: indexByRegime.shift.length,
    regimes: ["calm", "volatile", "stress", "shift"],
  };
}

function buildMethodRows(result, regimeInfo) {
  const methods = result.methods || [];
  const naiveStress = result.metrics?.naive?.stressMse || 0;

  return methods.map((method) => {
    const style = METHOD_STYLES[method.id] || {
      color: method.color || "#222222",
      dash: [],
      short: method.label,
      mechanism: "n/a",
      protect: "n/a",
      tradeoff: "n/a",
    };

    const metrics = result.metrics?.[method.id] || {};
    const diag = getDiagnostics(result, method.id);
    const sharpeByRegime = computeRegimeSharpe(diag.returns, regimeInfo.indexByRegime);

    return {
      id: method.id,
      label: method.label,
      style,
      stressMse: metrics.stressMse ?? 0,
      driftMse: metrics.driftMse ?? 0,
      maxDrawdown: metrics.maxDrawdown ?? 0,
      totalReturn: metrics.totalReturn ?? 0,
      stressRetention: improvement(naiveStress, metrics.stressMse ?? naiveStress),
      stressWeight:
        metrics.avgRiskyWeightStress ??
        meanByIndices(diag.riskyWeights, regimeInfo.indexByRegime.stress) ??
        0,
      driftWeight:
        metrics.avgRiskyWeightDrift ??
        meanByIndices(diag.riskyWeights, regimeInfo.indexByRegime.calm.concat(regimeInfo.indexByRegime.volatile)) ??
        0,
      worstStressDay: metrics.worstStressDay ?? minByIndices(diag.returns, regimeInfo.indexByRegime.stress),
      turnover: mean(diag.turnovers),
      recoveryDays: computeRecoveryDays(diag.equity),
      sharpeByRegime,
      equity: diag.equity,
      returns: diag.returns,
      riskyWeights: diag.riskyWeights,
    };
  });
}

function attachDeployScores(rows) {
  const stressRetention = normalizeHigherBetter(rows.map((row) => row.stressRetention));
  const drawdownControl = normalizeHigherBetter(rows.map((row) => row.maxDrawdown));
  const stressSharpe = normalizeHigherBetter(rows.map((row) => row.sharpeByRegime.stress || 0));
  const shiftSharpe = normalizeHigherBetter(rows.map((row) => row.sharpeByRegime.shift || 0));
  const turnoverControl = normalizeLowerBetter(rows.map((row) => row.turnover));
  const recoveryControl = normalizeLowerBetter(rows.map((row) => row.recoveryDays));

  for (let i = 0; i < rows.length; i += 1) {
    rows[i].deployScore =
      100 *
      (0.34 * stressRetention[i] +
        0.22 * drawdownControl[i] +
        0.18 * stressSharpe[i] +
        0.1 * shiftSharpe[i] +
        0.08 * turnoverControl[i] +
        0.08 * recoveryControl[i]);
  }
}

function getDiagnostics(result, methodId) {
  const diag = result.sharedDiagnostics?.[methodId] || {};
  const equity = cloneArray(diag.equity || result.equityCurves?.[methodId] || []);
  const returns = cloneArray(diag.returns || computeReturns(equity));
  const riskyWeights = cloneArray(diag.riskyWeights || []);
  const turnovers = cloneArray(diag.turnovers || []);

  return {
    equity: equity.length > 0 ? equity : [1],
    returns: returns.length > 0 ? returns : [0],
    riskyWeights: riskyWeights.length > 0 ? riskyWeights : Array(Math.max(1, equity.length)).fill(0),
    turnovers: turnovers.length > 0 ? turnovers : Array(Math.max(1, equity.length)).fill(0),
  };
}

function cloneArray(values) {
  if (!Array.isArray(values)) {
    return [];
  }
  return values.map((v) => Number(v) || 0);
}

function computeReturns(equity) {
  if (!Array.isArray(equity) || equity.length <= 1) {
    return [0];
  }

  const out = [];
  for (let i = 1; i < equity.length; i += 1) {
    const prev = Math.abs(equity[i - 1]) > 1e-12 ? equity[i - 1] : 1;
    out.push(equity[i] / prev - 1);
  }
  return out;
}

function computeRegimeSharpe(returns, indexByRegime) {
  const out = {};
  for (const regime of Object.keys(indexByRegime)) {
    const vals = indexByRegime[regime]
      .map((idx) => returns[idx])
      .filter((v) => Number.isFinite(v));
    out[regime] = annualizedSharpe(vals);
  }
  return out;
}

function annualizedSharpe(values) {
  if (!values || values.length < 2) {
    return 0;
  }

  const m = mean(values);
  const variance = mean(values.map((v) => (v - m) ** 2));
  const sigma = Math.sqrt(Math.max(variance, 1e-12));
  const raw = (m / sigma) * Math.sqrt(252);
  return Math.max(-5, Math.min(5, raw));
}

function computeRecoveryDays(equity) {
  if (!Array.isArray(equity) || equity.length === 0) {
    return 999;
  }

  let runningPeak = equity[0];
  let runningPeakAtWorst = equity[0];
  let worstDrawdown = 0;
  let troughIdx = 0;

  for (let i = 0; i < equity.length; i += 1) {
    if (equity[i] > runningPeak) {
      runningPeak = equity[i];
    }

    const dd = runningPeak > 0 ? equity[i] / runningPeak - 1 : 0;
    if (dd < worstDrawdown) {
      worstDrawdown = dd;
      troughIdx = i;
      runningPeakAtWorst = runningPeak;
    }
  }

  for (let j = troughIdx; j < equity.length; j += 1) {
    if (equity[j] >= runningPeakAtWorst) {
      return j - troughIdx;
    }
  }

  return 999;
}

function renderModeForFocus(id) {
  return activeFocus === "all" || activeFocus === id;
}

function focusAlpha(id) {
  return renderModeForFocus(id) ? 1 : 0.22;
}

function focusLineWidth(id) {
  return renderModeForFocus(id) ? 2.4 : 1.4;
}

function exportCurrentRun() {
  if (!latestResult) {
    setStatus("Run the demo before exporting.", true);
    return;
  }

  const report = {
    exportedAt: new Date().toISOString(),
    mode: activePreset,
    strategyLens: activeFocus,
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

function findRow(rows, id) {
  return rows.find((row) => row.id === id) || null;
}

function quantile(values, q) {
  if (!values || values.length === 0) {
    return 0;
  }

  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.max(0, Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * q)));
  return sorted[idx];
}

function normalizeLowerBetter(values) {
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo;

  if (!Number.isFinite(span) || span < 1e-12) {
    return values.map(() => 0.5);
  }

  return values.map((v) => (hi - v) / span);
}

function normalizeHigherBetter(values) {
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo;

  if (!Number.isFinite(span) || span < 1e-12) {
    return values.map(() => 0.5);
  }

  return values.map((v) => (v - lo) / span);
}

function mean(values) {
  if (!values || values.length === 0) {
    return 0;
  }
  let sum = 0;
  for (const v of values) {
    sum += Number(v) || 0;
  }
  return sum / values.length;
}

function meanByIndices(values, indices) {
  if (!values || values.length === 0 || !indices || indices.length === 0) {
    return 0;
  }

  let sum = 0;
  let count = 0;
  for (const idx of indices) {
    const v = values[idx];
    if (Number.isFinite(v)) {
      sum += v;
      count += 1;
    }
  }

  return count > 0 ? sum / count : 0;
}

function minByIndices(values, indices) {
  if (!values || values.length === 0 || !indices || indices.length === 0) {
    return 0;
  }

  let out = Infinity;
  for (const idx of indices) {
    const v = values[idx];
    if (Number.isFinite(v) && v < out) {
      out = v;
    }
  }

  return Number.isFinite(out) ? out : 0;
}

function improvement(base, current) {
  if (!Number.isFinite(base) || Math.abs(base) < 1e-12) {
    return 0;
  }
  return (base - current) / Math.abs(base);
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

function signed(x, digits = 2) {
  if (!Number.isFinite(x)) {
    return "n/a";
  }
  return `${x >= 0 ? "+" : ""}${x.toFixed(digits)}`;
}

function formatRecovery(days) {
  if (!Number.isFinite(days) || days >= 999) {
    return "not recovered";
  }
  return `${days}d`;
}
