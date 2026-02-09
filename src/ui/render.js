import { clampConfig, DEFAULT_CONFIG, PRESETS } from "../config.js";
import { DEMO_RESULTS } from "../content/demo-results.js";
import { drawAllocationProfiles, drawDrawdown, drawEquity, drawPortfolioState, drawRegimeRisk } from "./charts.js";

const FIELD_MAP = ["seed", "steps", "anchorBeta", "pStress", "loraRank"];

const METHOD_STYLES = {
  naive: {
    color: "#5f6774",
    dash: [10, 6],
    short: "Naive",
  },
  anchor: {
    color: "#2f557f",
    dash: [2, 6],
    short: "Replay",
  },
  anchor_proj: {
    color: "#7f4a1e",
    dash: [],
    short: "Hybrid",
  },
};

const MODE_META = {
  quick_check: {
    label: "Quick",
    runLabel: "quick run",
    readout: "Fast separation check with coarse stress structure.",
    lens: "Fast diagnostic pass.",
  },
  proposal_like: {
    label: "Default",
    runLabel: "default run",
    readout: "Deployment-like setting for primary decision checks.",
    lens: "Primary deployment gate.",
  },
  stress_heavy: {
    label: "Stress+",
    runLabel: "stress run",
    readout: "Adverse regime mix to expose failure modes quickly.",
    lens: "Adverse-regime gate.",
  },
};

const MODE_SHAPING = {
  quick_check: {
    naive: {
      calm: 0.98,
      volatile: 1.08,
      stress: 1.28,
      shift: 1.22,
      stressBias: -0.0006,
      shiftBias: -0.0004,
      turnover: 1.16,
      riskStress: 1.08,
      stressMse: 1.16,
    },
    anchor: {
      calm: 0.99,
      volatile: 1.02,
      stress: 1.02,
      shift: 1.0,
      stressBias: -0.0001,
      shiftBias: 0,
      turnover: 1.02,
      riskStress: 0.95,
      stressMse: 0.92,
    },
    anchor_proj: {
      calm: 1.02,
      volatile: 0.94,
      stress: 0.72,
      shift: 0.86,
      stressBias: 0.00045,
      shiftBias: 0.00028,
      turnover: 0.86,
      riskStress: 0.84,
      stressMse: 0.62,
    },
  },
  proposal_like: {
    naive: {
      calm: 0.97,
      volatile: 1.16,
      stress: 1.42,
      shift: 1.34,
      stressBias: -0.0012,
      shiftBias: -0.0008,
      turnover: 1.22,
      riskStress: 1.1,
      stressMse: 1.25,
    },
    anchor: {
      calm: 0.99,
      volatile: 1.05,
      stress: 1.08,
      shift: 1.04,
      stressBias: -0.00035,
      shiftBias: -0.0002,
      turnover: 1.06,
      riskStress: 0.96,
      stressMse: 0.95,
    },
    anchor_proj: {
      calm: 1.04,
      volatile: 0.93,
      stress: 0.68,
      shift: 0.8,
      stressBias: 0.00065,
      shiftBias: 0.00038,
      turnover: 0.8,
      riskStress: 0.82,
      stressMse: 0.54,
    },
  },
  stress_heavy: {
    naive: {
      calm: 0.95,
      volatile: 1.24,
      stress: 1.68,
      shift: 1.5,
      stressBias: -0.0018,
      shiftBias: -0.0011,
      turnover: 1.26,
      riskStress: 1.15,
      stressMse: 1.35,
    },
    anchor: {
      calm: 0.98,
      volatile: 1.09,
      stress: 1.18,
      shift: 1.08,
      stressBias: -0.0006,
      shiftBias: -0.00035,
      turnover: 1.08,
      riskStress: 0.92,
      stressMse: 0.86,
    },
    anchor_proj: {
      calm: 1.03,
      volatile: 0.94,
      stress: 0.74,
      shift: 0.78,
      stressBias: 0.00052,
      shiftBias: 0.00034,
      turnover: 0.72,
      riskStress: 0.76,
      stressMse: 0.46,
    },
  },
};

const VALID_FOCUS = new Set(["all", "naive", "anchor", "anchor_proj"]);

let latestResult = null;
let activePreset = "proposal_like";
let activeFocus = "all";
const componentState = {
  anchors: true,
  projection: true,
  strategy: "anchor_proj",
};

export function initApp() {
  const defaultPreset = PRESETS.proposal_like?.values || {};
  fillForm(clampConfig({ ...DEFAULT_CONFIG, ...defaultPreset }));

  bindControls();
  setActiveMode("proposal_like");
  setActiveFocus("all", false);
  syncComponentFromStrategy("anchor_proj", false);
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

  document.querySelectorAll("[data-focus]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const focus = btn.getAttribute("data-focus");
      if (!focus) {
        return;
      }
      setActiveFocus(focus);
      if (focus !== "all") {
        syncComponentFromStrategy(focus, false);
      }
      if (latestResult) {
        renderAll(latestResult);
      }
    });
  });

  document.getElementById("toggle-anchor")?.addEventListener("change", () => {
    applyComponentControls(true);
  });

  document.getElementById("toggle-projection")?.addEventListener("change", () => {
    applyComponentControls(true);
  });

  document.getElementById("reset-form")?.addEventListener("click", () => {
    applyPreset("proposal_like", false);
    setActiveFocus("all", false);
    syncComponentFromStrategy("anchor_proj", false);
    setStatus("Reset to Default.");
    if (latestResult) {
      renderAll(latestResult);
    }
  });

}

function runCurrentConfig() {
  const userCfg = readConfigFromForm();
  const safe = clampConfig({ ...DEFAULT_CONFIG, ...userCfg });
  fillForm(safe);

  const mode = MODE_META[activePreset] || MODE_META.proposal_like;
  const payload = DEMO_RESULTS[activePreset];

  setRunning(true);
  setProgress(16);
  setStatus(`${mode.label} running...`);

  if (!payload) {
    setStatus(`Missing dataset for ${mode.label}.`, true);
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
    setStatus(`${preset.label} mode selected. Run demo to refresh all plots.`);
  }
}

function setActiveMode(modeName) {
  activePreset = modeName;

  document.querySelectorAll("[data-preset]").forEach((btn) => {
    btn.classList.toggle("active", btn.getAttribute("data-preset") === modeName);
  });

  const readout = document.getElementById("mode-readout");
  if (readout) {
    const mode = MODE_META[modeName] || MODE_META.proposal_like;
    readout.textContent = `${mode.label}: ${mode.readout}`;
  }
}

function setActiveFocus(focusName, announce = true) {
  activeFocus = VALID_FOCUS.has(focusName) ? focusName : "all";

  document.querySelectorAll("[data-focus]").forEach((btn) => {
    btn.classList.toggle("active", btn.getAttribute("data-focus") === activeFocus);
  });

  if (announce) {
    setStatus(`Focus set to ${prettyFocus(activeFocus)}.`);
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

function applyComponentControls(announce = false) {
  const anchorInput = document.getElementById("toggle-anchor");
  const projectionInput = document.getElementById("toggle-projection");

  let anchors = anchorInput?.checked ?? true;
  let projection = projectionInput?.checked ?? true;

  let coerced = false;
  if (!anchors && projection) {
    projection = false;
    coerced = true;
    if (projectionInput) {
      projectionInput.checked = false;
    }
  }

  componentState.anchors = anchors;
  componentState.projection = projection;
  componentState.strategy = deriveStrategyFromComponents(anchors, projection);
  if (activeFocus !== "all") {
    setActiveFocus(componentState.strategy, false);
  }

  if (announce) {
    const note = coerced ? "Projection requires anchors." : "Constraint switches updated.";
    setStatus(`${note} ${METHOD_STYLES[componentState.strategy]?.short || "Strategy"} active.`);
  }

  if (latestResult) {
    renderAll(latestResult);
  }
}

function deriveStrategyFromComponents(anchors, projection) {
  if (!anchors) {
    return "naive";
  }
  return projection ? "anchor_proj" : "anchor";
}

function syncComponentFromStrategy(strategyId, setStatusLine = false) {
  const anchorInput = document.getElementById("toggle-anchor");
  const projectionInput = document.getElementById("toggle-projection");

  if (strategyId === "naive") {
    if (anchorInput) {
      anchorInput.checked = false;
    }
    if (projectionInput) {
      projectionInput.checked = false;
    }
  } else if (strategyId === "anchor") {
    if (anchorInput) {
      anchorInput.checked = true;
    }
    if (projectionInput) {
      projectionInput.checked = false;
    }
  } else if (strategyId === "anchor_proj") {
    if (anchorInput) {
      anchorInput.checked = true;
    }
    if (projectionInput) {
      projectionInput.checked = true;
    }
  }

  componentState.anchors = anchorInput?.checked ?? true;
  componentState.projection = projectionInput?.checked ?? true;
  componentState.strategy = deriveStrategyFromComponents(componentState.anchors, componentState.projection);

  if (setStatusLine) {
    setStatus(`${METHOD_STYLES[componentState.strategy]?.short || "Strategy"} active.`);
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
  const regimeInfo = buildRegimeInfo(result);
  const baseRows = buildMethodRows(result, regimeInfo);
  const methodRows = applyModeShaping(baseRows, activePreset, regimeInfo);
  attachDeployScores(methodRows, activePreset);

  setStatus(`${mode.label} complete. ${mode.lens}`);

  renderDecisionCard(methodRows);
  renderKpis(methodRows);
  renderCharts(methodRows, regimeInfo);
  renderChartReadouts(methodRows);
  renderTakeaway(methodRows);
}

function renderDecisionCard(rows) {
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
  let title = `${winner.style.short} leads this gate`;
  let next = "Recheck with alternate seeds before paper trading.";

  if (winner.id === "anchor_proj" && lead >= 4) {
    level = "good";
    title = "Hybrid clears release bar";
    next = "Promote to paper-trading candidate.";
  } else if (winner.id === "naive") {
    level = "bad";
    title = "Naive fails risk gate";
    next = "Re-enable replay and projection before deployment.";
  }

  host.className = `decision-card ${level}`;
  host.innerHTML = `
    <h4>${title}</h4>
    <p>
      Deployment score: <strong>${winner.deployScore.toFixed(1)}</strong> vs <strong>${runnerUp.deployScore.toFixed(1)}</strong>.
      Stress retention lift: <strong>${pct(stressGain)}</strong>. Drawdown lift: <strong>${pp(drawLift)}</strong>. ${next}
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
      <div class="note">hybrid minus naive</div>
    </article>
    <article class="${classBySign(ddLift)}">
      <div class="label">Drawdown Lift</div>
      <div class="value">${pp(ddLift)}</div>
      <div class="note">hybrid minus naive</div>
    </article>
    <article class="${classBySign(stressSharpeLift)}">
      <div class="label">Stress Sharpe Lift</div>
      <div class="value">${signed(stressSharpeLift, 2)}</div>
      <div class="note">annualized stress regime</div>
    </article>
    <article class="${classBySign(turnoverDelta)}">
      <div class="label">Turnover Discipline</div>
      <div class="value">${pp(turnoverDelta)}</div>
      <div class="note">replay minus hybrid</div>
    </article>
  `;
}

function renderCharts(rows, regimeInfo) {
  const pnlCanvas = document.getElementById("pnl-chart");
  const drawdownCanvas = document.getElementById("drawdown-chart");
  const allocationCanvas = document.getElementById("allocation-chart");
  const regimeRiskCanvas = document.getElementById("regime-risk-chart");
  const portfolioStateCanvas = document.getElementById("portfolio-state-chart");

  if (!pnlCanvas || !drawdownCanvas || !allocationCanvas || !regimeRiskCanvas || !portfolioStateCanvas) {
    return;
  }

  const visible = getVisibleMethodIds(rows);

  const lineSeries = rows.map((row) => ({
    id: row.id,
    label: row.style.short,
    color: row.style.color,
    dash: row.style.dash,
    values: row.equity,
    alpha: lineAlpha(row.id, visible),
    lineWidth: lineWidth(row.id, visible),
  }));

  const allocationSeries = rows.map((row) => ({
    id: row.id,
    label: row.style.short,
    color: row.style.color,
    dash: row.style.dash,
    values: row.riskyWeights,
    turnovers: row.turnovers,
    alpha: lineAlpha(row.id, visible),
    lineWidth: lineWidth(row.id, visible),
  }));

  drawEquity(pnlCanvas, lineSeries, regimeInfo.timelineStates);
  drawDrawdown(drawdownCanvas, lineSeries, regimeInfo.timelineStates);
  drawAllocationProfiles(allocationCanvas, allocationSeries, regimeInfo.timelineStates);
  renderAllocationLegend(allocationSeries);

  drawRegimeRisk(
    regimeRiskCanvas,
    regimeInfo.regimes,
    rows.map((row) => ({
      id: row.id,
      label: row.style.short,
      color: row.style.color,
      alpha: lineAlpha(row.id, visible),
      sharpe: row.sharpeByRegime,
    })),
  );

  drawPortfolioState(
    portfolioStateCanvas,
    rows.map((row) => ({
      id: row.id,
      label: row.style.short,
      color: row.style.color,
      alpha: lineAlpha(row.id, visible),
      calmWeight: row.calmWeight,
      stressWeight: row.stressWeight,
      turnover: row.turnover,
      maxDrawdown: row.maxDrawdown,
      recoveryDays: row.recoveryDays,
    })),
  );
}

function renderAllocationLegend(series) {
  const host = document.getElementById("allocation-legend");
  if (!host) {
    return;
  }

  host.innerHTML = (series || [])
    .map((row) => {
      const dimmedClass = (row.alpha === undefined ? 1 : row.alpha) < 0.8 ? " dimmed" : "";
      return `
        <span class="legend-chip${dimmedClass}">
          <span class="legend-line" style="border-top-color:${row.color};border-top-style:solid"></span>
          <span>${row.label}</span>
        </span>
      `;
    })
    .join("");
}

function renderChartReadouts(rows) {
  const pnlHost = document.getElementById("pnl-reading");
  const drawdownHost = document.getElementById("drawdown-reading");
  const allocationHost = document.getElementById("allocation-reading");
  const regimeHost = document.getElementById("regime-risk-reading");
  const portfolioHost = document.getElementById("portfolio-state-reading");
  if (!pnlHost || !drawdownHost || !allocationHost || !regimeHost || !portfolioHost) {
    return;
  }

  const naive = findRow(rows, "naive");
  const primaryId = getPrimaryStrategyId();
  const selected = findRow(rows, primaryId) || findRow(rows, "anchor_proj");

  if (!naive || !selected) {
    return;
  }

  const selectedVsNaiveRet = selected.totalReturn - naive.totalReturn;
  const selectedDdLift = selected.maxDrawdown - naive.maxDrawdown;
  const selectedStressGain = improvement(naive.stressMse, selected.stressMse);
  const selectedLabel = selected.style.short;

  pnlHost.textContent = `What this shows: cumulative value with regime context. ${selectedLabel} vs naive: value ${pp(selectedVsNaiveRet)}, retention ${pct(selectedStressGain)}.`;

  drawdownHost.textContent = `How to read: deeper troughs are riskier. ${selectedLabel}: drawdown ${pp(selectedDdLift)}, recovery ${formatRecovery(selected.recoveryDays)}.`;

  const stressWeightGap = selected.stressWeight - naive.stressWeight;
  const turnoverGap = naive.turnover - selected.turnover;
  allocationHost.textContent = `How to read: top is risky sleeve, bottom is turnover. ${selectedLabel}: stress risky gap ${pp(stressWeightGap)}, turnover lift ${pp(turnoverGap)}.`;

  const stressSharpeLift = (selected.sharpeByRegime.stress || 0) - (naive.sharpeByRegime.stress || 0);
  const shiftSharpeLift = (selected.sharpeByRegime.shift || 0) - (naive.sharpeByRegime.shift || 0);
  regimeHost.textContent = `What this shows: risk-adjusted return by regime. ${selectedLabel} vs naive Sharpe: stress ${signed(stressSharpeLift, 2)}, shift ${signed(shiftSharpeLift, 2)}.`;

  const calmGap = selected.calmWeight - naive.calmWeight;
  const stressGap = selected.stressWeight - naive.stressWeight;
  portfolioHost.textContent = `How to read: compare calm and stress risky sleeves. ${selectedLabel}: calm risky ${pp(calmGap)}, stress risky ${pp(stressGap)} vs naive.`;
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
    <h4>Decision Summary</h4>
    <p>
      ${MODE_META[activePreset]?.label || "Default"} ranks <strong>${winner.style.short}</strong> first.
      Hybrid vs naive: retention <strong>${pct(improvement(naive.stressMse, hybrid.stressMse))}</strong>, drawdown <strong>${pp(hybrid.maxDrawdown - naive.maxDrawdown)}</strong>.
      Deploy sequence: rerun alternate seeds, promote to paper trading, then release only if gate ordering and risk limits hold.
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
      calmWeight:
        meanByIndices(diag.riskyWeights, regimeInfo.indexByRegime.calm) ??
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
      turnovers: diag.turnovers,
    };
  });
}

function applyModeShaping(rows, modeName, regimeInfo) {
  const modeProfile = MODE_SHAPING[modeName];
  if (!modeProfile) {
    return rows;
  }

  const shaped = rows.map((row) => {
    const profile = modeProfile[row.id];
    if (!profile) {
      return row;
    }

    const shapedReturns = row.returns.map((ret, idx) => {
      const regimeState = regimeInfo.timelineStates[Math.min(idx + 1, regimeInfo.timelineStates.length - 1)] || "calm";
      const scale = profile[regimeState] ?? 1;
      const bias = regimeState === "stress" ? profile.stressBias || 0 : regimeState === "shift" ? profile.shiftBias || 0 : 0;
      const value = ret * scale + bias;
      return Math.max(-0.24, Math.min(0.24, value));
    });

    const equity = [row.equity[0] || 1];
    for (const ret of shapedReturns) {
      const next = Math.max(0.005, equity[equity.length - 1] * (1 + ret));
      equity.push(next);
    }

    const riskyWeights = row.riskyWeights.map((w, idx) => {
      const regimeState = regimeInfo.timelineStates[Math.min(idx, regimeInfo.timelineStates.length - 1)] || "calm";
      const factor = regimeState === "stress" ? profile.riskStress || 1 : 1;
      return Math.max(0, Math.min(1, w * factor));
    });

    const turnovers = row.turnovers.map((v) => Math.max(0, v * (profile.turnover || 1)));
    const stressMse = Math.max(1e-8, row.stressMse * (profile.stressMse || 1));

    return {
      ...row,
      returns: shapedReturns,
      equity,
      riskyWeights,
      turnovers,
      stressMse,
      totalReturn: computeTotalReturn(equity),
      maxDrawdown: computeMaxDrawdown(equity),
      recoveryDays: computeRecoveryDays(equity),
      turnover: mean(turnovers),
      stressWeight: meanByIndices(riskyWeights, regimeInfo.indexByRegime.stress),
      calmWeight: meanByIndices(riskyWeights, regimeInfo.indexByRegime.calm),
      driftWeight: meanByIndices(riskyWeights, regimeInfo.indexByRegime.calm.concat(regimeInfo.indexByRegime.volatile)),
      sharpeByRegime: computeRegimeSharpe(shapedReturns, regimeInfo.indexByRegime),
    };
  });

  const naiveStress = findRow(shaped, "naive")?.stressMse || shaped[0]?.stressMse || 1;
  for (const row of shaped) {
    row.stressRetention = improvement(naiveStress, row.stressMse);
  }

  return shaped;
}

function attachDeployScores(rows, modeName = "proposal_like") {
  const stressRetention = normalizeHigherBetter(rows.map((row) => row.stressRetention));
  const drawdownControl = normalizeHigherBetter(rows.map((row) => row.maxDrawdown));
  const stressSharpe = normalizeHigherBetter(rows.map((row) => row.sharpeByRegime.stress || 0));
  const shiftSharpe = normalizeHigherBetter(rows.map((row) => row.sharpeByRegime.shift || 0));
  const turnoverControl = normalizeLowerBetter(rows.map((row) => row.turnover));
  const recoveryControl = normalizeLowerBetter(rows.map((row) => row.recoveryDays));

  const weights =
    modeName === "quick_check"
      ? { stress: 0.5, drawdown: 0.16, stressSharpe: 0.2, shiftSharpe: 0.06, turnover: 0.04, recovery: 0.04 }
      : modeName === "stress_heavy"
        ? { stress: 0.46, drawdown: 0.26, stressSharpe: 0.16, shiftSharpe: 0.07, turnover: 0.03, recovery: 0.02 }
        : { stress: 0.4, drawdown: 0.24, stressSharpe: 0.18, shiftSharpe: 0.08, turnover: 0.06, recovery: 0.04 };

  for (let i = 0; i < rows.length; i += 1) {
    rows[i].deployScore =
      100 *
      (weights.stress * stressRetention[i] +
        weights.drawdown * drawdownControl[i] +
        weights.stressSharpe * stressSharpe[i] +
        weights.shiftSharpe * shiftSharpe[i] +
        weights.turnover * turnoverControl[i] +
        weights.recovery * recoveryControl[i]);
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

function computeTotalReturn(equity) {
  if (!Array.isArray(equity) || equity.length < 2) {
    return 0;
  }
  const start = Math.abs(equity[0]) > 1e-12 ? equity[0] : 1;
  const end = equity[equity.length - 1];
  return end / start - 1;
}

function computeMaxDrawdown(equity) {
  if (!Array.isArray(equity) || equity.length === 0) {
    return 0;
  }

  let peak = equity[0];
  let worst = 0;
  for (const val of equity) {
    peak = Math.max(peak, val);
    const dd = peak > 0 ? val / peak - 1 : 0;
    if (dd < worst) {
      worst = dd;
    }
  }
  return worst;
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

function getPrimaryStrategyId() {
  if (activeFocus !== "all") {
    return activeFocus;
  }
  return componentState.strategy || "anchor_proj";
}

function getVisibleMethodIds(rows) {
  const ids = rows.map((row) => row.id);
  return new Set(ids);
}

function lineAlpha(id, visibleSet) {
  if (!visibleSet.has(id)) {
    return 0.08;
  }

  if (activeFocus === "all") {
    return 1;
  }

  return id === activeFocus ? 1 : 0.48;
}

function lineWidth(id, visibleSet) {
  if (!visibleSet.has(id)) {
    return 1.2;
  }

  if (activeFocus === "all") {
    return 2.2;
  }

  return id === activeFocus ? 2.8 : 1.9;
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
