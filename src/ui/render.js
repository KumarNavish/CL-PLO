import { clampConfig, DEFAULT_CONFIG, PRESETS } from "../config.js?v=20260216r4";
import { DEMO_RESULTS } from "../content/demo-results.js?v=20260216r4";
import {
  drawAllocationProfiles,
  drawCostAttribution,
  drawDrawdown,
  drawEquity,
  drawGrossNet,
  drawPortfolioState,
  drawQualificationGate,
  drawRebalanceSweep,
  drawRegimeRisk,
} from "./charts.js?v=20260216r4";

const FIELD_MAP = [
  "seed",
  "steps",
  "anchorBeta",
  "pStress",
  "loraRank",
  "cLin",
  "cQuad",
  "aumScale",
  "rebalanceEveryK",
  "rebalanceThreshold",
  "qualTauW",
  "qualTauMu",
  "qualTauRank",
  "turnoverPenalty",
];

const SELECT_FIELDS = ["costModel", "rebalancePolicy"];
const CHECKBOX_FIELDS = ["qualificationEnabled", "qualificationAdvanced", "qualRequireFundamental"];

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
    readout: "Fast check with a coarse stress mix.",
    lens: "Quick gate.",
  },
  proposal_like: {
    label: "Default",
    runLabel: "default run",
    readout: "Primary deployment setting.",
    lens: "Primary gate.",
  },
  stress_heavy: {
    label: "Stress+",
    runLabel: "stress run",
    readout: "Adverse regime mix to expose failure modes.",
    lens: "Stress gate.",
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
    const baseline = clampConfig({ ...DEFAULT_CONFIG, ...(PRESETS.proposal_like?.values || {}) });
    fillForm(baseline);
    setActiveMode("proposal_like");
    setActiveFocus("all", false);
    syncComponentFromStrategy("anchor_proj", false);
    setStatus("Reset to Default.");
    if (latestResult) {
      renderAll(latestResult);
    }
  });

  document.querySelectorAll("#implementation-form input, #implementation-form select").forEach((node) => {
    node.addEventListener("change", () => {
      setStatus("Implementation realism settings updated. Run Demo to refresh.");
    });
  });

  document.getElementById("apply-realism-off")?.addEventListener("click", () => {
    applyRealismBaseline();
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
  setStatus(`${mode.label} run in progress...`);

  if (!payload) {
    setStatus(`No dataset found for ${mode.label}.`, true);
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

  for (const key of SELECT_FIELDS) {
    const input = document.querySelector(`[name="${key}"]`);
    if (!input) {
      continue;
    }
    cfg[key] = String(input.value || "");
  }

  for (const key of CHECKBOX_FIELDS) {
    const input = document.querySelector(`[name="${key}"]`);
    if (!input) {
      continue;
    }
    cfg[key] = Boolean(input.checked);
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

  for (const key of SELECT_FIELDS) {
    const input = document.querySelector(`[name="${key}"]`);
    if (!input || cfg[key] === undefined) {
      continue;
    }
    input.value = String(cfg[key]);
  }

  for (const key of CHECKBOX_FIELDS) {
    const input = document.querySelector(`[name="${key}"]`);
    if (!input || cfg[key] === undefined) {
      continue;
    }
    input.checked = Boolean(cfg[key]);
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
    setStatus(`${preset.label} mode selected. Run Demo to refresh plots.`);
  }
}

function applyRealismBaseline(announce = true) {
  const patch = {
    costModel: "off",
    cLin: 0,
    cQuad: 0,
    aumScale: 1,
    rebalancePolicy: "daily",
    rebalanceEveryK: 5,
    rebalanceThreshold: 0.04,
    qualificationEnabled: false,
    qualificationAdvanced: false,
    qualTauW: 0.04,
    qualTauMu: 0.01,
    qualTauRank: 0.2,
    qualRequireFundamental: false,
    turnoverPenalty: 0,
  };

  const merged = clampConfig({ ...DEFAULT_CONFIG, ...readConfigFromForm(), ...patch });
  fillForm(merged);

  if (announce) {
    setStatus("Implementation realism set to baseline: daily, zero cost, no qualification gate.");
  }

  if (latestResult) {
    renderAll(latestResult);
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
    setStatus(`Focus: ${prettyFocus(activeFocus)}.`);
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
    const note = coerced ? "Projection needs anchors." : "Constraint toggles updated.";
    setStatus(`${note} ${METHOD_STYLES[componentState.strategy]?.short || "Strategy"} selected.`);
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
    setStatus(`${METHOD_STYLES[componentState.strategy]?.short || "Strategy"} selected.`);
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

function renderErrorMessage(error) {
  if (!error) {
    return "Unknown error.";
  }
  if (typeof error === "string") {
    return error;
  }
  if (typeof error.message === "string" && error.message.trim()) {
    return error.message;
  }
  return "Unknown error.";
}

function pushRenderError(errors, label, error) {
  const message = `${label}: ${renderErrorMessage(error)}`;
  errors.push(message);
  console.error(`[CL-PLO render] ${message}`, error);
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
  applyExecutionRealism(methodRows, regimeInfo, result.config || DEFAULT_CONFIG);
  attachDeployScores(methodRows, activePreset);

  const errors = [];
  const safeRender = (label, renderFn) => {
    try {
      renderFn();
    } catch (error) {
      pushRenderError(errors, label, error);
    }
  };

  safeRender("Decision card", () => renderDecisionCard(methodRows));
  safeRender("Impact KPIs", () => renderKpis(methodRows));

  const chartErrors = renderCharts(methodRows, regimeInfo);
  if (Array.isArray(chartErrors) && chartErrors.length > 0) {
    errors.push(...chartErrors);
  }

  safeRender("Chart readouts", () => renderChartReadouts(methodRows));
  safeRender("Takeaway", () => renderTakeaway(methodRows));

  if (errors.length === 0) {
    setStatus(`${mode.label} run complete. ${mode.lens}`);
    return;
  }

  setStatus(`${mode.label} run complete with ${errors.length} render issue(s). Check console.`, true);
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
  const costLift = naive && hybrid ? naive.costDrag - hybrid.costDrag : 0;

  let level = "caution";
  let title = `${winner.style.short} leads this gate`;
  let next = "Recheck with alternate seeds before paper trading.";

  if (winner.id === "anchor_proj" && lead >= 4) {
    level = "good";
    title = "Hybrid clears the gate";
    next = "Move to paper trading.";
  } else if (winner.id === "naive") {
    level = "bad";
    title = "Naive fails the risk gate";
    next = "Turn replay and projection back on before deployment.";
  }

  host.className = `decision-card ${level}`;
  host.innerHTML = `
    <h4>${title}</h4>
    <p>
      Deployment score: <strong>${winner.deployScore.toFixed(1)}</strong> vs <strong>${runnerUp.deployScore.toFixed(1)}</strong>.
      Stress memory gain: <strong>${pct(stressGain)}</strong>. Drawdown gain: <strong>${pp(drawLift)}</strong>.
      Cost drag gain: <strong>${pp(costLift)}</strong>. ${next}
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
  const costDelta = naive.costDrag - hybrid.costDrag;
  const tradeRateDelta = naive.tradeRate - hybrid.tradeRate;

  host.innerHTML = `
    <article class="${classBySign(stressGain)}">
      <div class="label">Stress Memory Gain</div>
      <div class="value">${pct(stressGain)}</div>
      <div class="note">hybrid minus naive</div>
    </article>
    <article class="${classBySign(ddLift)}">
      <div class="label">Drawdown Gain</div>
      <div class="value">${pp(ddLift)}</div>
      <div class="note">hybrid minus naive</div>
    </article>
    <article class="${classBySign(stressSharpeLift)}">
      <div class="label">Stress Sharpe Gain</div>
      <div class="value">${signed(stressSharpeLift, 2)}</div>
      <div class="note">annualized stress regime</div>
    </article>
    <article class="${classBySign(turnoverDelta)}">
      <div class="label">Turnover Reduction</div>
      <div class="value">${pp(turnoverDelta)}</div>
      <div class="note">replay minus hybrid</div>
    </article>
    <article class="${classBySign(costDelta)}">
      <div class="label">Cost Drag Reduction</div>
      <div class="value">${pp(costDelta)}</div>
      <div class="note">naive minus hybrid</div>
    </article>
    <article class="${classBySign(tradeRateDelta)}">
      <div class="label">Trade-Rate Reduction</div>
      <div class="value">${pp(tradeRateDelta)}</div>
      <div class="note">naive minus hybrid</div>
    </article>
  `;
}

function renderCharts(rows, regimeInfo) {
  const errors = [];
  const pnlCanvas = document.getElementById("pnl-chart");
  const drawdownCanvas = document.getElementById("drawdown-chart");
  const allocationCanvas = document.getElementById("allocation-chart");
  const regimeRiskCanvas = document.getElementById("regime-risk-chart");
  const portfolioStateCanvas = document.getElementById("portfolio-state-chart");
  const grossNetCanvas = document.getElementById("gross-net-chart");
  const costAttrCanvas = document.getElementById("cost-attribution-chart");
  const qualificationCanvas = document.getElementById("qualification-chart");
  const rebalanceSweepCanvas = document.getElementById("rebalance-sweep-chart");

  if (!pnlCanvas || !drawdownCanvas || !allocationCanvas || !regimeRiskCanvas || !portfolioStateCanvas) {
    errors.push("Core chart canvases are missing from the page.");
    return errors;
  }

  const safeDraw = (label, drawFn) => {
    try {
      drawFn();
    } catch (error) {
      pushRenderError(errors, label, error);
    }
  };

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

  safeDraw("Q1 portfolio value", () => {
    drawEquity(pnlCanvas, lineSeries, regimeInfo.timelineStates);
  });
  safeDraw("Q2 drawdown", () => {
    drawDrawdown(drawdownCanvas, lineSeries, regimeInfo.timelineStates);
  });
  safeDraw("Q3 allocation and turnover", () => {
    drawAllocationProfiles(allocationCanvas, allocationSeries, regimeInfo.timelineStates);
  });
  safeDraw("Allocation legend", () => {
    renderAllocationLegend(allocationSeries);
  });

  safeDraw("Q4 regime split", () => {
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
  });

  safeDraw("Q5 deployment state", () => {
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
  });

  const primary = findRow(rows, getPrimaryStrategyId()) || findRow(rows, "anchor_proj") || rows[0];
  if (!primary) {
    errors.push("No strategy row is available for detail charts.");
    return errors;
  }

  if (grossNetCanvas) {
    safeDraw("Q6 gross vs net", () => {
      drawGrossNet(
        grossNetCanvas,
        {
          label: primary.style.short,
          gross: primary.grossEquity,
          net: primary.equity,
        },
        regimeInfo.timelineStates,
      );
    });
  }

  if (costAttrCanvas) {
    safeDraw("Q7 cost attribution", () => {
      drawCostAttribution(
        costAttrCanvas,
        rows.map((row) => ({
          id: row.id,
          label: row.style.short,
          color: row.style.color,
          alpha: lineAlpha(row.id, visible),
          grossReturn: row.grossTotalReturn,
          netReturn: row.totalReturn,
          costDrag: row.costDrag,
        })),
      );
    });
  }

  if (qualificationCanvas) {
    safeDraw("Q8 qualification gate", () => {
      drawQualificationGate(
        qualificationCanvas,
        {
          signalW: primary.signalMagnitudeW,
          threshold: Number(latestResult?.config?.qualTauW || 0),
          trades: primary.tradeFlags,
          fundamentals: primary.fundamentalFlags,
        },
        regimeInfo.timelineStates,
      );
    });
  }

  if (rebalanceSweepCanvas) {
    safeDraw("Q9 rebalance sweep", () => {
      drawRebalanceSweep(
        rebalanceSweepCanvas,
        buildRebalanceSweep(rows, latestResult?.config || DEFAULT_CONFIG),
      );
    });
  }

  return errors;
}

function renderAllocationLegend(series) {
  const host = document.getElementById("allocation-legend");
  if (!host) {
    return;
  }

  host.innerHTML = (series || [])
    .map((row) => {
      const dashed = Array.isArray(row.dash) && row.dash.length > 0;
      const dimmedClass = (row.alpha === undefined ? 1 : row.alpha) < 0.8 ? " dimmed" : "";
      return `
        <span class="legend-chip${dimmedClass}">
          <span class="legend-line" style="border-top-color:${row.color};border-top-style:${dashed ? "dashed" : "solid"}"></span>
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
  const grossNetHost = document.getElementById("gross-net-reading");
  const costHost = document.getElementById("cost-attribution-reading");
  const qualHost = document.getElementById("qualification-reading");
  const sweepHost = document.getElementById("rebalance-sweep-reading");
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

  pnlHost.textContent = `Shows portfolio value over time with regime shading. ${selectedLabel} vs Naive: value ${pp(selectedVsNaiveRet)}, stress memory ${pct(selectedStressGain)}.`;

  drawdownHost.textContent = `Read deeper negative values as larger losses from peak. ${selectedLabel}: drawdown ${pp(selectedDdLift)}, recovery ${formatRecovery(selected.recoveryDays)}.`;

  const stressWeightGap = selected.stressWeight - naive.stressWeight;
  const turnoverGap = naive.turnover - selected.turnover;
  allocationHost.textContent = `Top panel is risky allocation; bottom panel is turnover. ${selectedLabel}: stress risky gap ${pp(stressWeightGap)}, turnover gain ${pp(turnoverGap)}.`;

  const stressSharpeLift = (selected.sharpeByRegime.stress || 0) - (naive.sharpeByRegime.stress || 0);
  const shiftSharpeLift = (selected.sharpeByRegime.shift || 0) - (naive.sharpeByRegime.shift || 0);
  regimeHost.textContent = `Shows risk-adjusted return by regime. ${selectedLabel} vs Naive Sharpe: stress ${signed(stressSharpeLift, 2)}, shift ${signed(shiftSharpeLift, 2)}.`;

  const calmGap = selected.calmWeight - naive.calmWeight;
  const stressGap = selected.stressWeight - naive.stressWeight;
  portfolioHost.textContent = `Compare calm and stress risky sleeves for each strategy. ${selectedLabel}: calm risky ${pp(calmGap)}, stress risky ${pp(stressGap)} vs Naive.`;

  if (grossNetHost) {
    const grossNetGap = selected.grossTotalReturn - selected.totalReturn;
    grossNetHost.textContent = `Gross vs net isolates implementation drag. ${selectedLabel}: gross ${pp(selected.grossTotalReturn)}, net ${pp(selected.totalReturn)}, cost drag ${pp(grossNetGap)}.`;
  }

  if (costHost) {
    const selectedCost = selected.costDrag;
    const naiveCost = naive.costDrag;
    costHost.textContent = `Cost attribution compares each strategy's gross and net outcome. ${selectedLabel}: cost drag ${pp(naiveCost - selectedCost)} better than Naive.`;
  }

  if (qualHost) {
    const ignoredNoise = 1 - selected.qualifiedRate;
    qualHost.textContent = `Qualification gate: signal changes above threshold trigger trades. ${selectedLabel}: trade rate ${pct(selected.tradeRate)}, skipped noisy updates ${pct(ignoredNoise)}.`;
  }

  if (sweepHost) {
    sweepHost.textContent = "Mini sweep compares Daily, Weekly, Event, and Threshold policies using current run outputs. Read left-to-right as responsiveness vs implementation drag.";
  }
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
      Hybrid vs Naive: stress memory <strong>${pct(improvement(naive.stressMse, hybrid.stressMse))}</strong>, drawdown <strong>${pp(hybrid.maxDrawdown - naive.maxDrawdown)}</strong>, cost drag <strong>${pp(naive.costDrag - hybrid.costDrag)}</strong>.
      Next steps: rerun alternate seeds, move to paper trading, then release only if ordering and risk limits hold.
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
      grossTotalReturn: metrics.grossTotalReturn ?? metrics.totalReturn ?? 0,
      netTotalReturn: metrics.netTotalReturn ?? metrics.totalReturn ?? 0,
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
      tradeRate: metrics.tradeRate ?? 0,
      qualifiedRate: metrics.qualifiedRate ?? 0,
      precisionProxy: metrics.precisionProxy ?? 0,
      recallProxy: metrics.recallProxy ?? 0,
      costDrag: metrics.costDrag ?? 0,
      sharpeByRegime,
      equity: diag.equity,
      returns: diag.returns,
      grossEquity: diag.grossEquity || diag.equity,
      grossReturns: diag.grossReturns || diag.returns,
      costs: diag.costs || Array(diag.returns.length).fill(0),
      tradeFlags: diag.tradeFlags || Array(diag.returns.length).fill(1),
      qualifiedFlags: diag.qualifiedFlags || Array(diag.returns.length).fill(1),
      fundamentalFlags: diag.fundamentalFlags || Array(diag.returns.length).fill(0),
      signalMagnitudeW: diag.signalMagnitudeW || Array(diag.returns.length).fill(0),
      signalMagnitudeMu: diag.signalMagnitudeMu || Array(diag.returns.length).fill(0),
      signalRank: diag.signalRank || Array(diag.returns.length).fill(0),
      targetRiskyWeights: diag.targetRiskyWeights || diag.riskyWeights,
      proposedTurnovers: diag.proposedTurnovers || diag.turnovers,
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

function applyExecutionRealism(rows, regimeInfo, cfg) {
  for (const row of rows) {
    const sourceReturns = (row.returns || []).slice();
    const sourceRiskyWeights = (row.riskyWeights || []).slice();
    const sourceTurnovers = (row.turnovers || []).slice();
    const length = Math.min(sourceReturns.length, sourceRiskyWeights.length);
    if (length === 0) {
      continue;
    }

    const grossReturns = [];
    const netReturns = [];
    const costs = [];
    const turnovers = [];
    const executedRisky = [];
    const targetRisky = [];
    const tradeFlags = [];
    const qualifiedFlags = [];
    const fundamentalFlags = [];
    const signalMagnitudeW = [];
    const signalMagnitudeMu = [];
    const signalRank = [];
    const grossEquity = [1];
    const netEquity = [1];

    let prevRisk = Math.max(0, Math.min(1, sourceRiskyWeights[0] || 0));
    let prevTarget = prevRisk;
    let prevSignal = sourceReturns[0] || 0;
    let prevState = regimeInfo.timelineStates[0] || "calm";

    let tradeCount = 0;
    let qualifyCount = 0;
    let fundamentalCount = 0;
    let tradeOnFundamental = 0;

    for (let t = 0; t < length; t += 1) {
      const state = regimeInfo.timelineStates[Math.min(t, regimeInfo.timelineStates.length - 1)] || "calm";
      const proposedRisk = Math.max(0, Math.min(1, sourceRiskyWeights[t] || 0));
      const gross = sourceReturns[t] || 0;
      const baseTurn = Math.max(0, sourceTurnovers[t] || 0);

      const signalW = Math.abs(proposedRisk - prevTarget);
      const signalMu = Math.abs(gross - prevSignal);
      const signFlip = Math.sign(gross) !== Math.sign(prevSignal) ? 1 : 0;
      const rankProxy = Math.min(1.5, signFlip * 0.6 + Math.min(0.9, signalMu * 25));
      const fundamental = state === "shift" || state !== prevState;

      const rebalanceAllowed = shouldRebalanceLocal(cfg, t, signalW, fundamental);
      const qualified = passesQualificationLocal(cfg, signalW, signalMu, rankProxy, fundamental);
      const doTrade = rebalanceAllowed && qualified;
      const penalty = Number(cfg.turnoverPenalty || 0);
      const damp = doTrade ? 1 / (1 + penalty * baseTurn) : 0;
      const nextRisk = doTrade
        ? Math.max(0, Math.min(1, prevRisk + (proposedRisk - prevRisk) * damp))
        : prevRisk;

      const turnover = doTrade ? baseTurn * damp : 0;
      const missedAdaptation = doTrade ? 0 : missedAdaptationPenalty(signalW, state);
      const adjustedGross = gross - missedAdaptation;
      const cost = executionCostLocal(turnover, cfg);
      const net = adjustedGross - cost;

      grossReturns.push(adjustedGross);
      netReturns.push(net);
      costs.push(cost);
      turnovers.push(turnover);
      executedRisky.push(nextRisk);
      targetRisky.push(proposedRisk);
      tradeFlags.push(doTrade ? 1 : 0);
      qualifiedFlags.push(qualified ? 1 : 0);
      fundamentalFlags.push(fundamental ? 1 : 0);
      signalMagnitudeW.push(signalW);
      signalMagnitudeMu.push(signalMu);
      signalRank.push(rankProxy);

      grossEquity.push(Math.max(0.005, grossEquity[grossEquity.length - 1] * (1 + adjustedGross)));
      netEquity.push(Math.max(0.005, netEquity[netEquity.length - 1] * (1 + net)));

      if (doTrade) {
        tradeCount += 1;
      }
      if (qualified) {
        qualifyCount += 1;
      }
      if (fundamental) {
        fundamentalCount += 1;
      }
      if (doTrade && fundamental) {
        tradeOnFundamental += 1;
      }

      prevRisk = nextRisk;
      prevTarget = proposedRisk;
      prevSignal = gross;
      prevState = state;
    }

    row.grossReturns = grossReturns;
    row.baseGrossReturns = sourceReturns;
    row.baseRiskyWeights = sourceRiskyWeights;
    row.baseTurnovers = sourceTurnovers;
    row.returns = netReturns;
    row.costs = costs;
    row.turnovers = turnovers;
    row.riskyWeights = executedRisky;
    row.targetRiskyWeights = targetRisky;
    row.tradeFlags = tradeFlags;
    row.qualifiedFlags = qualifiedFlags;
    row.fundamentalFlags = fundamentalFlags;
    row.signalMagnitudeW = signalMagnitudeW;
    row.signalMagnitudeMu = signalMagnitudeMu;
    row.signalRank = signalRank;
    row.grossEquity = grossEquity;
    row.netEquity = netEquity;
    row.equity = netEquity;
    row.totalReturn = computeTotalReturn(netEquity);
    row.grossTotalReturn = computeTotalReturn(grossEquity);
    row.netTotalReturn = row.totalReturn;
    row.maxDrawdown = computeMaxDrawdown(netEquity);
    row.recoveryDays = computeRecoveryDays(netEquity);
    row.turnover = mean(turnovers);
    row.stressWeight = meanByIndices(executedRisky, regimeInfo.indexByRegime.stress);
    row.calmWeight = meanByIndices(executedRisky, regimeInfo.indexByRegime.calm);
    row.driftWeight = meanByIndices(executedRisky, regimeInfo.indexByRegime.calm.concat(regimeInfo.indexByRegime.volatile));
    row.sharpeByRegime = computeRegimeSharpe(netReturns, regimeInfo.indexByRegime);
    row.tradeRate = tradeCount / Math.max(1, length);
    row.qualifiedRate = qualifyCount / Math.max(1, length);
    row.precisionProxy = tradeOnFundamental / Math.max(1, tradeCount);
    row.recallProxy = tradeOnFundamental / Math.max(1, fundamentalCount);
    row.totalCost = sum(costs);
    const grossMagnitude = sum(row.grossReturns.map((v) => Math.abs(v)));
    row.costDrag = Math.min(2, row.totalCost / Math.max(1e-6, grossMagnitude));
  }
}

function shouldRebalanceLocal(cfg, t, signalW, fundamental) {
  const policy = String(cfg.rebalancePolicy || "daily");
  if (policy === "daily") {
    return true;
  }
  if (policy === "periodic") {
    const k = Math.max(1, Math.floor(Number(cfg.rebalanceEveryK) || 1));
    return t % k === 0;
  }
  if (policy === "event") {
    return fundamental || signalW >= Number(cfg.qualTauW || 0);
  }
  if (policy === "threshold") {
    return signalW >= Number(cfg.rebalanceThreshold || 0);
  }
  return true;
}

function passesQualificationLocal(cfg, signalW, signalMu, signalRank, fundamental) {
  if (!cfg.qualificationEnabled) {
    return true;
  }
  let pass = signalW >= Number(cfg.qualTauW || 0);
  if (cfg.qualificationAdvanced) {
    pass = pass &&
      signalMu >= Number(cfg.qualTauMu || 0) &&
      signalRank >= Number(cfg.qualTauRank || 0);
  }
  if (cfg.qualRequireFundamental) {
    pass = pass && fundamental;
  }
  return pass;
}

function executionCostLocal(turnover, cfg) {
  const model = String(cfg.costModel || "off");
  if (model === "off") {
    return 0;
  }
  const lin = Number(cfg.cLin || 0);
  const quad = Number(cfg.cQuad || 0) * Number(cfg.aumScale || 1);
  const linearCost = lin * turnover;
  if (model === "linear") {
    return linearCost;
  }
  return linearCost + quad * turnover * turnover;
}

function missedAdaptationPenalty(signalW, state) {
  if (!Number.isFinite(signalW) || signalW <= 0) {
    return 0;
  }
  const regimeMultiplier = state === "shift" ? 1.3 : state === "stress" ? 1.1 : 0.55;
  return Math.min(0.05, signalW * 0.12 * regimeMultiplier);
}

function attachDeployScores(rows, modeName = "proposal_like") {
  const stressRetention = normalizeHigherBetter(rows.map((row) => row.stressRetention));
  const drawdownControl = normalizeHigherBetter(rows.map((row) => row.maxDrawdown));
  const stressSharpe = normalizeHigherBetter(rows.map((row) => row.sharpeByRegime.stress || 0));
  const shiftSharpe = normalizeHigherBetter(rows.map((row) => row.sharpeByRegime.shift || 0));
  const turnoverControl = normalizeLowerBetter(rows.map((row) => row.turnover));
  const recoveryControl = normalizeLowerBetter(rows.map((row) => row.recoveryDays));
  const costControl = normalizeLowerBetter(rows.map((row) => row.costDrag));
  const qualifyPrecision = normalizeHigherBetter(rows.map((row) => row.precisionProxy || 0));

  const weights =
    modeName === "quick_check"
      ? { stress: 0.4, drawdown: 0.14, stressSharpe: 0.16, shiftSharpe: 0.06, turnover: 0.06, recovery: 0.06, cost: 0.07, qualify: 0.05 }
      : modeName === "stress_heavy"
        ? { stress: 0.38, drawdown: 0.24, stressSharpe: 0.14, shiftSharpe: 0.06, turnover: 0.04, recovery: 0.03, cost: 0.07, qualify: 0.04 }
        : { stress: 0.33, drawdown: 0.19, stressSharpe: 0.15, shiftSharpe: 0.07, turnover: 0.07, recovery: 0.06, cost: 0.08, qualify: 0.05 };

  for (let i = 0; i < rows.length; i += 1) {
    rows[i].deployScore =
      100 *
      (weights.stress * stressRetention[i] +
        weights.drawdown * drawdownControl[i] +
        weights.stressSharpe * stressSharpe[i] +
        weights.shiftSharpe * shiftSharpe[i] +
        weights.turnover * turnoverControl[i] +
        weights.recovery * recoveryControl[i] +
        weights.cost * costControl[i] +
        weights.qualify * qualifyPrecision[i]);
  }
}

function getDiagnostics(result, methodId) {
  const diag = result.sharedDiagnostics?.[methodId] || {};
  const equity = cloneArray(diag.equity || result.equityCurves?.[methodId] || []);
  const returns = cloneArray(diag.returns || computeReturns(equity));
  const riskyWeights = cloneArray(diag.riskyWeights || []);
  const turnovers = cloneArray(diag.turnovers || []);
  const grossEquity = cloneArray(diag.grossEquity || []);
  const grossReturns = cloneArray(diag.grossReturns || []);
  const costs = cloneArray(diag.costs || []);
  const tradeFlags = cloneArray(diag.tradeFlags || []);
  const qualifiedFlags = cloneArray(diag.qualifiedFlags || []);
  const fundamentalFlags = cloneArray(diag.fundamentalFlags || []);
  const signalMagnitudeW = cloneArray(diag.signalMagnitudeW || []);
  const signalMagnitudeMu = cloneArray(diag.signalMagnitudeMu || []);
  const signalRank = cloneArray(diag.signalRank || []);
  const targetRiskyWeights = cloneArray(diag.targetRiskyWeights || []);
  const proposedTurnovers = cloneArray(diag.proposedTurnovers || []);

  return {
    equity: equity.length > 0 ? equity : [1],
    returns: returns.length > 0 ? returns : [0],
    riskyWeights: riskyWeights.length > 0 ? riskyWeights : Array(Math.max(1, equity.length)).fill(0),
    turnovers: turnovers.length > 0 ? turnovers : Array(Math.max(1, equity.length)).fill(0),
    grossEquity: grossEquity.length > 0 ? grossEquity : equity,
    grossReturns: grossReturns.length > 0 ? grossReturns : returns,
    costs: costs.length > 0 ? costs : Array(Math.max(1, returns.length)).fill(0),
    tradeFlags: tradeFlags.length > 0 ? tradeFlags : Array(Math.max(1, returns.length)).fill(1),
    qualifiedFlags: qualifiedFlags.length > 0 ? qualifiedFlags : Array(Math.max(1, returns.length)).fill(1),
    fundamentalFlags: fundamentalFlags.length > 0 ? fundamentalFlags : Array(Math.max(1, returns.length)).fill(0),
    signalMagnitudeW: signalMagnitudeW.length > 0 ? signalMagnitudeW : Array(Math.max(1, returns.length)).fill(0),
    signalMagnitudeMu: signalMagnitudeMu.length > 0 ? signalMagnitudeMu : Array(Math.max(1, returns.length)).fill(0),
    signalRank: signalRank.length > 0 ? signalRank : Array(Math.max(1, returns.length)).fill(0),
    targetRiskyWeights: targetRiskyWeights.length > 0 ? targetRiskyWeights : riskyWeights,
    proposedTurnovers: proposedTurnovers.length > 0 ? proposedTurnovers : turnovers,
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

function buildRebalanceSweep(rows, cfg) {
  const policies = [
    { id: "daily", label: "Daily" },
    { id: "periodic", label: "Weekly" },
    { id: "event", label: "Event" },
    { id: "threshold", label: "Threshold" },
  ];

  return policies.map((policy) => {
    const methods = rows.map((row) => {
      const grossReturns = row.baseGrossReturns || row.grossReturns || row.returns || [];
      const baseRisky = row.baseRiskyWeights || row.riskyWeights || [];
      const fundamentals = row.fundamentalFlags || [];
      const proposedTurn = row.baseTurnovers || row.proposedTurnovers || row.turnovers || [];

      let equity = 1;
      let grossEquity = 1;
      let trades = 0;
      let totalCost = 0;
      let prevRisk = baseRisky[0] || 0;
      let prevGross = grossReturns[0] || 0;

      for (let t = 0; t < grossReturns.length; t += 1) {
        const currentRisk = baseRisky[t] || prevRisk;
        const sw = Math.abs(currentRisk - prevRisk);
        const fundamental = (fundamentals[t] || 0) === 1;
        const gross = grossReturns[t] || 0;
        const sm = Math.abs(gross - prevGross);
        const sr = Math.min(1.5, (Math.sign(gross) !== Math.sign(prevGross) ? 0.6 : 0) + Math.min(0.9, sm * 25));
        const turn = proposedTurn[t] || 0;
        const state = fundamental ? "shift" : "calm";

        const rebalanceAllowed = shouldRebalanceLocal({ ...cfg, rebalancePolicy: policy.id }, t, sw, fundamental);
        const qualified = passesQualificationLocal(cfg, sw, sm, sr, fundamental);
        const doTrade = rebalanceAllowed && qualified;
        const turnover = doTrade ? turn : 0;
        const adjustedGross = doTrade ? gross : gross - missedAdaptationPenalty(sw, state);
        const cost = executionCostLocal(turnover, cfg);
        const net = adjustedGross - cost;

        grossEquity *= 1 + adjustedGross;
        equity *= 1 + net;
        totalCost += cost;
        if (doTrade) {
          trades += 1;
        }
        prevRisk = currentRisk;
        prevGross = gross;
      }

      const grossMagnitude = sum(grossReturns.map((v) => Math.abs(v)));
      return {
        id: row.id,
        label: row.style.short,
        color: row.style.color,
        dash: row.style.dash,
        netReturn: equity - 1,
        tradeRate: trades / Math.max(1, grossReturns.length),
        costDrag: Math.min(2, totalCost / Math.max(1e-6, grossMagnitude)),
      };
    });

    return { ...policy, methods };
  });
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

function sum(values) {
  if (!values || values.length === 0) {
    return 0;
  }
  let out = 0;
  for (const value of values) {
    out += Number(value) || 0;
  }
  return out;
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
