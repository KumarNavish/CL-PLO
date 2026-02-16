import { generateRegimeBatch, generateReturns } from "./data.js";

function riskyWeightSum(w) {
  let s = 0;
  for (let i = 1; i < w.length; i += 1) {
    s += w[i];
  }
  return s;
}

function maxDrawdown(equity) {
  let peak = equity[0];
  let worst = 0;

  for (let i = 0; i < equity.length; i += 1) {
    if (equity[i] > peak) {
      peak = equity[i];
    }
    const dd = equity[i] / peak - 1;
    if (dd < worst) {
      worst = dd;
    }
  }

  return worst;
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i += 1) {
    s += a[i] * b[i];
  }
  return s;
}

function l1Diff(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i += 1) {
    s += Math.abs(a[i] - b[i]);
  }
  return s;
}

function l2Diff(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s);
}

function meanAbsRiskyRet(y) {
  if (!y || y.length <= 1) {
    return 0;
  }
  let s = 0;
  for (let i = 1; i < y.length; i += 1) {
    s += Math.abs(y[i]);
  }
  return s / Math.max(1, y.length - 1);
}

function quantile(values, q) {
  if (!values || values.length === 0) {
    return 0;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.max(0, Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * q)));
  return sorted[idx];
}

function ranks(values) {
  const pairs = values.map((v, idx) => ({ v, idx }));
  pairs.sort((a, b) => b.v - a.v);
  const out = new Array(values.length).fill(0);
  for (let i = 0; i < pairs.length; i += 1) {
    out[pairs[i].idx] = i + 1;
  }
  return out;
}

function spearmanCorr(a, b) {
  if (!a || !b || a.length !== b.length || a.length < 2) {
    return 1;
  }
  const ra = ranks(a);
  const rb = ranks(b);
  const n = ra.length;
  const meanRank = (n + 1) / 2;

  let cov = 0;
  let va = 0;
  let vb = 0;
  for (let i = 0; i < n; i += 1) {
    const da = ra[i] - meanRank;
    const db = rb[i] - meanRank;
    cov += da * db;
    va += da * da;
    vb += db * db;
  }
  const denom = Math.sqrt(Math.max(va * vb, 1e-12));
  return cov / denom;
}

function buildTargetWeights(mu, cfg) {
  const nAssets = mu.length;
  const wTgt = new Array(nAssets).fill(0);

  const scores = new Array(nAssets - 1);
  let sumScores = 0;
  for (let i = 1; i < nAssets; i += 1) {
    const s = Math.max(mu[i], 0);
    scores[i - 1] = s;
    sumScores += s;
  }

  if (sumScores <= 1e-12) {
    wTgt[0] = 1;
    return wTgt;
  }

  let riskySum = 0;
  for (let i = 1; i < nAssets; i += 1) {
    let w = scores[i - 1] / sumScores;
    w = Math.min(w, cfg.wMaxRisky);
    wTgt[i] = w;
    riskySum += w;
  }

  if (riskySum > 1) {
    for (let i = 1; i < nAssets; i += 1) {
      wTgt[i] /= riskySum;
    }
    riskySum = 1;
  }

  wTgt[0] = Math.max(0, 1 - riskySum);
  return wTgt;
}

function blendWeights(wPrev, wTarget, cfg) {
  const nAssets = wPrev.length;
  const w = new Array(nAssets).fill(0);
  const delta = l1Diff(wTarget, wPrev);
  const penalty = Number(cfg.turnoverPenalty || 0);
  const etaBase = Number(cfg.turnoverEta || 0.2);
  const etaEff = etaBase / (1 + penalty * delta);

  let total = 0;
  for (let i = 0; i < nAssets; i += 1) {
    const blended = (1 - etaEff) * wPrev[i] + etaEff * wTarget[i];
    w[i] = Math.max(0, blended);
    total += w[i];
  }

  if (total <= 1e-12) {
    w[0] = 1;
    for (let i = 1; i < nAssets; i += 1) {
      w[i] = 0;
    }
    return w;
  }

  for (let i = 0; i < nAssets; i += 1) {
    w[i] /= total;
  }

  return w;
}

function transactionCost(turnover, cfg) {
  const model = String(cfg.costModel || "off");
  if (model === "off") {
    return 0;
  }

  const lin = Number(cfg.cLin || 0);
  const quad = Number(cfg.cQuad || 0);
  const aumScale = Number(cfg.aumScale || 1);
  const quadEff = quad * aumScale;

  const linearCost = lin * turnover;
  if (model === "linear") {
    return linearCost;
  }
  return linearCost + quadEff * turnover * turnover;
}

function shouldRebalance({
  t,
  cfg,
  signalW,
  fundamentalEvent,
}) {
  const policy = String(cfg.rebalancePolicy || "daily");
  if (policy === "daily") {
    return true;
  }
  if (policy === "periodic") {
    const k = Math.max(1, Math.floor(Number(cfg.rebalanceEveryK) || 1));
    return t % k === 0;
  }
  if (policy === "event") {
    return fundamentalEvent || signalW >= Number(cfg.qualTauW || 0);
  }
  if (policy === "threshold") {
    return signalW >= Number(cfg.rebalanceThreshold || 0);
  }
  return true;
}

function passesQualification({
  cfg,
  signalW,
  signalMu,
  signalRank,
  fundamentalEvent,
}) {
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
    pass = pass && fundamentalEvent;
  }
  return pass;
}

function generateSyntheticStream({ cfg, WBase, WDrift, rng }) {
  const streamRegimes = new Array(cfg.simT);
  const Xs = new Array(cfg.simT);
  const Ys = new Array(cfg.simT);
  const eventMarkers = new Array(cfg.simT).fill(0);
  const volValues = new Array(cfg.simT).fill(0);

  for (let t = 0; t < cfg.simT; t += 1) {
    const regime = rng.uniform() < cfg.pStress ? "stress" : "drift";
    streamRegimes[t] = regime;

    const X = generateRegimeBatch({ n: 1, dSignal: cfg.dSignal, regime, rng })[0];
    const Y = generateReturns([X], regime === "stress" ? WBase : WDrift, cfg, rng)[0];
    Xs[t] = X;
    Ys[t] = Y;

    eventMarkers[t] = rng.uniform() < 0.06 ? 1 : 0;
    volValues[t] = meanAbsRiskyRet(Y);
  }

  const volThreshold = quantile(volValues, 0.7);
  const volStates = volValues.map((v) => v >= volThreshold);
  const fundamentalEvents = new Array(cfg.simT).fill(0);

  for (let t = 0; t < cfg.simT; t += 1) {
    const regimeChange = t > 0 && streamRegimes[t] !== streamRegimes[t - 1];
    const volStateChange = t > 0 && volStates[t] !== volStates[t - 1];
    const event = eventMarkers[t] === 1;
    fundamentalEvents[t] = regimeChange || volStateChange || event ? 1 : 0;
  }

  return {
    streamRegimes,
    Xs,
    Ys,
    eventMarkers,
    volValues,
    volStates,
    volThreshold,
    fundamentalEvents,
  };
}

export function portfolioPolicy(mu, wPrev, cfg) {
  const target = buildTargetWeights(mu, cfg);
  return blendWeights(wPrev, target, cfg);
}

export function simulateStream({ cfg, model, WBase, WDrift, rng }) {
  const shared = generateSyntheticStream({ cfg, WBase, WDrift, rng });
  const diag = evaluateOnSharedStream({ cfg, model, shared });

  let worstStressDay = 0;
  let stressDays = 0;
  let stressRiskySum = 0;
  let driftDays = 0;
  let driftRiskySum = 0;

  for (let t = 0; t < shared.streamRegimes.length; t += 1) {
    const regime = shared.streamRegimes[t];
    const ret = diag.returns[t] || 0;
    const rw = diag.riskyWeights[t] || 0;

    if (regime === "stress") {
      stressDays += 1;
      stressRiskySum += rw;
      if (ret < worstStressDay) {
        worstStressDay = ret;
      }
    } else {
      driftDays += 1;
      driftRiskySum += rw;
    }
  }

  return {
    totalReturn: diag.equity[diag.equity.length - 1] - 1,
    grossTotalReturn: diag.grossEquity[diag.grossEquity.length - 1] - 1,
    netTotalReturn: diag.netEquity[diag.netEquity.length - 1] - 1,
    maxDrawdown: maxDrawdown(diag.equity),
    worstStressDay,
    nStressDays: stressDays,
    avgRiskyWeightStress: stressDays > 0 ? stressRiskySum / stressDays : 0,
    avgRiskyWeightDrift: driftDays > 0 ? driftRiskySum / driftDays : 0,
    equity: diag.equity,
    grossEquity: diag.grossEquity,
    netEquity: diag.netEquity,
    costs: diag.costs,
    tradeRate: diag.tradeRate,
    qualifiedRate: diag.qualifiedRate,
    precisionProxy: diag.precisionProxy,
    recallProxy: diag.recallProxy,
    costDrag: diag.costDrag,
  };
}

export function buildSharedStream({ cfg, WBase, WDrift, rng }) {
  return generateSyntheticStream({ cfg, WBase, WDrift, rng });
}

export function evaluateOnSharedStream({ cfg, model, shared }) {
  const nAssets = cfg.nAssetsCash + cfg.nAssetsRisky;

  let wPrev = new Array(nAssets).fill(0);
  wPrev[0] = 1;

  let prevMu = null;
  let prevTarget = null;

  const grossEquity = [1];
  const netEquity = [1];
  const grossReturns = [];
  const netReturns = [];
  const costs = [];
  const turnovers = [];
  const riskyWeights = [];
  const tradeFlags = [];
  const qualifiedFlags = [];
  const fundamentalFlags = [];
  const signalMagnitudeW = [];
  const signalMagnitudeMu = [];
  const signalRank = [];
  const targetRiskyWeights = [];
  const proposedTurnovers = [];

  let tradeCount = 0;
  let qualifyCount = 0;
  let fundamentalCount = 0;
  let tradeOnFundamentalCount = 0;

  for (let t = 0; t < shared.streamRegimes.length; t += 1) {
    const mu = model.forwardSingle(shared.Xs[t]).y;
    const target = buildTargetWeights(mu, cfg);

    const sW = prevTarget ? l1Diff(target, prevTarget) : l1Diff(target, wPrev);
    const sMu = prevMu ? l2Diff(mu, prevMu) : 0;
    const sRank = prevMu ? Math.max(0, 1 - spearmanCorr(mu, prevMu)) : 0;
    const fundamentalEvent = shared.fundamentalEvents ? shared.fundamentalEvents[t] === 1 : false;

    const rebalanceAllowed = shouldRebalance({
      t,
      cfg,
      signalW: sW,
      fundamentalEvent,
    });

    const qualified = passesQualification({
      cfg,
      signalW: sW,
      signalMu: sMu,
      signalRank: sRank,
      fundamentalEvent,
    });

    const doTrade = rebalanceAllowed && qualified;
    const effectiveTarget = doTrade ? target : wPrev.slice();
    const w = blendWeights(wPrev, effectiveTarget, cfg);

    const proposedTurn = l1Diff(target, wPrev);
    const turnover = l1Diff(w, wPrev);
    const gross = dot(w, shared.Ys[t]);
    const cost = transactionCost(turnover, cfg);
    const net = gross - cost;

    grossReturns.push(gross);
    netReturns.push(net);
    costs.push(cost);
    turnovers.push(turnover);
    riskyWeights.push(riskyWeightSum(w));
    targetRiskyWeights.push(riskyWeightSum(target));
    proposedTurnovers.push(proposedTurn);

    tradeFlags.push(doTrade ? 1 : 0);
    qualifiedFlags.push(qualified ? 1 : 0);
    fundamentalFlags.push(fundamentalEvent ? 1 : 0);
    signalMagnitudeW.push(sW);
    signalMagnitudeMu.push(sMu);
    signalRank.push(sRank);

    grossEquity.push(grossEquity[grossEquity.length - 1] * (1 + gross));
    netEquity.push(netEquity[netEquity.length - 1] * (1 + net));

    if (doTrade) {
      tradeCount += 1;
    }
    if (qualified) {
      qualifyCount += 1;
    }
    if (fundamentalEvent) {
      fundamentalCount += 1;
    }
    if (doTrade && fundamentalEvent) {
      tradeOnFundamentalCount += 1;
    }

    wPrev = w;
    prevMu = mu;
    prevTarget = target;
  }

  const totalCost = costs.reduce((sum, v) => sum + v, 0);
  const grossMagnitude = grossReturns.reduce((sum, v) => sum + Math.abs(v), 0);

  return {
    equity: netEquity,
    returns: netReturns,
    turnovers,
    riskyWeights,
    grossEquity,
    netEquity,
    grossReturns,
    netReturns,
    costs,
    tradeFlags,
    qualifiedFlags,
    fundamentalFlags,
    signalMagnitudeW,
    signalMagnitudeMu,
    signalRank,
    targetRiskyWeights,
    proposedTurnovers,
    tradeRate: tradeCount / Math.max(1, shared.streamRegimes.length),
    qualifiedRate: qualifyCount / Math.max(1, shared.streamRegimes.length),
    precisionProxy: tradeOnFundamentalCount / Math.max(1, tradeCount),
    recallProxy: tradeOnFundamentalCount / Math.max(1, fundamentalCount),
    totalCost,
    costDrag: Math.min(2, totalCost / Math.max(1e-6, grossMagnitude)),
  };
}

export function runEquityOnSharedStream({ cfg, model, shared }) {
  return evaluateOnSharedStream({ cfg, model, shared }).equity;
}
