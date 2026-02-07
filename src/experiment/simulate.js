import { generateRegimeBatch, generateReturns } from "./data.js";

function riskyWeightSum(w) {
  let s = 0;
  for (let i = 1; i < w.length; i += 1) {
    s += w[i];
  }
  return s;
}

export function portfolioPolicy(mu, wPrev, cfg) {
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
  } else {
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
  }

  const w = new Array(nAssets).fill(0);
  let total = 0;
  for (let i = 0; i < nAssets; i += 1) {
    const blended = (1 - cfg.turnoverEta) * wPrev[i] + cfg.turnoverEta * wTgt[i];
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

export function simulateStream({ cfg, model, WBase, WDrift, rng }) {
  const nAssets = cfg.nAssetsCash + cfg.nAssetsRisky;
  let w = new Array(nAssets).fill(0);
  w[0] = 1;

  const equity = [1];
  let worstStressDay = 0;
  let stressDays = 0;

  let stressRiskySum = 0;
  let driftRiskySum = 0;
  let driftDays = 0;

  for (let t = 0; t < cfg.simT; t += 1) {
    const isStress = rng.uniform() < cfg.pStress;
    const regime = isStress ? "stress" : "drift";

    const X = generateRegimeBatch({ n: 1, dSignal: cfg.dSignal, regime, rng });
    const WTrue = isStress ? WBase : WDrift;
    const yTrue = generateReturns(X, WTrue, cfg, rng)[0];

    const mu = model.forwardSingle(X[0]).y;
    w = portfolioPolicy(mu, w, cfg);

    const rp = dot(w, yTrue);
    equity.push(equity[equity.length - 1] * (1 + rp));

    const rw = riskyWeightSum(w);
    if (isStress) {
      stressDays += 1;
      stressRiskySum += rw;
      if (rp < worstStressDay) {
        worstStressDay = rp;
      }
    } else {
      driftDays += 1;
      driftRiskySum += rw;
    }
  }

  return {
    totalReturn: equity[equity.length - 1] - 1,
    maxDrawdown: maxDrawdown(equity),
    worstStressDay,
    nStressDays: stressDays,
    avgRiskyWeightStress: stressDays > 0 ? stressRiskySum / stressDays : 0,
    avgRiskyWeightDrift: driftDays > 0 ? driftRiskySum / driftDays : 0,
    equity,
  };
}

export function buildSharedStream({ cfg, WBase, WDrift, rng }) {
  const streamRegimes = new Array(cfg.simT);
  const Xs = new Array(cfg.simT);
  const Ys = new Array(cfg.simT);

  for (let t = 0; t < cfg.simT; t += 1) {
    const regime = rng.uniform() < cfg.pStress ? "stress" : "drift";
    streamRegimes[t] = regime;
    const X = generateRegimeBatch({ n: 1, dSignal: cfg.dSignal, regime, rng })[0];
    const Y = generateReturns([X], regime === "stress" ? WBase : WDrift, cfg, rng)[0];
    Xs[t] = X;
    Ys[t] = Y;
  }

  return { streamRegimes, Xs, Ys };
}

function sumAbsDiff(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i += 1) {
    s += Math.abs(a[i] - b[i]);
  }
  return s;
}

export function evaluateOnSharedStream({ cfg, model, shared }) {
  const nAssets = cfg.nAssetsCash + cfg.nAssetsRisky;
  let wPrev = new Array(nAssets).fill(0);
  wPrev[0] = 1;

  const eq = [1];
  const returns = [];
  const turnovers = [];
  const riskyWeights = [];

  for (let t = 0; t < shared.streamRegimes.length; t += 1) {
    const mu = model.forwardSingle(shared.Xs[t]).y;
    const w = portfolioPolicy(mu, wPrev, cfg);
    const ret = dot(w, shared.Ys[t]);

    returns.push(ret);
    turnovers.push(sumAbsDiff(w, wPrev));
    riskyWeights.push(riskyWeightSum(w));
    eq.push(eq[eq.length - 1] * (1 + ret));

    wPrev = w;
  }

  return {
    equity: eq,
    returns,
    turnovers,
    riskyWeights,
  };
}

export function runEquityOnSharedStream({ cfg, model, shared }) {
  return evaluateOnSharedStream({ cfg, model, shared }).equity;
}
