import { zeros } from "./math.js";

function stackFeatures(xSignal, stress, drift) {
  return [...xSignal, stress, drift];
}

export function generateRegimeBatch({ n, dSignal, regime, rng }) {
  const X = new Array(n);

  for (let i = 0; i < n; i += 1) {
    const x = new Array(dSignal);
    for (let j = 0; j < dSignal; j += 1) {
      x[j] = rng.normal();
    }

    let stress = 0;
    let drift = 0;

    if (regime === "base") {
      for (let j = 0; j < dSignal; j += 1) {
        x[j] += 0.15 * rng.normal();
      }
    } else if (regime === "stress") {
      stress = 1;
      for (let j = 0; j < dSignal; j += 1) {
        x[j] = 1.8 * rng.normal();
      }
    } else if (regime === "drift") {
      drift = 1;
      for (let j = 0; j < dSignal; j += 1) {
        x[j] += 0.4;
      }
    } else {
      throw new Error(`Unknown regime: ${regime}`);
    }

    X[i] = stackFeatures(x, stress, drift);
  }

  return X;
}

export function makeTrueWeights(cfg, rng) {
  const dIn = cfg.dSignal + 2;
  const nAssets = cfg.nAssetsCash + cfg.nAssetsRisky;

  const WBase = zeros(nAssets, dIn);
  for (let i = 1; i < nAssets; i += 1) {
    for (let j = 0; j < cfg.dSignal; j += 1) {
      WBase[i][j] = 0.35 * rng.normal();
    }
  }

  const stressCol = cfg.dSignal;
  const driftCol = cfg.dSignal + 1;

  for (let i = 1; i < nAssets; i += 1) {
    WBase[i][stressCol] = -5.0;
    WBase[i][driftCol] = 0.0;
  }

  const rTrue = Math.max(1, Math.min(2, cfg.loraRank));
  const U = zeros(nAssets, rTrue);
  const V = zeros(dIn, rTrue);

  for (let i = 0; i < nAssets; i += 1) {
    for (let k = 0; k < rTrue; k += 1) {
      U[i][k] = rng.normal();
    }
  }
  for (let j = 0; j < dIn; j += 1) {
    for (let k = 0; k < rTrue; k += 1) {
      V[j][k] = rng.normal();
    }
  }

  const WDrift = zeros(nAssets, dIn);

  for (let i = 0; i < nAssets; i += 1) {
    for (let j = 0; j < dIn; j += 1) {
      let delta = 0;
      for (let k = 0; k < rTrue; k += 1) {
        delta += U[i][k] * V[j][k];
      }
      delta *= 0.25;

      if (i >= 1 && j === driftCol) {
        delta += 2.0;
      }

      WDrift[i][j] = WBase[i][j] + delta;
    }
  }

  return { WBase, WDrift };
}

export function generateReturns(X, WTrue, cfg, rng) {
  const n = X.length;
  const nAssets = WTrue.length;
  const dIn = WTrue[0].length;
  const Y = zeros(n, nAssets);

  for (let i = 0; i < n; i += 1) {
    const x = X[i];
    for (let a = 0; a < nAssets; a += 1) {
      let pred = 0;
      for (let j = 0; j < dIn; j += 1) {
        pred += x[j] * WTrue[a][j];
      }
      const noise = cfg.noiseStd > 0 ? cfg.noiseStd * rng.normal() : 0;
      Y[i][a] = (pred + noise) * cfg.returnScale;
    }
    Y[i][0] = 0.0;
  }

  return Y;
}

export function linearPredict(X, W) {
  const n = X.length;
  const nAssets = W.length;
  const dIn = W[0].length;
  const Y = zeros(n, nAssets);

  for (let i = 0; i < n; i += 1) {
    for (let a = 0; a < nAssets; a += 1) {
      let s = 0;
      for (let j = 0; j < dIn; j += 1) {
        s += X[i][j] * W[a][j];
      }
      Y[i][a] = s;
    }
    Y[i][0] = 0;
  }

  return Y;
}
