import { cloneMatrix, zeros } from "./math.js";

export class LoRALinear {
  constructor(W0, rank, rng, alpha = 1.0) {
    this.W0 = cloneMatrix(W0);
    this.nAssets = W0.length;
    this.dIn = W0[0].length;
    this.rank = rank;
    this.alpha = alpha;
    this.scale = this.alpha / Math.max(1, this.rank);

    this.A = zeros(this.rank, this.dIn);
    this.B = zeros(this.nAssets, this.rank);

    const bound = 1 / Math.sqrt(this.dIn);
    for (let r = 0; r < this.rank; r += 1) {
      for (let j = 0; j < this.dIn; j += 1) {
        this.A[r][j] = (rng.uniform() * 2 - 1) * bound;
      }
    }
  }

  forwardSingle(x) {
    const h = new Array(this.rank).fill(0);
    for (let r = 0; r < this.rank; r += 1) {
      let s = 0;
      for (let j = 0; j < this.dIn; j += 1) {
        s += this.A[r][j] * x[j];
      }
      h[r] = s;
    }

    const y = new Array(this.nAssets).fill(0);
    for (let a = 0; a < this.nAssets; a += 1) {
      let base = 0;
      for (let j = 0; j < this.dIn; j += 1) {
        base += this.W0[a][j] * x[j];
      }

      let delta = 0;
      for (let r = 0; r < this.rank; r += 1) {
        delta += this.B[a][r] * h[r];
      }
      y[a] = base + this.scale * delta;
    }

    return { y, h };
  }

  forwardBatch(X) {
    const out = new Array(X.length);
    for (let i = 0; i < X.length; i += 1) {
      out[i] = this.forwardSingle(X[i]).y;
    }
    return out;
  }
}

export function evalMse(model, X, Y) {
  const n = X.length;
  const nAssets = Y[0].length;
  let s = 0;

  for (let i = 0; i < n; i += 1) {
    const pred = model.forwardSingle(X[i]).y;
    for (let a = 0; a < nAssets; a += 1) {
      const e = pred[a] - Y[i][a];
      s += e * e;
    }
  }

  return s / (n * nAssets);
}
