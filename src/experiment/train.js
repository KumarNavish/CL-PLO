import {
  flattenGradients,
  l2Norm,
  projectAgem,
  unflattenGradients,
  zeros,
} from "./math.js";
import { LoRALinear } from "./model.js";

class Adam {
  constructor(model, lr) {
    this.model = model;
    this.lr = lr;

    this.t = 0;
    this.beta1 = 0.9;
    this.beta2 = 0.999;
    this.eps = 1e-8;

    this.mA = zeros(model.rank, model.dIn);
    this.vA = zeros(model.rank, model.dIn);

    this.mB = zeros(model.nAssets, model.rank);
    this.vB = zeros(model.nAssets, model.rank);
  }

  step(gradA, gradB) {
    this.t += 1;

    const b1 = this.beta1;
    const b2 = this.beta2;
    const corr1 = 1 - b1 ** this.t;
    const corr2 = 1 - b2 ** this.t;

    for (let r = 0; r < this.model.rank; r += 1) {
      for (let j = 0; j < this.model.dIn; j += 1) {
        const g = gradA[r][j];
        this.mA[r][j] = b1 * this.mA[r][j] + (1 - b1) * g;
        this.vA[r][j] = b2 * this.vA[r][j] + (1 - b2) * g * g;

        const mHat = this.mA[r][j] / corr1;
        const vHat = this.vA[r][j] / corr2;
        this.model.A[r][j] -= this.lr * (mHat / (Math.sqrt(vHat) + this.eps));
      }
    }

    for (let a = 0; a < this.model.nAssets; a += 1) {
      for (let r = 0; r < this.model.rank; r += 1) {
        const g = gradB[a][r];
        this.mB[a][r] = b1 * this.mB[a][r] + (1 - b1) * g;
        this.vB[a][r] = b2 * this.vB[a][r] + (1 - b2) * g * g;

        const mHat = this.mB[a][r] / corr1;
        const vHat = this.vB[a][r] / corr2;
        this.model.B[a][r] -= this.lr * (mHat / (Math.sqrt(vHat) + this.eps));
      }
    }
  }
}

function randomIndices(total, batchSize, rng) {
  const idx = new Array(batchSize);
  for (let i = 0; i < batchSize; i += 1) {
    idx[i] = rng.int(total);
  }
  return idx;
}

function computeBatchGradients(model, X, Y, indices) {
  const nAssets = model.nAssets;
  const rank = model.rank;
  const dIn = model.dIn;
  const scale = model.scale;
  const denom = indices.length * nAssets;

  const gradA = zeros(rank, dIn);
  const gradB = zeros(nAssets, rank);

  let loss = 0;

  for (let p = 0; p < indices.length; p += 1) {
    const idx = indices[p];
    const x = X[idx];
    const yTarget = Y[idx];

    const { y: pred, h } = model.forwardSingle(x);

    const weightedB = new Array(rank).fill(0);

    for (let a = 0; a < nAssets; a += 1) {
      const err = pred[a] - yTarget[a];
      loss += err * err;

      const gradPred = (2 * err) / denom;

      for (let r = 0; r < rank; r += 1) {
        gradB[a][r] += scale * gradPred * h[r];
        weightedB[r] += gradPred * model.B[a][r];
      }
    }

    for (let r = 0; r < rank; r += 1) {
      const w = scale * weightedB[r];
      for (let j = 0; j < dIn; j += 1) {
        gradA[r][j] += w * x[j];
      }
    }
  }

  return {
    loss: loss / denom,
    gradA,
    gradB,
  };
}

function combineGradients(a, b, beta) {
  const gradA = zeros(a.gradA.length, a.gradA[0].length);
  const gradB = zeros(a.gradB.length, a.gradB[0].length);

  for (let i = 0; i < gradA.length; i += 1) {
    for (let j = 0; j < gradA[i].length; j += 1) {
      gradA[i][j] = a.gradA[i][j] + beta * b.gradA[i][j];
    }
  }
  for (let i = 0; i < gradB.length; i += 1) {
    for (let j = 0; j < gradB[i].length; j += 1) {
      gradB[i][j] = a.gradB[i][j] + beta * b.gradB[i][j];
    }
  }

  return { gradA, gradB };
}

export function trainLora({
  cfg,
  W0,
  XDrift,
  YDrift,
  XAnchor,
  YAnchorTeacher,
  method,
  rng,
  onProgress,
}) {
  const model = new LoRALinear(W0, cfg.loraRank, rng, 1.0);
  const optimizer = new Adam(model, cfg.lr);

  let interferenceCount = 0;
  let distortionSum = 0;

  for (let step = 0; step < cfg.steps; step += 1) {
    const driftIdx = randomIndices(XDrift.length, cfg.batchSize, rng);
    const anchorIdx = randomIndices(XAnchor.length, cfg.anchorBatchSize, rng);

    const gNew = computeBatchGradients(model, XDrift, YDrift, driftIdx);

    let finalGradA = gNew.gradA;
    let finalGradB = gNew.gradB;

    if (method === "anchor" || method === "anchor_proj") {
      const gAnc = computeBatchGradients(model, XAnchor, YAnchorTeacher, anchorIdx);

      if (method === "anchor") {
        const combined = combineGradients(gNew, gAnc, cfg.anchorBeta);
        finalGradA = combined.gradA;
        finalGradB = combined.gradB;
      } else {
        const flatNew = flattenGradients(gNew.gradA, gNew.gradB);
        const flatAnc = flattenGradients(gAnc.gradA, gAnc.gradB);

        const proj = projectAgem(flatNew, flatAnc, 0);
        if (proj.dot < 0) {
          interferenceCount += 1;
        }

        const delta = new Array(flatNew.length);
        for (let i = 0; i < flatNew.length; i += 1) {
          delta[i] = proj.projected[i] - flatNew[i];
        }

        distortionSum += l2Norm(delta) / (l2Norm(flatNew) + 1e-12);

        const combo = new Array(flatNew.length);
        for (let i = 0; i < flatNew.length; i += 1) {
          combo[i] = proj.projected[i] + cfg.anchorBeta * flatAnc[i];
        }

        const unflat = unflattenGradients(combo, model.rank, model.dIn, model.nAssets);
        finalGradA = unflat.gradA;
        finalGradB = unflat.gradB;
      }
    }

    optimizer.step(finalGradA, finalGradB);

    if (onProgress && (step % Math.max(1, Math.floor(cfg.steps / 30)) === 0 || step === cfg.steps - 1)) {
      onProgress({ step: step + 1, totalSteps: cfg.steps, method });
    }
  }

  const logs = {};
  if (method === "anchor_proj") {
    logs.interferenceRate = interferenceCount / Math.max(1, cfg.steps);
    logs.updateDistortion = distortionSum / Math.max(1, cfg.steps);
  }

  return { model, logs };
}
