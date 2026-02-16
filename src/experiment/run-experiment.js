import { METHOD_SPECS } from "../config.js";
import { generateRegimeBatch, generateReturns, linearPredict, makeTrueWeights } from "./data.js";
import { createRng, zeros } from "./math.js";
import { evalMse } from "./model.js";
import {
  buildSharedStream,
  evaluateOnSharedStream,
  simulateStream,
} from "./simulate.js";
import { trainLora } from "./train.js";

function scaleWeights(W, scale) {
  const out = zeros(W.length, W[0].length);
  for (let i = 0; i < W.length; i += 1) {
    for (let j = 0; j < W[i].length; j += 1) {
      out[i][j] = W[i][j] * scale;
    }
  }
  return out;
}

export function runExperiment(cfg, onProgress) {
  const rootRng = createRng(cfg.seed);
  const { WBase, WDrift } = makeTrueWeights(cfg, rootRng);
  const W0 = scaleWeights(WBase, cfg.returnScale);

  const dataRng = createRng(cfg.seed + 17);

  const XStressAnchor = generateRegimeBatch({
    n: cfg.nAnchorStress,
    dSignal: cfg.dSignal,
    regime: "stress",
    rng: dataRng,
  });

  const YTeacher = linearPredict(XStressAnchor, W0);

  const XDrift = generateRegimeBatch({
    n: cfg.nTrainDrift,
    dSignal: cfg.dSignal,
    regime: "drift",
    rng: dataRng,
  });
  const YDrift = generateReturns(XDrift, WDrift, cfg, dataRng);

  const XDriftTest = generateRegimeBatch({
    n: cfg.nTestDrift,
    dSignal: cfg.dSignal,
    regime: "drift",
    rng: dataRng,
  });
  const YDriftTest = generateReturns(XDriftTest, WDrift, cfg, dataRng);

  const XStressTest = generateRegimeBatch({
    n: cfg.nTestStress,
    dSignal: cfg.dSignal,
    regime: "stress",
    rng: dataRng,
  });
  const YStressTeacherTest = linearPredict(XStressTest, W0);

  const metrics = {};
  const trainLogs = {};
  const models = {};

  for (let m = 0; m < METHOD_SPECS.length; m += 1) {
    const methodSpec = METHOD_SPECS[m];
    const method = methodSpec.id;

    const trainRng = createRng(cfg.seed + 1000 + m * 31);
    const { model, logs } = trainLora({
      cfg,
      W0,
      XDrift,
      YDrift,
      XAnchor: XStressAnchor,
      YAnchorTeacher: YTeacher,
      method,
      rng: trainRng,
      onProgress: (p) => {
        if (onProgress) {
          onProgress({
            kind: "training",
            method,
            methodLabel: methodSpec.label,
            methodIndex: m,
            totalMethods: METHOD_SPECS.length,
            step: p.step,
            totalSteps: p.totalSteps,
          });
        }
      },
    });

    models[method] = model;
    trainLogs[method] = logs;

    const driftMse = evalMse(model, XDriftTest, YDriftTest);
    const stressMse = evalMse(model, XStressTest, YStressTeacherTest);

    const simRng = createRng(cfg.seed + 5000 + m * 97);
    const sim = simulateStream({ cfg, model, WBase, WDrift, rng: simRng });

    metrics[method] = {
      driftMse,
      stressMse,
      totalReturn: sim.totalReturn,
      grossTotalReturn: sim.grossTotalReturn ?? sim.totalReturn,
      netTotalReturn: sim.netTotalReturn ?? sim.totalReturn,
      maxDrawdown: sim.maxDrawdown,
      worstStressDay: sim.worstStressDay,
      avgRiskyWeightStress: sim.avgRiskyWeightStress,
      avgRiskyWeightDrift: sim.avgRiskyWeightDrift,
      costDrag: sim.costDrag ?? 0,
      tradeRate: sim.tradeRate ?? 0,
      qualifiedRate: sim.qualifiedRate ?? 0,
      precisionProxy: sim.precisionProxy ?? 0,
      recallProxy: sim.recallProxy ?? 0,
    };
  }

  if (onProgress) {
    onProgress({ kind: "building_charts" });
  }

  const sharedRng = createRng(cfg.seed + 999);
  const shared = buildSharedStream({ cfg, WBase, WDrift, rng: sharedRng });

  const equityCurves = {};
  const sharedDiagnostics = {};
  for (let m = 0; m < METHOD_SPECS.length; m += 1) {
    const method = METHOD_SPECS[m].id;
    const diag = evaluateOnSharedStream({ cfg, model: models[method], shared });
    equityCurves[method] = diag.equity;
    sharedDiagnostics[method] = diag;
  }

  const keyResult = summarizeTakeaway(metrics);

  return {
    config: cfg,
    methods: METHOD_SPECS,
    metrics,
    trainLogs,
    equityCurves,
    sharedDiagnostics,
    streamRegimes: shared.streamRegimes,
    stressMarkers: shared.streamRegimes.map((regime, idx) => (regime === "stress" ? idx : -1)).filter((idx) => idx >= 0),
    keyResult,
  };
}

function summarizeTakeaway(metrics) {
  const entries = Object.entries(metrics).map(([method, m]) => ({ method, ...m }));

  const bestStress = entries.reduce((a, b) => (a.stressMse < b.stressMse ? a : b));
  const bestDrawdown = entries.reduce((a, b) => (a.maxDrawdown > b.maxDrawdown ? a : b));
  const bestReturn = entries.reduce((a, b) => (a.totalReturn > b.totalReturn ? a : b));

  return {
    bestStress: bestStress.method,
    bestDrawdown: bestDrawdown.method,
    bestReturn: bestReturn.method,
    stressGapVsNaive:
      metrics.naive && metrics.anchor_proj
        ? metrics.naive.stressMse - metrics.anchor_proj.stressMse
        : 0,
  };
}
