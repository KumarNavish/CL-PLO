export const METHOD_SPECS = [
  { id: "naive", label: "Naive online SGD", color: "#6e6e6e" },
  { id: "anchor", label: "Replay distill only", color: "#0f5fbf" },
  { id: "anchor_proj", label: "Constrained CL (distill + projection + gate)", color: "#b55400" },
];

export const DEFAULT_CONFIG = {
  seed: 24,

  dSignal: 8,
  nAssetsRisky: 4,
  nAssetsCash: 1,

  nTrainDrift: 1800,
  nAnchorStress: 384,
  nTestDrift: 1200,
  nTestStress: 768,

  loraRank: 4,
  lr: 0.03,
  steps: 700,
  batchSize: 96,
  anchorBatchSize: 96,
  anchorBeta: 0.05,
  stressLossTolerance: 0.02,
  stressLossFloor: 1e-3,

  simT: 280,
  pStress: 0.35,
  wMaxRisky: 0.35,
  turnoverEta: 0.2,

  noiseStd: 0.1,
  returnScale: 0.02,
};

export const PRESETS = {
  quick_check: {
    label: "Quick",
    values: {
      seed: 24,
      steps: 220,
      nTrainDrift: 760,
      nAnchorStress: 160,
      nTestDrift: 520,
      nTestStress: 320,
      simT: 150,
      pStress: 0.24,
      noiseStd: 0.12,
    },
  },
  proposal_like: {
    label: "Default",
    values: {
      seed: 24,
      steps: 1000,
      nTrainDrift: 2200,
      nAnchorStress: 512,
      nTestDrift: 1800,
      nTestStress: 960,
      simT: 320,
      batchSize: 128,
      anchorBatchSize: 128,
      anchorBeta: 0.06,
      pStress: 0.36,
    },
  },
  stress_heavy: {
    label: "Stress+",
    values: {
      seed: 24,
      pStress: 0.7,
      anchorBeta: 0.1,
      steps: 1200,
      simT: 380,
      noiseStd: 0.16,
      nAnchorStress: 640,
      nTestStress: 1400,
    },
  },
};

export function clampConfig(input) {
  const cfg = { ...DEFAULT_CONFIG, ...input };

  cfg.seed = Math.max(0, Math.floor(cfg.seed));

  cfg.dSignal = Math.max(2, Math.floor(cfg.dSignal));
  cfg.nAssetsRisky = Math.max(2, Math.floor(cfg.nAssetsRisky));
  cfg.nAssetsCash = Math.max(1, Math.floor(cfg.nAssetsCash));

  cfg.nTrainDrift = Math.max(200, Math.floor(cfg.nTrainDrift));
  cfg.nAnchorStress = Math.max(64, Math.floor(cfg.nAnchorStress));
  cfg.nTestDrift = Math.max(200, Math.floor(cfg.nTestDrift));
  cfg.nTestStress = Math.max(128, Math.floor(cfg.nTestStress));

  cfg.loraRank = Math.max(1, Math.floor(cfg.loraRank));
  cfg.lr = Math.min(0.2, Math.max(0.0005, Number(cfg.lr)));
  cfg.steps = Math.max(80, Math.floor(cfg.steps));
  cfg.batchSize = Math.max(16, Math.floor(cfg.batchSize));
  cfg.anchorBatchSize = Math.max(16, Math.floor(cfg.anchorBatchSize));
  cfg.anchorBeta = Math.min(0.5, Math.max(0.0, Number(cfg.anchorBeta)));
  cfg.stressLossTolerance = Math.min(0.5, Math.max(0.0, Number(cfg.stressLossTolerance)));
  cfg.stressLossFloor = Math.min(0.1, Math.max(0.0, Number(cfg.stressLossFloor)));

  cfg.simT = Math.max(40, Math.floor(cfg.simT));
  cfg.pStress = Math.min(0.9, Math.max(0.05, Number(cfg.pStress)));
  cfg.wMaxRisky = Math.min(1.0, Math.max(0.05, Number(cfg.wMaxRisky)));
  cfg.turnoverEta = Math.min(1.0, Math.max(0.01, Number(cfg.turnoverEta)));

  cfg.noiseStd = Math.min(0.8, Math.max(0.0, Number(cfg.noiseStd)));
  cfg.returnScale = Math.min(0.2, Math.max(0.001, Number(cfg.returnScale)));

  return cfg;
}
