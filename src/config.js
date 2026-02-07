export const METHOD_SPECS = [
  { id: "naive", label: "LoRA (naive)", color: "#cf4f37" },
  { id: "anchor", label: "LoRA + anchor", color: "#117864" },
  { id: "anchor_proj", label: "LoRA + anchor + projection", color: "#2856a4" },
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

  simT: 280,
  pStress: 0.35,
  wMaxRisky: 0.35,
  turnoverEta: 0.2,

  noiseStd: 0.1,
  returnScale: 0.02,
};

export const PRESETS = {
  quick_check: {
    label: "Quick Check",
    values: {
      seed: 24,
      steps: 280,
      nTrainDrift: 900,
      nAnchorStress: 192,
      nTestDrift: 700,
      nTestStress: 420,
      simT: 180,
    },
  },
  proposal_like: {
    label: "Default (High Signal)",
    values: {
      seed: 24,
      steps: 1200,
      nTrainDrift: 2000,
      nAnchorStress: 512,
      nTestDrift: 2000,
      nTestStress: 1024,
      simT: 300,
      batchSize: 128,
      anchorBatchSize: 128,
    },
  },
  stress_heavy: {
    label: "Stress Heavy",
    values: {
      seed: 24,
      pStress: 0.55,
      anchorBeta: 0.08,
      steps: 900,
      simT: 320,
      noiseStd: 0.12,
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

  cfg.simT = Math.max(40, Math.floor(cfg.simT));
  cfg.pStress = Math.min(0.9, Math.max(0.05, Number(cfg.pStress)));
  cfg.wMaxRisky = Math.min(1.0, Math.max(0.05, Number(cfg.wMaxRisky)));
  cfg.turnoverEta = Math.min(1.0, Math.max(0.01, Number(cfg.turnoverEta)));

  cfg.noiseStd = Math.min(0.8, Math.max(0.0, Number(cfg.noiseStd)));
  cfg.returnScale = Math.min(0.2, Math.max(0.001, Number(cfg.returnScale)));

  return cfg;
}
