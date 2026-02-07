import { clampConfig } from "../config.js";
import { runExperiment } from "../experiment/run-experiment.js";

self.onmessage = (event) => {
  const { type, payload } = event.data;

  if (type !== "run") {
    return;
  }

  try {
    const cfg = clampConfig(payload.config || {});

    const result = runExperiment(cfg, (progress) => {
      self.postMessage({ type: "progress", payload: progress });
    });

    self.postMessage({ type: "result", payload: result });
  } catch (error) {
    self.postMessage({
      type: "error",
      payload: {
        message: error instanceof Error ? error.message : String(error),
      },
    });
  }
};
