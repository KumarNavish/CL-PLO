# Minimal end-to-end validation (proof-of-concept)

This repo's validation loop is intentionally small but it is **genuinely
Transformer/LLM-based**:

- the market simulator emits a tokenized "text" stream whose distribution
  shifts with regimes,
- a tiny causal Transformer language model is pretrained on that token stream
  and then frozen (standing in for a production pretrained LLM),
- the only continually updated parameters are a low-rank LoRA adapter in the
  regression head that maps (x_t, z_t) to per-asset residual alphas.

Run (from the project root):

```bash
python example_code/run_validation_experiment.py
```

Outputs (written to `./figures/`):

- `validation_stress_loss.pdf` (stress-suite regression + rollback events)
- `validation_ic.pdf` (predictive adaptation signal)
- `validation_stress_es.pdf` (decision-layer ES feasibility check)
- `summary.json` (configuration + headline metrics)

These artifacts are the ones embedded and referenced in `main.pdf`.
