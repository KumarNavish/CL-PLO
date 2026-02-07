# Deployment Playbook
## Constraint-Aligned Continual PEFT

## Objective
Run continual PEFT updates that adapt to drift without violating stress-anchor behavior.

## Boundaries
- In scope: update loop, promotion gate, rollback logic, monitoring.
- Out of scope: replacing optimizer architecture or risk policy.

## Required inputs
1. Update-window feature tensors (market features; optional text embeddings).
2. Drift trigger definition (schedule or detector).
3. Stress anchor suite (`M_t`) with cached reference outputs from deployed checkpoint.
4. Replay slices for gate evaluation.

## Update cycle
1. Load deployed checkpoint and active adapters.
2. Build drift batch `D_t` and anchor batch `M_t`.
3. Compute gradients:
   - `g_new` (drift fit)
   - `g_anchor` (retention)
4. Project update direction to satisfy anchor-compatibility constraint.
5. Evaluate candidate on gate metrics.
6. Promote only if all gate checks pass.

## Gate policy template
Require all conditions:
- drift-fit change within tolerance,
- anchor regression below threshold,
- tail-risk proxy non-degradation,
- no increase in constraint violations.

Example policy skeleton:
- `stress_mse <= stress_mse_limit`
- `drift_mse <= drift_mse_baseline * (1 + delta)`
- `mdd >= mdd_baseline - tolerance`
- `violations <= violations_baseline`

## Monitoring per update
- Drift MSE, Stress MSE
- Max drawdown proxy, worst stress-day return
- Constraint violation counts
- Output drift vs previous checkpoint
- Promotion decision and rollback reason

## Rollback policy
Immediate rollback on:
- stress-regression breach,
- tail-risk breach,
- constraint-violation spike,
- unexplained output-distribution or attribution shift.

## Audit artifacts
Persist for every evaluated candidate:
- anchor suite hash,
- adapter delta checksum,
- gate metrics snapshot,
- config and data-window metadata,
- promotion decision record.

## Common failure modes
- Drift overfit: raise anchor weight, inspect anchor coverage.
- Anchor blind spots: refresh stress taxonomy with desk input.
- Over-constrained update: relax constraints and rerun replay.
- Pipeline mismatch: keep candidate in shadow mode until stable.

## Internship implementation order
1. Integrate adapter update job with current scheduler.
2. Add anchor suite storage + teacher-output cache.
3. Implement gate evaluator + decision logger.
4. Add monitoring and alert hooks.
5. Run shadow comparison against naive adaptation.

## Handoff success criteria
- Repeated stress-regression improvement vs naive baseline.
- Drift-fit remains inside policy tolerance.
- Tail-risk profile improves or remains stable on stress slices.
- Deployment decisions are reproducible and auditable.
