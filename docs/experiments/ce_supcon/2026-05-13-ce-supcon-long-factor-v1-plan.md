# Worklog: 2026-05-13 — CE + SupCon Long Factor V1 Plan

**Status**: In progress  
**Checkpoint**: `data/stage1a/binance/15m/checkpoints/ce_supcon_long_factor_v1`  
**Baseline to beat**: `ce_supcon_long_v1`

---

## Hypothesis

Factor-aware SupCon on `z_long_proj`, combined with low-weight pressure/maturity auxiliary heads, can reduce borderline intact -> confirmed break collapse while preserving the useful directionality and low symbol shortcut of `ce_supcon_long_v1`.

---

## Method

Baseline objective:

```text
CE(confirmed_state)
```

New objective:

```text
CE(confirmed_state)
+ 0.05 * SupCon(z_long_proj, factor_target)
+ 0.01 * CE(pressure_state)
+ 0.01 * CE(maturity_target)
```

Important changes from `ce_supcon_long_v1`:

- SupCon labels change from coarse `confirmed_state` to `factor_target`
- auxiliary heads are added on `z_fused`
- batch size is increased to `256` to reduce factor-class fragmentation

Design worklog:

- `docs/worklogs/2026-05-13-factor-aware-supcon-design.md`

---

## Training Command

```bash
PYTHONPATH=src python scripts/train_stage1a.py \
  --dataset-root data/stage1a/binance/15m \
  --epochs 50 \
  --batch-size 256 \
  --checkpoint-dir data/stage1a/binance/15m/checkpoints/ce_supcon_long_factor_v1 \
  --use-supcon \
  --supcon-embedding-key z_long_proj \
  --supcon-positive-mode factor \
  --supcon-weight 0.05 \
  --supcon-temperature 0.1 \
  --pressure-loss-weight 0.01 \
  --maturity-loss-weight 0.01
```

---

## Success Criteria

Required:

- val macro F1 should not materially degrade from `ce_supcon_long_v1`
- `z_long` / `z_fused` symbol agreement should remain low
- pressure probe should remain close to `ce_supcon_long_v1`
- maturity binary probe should not degrade
- borderline intact misclassification rate should fall
- misclassified borderline intact NN confirmed rate should fall

Guardrails:

- do not accept if pressure readability is erased
- do not accept if symbol shortcut returns
- do not accept if `z_long` label agreement collapses
- read SupCon anchor/positive coverage before interpreting metrics

---

## Planned Readout

Post-run artefacts:

- val/test latent exports
- latent analysis for `z_short`, `z_long`, `z_fused`
- pressure probe
- maturity binary probe
- fusion/head borderline diagnosis
- compare against `ce_only` and `ce_supcon_long_v1`
