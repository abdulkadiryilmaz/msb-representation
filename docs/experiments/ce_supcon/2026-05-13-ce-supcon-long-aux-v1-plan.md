# Worklog: 2026-05-13 — CE + SupCon Long Aux V1 Plan

**Status**: Completed  
**Checkpoint**: `data/stage1a/binance/15m/checkpoints/ce_supcon_long_aux_v1`  
**Baseline to beat**: `ce_supcon_long_v1`

---

## Hypothesis

Keeping the successful `ce_supcon_long_v1` SupCon pairing rule while adding low-weight pressure/maturity auxiliary heads can improve pressure/maturity readability and borderline behavior without reintroducing symbol shortcut.

---

## Method

Baseline:

```text
CE(confirmed_state)
+ 0.05 * SupCon(z_long_proj, confirmed_state)
```

New objective:

```text
CE(confirmed_state)
+ 0.05 * SupCon(z_long_proj, confirmed_state)
+ 0.01 * CE(pressure_state)
+ 0.01 * CE(maturity_target)
```

This intentionally does not use factor-aware SupCon pairing.

Difference from `ce_supcon_long_factor_v1`:

- SupCon positive rule returns to coarse `confirmed_state`
- auxiliary heads remain
- batch size returns to the historical default `32`

---

## Training Command

```bash
PYTHONPATH=src python scripts/train_stage1a.py \
  --dataset-root data/stage1a/binance/15m \
  --epochs 50 \
  --batch-size 32 \
  --checkpoint-dir data/stage1a/binance/15m/checkpoints/ce_supcon_long_aux_v1 \
  --use-supcon \
  --supcon-embedding-key z_long_proj \
  --supcon-positive-mode label \
  --supcon-weight 0.05 \
  --supcon-temperature 0.1 \
  --pressure-loss-weight 0.01 \
  --maturity-loss-weight 0.01
```

---

## Success Criteria

Required:

- preserve `ce_supcon_long_v1` low symbol agreement
- pressure probe improves or stays close to `ce_supcon_long_v1`
- maturity probe improves or stays close to `ce_supcon_long_v1`
- borderline misclassification and NN confirmed drift improve without erasing pressure
- val macro F1 should not materially degrade

Guardrail:

- reject if `z_long` / `z_fused` symbol agreement moves toward `long_factor_v1`
