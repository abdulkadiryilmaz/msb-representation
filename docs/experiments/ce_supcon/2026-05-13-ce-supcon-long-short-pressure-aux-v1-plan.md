# Experiment Plan: 2026-05-13 — CE + SupCon Long Short-Pressure Aux V1

**Status**: Completed  
**Checkpoint**: `data/stage1a/binance/15m/checkpoints/ce_supcon_long_short_pressure_aux_v1`  
**Baseline to beat**: `ce_supcon_long_v1`  

---

## Hypothesis

Routing pressure auxiliary supervision through `z_short` instead of `z_fused` can improve pressure readability without sending pressure gradients into `z_long`, preserving the cross-symbol structural geometry learned by branch-aware SupCon.

---

## Method

Objective:

```text
CE(confirmed_state)
+ 0.05 * SupCon(z_long_proj, confirmed_state)
+ 0.01 * CE(pressure_state from z_short)
```

Maturity auxiliary is disabled.

Difference from `ce_supcon_long_aux_v2`:

- same pressure weight
- same SupCon setup
- pressure head input changes from `z_fused` to `z_short`

---

## Training Command

```bash
PYTHONPATH=src python scripts/train_stage1a.py \
  --dataset-root data/stage1a/binance/15m \
  --epochs 50 \
  --batch-size 32 \
  --checkpoint-dir data/stage1a/binance/15m/checkpoints/ce_supcon_long_short_pressure_aux_v1 \
  --use-supcon \
  --supcon-embedding-key z_long_proj \
  --supcon-positive-mode label \
  --supcon-weight 0.05 \
  --supcon-temperature 0.1 \
  --pressure-loss-weight 0.01 \
  --pressure-head-input z_short \
  --maturity-loss-weight 0.0
```

---

## Success Criteria

Required:

- `z_long` symbol agreement remains close to `ce_supcon_long_v1`
- `z_long_proj` symbol agreement does not move toward `aux_v2`
- pressure probe improves versus `ce_supcon_long_v1`
- borderline misclassification does not materially worsen
- val macro F1 remains close to `ce_supcon_long_v1`

Decision:

- accept as stronger ablation if pressure gains remain while `z_long` geometry is preserved
- keep `ce_supcon_long_v1` as current best unless both `z_long` and `z_fused` geometry are comparable or better
