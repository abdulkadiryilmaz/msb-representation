# Experiment Plan: 2026-05-13 — CE + SupCon Long Aux V2

**Status**: Completed  
**Checkpoint**: `data/stage1a/binance/15m/checkpoints/ce_supcon_long_aux_v2`  
**Baseline to beat**: `ce_supcon_long_v1`  
**Previous ablation**: `ce_supcon_long_aux_v1`

---

## Hypothesis

`ce_supcon_long_aux_v1` pressure ve borderline davranışını iyileştirdi, fakat maturity auxiliary head `z_long` geometry üzerinde maliyet yarattı. Pressure-only auxiliary objective, aynı borderline/pressure kazanımını daha az `z_long` label agreement kaybıyla sağlayabilir.

---

## Method

Baseline:

```text
CE(confirmed_state)
+ 0.05 * SupCon(z_long_proj, confirmed_state)
```

`aux_v1`:

```text
CE(confirmed_state)
+ 0.05 * SupCon(z_long_proj, confirmed_state)
+ 0.01 * CE(pressure_state)
+ 0.01 * CE(maturity_target)
```

`aux_v2`:

```text
CE(confirmed_state)
+ 0.05 * SupCon(z_long_proj, confirmed_state)
+ 0.01 * CE(pressure_state)
```

Tek değişiklik: maturity auxiliary head kapatılıyor.

---

## Training Command

```bash
PYTHONPATH=src python scripts/train_stage1a.py \
  --dataset-root data/stage1a/binance/15m \
  --epochs 50 \
  --batch-size 32 \
  --checkpoint-dir data/stage1a/binance/15m/checkpoints/ce_supcon_long_aux_v2 \
  --use-supcon \
  --supcon-embedding-key z_long_proj \
  --supcon-positive-mode label \
  --supcon-weight 0.05 \
  --supcon-temperature 0.1 \
  --pressure-loss-weight 0.01 \
  --maturity-loss-weight 0.0
```

---

## Success Criteria

Required:

- preserve `ce_supcon_long_v1` low symbol agreement
- improve pressure probe versus `ce_supcon_long_v1`
- improve borderline misclassification versus `ce_supcon_long_v1`
- recover some of the `z_long` / `z_long_proj` label agreement lost in `aux_v1`

Decision rule:

- accept as stronger ablation than `aux_v1` if pressure/borderline gains remain while `z_long` geometry moves back toward `ce_supcon_long_v1`
- keep `ce_supcon_long_v1` as current best unless `z_long` and `z_fused` geometry are both at least comparable to it

Guardrail:

- reject if symbol agreement moves toward `long_factor_v1`
