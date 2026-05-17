# Experiment Plan: 2026-05-17 — CE + SupCon Long SupCon-Dominant Warmup V1

**Status**: Planned  
**Family**: `CE + SupCon`  
**Baseline**: `ce_supcon_long_supcon_warmup_v1`  
**Primary diagnostic seed**: `seed_41`  

---

## Motivation

`SupCon-only warmup v1` proved that early SupCon activation matters:

- it broke the early SupCon plateau
- it fixed `seed_41` `z_long` / `z_long_proj` symbol-heavy behavior

But hard CE shutdown caused a new failure mode:

- `z_short` degraded sharply
- `z_fused` became symbol-heavy
- effective long/short became excessively large

So the next question is whether we can keep the early SupCon benefit while preserving enough CE signal for `z_short` and fused balance.

---

## Hypothesis

A small nonzero CE weight during warmup should keep `z_short` and classifier head anchored, while SupCon remains strong enough to avoid the `seed_41` SupCon plateau.

Expected signs:

- `train_supcon_loss` breaks the 3.43 plateau before epoch 5-10
- `z_long` / `z_long_proj` symbol agreement remains close to SupCon-only warmup
- `z_short` label/symbol quality remains closer to frontload than to SupCon-only warmup
- `z_fused` symbol agreement does not spike as much as hard warmup
- effective long/short stays above frontload but far below hard warmup

---

## Intervention

Use SupCon-dominant, not SupCon-only, warmup:

```text
epoch 1-5:
  loss = 0.10 * CE + 0.10 * SupCon

epoch 6+:
  loss = 1.00 * CE + 0.05 * SupCon
```

This is between:

```text
frontload v1:
  1.00 * CE + 0.10 * SupCon

supcon-only warmup v1:
  0.00 * CE + 0.10 * SupCon
```

---

## Primary Run

Run first on `seed_41`, the diagnostic failure seed.

```bash
PYTHONPATH=src python scripts/train_stage1a.py \
  --dataset-root data/stage1a/binance/15m \
  --epochs 50 \
  --patience 100 \
  --batch-size 32 \
  --use-supcon \
  --ce-warmup-weight 0.10 \
  --ce-warmup-epochs 5 \
  --supcon-weight 0.05 \
  --supcon-frontload-weight 0.10 \
  --supcon-frontload-epochs 5 \
  --supcon-temperature 0.1 \
  --supcon-embedding-key z_long_proj \
  --seed 41 \
  --epoch-checkpoints 5 10 20 30 40 50 \
  --checkpoint-dir data/stage1a/binance/15m/checkpoints/ce_supcon_long_supcon_dominant_warmup_v1_seed_41_e50
```

If `seed_41` is promising, repeat on `seed_42`:

```bash
PYTHONPATH=src python scripts/train_stage1a.py \
  --dataset-root data/stage1a/binance/15m \
  --epochs 50 \
  --patience 100 \
  --batch-size 32 \
  --use-supcon \
  --ce-warmup-weight 0.10 \
  --ce-warmup-epochs 5 \
  --supcon-weight 0.05 \
  --supcon-frontload-weight 0.10 \
  --supcon-frontload-epochs 5 \
  --supcon-temperature 0.1 \
  --supcon-embedding-key z_long_proj \
  --seed 42 \
  --epoch-checkpoints 5 10 20 30 40 50 \
  --checkpoint-dir data/stage1a/binance/15m/checkpoints/ce_supcon_long_supcon_dominant_warmup_v1_seed_42_e50
```

---

## Planned Readout

Use the standard fixed-budget checkpoint selection protocol:

1. Export validation latents for `epoch_005/010/020/030/040/050`.
2. Run embedding compare and branch usage diagnostics for each validation checkpoint.
3. Select checkpoint by validation geometry.
4. Export test latents only for selected checkpoint.
5. Compare against:
   - `frontload_seed41`
   - `warmup_seed41`
   - `frontload_seed42`
   - `warmup_seed42`

---

## Success Criteria

Primary target is not maximum `z_long` quality alone. It is a better balance.

Desired relative to hard SupCon-only warmup:

- lower `z_fused` symbol agreement
- better `z_short` label agreement
- lower `z_short` symbol agreement
- lower effective long/short ratio

Desired relative to frontload:

- lower `z_long` symbol agreement, especially for `seed_41`
- stronger `z_long_proj` label/symbol geometry
- no return to the SupCon plateau

Failure interpretation:

- If SupCon still plateaus, CE weight 0.10 is already too high for `seed_41`.
- If `z_short` still collapses, any CE weight this small is insufficient; use staged warmup.
- If `z_long` improves and `z_short` holds, this becomes the new strongest candidate.
