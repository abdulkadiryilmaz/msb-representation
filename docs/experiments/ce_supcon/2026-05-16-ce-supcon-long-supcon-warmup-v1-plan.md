# Experiment Plan: 2026-05-16 — CE + SupCon Long SupCon-Only Warmup V1

**Status**: Planned  
**Family**: `CE + SupCon`  
**Baseline**: `ce_supcon_long_frontload_v1`  
**Diagnostic seed**: `seed_41`  

---

## Motivation

`long frontload v1` improved the training dynamics but did not fully solve seed stability.

The key failure case is `seed_41`:

- frontload kept `train_supcon_loss` near the 3.43 plateau until about epoch 18
- label geometry eventually recovered
- `z_long` remained symbol-heavy even at the validation-selected checkpoint

This suggests that increasing SupCon weight is not sufficient. In some seeds, CE/classifier dynamics may still settle before `z_long_proj` label geometry becomes active.

---

## Hypothesis

If `seed_41` fails because CE/classifier dynamics dominate before `z_long_proj` geometry forms, then a short SupCon-only warmup should:

- break the SupCon loss plateau before epoch 5-10
- reduce `z_long` / `z_long_proj` symbol agreement at the validation-selected checkpoint
- increase effective long/short contribution without making `z_long` more symbol-bound
- preserve label agreement after CE is restored

---

## Intervention

Use a two-phase schedule:

```text
epoch 1-5:
  loss = 0.00 * CE + 0.10 * SupCon

epoch 6+:
  loss = 1.00 * CE + 0.05 * SupCon
```

This differs from frontload v1:

```text
frontload v1 epoch 1-8:
  loss = 1.00 * CE + 0.10 * SupCon
```

The goal is to test whether `z_long_proj` can form a label-dominant geometry before CE creates a shortcut.

---

## Primary Run

Run first on diagnostic `seed_41`.

```bash
PYTHONPATH=src python scripts/train_stage1a.py \
  --dataset-root data/stage1a/binance/15m \
  --epochs 50 \
  --patience 100 \
  --batch-size 32 \
  --use-supcon \
  --ce-warmup-weight 0.0 \
  --ce-warmup-epochs 5 \
  --supcon-weight 0.05 \
  --supcon-frontload-weight 0.10 \
  --supcon-frontload-epochs 5 \
  --supcon-temperature 0.1 \
  --supcon-embedding-key z_long_proj \
  --seed 41 \
  --epoch-checkpoints 5 10 20 30 40 50 \
  --checkpoint-dir data/stage1a/binance/15m/checkpoints/ce_supcon_long_supcon_warmup_v1_seed_41_e50
```

---

## Planned Readout

Use the standard fixed-budget checkpoint selection protocol:

1. Export validation latents for `epoch_005/010/020/030/040/050`.
2. Run embedding compare and branch usage diagnostics for each validation checkpoint.
3. Select checkpoint by validation geometry.
4. Export test latents only for the selected checkpoint.
5. Compare against:
   - `ce_supcon_long_frontload_v1_seed_41_e50`
   - `ce_supcon_long_frontload_v1_seed_42_e50`
   - `ce_supcon_long_frontload_v1_seed_43_e70`
   - original `ce_supcon_long_v1`

---

## Success Criteria

Relative to `frontload_seed41 epoch_040.pt`:

| Metric | Frontload seed 41 baseline | Desired direction |
|---|---:|---|
| val `z_long` label agreement | 0.8024 | keep similar or improve |
| val `z_long` symbol agreement | 0.4888 | materially lower |
| val `z_long_proj` label agreement | 0.7980 | keep similar or improve |
| val `z_long_proj` symbol agreement | 0.3395 | lower |
| effective long/short | 0.2893 | higher, without symbol increase |

Training-dynamics success:

- `train_supcon_loss` should break the 3.43 plateau before epoch 5-10.

Failure interpretation:

- If SupCon loss still plateaus, the issue is not CE competition alone; inspect batch/pair composition or projection optimization.
- If SupCon loss drops but `z_long` remains symbol-heavy, add explicit branch/symbol guardrails next.
- If warmup improves geometry but CE restoration later degrades it, try SupCon-dominant staged warmup instead of hard switch.
