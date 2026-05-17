# Experiment Plan: 2026-05-16 — CE + SupCon Long Frontload V1

**Status**: Completed  
**Family**: `CE + SupCon`  
**Baseline**: `ce_supcon_long_v1`  
**Diagnostic precursor**: `ce_supcon_long_v1_seed_stability`  

---

## Motivation

Seed stability readout showed that `ce_supcon_long_v1` can find the desired geometry, but does not do so reliably.

The strongest training-dynamics signal:

- stable run (`seed_42`) starts reducing SupCon loss by epoch 2-3
- unstable runs (`seed_41`, `seed_43`) keep SupCon loss nearly flat for many epochs
- once CE/classifier dynamics settle first, long branch usage remains weaker even if SupCon later improves

This suggests the current objective is not wrong, but under-constrained in the early training phase.

---

## Hypothesis

Front-loading SupCon weight during the first epochs will make `z_long_proj` gradients effective earlier and reduce the probability that the fused classifier settles into a short-branch shortcut before long-branch geometry forms.

Expected signs:

- `train_supcon_loss` should start dropping in the first few epochs
- `z_long/z_short` norm ratio should not collapse
- effective classifier `long/short` contribution should move closer to reference / `seed_42`
- `z_long` and `z_long_proj` symbol agreement should stay below the unstable seed runs

---

## Intervention

Base objective remains unchanged except for an early SupCon schedule:

```text
epoch 1-8:
  loss = CE + 0.10 * SupCon

epoch 9+:
  loss = CE + 0.05 * SupCon
```

CE is not delayed in this experiment. This keeps the intervention narrower than a CE-delay / warm-start design.

---

## Primary Run

Use the previously unstable `seed_43`.

Initial plan used 40 epochs. During analysis, the run was extended to 70 epochs with dense checkpoints to inspect geometry progression:

```text
epoch_005, epoch_010, epoch_020, epoch_030, epoch_040, epoch_050, epoch_060, epoch_070
```

```bash
PYTHONPATH=src python scripts/train_stage1a.py \
  --dataset-root data/stage1a/binance/15m \
  --epochs 40 \
  --patience 100 \
  --batch-size 32 \
  --use-supcon \
  --supcon-weight 0.05 \
  --supcon-frontload-weight 0.10 \
  --supcon-frontload-epochs 8 \
  --supcon-temperature 0.1 \
  --supcon-embedding-key z_long_proj \
  --seed 43 \
  --checkpoint-dir data/stage1a/binance/15m/checkpoints/ce_supcon_long_frontload_v1_seed_43
```

Extended run:

```bash
PYTHONPATH=src python scripts/train_stage1a.py \
  --dataset-root data/stage1a/binance/15m \
  --epochs 70 \
  --patience 100 \
  --batch-size 32 \
  --use-supcon \
  --supcon-weight 0.05 \
  --supcon-frontload-weight 0.10 \
  --supcon-frontload-epochs 8 \
  --supcon-temperature 0.1 \
  --supcon-embedding-key z_long_proj \
  --seed 43 \
  --epoch-checkpoints 5 10 20 30 40 50 60 70 \
  --checkpoint-dir data/stage1a/binance/15m/checkpoints/ce_supcon_long_frontload_v1_seed_43_e70
```

---

## Post-Run Required Outputs

Initial required outputs were `best.pt` and `final.pt` test latents. After the run was extended, the readout used dense epoch checkpoints instead.

For future repeats of this design, export validation latents for each candidate epoch checkpoint first, choose the representation checkpoint on validation geometry, then export test latents only for the selected checkpoint.

Original `best.pt` / `final.pt` commands:

```bash
PYTHONPATH=src python scripts/export_stage1a_latents.py \
  --dataset-root data/stage1a/binance/15m \
  --checkpoint-dir data/stage1a/binance/15m/checkpoints/ce_supcon_long_frontload_v1_seed_43 \
  --checkpoint-name best.pt \
  --split test \
  --output-path data/stage1a/binance/15m/checkpoints/ce_supcon_long_frontload_v1_seed_43/analysis/best_test_latents.npz

PYTHONPATH=src python scripts/export_stage1a_latents.py \
  --dataset-root data/stage1a/binance/15m \
  --checkpoint-dir data/stage1a/binance/15m/checkpoints/ce_supcon_long_frontload_v1_seed_43 \
  --checkpoint-name final.pt \
  --split test \
  --output-path data/stage1a/binance/15m/checkpoints/ce_supcon_long_frontload_v1_seed_43/analysis/final_test_latents.npz
```

Run for both exported latent files:

```bash
PYTHONPATH=src python scripts/compare_stage1a_embedding_views.py \
  --latent-path <latent_path>

PYTHONPATH=src python scripts/diagnose_stage1a_branch_usage.py \
  --latent-path <latent_path> \
  --checkpoint-dir data/stage1a/binance/15m/checkpoints/ce_supcon_long_frontload_v1_seed_43 \
  --checkpoint-name <best.pt|final.pt>

PYTHONPATH=src python scripts/diagnose_stage1a_bucket_geometry.py \
  --latent-path <latent_path>
```

---

## Decision Rule

Accept as useful if frontload improves `seed_43` relative to `seed_43_e40 final`:

- lower `z_long` symbol agreement
- lower `z_long_proj` symbol agreement
- higher effective long/short contribution
- no major loss in label agreement

If frontload improves early SupCon loss but does not improve branch usage, the issue is not only SupCon gradient magnitude; next candidate should be CE-delay / warm-start or explicit branch usage regularization.
