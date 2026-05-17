# Experiment Readout: 2026-05-16 — CE + SupCon Long Frontload V1

**Status**: Useful / checkpoint-selection change required  
**Family**: `CE + SupCon`  
**Baseline**: `ce_supcon_long_v1`  
**Primary run**: `ce_supcon_long_frontload_v1_seed_43_e70`  
**Follow-up seeds**: `ce_supcon_long_frontload_v1_seed_41_e50`, `ce_supcon_long_frontload_v1_seed_42_e50`  
**Primary validation-selected checkpoint**: `seed_43 epoch_050.pt`  
**Retrospective test diagnostic peak**: `epoch_040.pt`  

---

## Setup

Seed stability readout showed that `ce_supcon_long_v1` can find the desired geometry, but the result is seed-sensitive.

This experiment used the previously unstable `seed_43` and changed only the SupCon weight schedule:

```text
epoch 1-8:
  CE + 0.10 * SupCon

epoch 9+:
  CE + 0.05 * SupCon
```

The run was extended to 70 epochs with dense epoch checkpoints:

```text
epoch_005, epoch_010, epoch_020, epoch_030, epoch_040, epoch_050, epoch_060, epoch_070
```

---

## Training Summary

Training metadata:

| Metric | Value |
|---|---:|
| best val macro F1 | 0.8893 |
| best val loss | 0.4230 |
| best val F1 epoch | 20 |
| best val loss epoch | 23 |
| final epoch | 70 |
| final val macro F1 | 0.8803 |
| final val loss | 0.4925 |

SupCon trend:

| Point | train SupCon |
|---|---:|
| epoch 1 | 3.4355 |
| epoch 10 | 2.9963 |
| epoch 40 | 2.9270 |
| epoch 70 | 2.8958 |

Readout:

- Frontload made SupCon active much earlier than the original unstable `seed_43` run.
- Classifier validation metrics peak earlier than latent geometry.
- Training loss and SupCon loss continue improving after validation loss stops improving.

---

## Validation Geometry Sweep

Checkpoint selection should be made on validation geometry. After the initial retrospective test sweep, the same saved epoch checkpoints were evaluated on validation split.

| Epoch | `z_long` label | `z_long` symbol | `z_long_proj` label | `z_long_proj` symbol | `z_fused` label | `z_fused` symbol | effective long/short |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.7796 | 0.5335 | 0.7715 | 0.3434 | 0.8202 | 0.2960 | 0.0692 |
| 10 | 0.7949 | 0.4391 | 0.7874 | 0.2739 | 0.8265 | 0.3031 | 0.2159 |
| 20 | 0.7955 | 0.3521 | 0.7947 | 0.2717 | 0.8397 | 0.3101 | 0.3465 |
| 30 | 0.8005 | 0.3407 | 0.7988 | 0.2703 | 0.8317 | 0.3191 | 0.4692 |
| 40 | 0.8023 | 0.3442 | 0.8015 | 0.2728 | 0.8356 | 0.3219 | 0.5130 |
| 50 | 0.8032 | 0.3408 | 0.8007 | 0.2713 | 0.8344 | 0.3117 | 0.5852 |
| 60 | 0.8065 | 0.3668 | 0.8044 | 0.2799 | 0.8335 | 0.3145 | 0.6187 |
| 70 | 0.8001 | 0.3817 | 0.7994 | 0.2894 | 0.8351 | 0.3171 | 0.6283 |

Validation readout:

- Epoch 40 and 50 are close.
- Epoch 50 is the cleaner validation-selected checkpoint: slightly better `z_long` label agreement than 40, slightly lower `z_long` symbol agreement, lower `z_fused` symbol agreement, and stronger effective long/short contribution.
- Epoch 60 has the best `z_long` label agreement, but symbol agreement rises enough to make it less attractive.
- Epoch 70 continues increasing effective long/short, but symbol agreement worsens further.

So under the corrected selection protocol, `epoch_050.pt` is preferred over the retrospectively observed `epoch_040.pt`.

---

## Retrospective Test Geometry Sweep

The following sweep was run on test split as a retrospective diagnostic. Future checkpoint selection should use validation geometry, then reserve test split for final confirmation.

| Epoch | val F1 | `z_long` label | `z_long` symbol | `z_long_proj` label | `z_long_proj` symbol | effective long/short |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.8717 | 0.7776 | 0.4770 | 0.7727 | 0.3247 | 0.0695 |
| 10 | 0.8778 | 0.7908 | 0.4054 | 0.7897 | 0.2773 | 0.2159 |
| 20 | 0.8893 | 0.7998 | 0.3334 | 0.7985 | 0.2746 | 0.3475 |
| 30 | 0.8748 | 0.7999 | 0.3336 | 0.8000 | 0.2743 | 0.4732 |
| 40 | 0.8827 | 0.8051 | 0.3158 | 0.8024 | 0.2613 | 0.5250 |
| 50 | 0.8823 | 0.8011 | 0.3225 | 0.8019 | 0.2682 | 0.5909 |
| 60 | 0.8756 | 0.8025 | 0.3385 | 0.7998 | 0.2725 | 0.6238 |
| 70 | 0.8803 | 0.8005 | 0.3489 | 0.7970 | 0.2733 | 0.6360 |

Epoch 40 is the best observed balance on this retrospective test sweep:

- `z_long` label agreement is highest.
- `z_long` symbol agreement remains near reference quality.
- `z_long_proj` symbol agreement reaches reference-level suppression.
- effective long/short contribution is close to the reference `ce_supcon_long_v1`.

Epochs 50-70 continue increasing effective long/short contribution, but symbol agreement worsens. That means later epochs use the long branch more, but not necessarily in a cleaner structural way.

---

## Selected Checkpoint Readout

### Validation split, `epoch_050.pt`

| Embedding | label agree | symbol agree |
|---|---:|---:|
| `z_short` | 0.7723 | 0.3070 |
| `z_long` | 0.8032 | 0.3408 |
| `z_fused` | 0.8344 | 0.3117 |
| `z_long_proj` | 0.8007 | 0.2713 |

Branch usage:

| Metric | Value |
|---|---:|
| `z_long / z_short` norm ratio | 0.4218 |
| classifier long/short weight ratio | 1.3873 |
| effective long/short ratio | 0.5852 |
| `z_long` centroid symbol/label ratio | 0.0318 |
| `z_long_proj` centroid symbol/label ratio | 0.0285 |

### Test split, `epoch_050.pt`

| Embedding | label agree | symbol agree |
|---|---:|---:|
| `z_short` | 0.7834 | 0.3001 |
| `z_long` | 0.8011 | 0.3225 |
| `z_fused` | 0.8365 | 0.3009 |
| `z_long_proj` | 0.8019 | 0.2682 |

Branch usage:

| Metric | Value |
|---|---:|
| `z_long / z_short` norm ratio | 0.4259 |
| classifier long/short weight ratio | 1.3873 |
| effective long/short ratio | 0.5909 |
| `z_long` centroid symbol/label ratio | 0.0237 |
| `z_long_proj` centroid symbol/label ratio | 0.0211 |

---

## Decision

Frontload is useful. It does not fully solve seed stability yet, but it directly improves the previously unstable `seed_43` trajectory:

- SupCon activates earlier.
- `z_long` and `z_long_proj` recover close to reference geometry.
- effective long/short contribution recovers without immediately becoming symbol-dominant.

Follow-up seed runs refine the decision:

| Run | Selected checkpoint | Split | `z_long` label | `z_long` symbol | `z_long_proj` label | `z_long_proj` symbol | `z_fused` label | `z_fused` symbol | effective long/short |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `frontload_seed41` | `epoch_040.pt` | val | 0.8024 | 0.4888 | 0.7980 | 0.3395 | 0.8423 | 0.3412 | 0.2893 |
| `frontload_seed41` | `epoch_040.pt` | test | 0.8068 | 0.4860 | 0.8002 | 0.3217 | 0.8413 | 0.3054 | 0.2863 |
| `frontload_seed42` | `epoch_040.pt` | val | 0.8107 | 0.3062 | 0.8086 | 0.2690 | 0.8390 | 0.2962 | 0.6414 |
| `frontload_seed42` | `epoch_040.pt` | test | 0.8089 | 0.3001 | 0.8067 | 0.2607 | 0.8361 | 0.2861 | 0.6489 |
| `frontload_seed43` | `epoch_050.pt` | val | 0.8032 | 0.3408 | 0.8007 | 0.2713 | 0.8344 | 0.3117 | 0.5852 |
| `frontload_seed43` | `epoch_050.pt` | test | 0.8011 | 0.3225 | 0.8019 | 0.2682 | 0.8365 | 0.3009 | 0.5909 |

Interpretation:

- `seed_42` remains excellent under frontload; frontload does not damage the previously good seed.
- `seed_43` is substantially repaired versus the original unstable seed run.
- `seed_41` improves its label geometry, but remains symbol-heavy in `z_long`; frontload alone does not fully solve seed stability.

So the revised decision is:

```text
frontload_v1 is useful and improves stability,
but it is not sufficient as the final seed-stability solution.
```

The more important process decision:

```text
For Stage 1A representation runs, best.pt by val loss is not the final representation checkpoint.
Use fixed-budget training, save dense epoch checkpoints, then select a geometry checkpoint from validation diagnostics.
```

---

## Next Direction

1. Update experiment workflow to make geometry checkpoint selection explicit.
2. Run future Stage 1A representation experiments with fixed epoch budget and dense checkpoint saves.
3. Select checkpoints on validation geometry, not test geometry.
4. Use test split only after selecting the checkpoint.
5. Investigate why `seed_41` remains symbol-heavy despite early SupCon activation.
6. Next candidate should target branch usage stability more directly: CE-delay / warm-start or explicit branch usage regularization.
