# Experiment Readout: 2026-05-17 — CE + SupCon Long SupCon-Only Warmup V1

**Status**: Diagnostic success / not final candidate  
**Family**: `CE + SupCon`  
**Baseline**: `ce_supcon_long_frontload_v1`  
**Runs**: `seed_41`, `seed_42`  

---

## Setup

This experiment targeted the `seed_41` failure mode from `frontload_v1`.

Frontload improved stability, but in `seed_41` the SupCon loss still stayed near the 3.43 plateau until late training and `z_long` remained symbol-heavy.

SupCon-only warmup used:

```text
epoch 1-5:
  loss = 0.00 * CE + 0.10 * SupCon

epoch 6+:
  loss = 1.00 * CE + 0.05 * SupCon
```

Dense checkpoints:

```text
epoch_005, epoch_010, epoch_020, epoch_030, epoch_040, epoch_050
```

---

## Training Dynamics

SupCon-only warmup broke the SupCon plateau early in both seeds.

| Run | epoch 1 train SupCon | epoch 5 train SupCon | epoch 10 train SupCon | epoch 50 train SupCon |
|---|---:|---:|---:|---:|
| `warmup_seed41` | 3.1653 | 2.9739 | 2.9101 | 2.7873 |
| `warmup_seed42` | 3.2156 | 2.9763 | 2.9167 | 2.7932 |

Classifier behavior is expected:

- during epoch 1-5, CE is disabled and macro F1 is not meaningful
- after CE returns at epoch 6, classifier performance quickly recovers

This confirms that the seed 41 failure was at least partly an early SupCon activation problem.

---

## Validation Sweep

### `seed_41`

| Epoch | `z_long` label | `z_long` symbol | `z_long_proj` label | `z_long_proj` symbol | `z_fused` label | `z_fused` symbol | effective long/short |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.8105 | 0.4567 | 0.8108 | 0.2753 | 0.8195 | 0.5885 | 0.8212 |
| 10 | 0.8267 | 0.2854 | 0.8278 | 0.2562 | 0.8306 | 0.6717 | 14.6419 |
| 20 | 0.8389 | 0.3581 | 0.8370 | 0.2730 | 0.8392 | 0.6424 | 16.4474 |
| 30 | 0.8378 | 0.2959 | 0.8370 | 0.2699 | 0.8410 | 0.6100 | 15.1470 |
| 40 | 0.8303 | 0.2822 | 0.8311 | 0.2650 | 0.8348 | 0.5491 | 10.7696 |
| 50 | 0.8361 | 0.2720 | 0.8353 | 0.2582 | 0.8415 | 0.5552 | 7.6147 |

Validation-selected checkpoint: `epoch_050.pt`.

### `seed_42`

| Epoch | `z_long` label | `z_long` symbol | `z_long_proj` label | `z_long_proj` symbol | `z_fused` label | `z_fused` symbol | effective long/short |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.8108 | 0.4169 | 0.8089 | 0.2594 | 0.8149 | 0.5847 | 0.6733 |
| 10 | 0.8254 | 0.3125 | 0.8242 | 0.2706 | 0.8284 | 0.6274 | 5.2699 |
| 20 | 0.8307 | 0.3605 | 0.8323 | 0.2913 | 0.8353 | 0.5078 | 8.1257 |
| 30 | 0.8407 | 0.2841 | 0.8403 | 0.2544 | 0.8459 | 0.5338 | 6.1786 |
| 40 | 0.8394 | 0.2864 | 0.8375 | 0.2551 | 0.8437 | 0.4544 | 5.1639 |
| 50 | 0.8385 | 0.3182 | 0.8375 | 0.2646 | 0.8424 | 0.4317 | 4.6683 |

Validation-selected checkpoint: `epoch_040.pt`.

---

## Selected Checkpoint Test Readout

| Run | Checkpoint | `z_short` label | `z_short` symbol | `z_long` label | `z_long` symbol | `z_long_proj` label | `z_long_proj` symbol | `z_fused` label | `z_fused` symbol | effective long/short |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `warmup_seed41` | `epoch_050.pt` | 0.5174 | 0.7091 | 0.8380 | 0.2701 | 0.8382 | 0.2562 | 0.8438 | 0.5384 | 7.8831 |
| `warmup_seed42` | `epoch_040.pt` | 0.6137 | 0.5776 | 0.8364 | 0.2771 | 0.8358 | 0.2592 | 0.8397 | 0.4446 | 5.4168 |

---

## Comparison Against Frontload

| Run | `z_short` label | `z_short` symbol | `z_long` label | `z_long` symbol | `z_fused` label | `z_fused` symbol | effective long/short |
|---|---:|---:|---:|---:|---:|---:|---:|
| `frontload_seed41` | 0.8284 | 0.2769 | 0.8068 | 0.4860 | 0.8413 | 0.3054 | 0.2863 |
| `warmup_seed41` | 0.5174 | 0.7091 | 0.8380 | 0.2701 | 0.8438 | 0.5384 | 7.8831 |
| `frontload_seed42` | 0.7827 | 0.2761 | 0.8089 | 0.3001 | 0.8361 | 0.2861 | 0.6489 |
| `warmup_seed42` | 0.6137 | 0.5776 | 0.8364 | 0.2771 | 0.8397 | 0.4446 | 5.4168 |

---

## Decision

SupCon-only warmup is a diagnostic success:

- it breaks the early SupCon plateau
- it repairs `z_long` and `z_long_proj` label-dominant geometry
- it removes the `seed_41` `z_long` symbol-heavy failure mode

But it is not a final candidate:

- `z_short` geometry degrades sharply
- `z_fused` becomes more symbol-heavy
- effective long/short becomes excessively large
- branch balance is lost

Interpretation:

```text
Hard SupCon-only warmup proves early SupCon activation matters,
but CE=0 for 5 epochs is too aggressive for balanced fused representation learning.
```

---

## Next Direction

Try SupCon-dominant warmup instead of hard SupCon-only warmup:

```text
epoch 1-5:
  loss = 0.10 * CE + 0.10 * SupCon

epoch 6+:
  loss = 1.00 * CE + 0.05 * SupCon
```

Expected improvement:

- keep early SupCon activation
- preserve more `z_short` / fused balance
- avoid excessive effective long/short ratio
