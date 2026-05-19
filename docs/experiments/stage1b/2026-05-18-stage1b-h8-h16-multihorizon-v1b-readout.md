# Stage 1B H8 + H16 Multi-Horizon Predictor v1b Readout

**Date**: 2026-05-18
**Status**: completed
**Plan**: `2026-05-18-stage1b-h8-h16-multihorizon-v1b-plan.md`
**Checkpoint**: `data/stage1b/binance/15m/checkpoints/stage1b_h8_h16_multihorizon_v1b_z_fused`

## Goal

Test whether H16 follow-through supervision improves or stabilizes the H8 primary predictor.

V1b question:

> Can H8 near-term break prediction and H16 wider follow-through be learned together without degrading H8?

## Setup

Stage 1A source:

```text
ce_supcon_long_branch_ce_aux_v1_seed_41_e50/epoch_020.pt
```

Input:

```text
z_fused
```

Targets:

```text
h8_future_break_direction
h16_future_break_direction
```

Primary horizon:

```text
H8
```

Objective:

```text
loss =
  CE(break_occurs_8)
  + CE(direction_8 | break_8)
  + CE(break_occurs_16)
  + CE(direction_16 | break_16)
```

Training:

```text
epochs=50
patience=10
batch_size=256
seed=42
selection_metric=H8 val reconstructed 3-class macro F1
```

Training stopped at epoch 32. Best validation checkpoint was epoch 22.

## Results

### H8 Primary Horizon

| Split | Reconstructed macro F1 | Break macro F1 | Direction-given-break macro F1 |
|---|---:|---:|---:|
| val | 0.7340 | 0.7480 | 0.9018 |
| test | 0.7270 | 0.7463 | 0.8968 |

Test reconstructed 3-class confusion matrix:

Labels: `none / bullish / bearish`

```text
none    -> [3415,  456,  461]
bullish -> [ 746, 1701,   81]
bearish -> [ 790,   89, 1948]
```

Test break occurrence confusion matrix:

Labels: `none / break`

```text
none  -> [3415,  917]
break -> [1536, 3819]
```

Test direction-given-break confusion matrix:

Labels: `bullish / bearish`

```text
bullish -> [2309,  219]
bearish -> [ 333, 2494]
```

### H16 Secondary Horizon

| Split | Reconstructed macro F1 | Break macro F1 | Direction-given-break macro F1 |
|---|---:|---:|---:|
| val | 0.6568 | 0.6708 | 0.8486 |
| test | 0.6607 | 0.6880 | 0.8359 |

H16 secondary head remains usable, but it is weaker than the earlier H16-primary predictor runs.

## Baseline Comparison

| Model | Horizon | Test reconstructed macro F1 | Test break macro F1 | Test direction macro F1 |
|---|---|---:|---:|---:|
| H8 predictor v1a | H8 | 0.7307 | 0.7497 | 0.8955 |
| H8+H16 v1b | H8 | 0.7270 | 0.7463 | 0.8968 |
| H16 predictor v1a | H16 | 0.6731 | 0.7015 | 0.8330 |
| H8+H16 v1b | H16 | 0.6607 | 0.6880 | 0.8359 |

H16 supervision does not improve H8. It slightly degrades H8 reconstructed and break metrics while leaving H8 direction roughly intact.

## Threshold Sweep

Break threshold sweep H8 primary horizon üzerinde yapıldı.

Result:

| Threshold | Split | Reconstructed macro F1 | Break macro F1 | Break pred rate |
|---:|---|---:|---:|---:|
| 0.50 | val | 0.7340 | 0.7480 | 0.5260 |
| 0.50 | test | 0.7270 | 0.7463 | 0.4889 |

Default threshold is also the validation-selected threshold.

## Interpretation

H8+H16 v1b is a non-catastrophic but not useful multi-horizon variant.

Main observations:

- H8 remains strong, but does not improve over H8-only v1a.
- H16 secondary head is weaker than H16-primary v1a.
- Equal-weight multi-horizon training likely makes H16 act as a noisy auxiliary objective rather than useful follow-through guidance.

This differs from the earlier H16+H32 run, where H32 did not degrade H16 and slightly improved it. For H8, the H16 objective appears to dilute the cleaner near-term signal.

## Decision

Reject H8+H16 v1b as the primary Stage 1B candidate.

Keep:

```text
stage1b_h8_predictor_v1a_z_fused
```

as the current primary Stage 1B baseline.

Next options:

1. Analyze H8 predictor errors before adding more horizons.
2. Try H4 + H8 if we want a trigger/near-term auxiliary head.
3. If revisiting H8 + H16, use lower H16 loss weight instead of equal weights.

## Artifacts

```text
data/stage1b/binance/15m/checkpoints/stage1b_h8_h16_multihorizon_v1b_z_fused/best.pt
data/stage1b/binance/15m/checkpoints/stage1b_h8_h16_multihorizon_v1b_z_fused/final.pt
data/stage1b/binance/15m/checkpoints/stage1b_h8_h16_multihorizon_v1b_z_fused/history.json
data/stage1b/binance/15m/checkpoints/stage1b_h8_h16_multihorizon_v1b_z_fused/metadata.json
data/stage1b/binance/15m/checkpoints/stage1b_h8_h16_multihorizon_v1b_z_fused/train_config.json
data/stage1b/binance/15m/checkpoints/stage1b_h8_h16_multihorizon_v1b_z_fused/threshold_sweep.json
```
