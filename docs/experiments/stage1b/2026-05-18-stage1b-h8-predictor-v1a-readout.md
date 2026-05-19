# Stage 1B H8 Predictor v1a Readout

**Date**: 2026-05-18
**Status**: completed
**Plan**: `2026-05-18-stage1b-h8-predictor-v1a-plan.md`
**Checkpoint**: `data/stage1b/binance/15m/checkpoints/stage1b_h8_predictor_v1a_z_fused`

## Goal

Train the first H8-focused Stage 1B two-head predictor.

Question:

> Can the selected Stage 1A representation support a practical predictor for the next 8 bars?

## Setup

Stage 1A source:

```text
ce_supcon_long_branch_ce_aux_v1_seed_41_e50/epoch_020.pt
```

Input:

```text
z_fused
```

Target:

```text
h8_future_break_direction
```

Model:

```text
Stage1BForwardPredictor(
  input_dim=96,
  hidden_dim=64,
  dropout=0.1,
  heads:
    break_logits: 2
    direction_logits: 2
)
```

Objective:

```text
loss =
  CE(break_occurs_8)
  + CE(direction_8 | break_8)
```

Training:

```text
epochs=50
patience=10
batch_size=256
seed=42
selection_metric=val reconstructed 3-class macro F1
```

Training stopped at epoch 44. Best validation checkpoint was epoch 34.

## Results

### Validation

| Metric | Value |
|---|---:|
| reconstructed 3-class macro F1 | 0.7364 |
| reconstructed 3-class accuracy | 0.7341 |
| break occurrence macro F1 | 0.7496 |
| break occurrence accuracy | 0.7525 |
| direction-given-break macro F1 | 0.9051 |
| direction-given-break accuracy | 0.9052 |

Validation reconstructed 3-class confusion matrix:

Labels: `none / bullish / bearish`

```text
none    -> [2844,  421,  440]
bullish -> [ 672, 1787,   63]
bearish -> [ 652,  100, 1851]
```

### Test

| Metric | Value |
|---|---:|
| reconstructed 3-class macro F1 | 0.7307 |
| reconstructed 3-class accuracy | 0.7327 |
| break occurrence macro F1 | 0.7497 |
| break occurrence accuracy | 0.7502 |
| direction-given-break macro F1 | 0.8955 |
| direction-given-break accuracy | 0.8955 |

Test reconstructed 3-class confusion matrix:

Labels: `none / bullish / bearish`

```text
none    -> [3426,  458,  450]
bullish -> [ 737, 1717,   80]
bearish -> [ 777,   89, 1961]
```

Test break occurrence confusion matrix:

Labels: `none / break`

```text
none  -> [3426,  908]
break -> [1514, 3847]
```

Test direction-given-break confusion matrix:

Labels: `bullish / bearish`

```text
bullish -> [2324,  210]
bearish -> [ 350, 2477]
```

## Baseline Comparison

| Model | Target | Test reconstructed macro F1 |
|---|---|---:|
| H8 linear probe on `z_fused` | H8 3-class direction | 0.7233 |
| H8 MLP probe on `z_fused` | H8 3-class direction | 0.7284 |
| H8 predictor v1a reconstructed | H8 3-class direction | 0.7307 |
| H16 predictor v1a reconstructed | H16 3-class direction | 0.6731 |
| H16+H32 predictor v1b reconstructed | H16 3-class direction | 0.6765 |

The H8 predictor slightly beats the H8 probe baselines while exposing separate:

```text
p_break_8
p_direction_8_given_break
```

This is the cleanest Stage 1B predictor result so far.

## Threshold Sweep

Break threshold sweep validation split üzerinde yapıldı:

```text
threshold range: 0.05 - 0.95
step: 0.01
selection metric: val reconstructed 3-class macro F1
```

Sonuç:

| Threshold | Split | Reconstructed macro F1 | Break macro F1 | Break pred rate |
|---:|---|---:|---:|---:|
| 0.50 | test | 0.7307 | 0.7497 | 0.4905 |
| 0.49 | val | 0.7365 | 0.7499 | 0.5362 |
| 0.49 | test | 0.7296 | 0.7488 | 0.4982 |

Val üzerinde en iyi threshold `0.49` oldu. Ancak test split'te default `0.50` daha iyi kaldı. H8 v1a için default threshold korunmalı.

## Interpretation

H8 is currently the best primary Stage 1B horizon.

Key points:

- H8 label distribution is much more balanced than H16.
- H8 predictor beats H8 probe baselines slightly.
- Break occurrence macro F1 is materially better than H16 predictor.
- Direction-given-break is strong (`0.8955` test macro F1).
- Threshold tuning does not provide a robust gain.

Remaining issue:

`break -> none` false negatives are still substantial:

```text
break -> none = 1514
```

But compared with H16, the break/no-break problem is cleaner and less dominated by break-heavy prior.

## Decision

Accept `stage1b_h8_predictor_v1a_z_fused` as the current primary Stage 1B predictor baseline.

Next step:

1. Run H8 + H16 multi-horizon predictor.
2. Check whether H16 follow-through supervision improves H8 without degrading it.
3. Analyze H8=`none`, H16=`bullish/bearish` delayed segment if multi-horizon is accepted.

## Artifacts

```text
data/stage1b/binance/15m/checkpoints/stage1b_h8_predictor_v1a_z_fused/best.pt
data/stage1b/binance/15m/checkpoints/stage1b_h8_predictor_v1a_z_fused/final.pt
data/stage1b/binance/15m/checkpoints/stage1b_h8_predictor_v1a_z_fused/history.json
data/stage1b/binance/15m/checkpoints/stage1b_h8_predictor_v1a_z_fused/metadata.json
data/stage1b/binance/15m/checkpoints/stage1b_h8_predictor_v1a_z_fused/train_config.json
data/stage1b/binance/15m/checkpoints/stage1b_h8_predictor_v1a_z_fused/threshold_sweep.json
```
