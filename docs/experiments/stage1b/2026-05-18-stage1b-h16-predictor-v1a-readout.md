# Stage 1B H16 Predictor v1a Readout

**Date**: 2026-05-18
**Status**: completed
**Plan**: `2026-05-18-stage1b-h16-predictor-v1a-plan.md`
**Checkpoint**: `data/stage1b/binance/15m/checkpoints/stage1b_h16_predictor_v1a_z_fused`

## Goal

İlk Stage 1B predictor baseline'ını kurmak:

> Önümüzdeki 16 bar içinde structural break olacak mı? Olacaksa yönü ne?

Bu çalışma Stage 2 / TradePlan üretmez. Amaç, probe'da gözlenen H16 forward direction sinyalini iki-head predictor formatına taşımaktır.

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
h16_future_break_direction
```

Excluded labels:

```text
ambiguous
insufficient_future
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
  CE(break_occurs_16)
  + CE(direction_16 | break_16)
```

Direction loss is masked for `none` samples.

Training:

```text
epochs=50
patience=10
batch_size=256
seed=42
selection_metric=val_reconstructed_3class_macro_f1
```

Training stopped at epoch 11. Best validation checkpoint was epoch 1.

## Results

### Validation

| Metric | Value |
|---|---:|
| reconstructed 3-class macro F1 | 0.6649 |
| reconstructed 3-class accuracy | 0.6700 |
| break occurrence macro F1 | 0.6823 |
| break occurrence accuracy | 0.7322 |
| direction-given-break macro F1 | 0.8401 |
| direction-given-break accuracy | 0.8401 |

Validation reconstructed 3-class confusion matrix:

Labels: `none / bullish / bearish`

```text
none    -> [1484,  553,  550]
bullish -> [ 626, 2237,  237]
bearish -> [ 636,  312, 2195]
```

### Test

| Metric | Value |
|---|---:|
| reconstructed 3-class macro F1 | 0.6731 |
| reconstructed 3-class accuracy | 0.6740 |
| break occurrence macro F1 | 0.7015 |
| break occurrence accuracy | 0.7375 |
| direction-given-break macro F1 | 0.8330 |
| direction-given-break accuracy | 0.8332 |

Test reconstructed 3-class confusion matrix:

Labels: `none / bullish / bearish`

```text
none    -> [1891,  586,  548]
bullish -> [ 645, 2205,  265]
bearish -> [ 764,  350, 2433]
```

Test break occurrence confusion matrix:

Labels: `none / break`

```text
none  -> [1891, 1134]
break -> [1409, 5253]
```

Test direction-given-break confusion matrix:

Labels: `bullish / bearish`

```text
bullish -> [2651,  464]
bearish -> [ 647, 2900]
```

## Baseline Comparison

Probe reference:

| Model | Target | Test macro F1 |
|---|---|---:|
| linear probe on `z_fused` | H16 3-class direction | 0.6768 |
| Stage 1B predictor v1a reconstructed | H16 3-class direction | 0.6731 |

The predictor is essentially at the linear probe level on reconstructed 3-class output while exposing separate:

```text
p_break_16
p_direction_16_given_break
```

This is the intended benefit over a single probe classifier.

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
| 0.50 | test | 0.6731 | 0.7015 | 0.6593 |
| 0.54 | val | 0.6681 | 0.6862 | 0.6410 |
| 0.54 | test | 0.6729 | 0.7004 | 0.6147 |

Val üzerinde en iyi threshold `0.54` oldu. Ancak bu threshold test split'te default `0.50` üstüne anlamlı kazanç getirmedi.

Test confusion matrix at `threshold=0.54`:

Labels: `none / bullish / bearish`

```text
none    -> [2060,  502,  463]
bullish -> [ 761, 2123,  231]
bearish -> [ 911,  314, 2322]
```

Interpretation:

- Threshold yükselince `none` recall artıyor.
- Buna karşılık bullish/bearish break örneklerinden daha fazlası `none` olarak kaçıyor.
- Net test macro F1 değişmiyor; default `0.50` daha iyi kalıyor.

Bu nedenle v1a için default threshold `0.50` korunmalı. Threshold tuning şu koşuda ana iyileştirme yolu değil; break occurrence head'in kendisi veya multi-horizon sinyal daha önemli görünüyor.

## Interpretation

V1a is a valid Stage 1B baseline.

Key points:

- Reconstructed 3-class performance is close to the H16 linear probe.
- Break occurrence is readable but still the main error source.
- Direction-given-break is strong and does not collapse to one side.
- Test is slightly better than validation on reconstructed macro F1, so the first run does not show a val-only artifact.

The main remaining issue is `none` vs `break` separation. The direction head is substantially stronger than the occurrence head, which suggests the predictor often knows direction once a break exists, but deciding whether a break occurs inside H16 remains harder.

## Decision

Accept `stage1b_h16_predictor_v1a_z_fused` as the first Stage 1B predictor baseline.

Do not move to Stage 2 yet.

Next step:

1. Keep default `p_break_16 >= 0.50` threshold for v1a.
2. Run H16 + H32 multi-horizon v1b and compare whether H32 improves delayed-setup behavior without degrading H16.
3. Revisit threshold tuning only if v1b changes break probability calibration.

## Artifacts

```text
data/stage1b/binance/15m/checkpoints/stage1b_h16_predictor_v1a_z_fused/best.pt
data/stage1b/binance/15m/checkpoints/stage1b_h16_predictor_v1a_z_fused/final.pt
data/stage1b/binance/15m/checkpoints/stage1b_h16_predictor_v1a_z_fused/history.json
data/stage1b/binance/15m/checkpoints/stage1b_h16_predictor_v1a_z_fused/metadata.json
data/stage1b/binance/15m/checkpoints/stage1b_h16_predictor_v1a_z_fused/train_config.json
data/stage1b/binance/15m/checkpoints/stage1b_h16_predictor_v1a_z_fused/threshold_sweep.json
```
