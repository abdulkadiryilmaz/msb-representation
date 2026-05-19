# Stage 1B H16 + H32 Multi-Horizon Predictor v1b Readout

**Date**: 2026-05-18
**Status**: completed
**Plan**: `2026-05-18-stage1b-h16-h32-multihorizon-v1b-plan.md`
**Checkpoint**: `data/stage1b/binance/15m/checkpoints/stage1b_h16_h32_multihorizon_v1b_z_fused`

## Goal

H16-only predictor v1a üstüne H32 supervision eklemenin H16 hedefini bozup bozmadığını ve daha geniş intraday horizon için ek sinyal üretip üretmediğini test etmek.

Ana soru:

> H16 yakın-break davranışı korunurken H32 delayed / wider structure bias birlikte öğrenilebilir mi?

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
h16_future_break_direction
h32_future_break_direction
```

Model:

```text
Stage1BForwardPredictor(
  input_dim=96,
  hidden_dim=64,
  dropout=0.1,
  num_horizons=2,
  heads per horizon:
    break_logits: 2
    direction_logits: 2
)
```

Objective:

```text
loss =
  CE(break_occurs_16)
  + CE(direction_16 | break_16)
  + CE(break_occurs_32)
  + CE(direction_32 | break_32)
```

Training:

```text
epochs=50
patience=10
batch_size=256
seed=42
primary_horizon=H16
selection_metric=H16 reconstructed 3-class macro F1
```

Training stopped at epoch 24. Best validation checkpoint was epoch 14.

## Results

### H16 Primary Horizon

| Split | Reconstructed macro F1 | Break macro F1 | Direction-given-break macro F1 |
|---|---:|---:|---:|
| val | 0.6684 | 0.6861 | 0.8463 |
| test | 0.6765 | 0.7048 | 0.8373 |

Test reconstructed 3-class confusion matrix:

Labels: `none / bullish / bearish`

```text
none    -> [1866,  611,  546]
bullish -> [ 614, 2233,  254]
bearish -> [ 721,  368, 2458]
```

Test break occurrence confusion matrix:

Labels: `none / break`

```text
none  -> [1866, 1157]
break -> [1335, 5313]
```

Test direction-given-break confusion matrix:

Labels: `bullish / bearish`

```text
bullish -> [2661,  440]
bearish -> [ 640, 2907]
```

### H32 Secondary Horizon

| Split | Reconstructed macro F1 | Break macro F1 | Direction-given-break macro F1 |
|---|---:|---:|---:|
| val | 0.5713 | 0.5774 | 0.8004 |
| test | 0.5719 | 0.5908 | 0.7892 |

H32 direction-given-break signal is usable, but H32 break occurrence is weak. The reconstructed 3-class macro F1 is therefore much lower than H16.

## Baseline Comparison

| Model | Horizon | Test reconstructed macro F1 | Test break macro F1 | Test direction macro F1 |
|---|---|---:|---:|---:|
| v1a H16-only | H16 | 0.6731 | 0.7015 | 0.8330 |
| v1b H16+H32 | H16 | 0.6765 | 0.7048 | 0.8373 |
| v1b H16+H32 | H32 | 0.5719 | 0.5908 | 0.7892 |

V1b does not degrade H16. It slightly improves H16 test metrics, but the improvement is small enough to treat as tentative until repeated with another seed or architecture variant.

## Threshold Sweep

Break threshold sweep H16 primary horizon üzerinde yapıldı:

```text
threshold range: 0.05 - 0.95
step: 0.01
selection metric: val reconstructed 3-class macro F1
```

Sonuç:

| Threshold | Split | Reconstructed macro F1 | Break macro F1 | Break pred rate |
|---:|---|---:|---:|---:|
| 0.50 | test | 0.6765 | 0.7048 | 0.6690 |
| 0.58 | val | 0.6782 | 0.6953 | 0.6214 |
| 0.58 | test | 0.6759 | 0.7016 | 0.5859 |

Val üzerinde en iyi threshold `0.58` oldu. Ancak test split'te default `0.50` üstüne iyileşme getirmedi. Bu nedenle v1b için de default threshold `0.50` korunmalı.

## Interpretation

V1b is a valid continuation of v1a.

Main observations:

- H32 supervision H16 performansını bozmadı.
- H16 metrics v1a üstüne çok küçük bir artış gösterdi.
- H32 direction head anlamlı sinyal taşıyor.
- H32 break occurrence head zayıf kaldığı için H32 reconstructed macro F1 düşük.

Bu sonuç, multi-horizon supervision fikrini tamamen doğrulamaz; ama H16 guardrail'i geçtiği için sonraki Stage 1B iterasyonları için makul bir zemin sağlar.

## Decision

Accept v1b as a non-regressing multi-horizon baseline.

Do not move to Stage 2 yet.

Next step:

1. H16=`none`, H32=`bullish/bearish` segmentini ayrıca incele.
2. H32 break occurrence zayıflığının label prior / horizon imbalance kaynaklı olup olmadığını kontrol et.
3. Predictor v1c için `H16 + H32` loss ağırlıklarını veya horizon-specific heads kapasitesini değerlendirmeden önce bu segment analizini tamamla.

## Artifacts

```text
data/stage1b/binance/15m/checkpoints/stage1b_h16_h32_multihorizon_v1b_z_fused/best.pt
data/stage1b/binance/15m/checkpoints/stage1b_h16_h32_multihorizon_v1b_z_fused/final.pt
data/stage1b/binance/15m/checkpoints/stage1b_h16_h32_multihorizon_v1b_z_fused/history.json
data/stage1b/binance/15m/checkpoints/stage1b_h16_h32_multihorizon_v1b_z_fused/metadata.json
data/stage1b/binance/15m/checkpoints/stage1b_h16_h32_multihorizon_v1b_z_fused/train_config.json
data/stage1b/binance/15m/checkpoints/stage1b_h16_h32_multihorizon_v1b_z_fused/threshold_sweep.json
```
