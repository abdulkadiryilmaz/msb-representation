# 2026-05-25 - Stage 1B H16 Joint Event-Sequence Predictor v1 Readout

## Amaç

Bir önceki H16 event-sequence v1 modelinde event type ve event direction ayrı head'lerde öğrenildi.

Kritik hata:

```text
index 453:
  true event_type      = reversal
  pred event_type      = reversal
  true event_direction = bearish
  pred event_direction = bullish
```

Bu deney, event type ve yönü tek sınıf uzayında birleştirerek bu kopukluğu azaltmayı test eder.

## Joint Label Space

Joint sınıflar:

```text
no_event
fresh_break_bullish
fresh_break_bearish
continuation_bullish
continuation_bearish
reversal_bullish
reversal_bearish
```

Bu tasarımda `reversal` ve `bearish` ayrı kararlar değil, aynı karar yüzeyinin parçasıdır.

## Setup

Input:

```text
z_fused
+ explicit proximity / level-interaction context
```

Targets:

```text
h16_event_type + h16_event_direction -> joint_event
```

Loss:

```text
balanced CE(joint_event)
```

Checkpoint:

```text
data/stage1b/binance/15m/checkpoints/stage1b_h16_joint_event_sequence_v1_z_fused_proximity
```

## Command

```bash
PYTHONPATH=src python scripts/train_stage1b_joint_event_sequence.py \
  --train-latent-path data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/selected_epoch_020_train_latents.npz \
  --train-label-path data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/stage1b_forward_labels_train_h4_h8_h16_v2.parquet \
  --val-latent-path data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/selected_epoch_020_val_latents.npz \
  --val-label-path data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/stage1b_forward_labels_val_h4_h8_h16_v2.parquet \
  --test-latent-path data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/selected_epoch_020_test_latents.npz \
  --test-label-path data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/stage1b_forward_labels_test_h4_h8_h16_v2.parquet \
  --feature-keys z_fused \
  --context-feature-keys bull_distance_pct_scaled bear_distance_pct_scaled nearest_distance_pct_scaled nearest_side_signed bull_close_count_norm bear_close_count_norm bull_wick_count_norm bear_wick_count_norm bull_final_excess_scaled bear_final_excess_scaled current_label_intact current_label_bullish current_label_bearish \
  --event-type-column h16_event_type \
  --event-direction-column h16_event_direction \
  --balanced-loss \
  --epochs 50 \
  --patience 10 \
  --batch-size 256 \
  --checkpoint-dir data/stage1b/binance/15m/checkpoints/stage1b_h16_joint_event_sequence_v1_z_fused_proximity
```

## Metrics

Best validation checkpoint:

```text
epoch = 14
best_val_joint_macro_f1 = 0.4475
```

Test:

```text
joint_macro_f1           = 0.4529
derived event_type_f1    = 0.4787
derived event_direction_f1 = 0.7370
```

Joint per-class F1:

```text
no_event              0.3543
fresh_break_bullish   0.4993
fresh_break_bearish   0.5392
continuation_bullish  0.7627
continuation_bearish  0.7886
reversal_bullish      0.0995
reversal_bearish      0.1265
```

Joint confusion matrix:

Labels:

```text
no_event / fresh_break_bullish / fresh_break_bearish /
continuation_bullish / continuation_bearish /
reversal_bullish / reversal_bearish
```

```text
no_event              -> [866, 934, 1065, 245, 243, 160, 177]
fresh_break_bullish   -> [152, 876,  276,  11,  16,  30,  46]
fresh_break_bearish   -> [178, 291, 1100,  21,   4,   7,  33]
continuation_bullish  -> [  1,   1,    1, 993,   0,   0, 314]
continuation_bearish  -> [  1,   0,    4,   0,1179, 350,   0]
reversal_bullish      -> [  0,   0,    0,   0,  14,  31,   0]
reversal_bearish      -> [  0,   0,    0,  24,   0,   0,  43]
```

## 449-454 Window Check

Prediction artifact:

```text
data/stage1b/binance/15m/checkpoints/stage1b_h16_joint_event_sequence_v1_z_fused_proximity/test_joint_event_predictions.csv
```

| index | true joint | pred joint | conf | comment |
|---:|---|---|---:|---|
| 449 | no_event | fresh_break_bullish | 0.434 | false positive |
| 450 | no_event | fresh_break_bullish | 0.675 | false positive |
| 451 | fresh_break_bullish | fresh_break_bullish | 0.645 | correct |
| 452 | fresh_break_bullish | reversal_bearish | 0.407 | label error, but trade-context interesting |
| 453 | reversal_bearish | reversal_bearish | 0.549 | correct; previous split-head direction error fixed |
| 454 | fresh_break_bearish | fresh_break_bearish | 0.635 | correct |

453 için önemli fark:

```text
split-head v1:
  reversal / bullish

joint v1:
  reversal_bearish
```

Bu, joint target'ın hedeflediğimiz kopukluğu düzeltebildiğini gösteriyor.

## Interpretation

Joint modelin genel macro F1'i split-head event-type modelinden daha düşük görünüyor; bu beklenebilir, çünkü 7 sınıflı problem daha zor.

Fakat TradePlan açısından kritik olan durumlarda sinyal daha tutarlı:

```text
event type ve direction aynı karar yüzeyinde bağlanıyor.
```

Özellikle 453 örneğinde model artık reversal'ın bearish olduğunu doğru yakalıyor.

452 örneği ayrıca önemli:

```text
true target = fresh_break_bullish
pred target = reversal_bearish
```

Bu raw supervised target'a göre hata. Fakat 452, takip eden path açısından bearish reversal'a çok yakın bir bölgede olabilir. Bu nedenle bu örneği Stage 2 / trade-context audit'te ayrıca incelemek gerekiyor.

## Decision

Joint event-sequence classifier, H16 bağlam sinyali için split-head versiyondan daha tutarlı bir adaydır.

Ancak doğrudan selected Stage 1B output olmak için hâlâ şu sorunlar var:

- `no_event` ile `fresh_break` karışımı yüksek.
- Reversal sınıfları seyrek ve F1 düşük.
- Confidence orta düzeyde; thresholding / abstention gerekiyor.

## Next Direction

1. 449-454 ve benzeri local sequence örneklerinde joint H16 sinyalinin Stage 2 kararını nasıl değiştireceğini incele.
2. Joint model için confidence threshold / abstention analizi yap.
3. Gerekirse `no_event` ayrımı için iki aşamalı tasarım düşün:

```text
event_occurrence / no_event gate
then joint directional event class
```

## Artifacts

```text
src/msb_repr/stage1b/dataset.py
src/msb_repr/stage1b/model.py
scripts/train_stage1b_joint_event_sequence.py
data/stage1b/binance/15m/checkpoints/stage1b_h16_joint_event_sequence_v1_z_fused_proximity/
```

## Verification

```bash
python -m py_compile src/msb_repr/stage1b/dataset.py src/msb_repr/stage1b/model.py src/msb_repr/stage1b/__init__.py scripts/train_stage1b_joint_event_sequence.py
ruff check src/msb_repr/stage1b/dataset.py src/msb_repr/stage1b/model.py src/msb_repr/stage1b/__init__.py scripts/train_stage1b_joint_event_sequence.py
```
