# 2026-05-25 - Stage 1B H16 Multi-Head Event-Sequence v1 Readout

## Amaç

Önceki iki H16 deneyi farklı soruları cevapladı:

```text
event-joint v1:
  horizon içindeki ilk structural event ne?

outcome-joint v1:
  horizon sonunda post-break path hangi yapısal sonuca bağlanıyor?
```

Tek target iki bilgiyi aynı anda taşıyamıyor. Bu deney, aynı input trunk üzerinden üç head eğiterek iki semantiği birlikte korumayı test eder.

## Model

Input:

```text
z_fused
+ explicit proximity / level-interaction context
```

Heads:

```text
first_event_joint:
  no_event
  fresh_break_bullish
  fresh_break_bearish
  continuation_bullish
  continuation_bearish
  reversal_bullish
  reversal_bearish

outcome_joint:
  no_event
  fresh_break_bullish
  fresh_break_bearish
  continuation_bullish
  continuation_bearish
  reversal_bullish
  reversal_bearish

dominant_direction:
  none
  bullish
  bearish
```

Loss:

```text
1.0 * CE(first_event_joint)
+ 0.7 * CE(outcome_joint)
+ 0.4 * CE(dominant_direction)
```

Selection score:

```text
0.45 * first_event_joint_macro_f1
+ 0.35 * outcome_joint_macro_f1
+ 0.20 * dominant_direction_macro_f1
```

Checkpoint:

```text
data/stage1b/binance/15m/checkpoints/stage1b_h16_multihead_event_sequence_v1_z_fused_proximity
```

## Command

```bash
PYTHONPATH=src python scripts/train_stage1b_multihead_event_sequence.py \
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
  --outcome-column h16_post_break_outcome \
  --dominant-direction-column h16_dominant_forward_direction \
  --balanced-loss \
  --epochs 50 \
  --patience 10 \
  --batch-size 256 \
  --checkpoint-dir data/stage1b/binance/15m/checkpoints/stage1b_h16_multihead_event_sequence_v1_z_fused_proximity
```

## Metrics

Best validation checkpoint:

```text
epoch = 13
best_val_selection_score = 0.4212
```

Test:

```text
first_event_joint_macro_f1      = 0.4450
outcome_joint_macro_f1          = 0.2726
dominant_direction_macro_f1     = 0.6604
```

Comparison:

| Model | first-event F1 | outcome F1 | dominant direction F1 |
|---|---:|---:|---:|
| H16 joint event v1 | 0.4529 | n/a | n/a |
| H16 outcome-joint v1 | n/a | 0.2840 | n/a |
| H16 multi-head v1 | 0.4450 | 0.2726 | 0.6604 |

Multi-head v1 ayrı single-task modelleri metrik olarak geçmedi; fakat iki sinyali aynı artifact içinde üretiyor.

## 449-454 Window Check

Prediction artifact:

```text
data/stage1b/binance/15m/checkpoints/stage1b_h16_multihead_event_sequence_v1_z_fused_proximity/test_multihead_event_predictions.csv
```

| index | first true | first pred | outcome true | outcome pred | dominant true | dominant pred | comment |
|---:|---|---|---|---|---|---|---|
| 449 | no_event | fresh_break_bullish | no_event | no_event | none | none | first-event false positive, outcome clean |
| 450 | no_event | fresh_break_bullish | no_event | no_event | none | none | first-event false positive, outcome clean |
| 451 | fresh_break_bullish | fresh_break_bullish | reversal_bearish | continuation_bullish | bearish | bullish | first event correct, outcome misses reversal |
| 452 | fresh_break_bullish | fresh_break_bullish | reversal_bearish | reversal_bearish | bearish | bullish | first + outcome correct, direction head wrong |
| 453 | reversal_bearish | reversal_bearish | reversal_bearish | reversal_bearish | bearish | bullish | first + outcome correct, direction head wrong |
| 454 | fresh_break_bearish | fresh_break_bearish | continuation_bearish | reversal_bullish | bearish | bearish | first correct, outcome wrong, direction correct |

## Interpretation

Multi-head model istenen contract yapısını teknik olarak üretiyor:

```text
first_event_signal + outcome_signal + dominant_direction
```

Ancak ilk versiyon metrik olarak single-task head'lerden daha iyi değil:

- `first_event_joint` hafif geriledi.
- `outcome_joint` hafif geriledi.
- `dominant_direction` 451-453 gibi kritik bölgede bullish'e kayabiliyor.

Yine de 452-453 segmenti için önemli bir kazanç var:

```text
first_event = fresh/reversal context
outcome = reversal_bearish
```

Bu, Stage 2 tarafında "ilk trigger" ile "path bias" ayrımının mümkün olduğunu gösteriyor.

## Decision

Multi-head v1 selected model değil.

Ama contract yönü doğru:

```text
Stage 1B H16 output'u çok-output olmalı.
```

Mevcut en güvenilir H16 inference adayı hâlâ:

```text
stage1b_h16_joint_event_sequence_v1_z_fused_proximity
```

Multi-head v1 ise sonraki model tasarımı için baseline olarak tutulmalı.

## Next Direction

İyileştirme adayları:

1. Loss ağırlıklarını first-event lehine biraz artırmak:

```text
first_event=1.0
outcome=0.4
dominant=0.2
```

2. Dominant direction head'i auxiliary olarak tutup Stage 2 kararında düşük ağırlık vermek.
3. Outcome head'i sadece `continuation/reversal/no_event` üçlüsüne indirgemek; support almayan `fresh_break_*` sınıflarını outcome head'den çıkarmak.
4. Reversal sınıfları için daha geniş symbol universe veya targeted sampling düşünmek.

## Artifacts

```text
src/msb_repr/stage1b/dataset.py
src/msb_repr/stage1b/model.py
src/msb_repr/stage1b/__init__.py
scripts/train_stage1b_multihead_event_sequence.py
data/stage1b/binance/15m/checkpoints/stage1b_h16_multihead_event_sequence_v1_z_fused_proximity/
```

## Verification

```bash
python -m py_compile src/msb_repr/stage1b/dataset.py src/msb_repr/stage1b/model.py src/msb_repr/stage1b/__init__.py scripts/train_stage1b_multihead_event_sequence.py
ruff check src/msb_repr/stage1b/dataset.py src/msb_repr/stage1b/model.py src/msb_repr/stage1b/__init__.py scripts/train_stage1b_multihead_event_sequence.py
```
