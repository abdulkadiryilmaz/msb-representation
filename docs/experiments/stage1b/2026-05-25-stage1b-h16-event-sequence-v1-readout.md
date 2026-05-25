# 2026-05-25 - Stage 1B H16 Event-Sequence Predictor v1 Readout

## Amaç

Bu deney, H8 fresh-break predictor'ı tamamlayacak daha geniş horizon sinyalini test eder.

Soru:

```text
Önümüzdeki H16 içinde yapı no_event / fresh_break / continuation / reversal mı olacak,
ve yönü ne olacak?
```

Bu predictor'ın Stage 2'ye vermesi beklenen sinyal, doğrudan entry/stop/target değil; continuation / reversal bağlamıdır.

## Setup

Selected Stage 1A latent:

```text
ce_supcon_long_branch_ce_aux_v1_seed_41_e50/selected_epoch_020
```

Input:

```text
z_fused
+ explicit proximity / level-interaction context
```

Context feature'ları:

```text
bull_distance_pct_scaled
bear_distance_pct_scaled
nearest_distance_pct_scaled
nearest_side_signed
bull_close_count_norm
bear_close_count_norm
bull_wick_count_norm
bear_wick_count_norm
bull_final_excess_scaled
bear_final_excess_scaled
current_label_intact
current_label_bullish
current_label_bearish
```

Targets:

```text
h16_event_type
h16_event_direction
```

Event type classes:

```text
no_event / fresh_break / continuation / reversal
```

Direction classes:

```text
bullish / bearish
```

Direction loss sadece `event_direction != none` örneklerinde hesaplandı.

Loss:

```text
balanced CE(event_type) + balanced CE(event_direction)
```

## Command

```bash
PYTHONPATH=src python scripts/train_stage1b_event_sequence.py \
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
  --balanced-event-type-loss \
  --balanced-direction-loss \
  --epochs 50 \
  --patience 10 \
  --batch-size 256 \
  --checkpoint-dir data/stage1b/binance/15m/checkpoints/stage1b_h16_event_sequence_v1_z_fused_proximity
```

## Dataset Counts

Train:

```text
no_event      51818
fresh_break   40675
continuation  36906
reversal       1633
```

Val:

```text
no_event       3286
fresh_break    2758
continuation   2649
reversal        137
```

Reversal sınıfı hâlâ seyrek, fakat H8'e göre daha kullanılabilir.

## Metrics

Best validation checkpoint:

```text
epoch = 7
best_val_event_type_macro_f1 = 0.5132
best_val_event_direction_macro_f1 = 0.8519
```

Test:

```text
event_type_macro_f1      = 0.5210
event_direction_macro_f1 = 0.8479
```

Event type per-class F1:

```text
no_event      0.5427
fresh_break   0.6441
continuation  0.7710
reversal      0.1265
```

Test event type confusion matrix:

Labels: `no_event / fresh_break / continuation / reversal`

```text
no_event      -> [1727, 1263,  421, 279]
fresh_break   -> [ 884, 2070,   38,  49]
continuation  -> [  59,   52, 2092, 641]
reversal      -> [   5,    2,   32,  73]
```

Test direction confusion matrix:

Labels: `bullish / bearish`

```text
bullish -> [2370,  392]
bearish -> [ 517, 2718]
```

## 449-454 Window Check

Prediction artifact:

```text
data/stage1b/binance/15m/checkpoints/stage1b_h16_event_sequence_v1_z_fused_proximity/test_event_predictions.csv
```

Key rows:

| index | true event | pred event | event conf | true dir | pred dir | dir conf | note |
|---:|---|---|---:|---|---|---:|---|
| 449 | no_event | fresh_break | 0.555 | none | bullish | 0.747 | false positive |
| 450 | no_event | fresh_break | 0.650 | none | bullish | 0.854 | false positive |
| 451 | fresh_break | fresh_break | 0.735 | bullish | bullish | 0.802 | correct |
| 452 | fresh_break | fresh_break | 0.708 | bullish | bullish | 0.835 | correct |
| 453 | reversal | reversal | 0.586 | bearish | bullish | 0.893 | event type correct, direction wrong |
| 454 | fresh_break | fresh_break | 0.603 | bearish | bearish | 0.631 | correct |

## Interpretation

Bu model H16 seviyesinde iki önemli şeyi gösterdi:

1. `continuation` sınıfı oldukça okunabilir.
2. `reversal` sınıfı event type olarak yakalanabiliyor, fakat yön bağlama hâlâ zayıf.

453 örneği bunun net göstergesi:

```text
true event_type      = reversal
pred event_type      = reversal
true event_direction = bearish
pred event_direction = bullish
```

Yani model "burada reversal var" diyebiliyor, fakat reversal'ın hangi yöne çözüldüğünü ayrı direction head'de güvenilir bağlamıyor.

## Decision

Bu deney H16 event-sequence sinyalinin faydalı olduğunu gösteriyor, fakat v1 doğrudan Stage 2 handoff için yeterli değil.

Kabul edilen çıkarım:

```text
H16 event_type signal useful as diagnostic / context signal.
H16 event_direction needs stronger coupling to event_type.
```

## Next Direction

Bir sonraki mantıklı modelleme adımı:

```text
joint event-sequence classifier
```

Yani event type ve yönü ayrı head'lerde değil, ortak sınıf uzayında öğrenmek:

```text
no_event
fresh_break_bullish
fresh_break_bearish
continuation_bullish
continuation_bearish
reversal_bullish
reversal_bearish
```

Bu, 453 gibi örneklerde `reversal` ve `bearish` bilgisini aynı karar yüzeyinde bağlamayı hedefler.

## Artifacts

```text
src/msb_repr/stage1b/dataset.py
src/msb_repr/stage1b/model.py
scripts/train_stage1b_event_sequence.py
data/stage1b/binance/15m/checkpoints/stage1b_h16_event_sequence_v1_z_fused_proximity/
```

## Verification

```bash
python -m py_compile src/msb_repr/stage1b/dataset.py src/msb_repr/stage1b/model.py src/msb_repr/stage1b/__init__.py scripts/train_stage1b_event_sequence.py
ruff check src/msb_repr/stage1b/dataset.py src/msb_repr/stage1b/model.py src/msb_repr/stage1b/__init__.py scripts/train_stage1b_event_sequence.py
```
