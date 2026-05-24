# 2026-05-24 - Stage 1B Proximity-Enriched Predictor v3 Readout

## Amaç

Bu deney, frozen Stage 1A latent'ine explicit proximity / level-interaction feature'ları eklemenin Stage 1B forward prediction kalitesini artırıp artırmadığını test eder.

Plan:

```text
docs/experiments/stage1b/2026-05-24-stage1b-proximity-enriched-v3-plan.md
```

## Hipotez

Stage 1B v2 bazı pencerelerde break occurrence sinyalini yakalıyor fakat yönü ters okuyordu.

Örnek:

```text
index = 454
current_label = intact
target true = bearish
v2 decision = bullish
bear_distance_pct = 0.087%
```

Stage 1A latent proximity probe v1, nearest level side bilgisinin latentte var olduğunu fakat distance hassasiyetinin zayıf olduğunu gösterdi.

Bu nedenle v3 input'u:

```text
z_fused + proximity / level-interaction context features
```

## Input Features

Latent:

```text
z_fused
```

Context features:

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

Feature source:

- distance / current label: Stage 1B label context parquet'leri
- close/wick count and final excess: Stage 1A latent export extra arrays

No future label fields were used as features.

Input dimension:

```text
z_fused = 96
context = 13
total = 109
```

## H8 Fresh Break v3

Target:

```text
h8_fresh_break_direction
```

Checkpoint:

```text
data/stage1b/binance/15m/checkpoints/stage1b_h8_fresh_break_v3_z_fused_proximity
```

Training:

```text
best_epoch = 16
break_threshold = 0.57
balanced_break_loss = true
balanced_direction_loss = true
```

### Test Metrics

| Metric | v2 fresh balanced | v3 proximity |
|---|---:|---:|
| reconstructed macro F1 | 0.4472 | 0.5971 |
| break macro F1 | 0.5933 | 0.6978 |
| direction macro F1 | 0.7234 | 0.8412 |
| reconstructed accuracy | - | 0.7586 |
| break accuracy | - | 0.7767 |
| direction accuracy | - | 0.8415 |

v3, fresh-break target'ta v2'yi belirgin şekilde geçti.

### Test Confusion Matrices

Reconstructed labels:

Labels: `none / bullish / bearish`

```text
none    -> [6242,  857,  773]
bullish -> [ 246,  543,   67]
bearish -> [ 289,  108,  570]
```

Break occurrence:

Labels: `none / break`

```text
none  -> [6242, 1630]
break -> [ 535, 1288]
```

Direction:

Labels: `bullish / bearish`

```text
bullish -> [731, 125]
bearish -> [164, 803]
```

### Error Audit

At threshold `0.57`:

| Segment | v2 | v3 |
|---|---:|---:|
| correct | 6510 | 7355 |
| false positive break | 2002 | 1630 |
| false negative break | 910 | 535 |
| wrong direction | 273 | 175 |

High-confidence bucket:

| Bucket | v2 accuracy | v3 accuracy |
|---|---:|---:|
| 0.90-1.00 | 0.9854 | 0.9903 |

v3 yalnızca genel skoru artırmadı; yanlış yön ve false-negative hatalarını da düşürdü.

## Index 454 Recheck

Önceki problemli örnek:

```text
index = 454
timestamp = 2026-01-18 23:00
current_label = intact
target true = bearish
h8_break_semantic_label = fresh_bearish_break
h8_post_break_outcome = bearish_continuation
bear_distance_pct = 0.087%
```

v2:

```text
pred_label = bullish
p_break = 0.637
pred_direction = bullish
direction_confidence = 0.812
error_type = wrong_direction
```

v3:

```text
pred_label = bearish
p_break = 0.694
pred_direction = bearish
direction_confidence = 0.875
error_type = correct
```

Bu örnek, proximity feature'larının hedeflenen hatayı düzelttiğini gösteren doğrudan bir sanity check'tir.

## H16 Dominant Direction v3

Target:

```text
h16_dominant_forward_direction
```

Checkpoint:

```text
data/stage1b/binance/15m/checkpoints/stage1b_h16_dominant_v3_z_fused_proximity
```

Training:

```text
best_epoch = 5
break_threshold = 0.50
balanced_break_loss = true
balanced_direction_loss = true
```

### Test Metrics

| Metric | v3 |
|---|---:|
| reconstructed macro F1 | 0.6701 |
| break macro F1 | 0.7103 |
| direction macro F1 | 0.8525 |
| reconstructed accuracy | 0.6815 |
| break accuracy | 0.7103 |
| direction accuracy | 0.8540 |

Reconstructed labels:

Labels: `none / bullish / bearish`

```text
none    -> [3505,  544,  562]
bullish -> [ 738, 1437,  137]
bearish -> [ 962,  142, 1660]
```

Direction:

Labels: `bullish / bearish`

```text
bullish -> [1913,  399]
bearish -> [ 342, 2422]
```

Yorum:

H16 dominant target'ta direction head güçlü. Bu target, Stage 2 için first-break yönünden daha trade-relevant olabilir; fakat break occurrence tarafında hâlâ ciddi false positive / false negative var.

## Değerlendirme

v3 hipotezi desteklendi.

Proximity / level-interaction context, Stage 1B predictor kalitesini belirgin artırdı. Bu sonuç, Stage 1A latent'in faydalı olduğunu ama Stage 1B karar katmanının explicit price-level context'ten yararlandığını gösteriyor.

Önemli ayrım:

```text
Stage 1A temsil hedefi: structural latent öğrenmek
Stage 1B karar hedefi: latent + structural context ile forward setup okumak
```

Bu nedenle explicit context feature kullanımı Stage 1A başarısını ikame etmiyor; Stage 1B decision module'ı zenginleştiriyor.

## Sonraki Adımlar

1. H8 v3 için threshold sweep yapılmalı.
2. H8 v3 proximity audit arayüzde incelenmeli.
3. Stage 1B output contract, `first_break_signal` ve `dominant_direction_signal` ayrımıyla güncellenmeli.
4. Stage 2'ye geçmeden önce, v3 çıktılarının TradePlan açısından hangi sinyali temsil ettiği netleştirilmeli:
   - fresh trigger sinyali
   - dominant forward direction
   - no-trade / wait için kullanılacak confidence ve context alanları

## Artifacts

```text
src/msb_repr/stage1b/dataset.py
scripts/train_stage1b_predictor.py
scripts/audit_stage1b_predictor_errors.py

data/stage1b/binance/15m/checkpoints/stage1b_h8_fresh_break_v3_z_fused_proximity/
data/stage1b/binance/15m/checkpoints/stage1b_h16_dominant_v3_z_fused_proximity/

data/stage1b/binance/15m/checkpoints/stage1b_h8_fresh_break_v3_z_fused_proximity/error_audit_test_t057/
data/stage1b/binance/15m/checkpoints/stage1b_h8_fresh_break_v3_z_fused_proximity/proximity_audit_test_t057/
```

## Verification

```bash
python -m py_compile src/msb_repr/stage1b/dataset.py scripts/train_stage1b_predictor.py scripts/audit_stage1b_predictor_errors.py
ruff check src/msb_repr/stage1b/dataset.py scripts/train_stage1b_predictor.py scripts/audit_stage1b_predictor_errors.py
```
