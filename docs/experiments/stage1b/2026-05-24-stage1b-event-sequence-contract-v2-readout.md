# 2026-05-24 - Stage 1B Event-Sequence Contract v2 Readout

## Amaç

Stage 1B output contract'ı, yalnızca `fresh_break_direction` üretmek yerine event-sequence semantiğini taşıyacak şekilde güncellendi.

Motivasyon:

```text
fresh_break_signal != trade-relevant structural event
```

Özellikle already-broken yapılarda, fresh-break target `none` olabilir; fakat forward path bir continuation veya reversal olabilir.

## Contract Update

Güncellenen foundation:

```text
docs/foundations/stage1b-output-contract.md
```

Yeni ana alanlar:

```text
event_type:
  no_event
  fresh_break
  continuation
  reversal

event_direction:
  none
  bullish
  bearish
```

Bu alanlar Stage 2'ye çıplak direction yerine event nedeniyle birlikte aktarılacak.

## Label Generator Update

`scripts/generate_stage1b_forward_labels.py` artık her horizon için şu alanları üretir:

```text
h{H}_event_type
h{H}_event_direction
```

Mapping:

| Source | event_type | event_direction |
|---|---|---|
| `fresh_bullish_break` | `fresh_break` | `bullish` |
| `fresh_bearish_break` | `fresh_break` | `bearish` |
| `bullish_continuation` | `continuation` | `bullish` |
| `bearish_continuation` | `continuation` | `bearish` |
| `bullish_to_bearish_reversal` | `reversal` | `bearish` |
| `bearish_to_bullish_reversal` | `reversal` | `bullish` |
| `failed_bullish_break` / `failed_bearish_break` | `no_event` | `none` |

`failed_*` outcome'ları v2'de konservatif olarak `no_event` altında tutuldu.

## Regenerated Artifacts

```text
data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/stage1b_forward_labels_train_h4_h8_h16_v2.parquet
data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/stage1b_forward_labels_val_h4_h8_h16_v2.parquet
data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/stage1b_forward_labels_test_h4_h8_h16_v2.parquet
```

## Label Distribution

### H8 event_type

| Split | no_event | fresh_break | continuation | reversal | ambiguous | insufficient_future |
|---|---:|---:|---:|---:|---:|---:|
| train | 66688 | 24648 | 39399 | 300 | 5 | 0 |
| val | 4235 | 1715 | 2849 | 31 | 2 | 0 |
| test | 4849 | 1823 | 2998 | 25 | 1 | 8 |

### H16 event_type

| Split | no_event | fresh_break | continuation | reversal | ambiguous | insufficient_future |
|---|---:|---:|---:|---:|---:|---:|
| train | 51818 | 40675 | 36906 | 1633 | 8 | 0 |
| val | 3286 | 2758 | 2649 | 137 | 2 | 0 |
| test | 3690 | 3041 | 2844 | 112 | 1 | 16 |

Yorum:

- H8 reversal sınıfı çok seyrek. Tek başına 4-class event_type predictor için sınıf dengesizliği ciddi risk.
- H16 reversal sınıfı daha anlamlı sayıda. Reversal / continuation çalışması için H16 daha uygun ilk aday.
- H8 hâlâ near-trigger / fresh-break için daha uygun.

## Sanity Checks

### Index 453

```text
current_label = bullish
h8_fresh_break_direction = none
h8_post_break_outcome = bullish_to_bearish_reversal
h8_dominant_forward_direction = bearish
h8_event_type = reversal
h8_event_direction = bearish
```

Bu örnek artık fresh-break `none` olarak kalırken, event-sequence contract altında doğru şekilde `reversal / bearish` olarak görünür.

### Index 454

```text
current_label = intact
h8_fresh_break_direction = bearish
h8_post_break_outcome = bearish_continuation
h8_dominant_forward_direction = bearish
h8_event_type = fresh_break
h8_event_direction = bearish
```

Bu örnek, intact yapıdan fresh bearish break olarak ayrışır.

## UI Update

`scripts/inspect_stage1b.py` içinde label context başlığı:

```text
Event-sequence context
```

olarak güncellendi ve `event_type` / `event_direction` alanları varsa gösterilecek.

## Sonuç

Stage 1B'nin Stage 2'ye taşıması gereken bilgi artık daha net:

```text
fresh trigger mı?
continuation mı?
reversal mı?
no_event mi?
```

Bu, Stage 2 actionability tasarımında `actionable_break_candidate`, `actionable_reversal_candidate` ve `actionable_continuation_candidate` ayrımının temelini oluşturur.

## Sonraki Adım

İlk modelleme adımı:

```text
H16 event-sequence predictor v1
```

Önerilen hedef:

```text
h16_event_type
h16_event_direction
```

H8 için ise fresh-break v3 predictor korunmalı; H8 reversal sınıfı şu an çok seyrek.
