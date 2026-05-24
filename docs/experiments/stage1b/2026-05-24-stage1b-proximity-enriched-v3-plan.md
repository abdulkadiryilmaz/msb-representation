# 2026-05-24 - Stage 1B Proximity-Enriched Predictor v3 Plan

## Amaç

Stage 1B fresh-break v2 ve event-sequence v2 incelemelerinde bazı pencerelerde modelin break occurrence sinyalini yakaladığı, fakat yönü ters okuduğu görüldü.

Örnek:

```text
index = 454
current_label = intact
target true = bearish
predictor decision = bullish
bear_distance_pct = 0.087%
```

Stage 1A latent proximity probe v1 şunu gösterdi:

- nearest level side latentten okunabiliyor: `z_fused` linear macro F1 `0.7644`
- nearest distance bucket daha zayıf: `z_fused` linear macro F1 `0.4295`
- wick breach side orta güçte okunuyor: `z_fused` MLP macro F1 `0.5953`

Bu sonuç, proximity bilgisinin latentte kısmen var olduğunu fakat Stage 1B predictor'ın bunu yeterince güvenilir kullanmadığını düşündürüyor.

## Hipotez

Frozen Stage 1A latent'ine explicit proximity / level-interaction feature'ları eklemek, Stage 1B forward direction kararlarını iyileştirir.

```text
v2 input = z_fused
v3 input = z_fused + proximity_features
```

Bu çalışma Stage 1A representation başarısını yeniden ölçmez. Stage 1B karar modülünün, frozen latent yanında explicit structural context kullanıp kullanmamasını test eder.

## Feature Set

İlk v3 feature set:

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

Ölçekleme:

- distance / excess alanları `* 100`
- close/wick count alanları `/ 12`
- nearest side: `bull = +1`, `bear = -1`
- current label one-hot

Bu feature'lar predictor'a “hangi level yakın?”, “ne kadar yakın?”, “wick/close interaction var mı?” ve “mevcut Stage 1A state ne?” bilgisini açık verir.

## İlk Target'lar

İki ayrı deney:

1. `h8_fresh_break_direction`
   - immediate fresh-break target
   - Stage 2 trigger tarafına en yakın sinyal

2. `h16_dominant_forward_direction`
   - event-sequence v2 ile bulunan daha trade-relevant yön sinyali
   - first-break sonrası continuation/reversal davranışını daha iyi yakalayabilir

## Karşılaştırma

Her target için minimum karşılaştırma:

| Variant | Input |
|---|---|
| baseline | `z_fused` |
| v3 | `z_fused + proximity_features` |

Başarı kriterleri:

- reconstructed macro F1 artmalı
- direction macro F1 artmalı
- wrong-direction segment azalmalı
- near-bear-trigger / near-bull-trigger yanlış yön örnekleri özel audit'te azalmalı

## Kabul / Red

Accept:

- v3, aynı target üzerinde `z_fused` baseline'ı anlamlı biçimde geçerse
- özellikle wrong-direction ve near-trigger hataları azalırsa

Reject:

- genel macro F1 artışı yoksa
- improvement sadece no-break / majority davranışından geliyorsa
- proximity feature'ları false-positive break oranını belirgin artırıyorsa

## Not

Bu çalışma Stage 2 değildir. Stage 2 hâlâ şu sorunun sahibidir:

```text
Bu predicted setup işleme alınabilir mi, yoksa no_trade / wait mi?
```

Stage 1B v3 yalnızca Stage 2'ye daha doğru forward structural signal vermeyi hedefler.
