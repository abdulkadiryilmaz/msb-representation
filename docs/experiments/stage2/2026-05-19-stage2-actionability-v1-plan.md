# Stage 2 Actionability v1 Plan

**Date**: 2026-05-19
**Status**: initial label/audit completed
**Foundation**: `docs/foundations/stage2-actionability-framing.md`
**Stage 1B source**: `stage1b_h8_predictor_v1a_z_fused`

## Amaç

Bu deneyin amacı Stage 1B H8 predictor çıktısını doğrudan trade sinyali yapmak değildir.

Amaç, Stage 2 için ilk actionability label/proxy setini üretmek ve şu soruyu test edilebilir hale getirmektir:

> Stage 1B'nin işaret ettiği yakın break, uygulanabilir TradePlan adayına dönüşebilecek kalite taşıyor mu?

## Neden Bu Deney Gerekli?

Stage 1B şu bilgiyi verebiliyor:

- H8 içinde break olasılığı
- yön
- confidence
- break seviyesine yakınlık

Ama Stage 1B şunları söylemiyor:

- şu fiyattan gir
- şu seviyede stop koy
- şu target mantıklı
- bu yapı trade edilmeye değer
- bekle veya no-trade

Stage 2 v1 bu boşluğu ilk defa ölçülebilir hale getirecek.

## Input

Bir örnek için beklenen input:

```text
symbol
timestamp
current_price
Stage1BSignal:
  side
  p_break
  p_direction
  break_confidence
  direction_confidence
Stage1BContext:
  bull_level
  bear_level
  distance_to_bull_level
  distance_to_bear_level
  predicted_side_distance_to_level
future OHLCV path
```

İlk versiyonda model eğitmeden önce bu inputlardan label üretilecek.

## Output Label v1

Hedef actionability sınıfları:

```text
0 = no_trade
1 = wait_for_break
2 = wait_for_retest
3 = actionable_break_candidate
```

Bu sınıflar TradePlan değildir. TradePlan'a geçiş için ara karar katmanıdır.

## Label Generation Sketch

### 1. Candidate side seçimi

Stage 1B `side=none` ise örnek default olarak `no_trade` adayıdır.

Stage 1B `side=long` ise:

```text
trigger_level = bull_level
```

Stage 1B `side=short` ise:

```text
trigger_level = bear_level
```

### 2. Distance filter

İlk rule-based eşik:

```text
near_trigger = abs(current_price - trigger_level) / current_price <= distance_threshold
```

Başlangıç sweep adayları:

```text
0.25%, 0.50%, 1.00%, 1.50%
```

### 3. Break confirmation

Beklenen yönde close-confirmed break:

```text
long: future close > trigger_level * (1 + effective_break_pct)
short: future close < trigger_level * (1 - effective_break_pct)
```

### 4. Invalidation proxy

V1 için rule-based invalidation:

```text
long: recent structural low veya bear_level fallback
short: recent structural high veya bull_level fallback
```

Fallback gerekiyorsa volatility-scaled stop kullanılabilir.

### 5. MFE / MAE proxy

Break veya anchor sonrası path için:

```text
MFE = expected direction max favorable excursion
MAE = expected direction max adverse excursion
```

İlk kabul eşiği:

```text
MFE / MAE veya target_R >= min_rr_proxy
```

Başlangıç sweep adayları:

```text
1.0R, 1.5R, 2.0R
```

## İlk Karar Kuralları

```text
no_trade:
  low confidence
  veya trigger çok uzak
  veya expected break yok
  veya adverse path önce invalidation'a gidiyor
  veya RR proxy yetersiz

wait_for_break:
  trigger yakın
  ve future path beklenen yönde break gösteriyor
  fakat anchor anında break yok

wait_for_retest:
  trigger kırılmış veya çok yakın
  fakat immediate entry chase riski taşıyor
  ve path retest/hold davranışı gösteriyor

actionable_break_candidate:
  confidence yeterli
  trigger yakın veya kırılmış
  invalidation'dan önce target/R proxy sağlanıyor
```

## İlk Audit

Model eğitmeden önce label audit zorunlu:

- class distribution
- symbol distribution
- confidence bucket distribution
- distance bucket distribution
- MFE/MAE summary by class
- no_trade seçilen Stage 1B false-positive segmenti azalıyor mu?
- actionable seçilen örneklerde future path kalitesi daha iyi mi?

## Başarı Kriteri

Bu deney başarılı sayılırsa:

- label dağılımı tek sınıfa çökmez
- high-confidence Stage 1B segmentinde daha yüksek actionable oranı çıkar
- actionable sınıfı MFE/MAE açısından no_trade'den daha iyi ayrışır
- false positive break segmentinin önemli kısmı no_trade veya wait sınıflarına düşer

Bu sağlanmadan Stage 2 predictor eğitmek erken olur.

## İlk Implementasyon Adımları

1. `generate_stage2_actionability_labels.py`
2. `audit_stage2_actionability_labels.py`
3. Stage 2 v1 readout
4. Gerekirse rule threshold sweep
5. Ancak sonra baseline predictor / probe

## Guardrails

- Bu deney PnL optimizasyonu değildir.
- Bu deney coin-oracle execution mantığı içermez.
- Stage 1A / Stage 1B başarısını doğrudan geçersiz kılmaz.
- Stage 2 label'ı rule-based başladığı için ilk sonuçlar label kalitesiyle birlikte okunmalıdır.
