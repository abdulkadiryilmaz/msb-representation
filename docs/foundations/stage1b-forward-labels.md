# Stage 1B Forward Labels

**Status**: Draft v2, event-sequence fields active
**Scope**: Stage 1B research labels for forward structure forecasting
**Primary timeframe**: 15m
**Initial horizons**: `H = 16, 32, 48`

Bu belge Stage 1B için forward label sözleşmesini tanımlar. Amaç, Stage 1A temsilinden yakın gelecekteki yapısal akışın okunup okunamadığını ölçmek ve Stage 1B predictor için aynı label semantiğini sabitlemektir.

Not: 2026-05-22 H8 fresh-break semantics audit'i, v1 label'ın fresh break ile already-broken / continuation durumlarını aynı directional sınıfta topladığını gösterdi. Bu nedenle v1 sonuçları forward structural state okuması olarak değerlidir, fakat doğrudan "henüz gelmemiş trigger oluşacak" anlamında kullanılmamalıdır.

Not 2: 2026-05-23 event-sequence readout'u, ilk break yönünün TradePlan için tek başına yeterli olmadığını gösterdi. Stage 1B v2, ilk break sonrasındaki continuation / failed break / reversal outcome bilgisini de taşımalıdır.

Bu label seti TradePlan label'ı değildir. Entry, stop, target, leverage, position sizing veya PnL sonucu üretmez.

---

## Stage 1B Sorusu

Stage 1B'nin bu aşamadaki sorusu:

> Mevcut anchor'daki market structure, yakın gelecekte ilk olarak hangi yönde close-confirmed break'e evriliyor?

Bu soru TradePlan'ın tamamını değil, TradePlan'a giden yön / forward-structure bileşenini test eder.

---

## Primary Label

Ana label:

```text
future_break_direction_H
```

İlk horizonlar:

```text
h16_future_break_direction
h32_future_break_direction
h48_future_break_direction
```

15m timeframe'de yaklaşık karşılıkları:

| Horizon | Süre |
|---|---:|
| H16 | 4 saat |
| H32 | 8 saat |
| H48 | 12 saat |

---

## Label Values

V1 label değerleri:

```text
none
bullish
bearish
ambiguous
insufficient_future
```

Anlamları:

| Value | Anlam |
|---|---|
| `none` | Horizon içinde close-confirmed bullish veya bearish break yok |
| `bullish` | İlk close-confirmed break bullish yönde |
| `bearish` | İlk close-confirmed break bearish yönde |
| `ambiguous` | Bullish ve bearish confirmation aynı veya çok yakın pencerede geliyor |
| `insufficient_future` | Anchor sonrasında horizon kadar bar yok |

Probe / predictor eğitiminde `insufficient_future` her zaman dışarıda bırakılmalıdır.

`ambiguous` sınıfı v1'de üretilir fakat çok nadir olabilir. İlk probe okumalarında ana metrik çoğunlukla 3-class olarak okunmalıdır:

```text
none / bullish / bearish
```

---

## Level Construction

Her anchor için Stage 1A ile aynı structure-level mantığı kullanılır.

Girdi:

- Stage 1A short window
- Stage 1A label config:
  - `structure_bars`
  - `recent_bars`
  - `pivot_n`
  - `min_break_pct`
  - `atr_factor`
  - `min_recent_break_bars`

Anchor seviyesinde:

```text
struct_highs = short_window.high[:structure_bars]
struct_lows  = short_window.low[:structure_bars]
ref_close    = short_window.close[structure_bars - 1]
atr          = mean(struct_highs - struct_lows)
effective_break_pct = max(min_break_pct, atr_factor * atr / ref_close)
```

Son confirmed pivot seviyeleri:

```text
swing_high = last pivot high inside struct_highs
swing_low  = last pivot low inside struct_lows
```

Break seviyeleri:

```text
bull_level = swing_high * (1 + effective_break_pct)
bear_level = swing_low  * (1 - effective_break_pct)
```

Eğer ilgili swing seviyesi yoksa o yönde break aranmaz.

---

## Forward Confirmation

Anchor timestamp'inden sonraki future close serisi kullanılır:

```text
future_closes = close[t + 1 : t + H]
```

Bullish confirmation:

```text
future_close > bull_level
```

Bearish confirmation:

```text
future_close < bear_level
```

V1'de bu confirmation yalnızca future close'lara bakar. Anchor anındaki `current_close` değerinin pre-break tarafta olup olmadığı kontrol edilmez.

Bu yüzden v1'de şu iki durum aynı label'a düşebilir:

```text
fresh break:
  current_close pre-break tarafta
  future_close break seviyesini sonra geçiyor

already-broken / continuation:
  current_close zaten break seviyesinin ötesinde
  future_close horizon içinde orada kalıyor veya devam ediyor
```

TradePlan'a yakın semantik için bu ayrım gereklidir. Fresh break için ek koşul:

```text
bullish fresh candidate:
  current_close < bull_level
  future_close > bull_level

bearish fresh candidate:
  current_close > bear_level
  future_close < bear_level
```

Bir direction'ın confirm olması için horizon içinde en az:

```text
min_recent_break_bars
```

adet close hit'i gerekir.

Confirmation bar:

```text
ilk min_recent_break_bars'inci close-hit barı
```

Örnek: `min_recent_break_bars = 2` ise bullish confirmation barı, horizon içindeki ikinci `future_close > bull_level` barıdır.

---

## Direction Assignment

Her horizon için:

```text
bull_confirm_bar
bear_confirm_bar
```

hesaplanır.

Atama:

```text
if neither exists:
    none
elif only bull exists:
    bullish
elif only bear exists:
    bearish
else:
    if abs(bull_confirm_bar - bear_confirm_bar) <= ambiguity_bars:
        ambiguous
    elif bull_confirm_bar < bear_confirm_bar:
        bullish
    else:
        bearish
```

V1 default:

```text
ambiguity_bars = 1
```

---

## Secondary Fields

Label artifact'ında ana label yanında şu alanlar da tutulur:

```text
h{H}_future_break_occurs
h{H}_time_to_break
h{H}_bull_confirm_bar
h{H}_bear_confirm_bar
h{H}_bull_close_count
h{H}_bear_close_count
```

Bunlar Stage 1B predictor ve Stage 2 tasarımında yardımcı olabilir, fakat v1 ana hedefi değildir.

V2 için eklenmesi gereken alan:

```text
h{H}_break_anchor_status
```

Önerilen değerler:

| Value | Anlam |
|---|---|
| `fresh_candidate` | Anchor pre-break tarafta; horizon içinde yeni break oluşuyor |
| `already_broken` | Anchor anında fiyat zaten ilgili break seviyesinin ötesinde |
| `not_directional` | Directional break label yok |

Alternatif daha güçlü sözleşme, ana label'ı doğrudan şu sınıflara bölmektir:

```text
fresh_bullish_break
fresh_bearish_break
already_broken_bullish
already_broken_bearish
none
ambiguous
insufficient_future
```

V2 event-sequence alanları:

```text
h{H}_post_break_outcome
h{H}_dominant_forward_direction
h{H}_event_type
h{H}_event_direction
```

`post_break_outcome` değerleri:

| Value | Anlam |
|---|---|
| `bullish_continuation` | İlk break bullish; horizon sonunda bullish break tarafı korunuyor; bearish confirm yok |
| `bearish_continuation` | İlk break bearish; horizon sonunda bearish break tarafı korunuyor; bullish confirm yok |
| `failed_bullish_break` | İlk break bullish; horizon sonunda bull level korunmuyor; bearish confirm yok |
| `failed_bearish_break` | İlk break bearish; horizon sonunda bear level korunmuyor; bullish confirm yok |
| `bullish_to_bearish_reversal` | İlk break bullish; sonra bearish close-confirmed break geliyor |
| `bearish_to_bullish_reversal` | İlk break bearish; sonra bullish close-confirmed break geliyor |
| `none` | Horizon içinde directional break yok |
| `ambiguous` | Direction assignment ambiguous |
| `insufficient_future` | Horizon kadar future bar yok |

`dominant_forward_direction` değerleri:

```text
none / bullish / bearish / ambiguous / insufficient_future
```

Bu alan TradePlan yönü değildir. Ancak Stage 2'ye, ilk event'in devam mı ettiği, başarısız mı olduğu, yoksa karşı yöne mi döndüğü hakkında daha trade-relevant forward structure sinyali verir.

`event_type` değerleri:

| Value | Anlam |
|---|---|
| `no_event` | Horizon içinde trade-relevant event yok veya failed break outcome'u konservatif olarak trade event'e dönüştürülmedi |
| `fresh_break` | Anchor pre-break tarafta; horizon içinde yeni close-confirmed break oluşuyor |
| `continuation` | Anchor zaten broken veya ilk event sonrası aynı yön korunuyor |
| `reversal` | Anchor/ilk event sonrası karşı yönde close-confirmed break geliyor |
| `ambiguous` | Direction assignment ambiguous |
| `insufficient_future` | Horizon kadar future bar yok |

`event_direction` değerleri:

```text
none / bullish / bearish / ambiguous / insufficient_future
```

Örnek mapping:

| Durum | event_type | event_direction |
|---|---|---|
| `fresh_bullish_break` | `fresh_break` | `bullish` |
| `fresh_bearish_break` | `fresh_break` | `bearish` |
| `bullish_continuation` | `continuation` | `bullish` |
| `bearish_continuation` | `continuation` | `bearish` |
| `bullish_to_bearish_reversal` | `reversal` | `bearish` |
| `bearish_to_bullish_reversal` | `reversal` | `bullish` |
| `failed_bullish_break` / `failed_bearish_break` | `no_event` | `none` |

---

## What This Label Does Not Mean

`future_break_direction_H = bullish` şu anlama gelmez:

```text
long trade alınmalı
```

Bu label yalnızca horizon içinde ilk structural break yönünü söyler. Şunları ölçmez:

- anchor anında trigger seviyesinin henüz gelmemiş olup olmadığını
- ilk break sonrası continuation / failed break / reversal davranışını
- entry kalitesi
- stop / invalidation seviyesi
- target seviyesi
- risk / reward
- post-break continuation
- rejection / failed break
- MFE / MAE profili
- fees / slippage sonrası edge

Bu nedenle Stage 1B direction label'ı Stage 2 tradability label'ı değildir.

---

## Initial Readout

İlk v1 probe readout:

- `docs/experiments/stage1b/2026-05-18-stage1b-forward-direction-probe-v1-readout.md`

Özet:

| Target | Embedding | Linear macro F1 |
|---|---|---:|
| `h16_future_break_direction` 3-class | `z_fused` | 0.6768 |
| `h32_future_break_direction` 3-class | `z_fused` | 0.6183 |
| `h48_future_break_direction` 3-class | `z_fused` | 0.5807 |

Bu sonuçlar Stage 1A temsilinde forward direction sinyalinin okunabilir olduğunu gösterir, fakat predictor veya TradePlan başarısı kanıtı değildir.

---

## Guardrails

Stage 1B predictor'a geçmeden önce bu guardrail'ler korunmalıdır:

1. `insufficient_future` eğitim ve değerlendirme dışında bırakılmalı.
2. `ambiguous` ayrı raporlanmalı; ana metrik 3-class ve 4-class olarak ayrı okunmalı.
3. Horizon bazında label distribution raporlanmalı.
4. Majority baseline her readout'ta yazılmalı.
5. `z_short`, `z_long`, `z_fused` ayrı okunmalı.
6. Val split ile model/label tasarımı seçilmeli; test split final confirmation olarak kullanılmalı.
7. Bu label doğrudan PnL veya tradability metriği gibi okunmamalı.
