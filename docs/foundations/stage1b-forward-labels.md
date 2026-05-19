# Stage 1B Forward Labels

**Status**: Draft v1
**Scope**: Stage 1B research labels for forward structure forecasting
**Primary timeframe**: 15m
**Initial horizons**: `H = 16, 32, 48`

Bu belge Stage 1B için ilk forward label sözleşmesini tanımlar. Amaç, Stage 1A temsilinden yakın gelecekteki yapısal akışın okunup okunamadığını ölçmek ve ileride eğitilecek Stage 1B predictor için aynı label semantiğini sabitlemektir.

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

---

## What This Label Does Not Mean

`future_break_direction_H = bullish` şu anlama gelmez:

```text
long trade alınmalı
```

Bu label yalnızca horizon içinde ilk structural break yönünü söyler. Şunları ölçmez:

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
