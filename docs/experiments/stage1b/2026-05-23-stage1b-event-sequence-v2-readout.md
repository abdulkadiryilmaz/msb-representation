# Stage 1B Event Sequence v2 Readout

**Tarih**: 2026-05-23
**Durum**: Completed
**Amaç**: `first_break_direction` hedefinin TradePlan için yetersiz kaldığı örneklerden hareketle Stage 1B label sözleşmesine post-break outcome ve dominant forward direction alanlarını eklemek.

---

## Motivation

Görsel incelemede örnek bir H8 pencerede modelin bullish break'i doğru yakaladığı, fakat trade açısından daha önemli olayın hemen ardından gelen bearish reversal olduğu görüldü.

Bu örnek şu ayrımı netleştirdi:

```text
first_break_direction = bullish
trade-relevant scenario = bullish break failed, bearish reversal
```

Bu nedenle Stage 1B'nin yalnızca ilk break yönünü değil, event sonrası structural outcome bilgisini de üretmesi gerekiyor.

---

## Label Additions

`scripts/generate_stage1b_forward_labels.py` mevcut v1/v2 kolonlarını koruyarak her horizon için şu alanları ekler:

```text
h{H}_post_break_outcome
h{H}_dominant_forward_direction
```

Outcome değerleri:

```text
none
bullish_continuation
bearish_continuation
failed_bullish_break
failed_bearish_break
bullish_to_bearish_reversal
bearish_to_bullish_reversal
ambiguous
insufficient_future
```

Dominant direction değerleri:

```text
none
bullish
bearish
ambiguous
insufficient_future
```

Tanım:

- `*_continuation`: ilk break sonrası horizon final close hâlâ break tarafında ve opposite confirm yok
- `failed_*_break`: ilk break geliyor ama horizon sonunda ilgili break level korunmuyor; opposite confirm yok
- `*_to_*_reversal`: ilk break sonrası karşı yönde close-confirmed break geliyor

---

## Reference Example

İncelenen örnek:

```text
index: 12
symbol: BTC_USDT_15m
timestamp: 2025-12-31 13:00
```

Yeni label okuması:

| Horizon | First break | Bull confirm | Bear confirm | Outcome | Dominant |
|---|---|---:|---:|---|---|
| H4 | bullish | 4 | -1 | bullish_continuation | bullish |
| H8 | bullish | 4 | -1 | failed_bullish_break | none |
| H16 | bullish | 4 | 9 | bullish_to_bearish_reversal | bearish |

Bu artık görsel yorumla uyumlu:

```text
İlk event bullish break.
Kısa horizon içinde bullish devam zayıflıyor.
Daha geniş H16 içinde bearish reversal confirm oluyor.
```

---

## Distribution Snapshot

Test split H8:

| Outcome | Count |
|---|---:|
| none | 4334 |
| bearish_continuation | 2298 |
| bullish_continuation | 2014 |
| failed_bearish_break | 515 |
| failed_bullish_break | 496 |
| bullish_to_bearish_reversal | 24 |
| bearish_to_bullish_reversal | 14 |
| ambiguous | 1 |
| insufficient_future | 8 |

Test split H16:

| Outcome | Count |
|---|---:|
| none | 3025 |
| bearish_continuation | 2641 |
| bullish_continuation | 2216 |
| failed_bearish_break | 810 |
| failed_bullish_break | 776 |
| bullish_to_bearish_reversal | 123 |
| bearish_to_bullish_reversal | 96 |
| ambiguous | 1 |
| insufficient_future | 16 |

Yorum: Reversal örnekleri H8 içinde nadir, H16 içinde daha görünür. Bu beklenen bir sonuç; ilk break sonrası karşı yönde confirmed structure oluşması daha fazla bar gerektiriyor.

---

## Dominant Direction Probe

Target:

```text
h16_dominant_forward_direction
```

Embedding:

```text
z_fused
```

Dropped:

```text
ambiguous
insufficient_future
```

| Split | Model | Macro F1 | Accuracy | Majority Macro F1 |
|---|---|---:|---:|---:|
| Val | linear | 0.6023 | 0.5998 | 0.2141 |
| Val | MLP | 0.6148 | 0.6365 | 0.2141 |
| Test | linear | 0.6181 | 0.6181 | 0.2150 |
| Test | MLP | 0.6169 | 0.6423 | 0.2150 |

Bu, `dominant_forward_direction` hedefinin fresh-break hedefinden belirgin şekilde daha okunabilir olduğunu gösteriyor.

---

## Interpretation

Stage 1B için daha doğru sözleşme:

```text
first_break_direction:
  İlk close-confirmed structural event hangi yönde?

post_break_outcome:
  İlk event continuation mı, failed break mi, opposite reversal mı?

dominant_forward_direction:
  TradePlan açısından forward path hangi yöne ağırlık veriyor?
```

Bu ayrım Stage 2'nin görevini daha doğru sınırlar.

Stage 2 artık ilk break yönünden trade yönü çıkarmamalı. Bunun yerine Stage 1B'den gelen scenario bilgisini actionability'ye çevirmeli:

```text
bullish_continuation -> long actionability adayı
failed_bullish_break -> long no-trade / early short-rejection adayı
bullish_to_bearish_reversal -> short reversal adayı
```

---

## Decision

Stage 1B output contract v2 şu modları desteklemeli:

```text
first_break_direction
break_anchor_status
post_break_outcome
dominant_forward_direction
confidence fields
level context
```

Stage 2 actionability tasarımı bu sözleşmeye göre revize edilmeli.

---

## Next

1. `stage1b-output-contract.md` v2 olarak güncellenmeli.
2. Predictor tarafında `h16_dominant_forward_direction` veya multi-head event-sequence hedefi denenmeli.
3. Stage 2 label tasarımı continuation / failed-break / reversal modlarına ayrılmalı.
