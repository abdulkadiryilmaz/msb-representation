# Stage 1B H8 Fresh Break Semantics Audit v1

**Tarih**: 2026-05-22
**Durum**: Completed
**Amaç**: H8 `future_break_direction` label'ının gerçekten fresh break mi, yoksa anchor anında zaten break seviyesinin ötesinde olan continuation örneklerini mi temsil ettiğini ölçmek.

---

## Soru

Stage 2 görsel incelemesinde bazı `actionable_break_candidate` örneklerinde fiyatın, short yön için `bear_level` altında veya long yön için `bull_level` üstünde olduğu görüldü.

Bu şu soruyu doğurdu:

> Stage 1B `future_break_direction_H` label'ı, anchor sonrasında oluşan yeni bir kırılımı mı söylüyor, yoksa anchor anında zaten kırılmış olan yapının horizon içinde devam ettiğini de aynı sınıfa mı koyuyor?

Bu ayrım TradePlan açısından kritik. Çünkü:

- fresh break: henüz trigger gelmemiştir, giriş trigger seviyesinden planlanabilir
- already-broken / continuation: trigger geçmiştedir, aynı entry proxy artık geçerli değildir

---

## Method

Audit script:

```text
scripts/audit_stage1b_fresh_break_semantics.py
```

Input label artifact:

```text
data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/stage1b_forward_labels_test_h4_h8_h16.parquet
```

Input prediction artifact:

```text
data/stage1b/binance/15m/checkpoints/stage1b_h8_predictor_v1a_z_fused/proximity_audit_test/predictions_with_proximity.csv
```

Output:

```text
data/stage1b/binance/15m/stage1b_h8_fresh_break_semantics_audit_test/
```

Status tanımı:

```text
bullish fresh_candidate:
  current_close < bull_level

bullish already_broken:
  current_close >= bull_level

bearish fresh_candidate:
  current_close > bear_level

bearish already_broken:
  current_close <= bear_level
```

`none`, `ambiguous`, `insufficient_future` örnekleri directional audit dışında tutuldu.

---

## Label Semantics Result

H8 label artifact:

| Metric | Value |
|---|---:|
| Total support | 9704 |
| Directional support | 5361 |
| Fresh candidate | 1823 |
| Already broken | 3538 |
| Fresh candidate rate | 34.00% |
| Already broken rate | 66.00% |

Direction split:

| Label | Fresh candidate | Already broken |
|---|---:|---:|
| bullish | 856 | 1678 |
| bearish | 967 | 1860 |

Sonuç: mevcut H8 directional label'larının yaklaşık üçte ikisi fresh break değil. Anchor anında fiyat zaten ilgili break seviyesinin ötesinde.

---

## Prediction Semantics Result

Stage 1B H8 predictor output:

| Metric | Value |
|---|---:|
| Total support | 9695 |
| Directional prediction support | 4755 |
| Fresh candidate predictions | 1335 |
| Already broken predictions | 3392 |
| Missing level | 28 |
| Fresh candidate rate | 28.08% |
| Already broken rate | 71.34% |

Direction split:

| Pred label | Fresh candidate | Already broken | Missing level |
|---|---:|---:|---:|
| bullish | 688 | 1565 | 11 |
| bearish | 647 | 1827 | 17 |

Sonuç: predictor da çoğunlukla already-broken / continuation durumlarını directional break olarak işaretliyor. Bu sadece predictor hatası değil; label semantiğinin eğitim hedefi haline getirdiği şey bu.

---

## Interpretation

Mevcut Stage 1B label v1 şu soruyu cevaplıyor:

> H8 içinde fiyat, ilgili break seviyesinin ötesinde close-confirmed durumda mı?

Fakat TradePlan'a daha yakın olan soru şudur:

> Anchor anında fiyat henüz pre-break taraftayken, H8 içinde yeni bir close-confirmed break oluşuyor mu?

Bu iki soru aynı değil.

Stage 1B v1 sonuçları bu yüzden tamamen geçersiz değildir. H8 predictor gerçekten yakın horizon içinde directional structural state'i yakalıyor. Ancak bu state, fresh trigger ile continuation'ı ayırmadığı için Stage 2 entry proxy'sine doğrudan bağlanamaz.

---

## Decision

Mevcut H8 predictor v1a, Stage 2 için tek başına `break trigger oluşacak` sinyali olarak kullanılmamalı.

Stage 1B label sözleşmesi v2'de anchor-side semantiği açıkça ayrılmalı:

```text
fresh_bullish_break
fresh_bearish_break
already_broken_bullish
already_broken_bearish
none
ambiguous
insufficient_future
```

Alternatif olarak ana direction label korunup ek bir alan eklenebilir:

```text
h{H}_break_anchor_status:
  fresh_candidate
  already_broken
  not_directional
```

Stage 2 ise `already_broken` örneklerde trigger-level entry proxy'si kullanmamalı; ayrı continuation/retest mantığı gerektirir.

---

## Next

1. `stage1b-forward-labels.md` içine v1 caveat ve v2 direction yazılmalı.
2. `generate_stage1b_forward_labels.py` v2 için anchor-side alanları üretecek şekilde güncellenmeli.
3. H8 predictor, fresh-break hedefi veya multi-head fresh/continuation hedefiyle yeniden denenmeli.
4. Stage 2 actionability label v1, `already_broken` durumunu ayrı ele alacak şekilde revize edilmeli.
