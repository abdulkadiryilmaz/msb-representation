# 2026-05-24 - Stage 1B Process Summary

Bu doküman, Stage 1B tarafında Stage 1A downstream aktarımından event-sequence contract'a kadar izlenen yolu kısa ve sıralı şekilde özetler.

Amaç, ileride süreci yeniden değerlendirirken hangi sorudan hangi sonuca geçildiğini hızlıca hatırlamaktır.

## 1. Stage 1A'dan Downstream'e Geçiş

Stage 1A'da seçili temsil modeli:

```text
ce_supcon_long_branch_ce_aux_v1_seed_41_e50/epoch_020.pt
```

Stage 1B için bu modelin frozen latent çıktıları kullanıldı:

```text
z_short
z_long
z_fused
```

Stage 1B'de ana input çoğunlukla:

```text
z_fused
```

Amaç current structure'ı tekrar sınıflamak değil, bu temsilden yakın gelecekteki structural flow'un okunup okunamadığını test etmekti.

## 2. İlk Forward Label: Future Break Direction

İlk Stage 1B sorusu:

```text
Önümüzdeki H bar içinde ilk close-confirmed break hangi yönde?
```

İlk label:

```text
future_break_direction_H
```

Değerler:

```text
none / bullish / bearish
```

Başlangıç horizonları:

```text
H16 / H32 / H48
```

Probe sonuçlarında H16 daha okunabilir çıktı. Daha sonra H4/H8/H16 kısa horizon denemelerinde H8, near-term yapı için daha iyi denge verdi.

## 3. İlk Predictor'lar

Stage 1B predictor iki head ile kuruldu:

```text
break occurrence head
direction head
```

H16 predictor makul sonuç verdi ama break-heavy yapı nedeniyle Stage 2'ye doğrudan taşımak zor görünüyordu.

H8 predictor v1a daha güçlü çıktı:

```text
test reconstructed macro F1 ~= 0.7307
test direction F1 ~= 0.8955
```

Bu aşamada H8, Stage 2 için ilk aday gibi değerlendirildi.

## 4. H8 Error / Proximity Audit

H8 predictor görsel ve sayısal audit ile incelendi.

Önemli bulgu:

```text
yüksek confidence örnekleri çoğunlukla level proximity ile ilişkili
```

Bu, Stage 2 için price/level context ihtiyacını gösterdi.

Ancak bu aşamada hâlâ önemli bir semantik problem vardı: H8 label'ı fresh break ile already-broken continuation örneklerini aynı directional sınıfta topluyordu.

## 5. Fresh Break Ayrımı

Fresh-break tanımı eklendi:

```text
fresh break = anchor henüz kırılım tarafında değilken horizon içinde yeni break oluşması
```

Yeni hedef:

```text
h{H}_fresh_break_direction
```

Bu daha doğru bir trigger semantiği verdi ama problem belirgin şekilde zorlaştı.

H8 fresh-break v2 balanced predictor:

```text
test reconstructed macro F1 ~= 0.4472
```

Yorum:

```text
fresh-trigger hedefi trade'e daha yakın ama okunması daha zor.
```

## 6. Event-Sequence İhtiyacı

Görsel incelemelerde ilk break yönünün TradePlan için tek başına yeterli olmadığı görüldü.

Örnek durum:

```text
ilk event = bullish break
sonraki path = bearish reversal
```

TradePlan açısından asıl önemli bilgi çoğu zaman ilk break yönü değil, ilk event sonrası yapının ne yaptığıdır:

```text
continuation mı?
failed break mi?
reversal mı?
```

Bu nedenle ek alanlar üretildi:

```text
h{H}_post_break_outcome
h{H}_dominant_forward_direction
```

Örnek outcome'lar:

```text
bullish_continuation
bearish_continuation
failed_bullish_break
failed_bearish_break
bullish_to_bearish_reversal
bearish_to_bullish_reversal
```

## 7. Stage 1A Latent Proximity Probe

Sonra şu soru test edildi:

```text
Stage 1A latent, level proximity bilgisini zaten taşıyor mu?
```

Sonuç:

```text
nearest level side:
  z_fused linear macro F1 ~= 0.7644

nearest distance bucket:
  z_fused linear macro F1 ~= 0.4295

wick breach side:
  z_fused MLP macro F1 ~= 0.5953
```

Yorum:

```text
proximity bilgisi latentte var,
fakat distance hassasiyeti ve karar katmanı kullanımı zayıf.
```

Bu nedenle Stage 1B predictor'a explicit proximity / level-interaction context eklemek makul görüldü.

## 8. Proximity-Enriched Predictor v3

Stage 1B input'u genişletildi:

```text
z_fused
+ bull/bear distance
+ nearest side/distance
+ close/wick interaction counts
+ final excess
+ current label one-hot
```

H8 fresh-break v3 sonuçları:

```text
reconstructed macro F1: 0.4472 -> 0.5971
break macro F1:        0.5933 -> 0.6978
direction macro F1:    0.7234 -> 0.8412
```

Önemli sanity check:

```text
index 454

v2:
  bullish / wrong_direction

v3:
  bearish / correct
```

Yorum:

```text
Stage 1A latent faydalı,
ama Stage 1B decision module explicit structural context ile belirgin iyileşiyor.
```

## 9. 453 Örneği ve Output Contract Problemi

Bir önceki örnekte şu yapı görüldü:

```text
current_state = bullish
fresh_break_target = none
future path = bullish_to_bearish_reversal
```

Fresh-break predictor'ın `none` demesi doğruydu; çünkü anchor zaten bullish break seviyesinin üstündeydi.

Fakat trade açısından bu pencere önemliydi:

```text
already-broken bullish yapı bearish reversal'a dönüyordu.
```

Bu, Stage 1B output contract'ın yalnızca fresh-break sinyali taşımasının yetersiz olduğunu gösterdi.

## 10. Event-Sequence Output Contract v2

Stage 1B output contract güncellendi.

Yeni alanlar:

```text
h{H}_event_type
h{H}_event_direction
```

`event_type`:

```text
no_event
fresh_break
continuation
reversal
```

`event_direction`:

```text
none
bullish
bearish
```

Sanity checks:

```text
index 453:
  h8_event_type = reversal
  h8_event_direction = bearish

index 454:
  h8_event_type = fresh_break
  h8_event_direction = bearish
```

## Current Interpretation

Stage 1B artık tek bir predictor problemi gibi değil, iki farklı okuma gibi ele alınmalı:

```text
H8 fresh-break v3:
  yakın trigger / fresh break için iyi aday

H16 event-sequence:
  continuation / reversal / dominant forward flow için daha uygun aday
```

H8 reversal sınıfı test split'te çok seyrek:

```text
H8 reversal support: 25
```

H16 reversal daha kullanılabilir:

```text
H16 reversal support: 112
```

Bu nedenle sıradaki modelleme adımı için H16 daha uygun görünür.

## Next Candidate

Önerilen sonraki deney:

```text
H16 event-sequence predictor v1
```

Olası hedefler:

```text
h16_event_type
h16_event_direction
```

Bu model, fresh-break H8 predictor'ın yerine geçmek zorunda değildir. Daha doğru ayrım:

```text
H8 fresh trigger signal
H16 event-sequence / reversal-continuation signal
```

Stage 2'ye taşınacak nihai bilgi de bu iki sinyalin birlikte yorumlanmasıyla oluşmalıdır.

## 11. H16 Event-Sequence Predictor Denemeleri

H16 için üç modelleme hattı denendi.

İlk deneme split-head yapıydı:

```text
event_type head
event_direction head
```

Sonuç:

```text
event_type test macro F1 ~= 0.5210
event_direction test macro F1 ~= 0.8479
```

Ancak kritik örnekte kopukluk vardı:

```text
index 453:
  type doğru: reversal
  direction yanlış: bullish
```

Bu nedenle joint sınıf denendi:

```text
no_event
fresh_break_bullish
fresh_break_bearish
continuation_bullish
continuation_bearish
reversal_bullish
reversal_bearish
```

Joint model:

```text
test joint macro F1 ~= 0.4529
index 453 = reversal_bearish doğru
```

Bu model selected H16 context adayı olarak tutuldu.

## 12. Outcome ve Multi-Head Tanısı

Sonraki soru şuydu:

```text
ilk event sinyali trade-relevant path outcome'u taşımaya yetiyor mu?
```

Outcome-joint modeli `h16_post_break_outcome` üzerinden denendi.

Sonuç:

```text
outcome joint macro F1 ~= 0.2840
451-453 segmentinde reversal_bearish düzeldi
```

Yorum:

```text
outcome target doğru semantiği gösteriyor,
ama tek başına selected model kalitesinde değil.
```

Ardından multi-head model denendi:

```text
first_event_joint
outcome_joint
dominant_direction
```

Sonuç:

```text
first_event F1 ~= 0.4450
outcome F1 ~= 0.2726
dominant direction F1 ~= 0.6604
```

Multi-head v1, single-task modelleri geçmedi. Fakat contract açısından şu ayrımı doğruladı:

```text
first_event_signal != outcome_signal
```

## Current Selected Stage 1B Interpretation

Stage 2'ye taşınacak pratik okuma:

```text
H8:
  primary near-trigger / fresh-break signal

H16:
  selected implementation = first_event_signal + candidate_gate
  model = stage1b_h16_joint_event_sequence_v1_z_fused_proximity
```

H16 contract ideal olarak şunları içerir:

```text
first_event_signal
outcome_signal
dominant_direction
```

Ancak mevcut selected-quality model yalnızca `first_event_signal` alanını güvenilir biçimde doldurur.

Bu nedenle şimdilik:

```text
outcome_signal = unknown
dominant_direction = unknown
```

Bu alanları zayıf predictor çıktısıyla doldurmak yerine boş bırakmak daha doğru kabul edildi.

## Stage 2'ye Devreden Açık Nokta

Stage 1B sinyalleri TradePlan değildir.

Stage 2'nin ilk görevi:

```text
H8/H16 candidate sinyallerini
price confirmation,
entry/stop/target guardrail,
fee/slippage-aware net target kontrolü
ile actionability kararına çevirmek.
```
