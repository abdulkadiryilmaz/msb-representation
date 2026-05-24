# Stage 2 Actionability Framing

**Status**: Draft v1, Stage 1B event-sequence contract sonrası revizyon gerekli
**Scope**: Stage 1B sinyalinden TradePlan adayına geçiş çerçevesi
**Upstream input**: `Stage1BSignal + Stage1BContext`

Not: Bu belge ilk Stage 2 tasarımını, H8 `break + direction` Stage 1B contract'ı üzerinden kurdu. 2026-05-24 Stage 1B output contract v2 ile Stage 1B artık `event_type + event_direction` ayrımını taşıyor. Bu nedenle Stage 2'nin sonraki revizyonu `actionable_break_candidate`, `actionable_reversal_candidate` ve `actionable_continuation_candidate` ayrımını içermelidir.

## Amaç

Stage 2'nin görevi, Stage 1B'nin verdiği yakın dönem break olasılığını doğrudan trade sinyali gibi kullanmak değildir.

Stage 2 şu soruyu yanıtlar:

> Stage 1B'nin işaret ettiği teknik veya öngörülen break, uygulanabilir bir TradePlan adayına dönüşebilir mi?

Bu nedenle Stage 2, program zincirinde ilk kez `no_trade` kararının gerçek sahibi olur.

## Stage 1B'den Gelen Girdi

Stage 1B'nin mevcut seçili çıktısı H8 predictor'dır:

```text
Stage1BSignal:
  horizon: H8
  side: long | short | none
  p_break
  p_direction
  break_confidence
  direction_confidence
```

Stage 2 bu sinyali tek başına kullanamaz. Aşağıdaki fiyat ve yapı bağlamına ihtiyaç duyar:

```text
Stage1BContext:
  current_price
  bull_level
  bear_level
  effective_break_pct
  distance_to_bull_level
  distance_to_bear_level
  predicted_side_distance_to_level
  symbol
  timestamp
  time_horizon_bars
```

Temel kural:

```text
break + direction != entry
```

Örneğin mevcut fiyat `10.00`, bullish break seviyesi `11.00` ise Stage 1B'nin `long` demesi `10.00`dan al anlamına gelmez. Bu, bullish yapısal kırılımın ilgili seviyeye yakın bir yerde veya sonrasında actionable hale gelebileceğini söyler.

## İlk Stage 2 Çıktısı

Stage 2 v1, doğrudan tam TradePlan üretmemelidir. İlk çıktı bir actionability sınıfı olmalıdır:

```text
Stage2Actionability:
  decision:
    no_trade
    wait_for_break
    wait_for_retest
    actionable_break_candidate
  side: long | short | none
  quality_score
  confidence
  reason_codes
```

Bu çıktı, final TradePlan'ın ön aşamasıdır.

## Sınıf Anlamları

### no_trade

Stage 1B sinyali olsa bile setup uygulanabilir değildir.

Olası nedenler:

- break confidence düşük
- fiyat ilgili seviyeden çok uzak
- risk / ödül alanı sıkışık
- yakın geçmişte ters yönde güçlü rejection var
- yapı çok gürültülü veya belirsiz

### wait_for_break

Break henüz gerçekleşmemiştir, fakat yapı break seviyesine doğru takip edilmeye değerdir.

Bu sınıf entry değildir. Beklenen şey:

- fiyatın bull/bear level'a yaklaşması
- seviyenin close-confirmed şekilde kırılması
- kırılımın zayıf fakeout gibi görünmemesi

### wait_for_retest

Break gerçekleşmiş veya gerçekleşmeye çok yakındır, fakat doğrudan chase etmek yerine retest / confirmation beklemek daha makuldür.

Bu sınıf özellikle şu durumlarda anlamlıdır:

- fiyat seviyeyi çoktan geçmiş
- immediate entry kötü risk/reward üretiyor
- kırılım mumunun displacement'ı yüksek
- stop seviyesi çok geniş kalıyor

### actionable_break_candidate

Break yönü, seviye yakınlığı, confidence ve ilk risk/reward koşulları birlikte yeterlidir.

Bu sınıf hâlâ emir değildir. Sonraki adım TradePlan alanlarını doldurmaktır:

```text
entry_condition
entry_zone
invalidation / stop_loss
target / exit_condition
expected_rr
```

## İlk Label / Proxy Tasarım İlkesi

Stage 2 label'ı doğrudan PnL label'ı olarak başlamamalıdır. İlk aşamada yapısal actionability proxy'si gerekir.

Neden:

- Stage 1B sinyali yakın break tahminidir, trade outcome değildir.
- PnL; entry, stop, target, fee, slippage ve execution policy'ye bağlıdır.
- Erken PnL optimizasyonu Stage 1A / Stage 1B sinyal kalitesini yanlış cezalandırabilir.

Buradaki ayrım önemlidir:

```text
Stage 1B question:
  Yakın gelecekte beklenen yönde yapısal break olur mu?

Stage 2 v1 question:
  Bu break, yönetilebilir bir trade planı adayına dönüşebilecek path kalitesi taşıyor mu?

Final execution / replay question:
  Belirli entry, stop, target, fee ve slippage varsayımlarıyla para kazandırıyor mu?
```

Bu üç soru aynı değildir. Aynı Stage 1B sinyali, farklı entry/stop/target politikalarıyla farklı PnL sonuçları üretebilir. Bu yüzden Stage 2'nin ilk label'ı "trade kazandı mı?" yerine "bu yapıdan plan çıkar mı?" sorusunu ölçmelidir.

### PnL Label'ının Erken Olmasının Riski

Örnek:

```text
Stage 1B side = long
current_price = 10.00
bull_level = 10.20
future path önce 10.22 üstüne close ediyor, sonra 10.45'e gidiyor
```

Bu yapısal olarak doğru ve potansiyel olarak actionable olabilir. Fakat kötü seçilmiş bir entry veya çok dar stop nedeniyle trade zarar yazabilir. Böyle bir durumda PnL label'ı Stage 1B / Stage 2 sinyalini haksız yere cezalandırır.

Tersi de olabilir:

```text
Stage 1B side = long
fiyat rastlantısal olarak yukarı spike atıyor
sonra seviyenin altına hızlı dönüyor
```

Bu örnek kısa vadeli PnL'de kârlı görünebilir, fakat yapısal olarak güvenilir bir TradePlan adayı olmayabilir. Stage 2'nin amacı bu ayrımı yakalamaktır.

### Actionability Proxy Ne Ölçer?

Actionability proxy, fiyat yolunun trade planına dönüşmeye elverişli olup olmadığını ölçer.

Ölçmeye çalıştığı şeyler:

- beklenen yönde close-confirmed break oluştu mu?
- break seviyesi anchor fiyatına göre takip edilebilir yakınlıkta mı?
- break sonrası fiyat yeterli favorable excursion verdi mi?
- adverse excursion, makul invalidation seviyesine göre kabul edilebilir mi?
- fiyat önce invalidation'a mı gitti, yoksa önce target/R alanı mı verdi?
- immediate entry mi, break bekleme mi, retest bekleme mi daha anlamlı?

Bu nedenle Stage 2 v1 label'ı PnL değil, path-quality label'ıdır.

### Örnek Karar Ayrımları

Long setup için:

```text
current_price = 10.00
bull_level = 10.20
future path: 10.20 üstü close, sonra 10.45, invalidation'a gitmiyor
```

Bu örnek `actionable_break_candidate` olabilir. Çünkü trigger yakın, beklenen yönde break var, adverse path sınırlı ve favorable path ölçülebilir.

Başka bir long setup:

```text
current_price = 10.00
bull_level = 11.00
future path: 10.25'e kadar gidiyor, break yok
```

Bu örnek `no_trade` veya en fazla `wait_for_break` olabilir. Yönlü baskı olabilir, fakat trade planı üretilecek seviyeye henüz gelmemiştir.

Break sonrası chase riski:

```text
current_price = 10.00
bull_level = 10.05
future path: hızlıca 10.30'a spike, sonra 10.04'e retest
```

Bu örnek `wait_for_retest` olabilir. Break var, fakat immediate entry kötü risk/reward yaratabilir; retest/hold davranışı beklemek daha doğru karar sınıfı olabilir.

Bu yüzden Stage 2 v1 label'ı fiyat yolu üzerinden şu soruları ölçmelidir:

```text
1. Beklenen yönde break oldu mu?
2. Break, giriş yapılabilir bir seviyede mi oluştu?
3. Break sonrası adverse excursion kabul edilebilir mi?
4. Minimum favorable excursion oluştu mu?
5. Setup, ters yönde invalidation'a daha önce mi gitti?
```

## Önerilen İlk Proxy Alanları

Stage 2 v1 için label üretiminde aşağıdaki alanlar hesaplanmalıdır:

```text
side
current_price
trigger_level
distance_to_trigger
break_time
break_price
max_favorable_excursion
max_adverse_excursion
invalidation_level
target_level
rr_proxy
path_outcome
```

### trigger_level

`side=long` için `bull_level`, `side=short` için `bear_level`.

### invalidation_level

İlk v1 için rule-based başlamalıdır.

Başlangıç seçenekleri:

- long için yakın swing low / bear level / structure low
- short için yakın swing high / bull level / structure high
- fallback olarak volatility-scaled stop

### target_level

İlk v1 için öğrenilmiş target yerine rule-based proxy daha doğru olur.

Başlangıç seçenekleri:

- fixed R multiple, örn. `1R`, `1.5R`, `2R`
- next opposite structure level
- recent range projection

## Initial Actionability Label v1

İlk uygulanabilir label taslağı:

```text
no_trade:
  p_break düşük
  veya trigger_level çok uzak
  veya future path önce invalidation'a gidiyor
  veya minimum RR proxy sağlanmıyor

wait_for_break:
  break henüz yok
  fakat trigger_level yakın
  ve future path beklenen yönde break'e gidiyor
  ve break sonrası minimum MFE oluşuyor

wait_for_retest:
  anchor anında fiyat trigger_level'ın ötesinde veya çok yakın
  fakat immediate RR zayıf
  ve future path retest/hold davranışı gösteriyor

actionable_break_candidate:
  trigger_level yakın veya kırılmış
  direction confidence yeterli
  future path invalidation'dan önce minimum target/R eşiğine ulaşıyor
```

Bu label seti nihai değildir. Ama Stage 2'nin ilk deneyini ölçülebilir hale getirir.

## İlk Deney Kapsamı

İlk Stage 2 deneyinde kapsam dar tutulmalıdır:

- sadece H8 Stage 1B baseline
- sadece `z_fused` kaynaklı Stage 1B çıktısı
- sadece high-confidence ve medium-confidence segmentleri ayrı raporlanmalı
- önce rule-based actionability label üretimi
- ardından basit baseline / probe

İlk model hedefi:

```text
Input:
  Stage1BSignal
  Stage1BContext
  optional Stage1A latent summary

Output:
  Stage2Actionability class
```

## Başarı Ölçümü

Stage 2 v1 başarı metriği doğrudan PnL değildir.

İlk başarı ölçümleri:

- actionability label distribution sağlıklı mı?
- high-confidence Stage 1B segmentinde actionable ayrımı daha temiz mi?
- `no_trade` sınıfı false positive break'leri azaltıyor mu?
- actionable seçilen örneklerde MFE/MAE profili seçilmeyenlere göre daha iyi mi?
- sınıf kararları sembol bağımlı mı, yoksa yapı bağımlı mı?

PnL / replay ölçümü daha sonra coin-oracle tarafında yapılmalıdır.

## Stage 2'nin Üretmeyeceği Şeyler

Stage 2 v1 şunları üretmemelidir:

- position size
- leverage
- execution order type
- live order lifecycle
- full replay PnL optimization

Bu alanlar `coin-oracle` ve sonraki TradePlan contract aşamalarına aittir.

## Yakın Dönem Karar

Stage 2'ye başlamanın en güvenli yolu:

1. actionability label v1'i üret
2. label dağılımını ve path mantığını audit et
3. Stage 1B high-confidence segmentinde proxy'nin anlamlı ayrışıp ayrışmadığını ölç
4. ancak bundan sonra predictor / classifier eğit

Bu sıra, Stage 2'nin doğrudan gürültülü PnL label'ına bağlanmasını engeller.
