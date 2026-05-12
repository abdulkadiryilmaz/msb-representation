# Market Structure Edge Program

**Status**: Draft  
**Created**: 2026-05-12  
**Scope**: Birden fazla repo, stage ve araştırma hattını birleştiren üst seviye amaç tanımı

---

## Agent Summary

This document defines the repo-level umbrella program, not a single implementation project.

- Program name: `Market Structure Edge Program`
- Mission: extract systematic, measurable, executable trade edge from market structure breaks
- Thesis: MSB semantics can be modeled through learned structural representations and used for forward structure and tradability modeling
- North Star output: `no_trade` or an executable `TradePlan`
- Current `TradePlan` fields are a draft target contract, not a stable implementation contract
- Stage 1A should not be judged directly by PnL; it should be judged by representation quality and downstream usefulness

---

## Neden Bu Belge Var?

`msb-representation`, `coin-oracle` ve ileride eklenecek Stage 1B / Stage 2 çalışmaları ayrı repo veya modüller olarak gelişebilir. Bu ayrım teknik olarak doğru olabilir; ancak üst amaç net yazılmadığında her parça kendi yerel metriğine optimize olup ana hedeften uzaklaşabilir.

Bu belge, tek bir repo planı değildir. Amaç, bütün çalışmaların hizmet ettiği üst programı tanımlamaktır:

- neden bu çalışmaları yapıyoruz
- hangi büyük hipotezi test ediyoruz
- sonunda sistemden hangi çıktı bekleniyor
- alt projeler bu çıktıya nasıl bağlanıyor
- ara sonuçları hangi çerçevede değerlendireceğiz

---

## İsimlendirme

Bu çalışmanın repo üstü çatı adı:

> **Market Structure Edge Program**

Burada `program`, tek bir kod reposundan daha geniş bir girişimi ifade eder.

- `msb-representation`: Stage 1A temsil öğrenimi araştırma projesi
- Stage 1B: forward structure forecasting hattı
- Stage 2: tradability / quality modeling hattı
- `coin-oracle`: replay, execution ve uygulama hattı
- `Market Structure Edge Program`: bu parçaların tamamını yöneten üst amaç

Bu ayrım önemli:

- repo = belirli teknik kapsam
- stage = belirli araştırma / modelleme sorusu
- program = parçaların birlikte test ettiği büyük iddia

---

## Mission

Piyasa yapısı kırılımlarından sistemli, ölçülebilir ve uygulanabilir trade edge'i çıkarmak.

Bu mission, doğrudan "her break'te işlem aç" anlamına gelmez. Amaç:

- yapısal bağlamı anlamak
- gelecekteki yapısal çözülmeyi öngörmek
- bu öngörünün işlem kalitesini değerlendirmek
- yalnızca yeterli kalite taşıyan durumlarda uygulanabilir trade planı üretmek

---

## Research Thesis

Market Structure Break semantiği, öğrenilmiş yapısal temsiller üzerinden modellenebilir.

Bu temsiller:

- sadece mevcut `bullish / bearish / intact` durumunu ayırmakla kalmamalı
- farklı coinlerde benzer yapıları hizalayabilmeli
- gelecekteki yapısal davranışa bilgi taşımalı
- tradability / quality tahmini için faydalı olmalı

Programın araştırma iddiası şudur:

> MSB semantiğini taşıyan latent temsiller, forward structure ve tradability kalitesi ile birleştirildiğinde, tekrar edebilir trade edge'i üretmek için kullanılabilir.

Bu iddia Stage 1A'da tek başına kanıtlanmaz. Stage 1A yalnızca zincirin ilk halkasını test eder.

---

## North Star Output

Programın nihai teknik çıktısı, her karar anında aşağıdaki iki sonuçtan birini üretebilen sistemdir:

1. `no_trade`
2. uygulanabilir `TradePlan`

Bu yüzden nihai çıktı yalnızca bir yön tahmini değildir.

Yanlış nihai çıktı örnekleri:

- `BTC bullish olabilir`
- `break olasılığı yüksek`
- `latent bullish kümeye yakın`

Bunlar ara sinyaldir. Programın üst çıktısı değildir.

Doğru nihai çıktı şu forma yaklaşmalıdır:

```text
symbol: BTCUSDT
timeframe: 15m
decision: long | short | no_trade
entry_condition: ...
entry_zone: ...
stop_loss / invalidation: ...
target / exit_condition: ...
expected_rr: ...
quality_score: ...
confidence: ...
time_horizon: ...
structural_context: ...
reason_codes: ...
```

Bu çıktı, doğrudan emir açma zorunluluğu taşımaz. Ancak emir açılacaksa `coin-oracle` tarafının uygulayabileceği kadar somut olmalıdır.

---

## Draft TradePlan Contract

`TradePlan`, programın repo ve stage sınırları arasında taşınabilir nihai karar sözleşmesi olmayı hedefleyen taslak çıktıdır.

Bu bölüm şu an stable / versioned contract değildir. Stage 1B ve Stage 2 tasarımları netleşene kadar aşağıdaki alanlar hedef sözleşme taslağı olarak okunmalıdır.

İlk ayrım:

- `Core fields`: karar nesnesinin var olabilmesi için zorunlu çekirdek alanlar
- `Candidate fields`: Stage 1B / Stage 2 / coin-oracle tasarımları ilerledikçe netleşecek alanlar

Core fields:

- `decision`: `long`, `short`, `no_trade`
- `symbol`
- `timeframe`
- `anchor_time`
- `direction`: `long`, `short` veya `none`

Candidate fields:

- `entry_condition` veya `entry_zone`
- `stop_loss` veya `invalidation_condition`
- `target` veya `exit_condition`
- `expected_rr`
- `quality_score`
- `confidence`
- `time_horizon`
- `structural_state`
- `forecast_summary`
- `reason_codes`

Bu taslak Stage 1A'nın doğrudan üretmesi gereken bir şey değildir. Ama Stage 1A'nın ürettiği latent temsil, ileride bu sözleşmenin daha kaliteli doldurulmasına hizmet etmelidir.

---

## Top-Down Stage Haritası

### Program Seviyesi

Soru:

> Ham OHLCV verisinden ölçülebilir ve uygulanabilir trade planı üretilebilir mi?

Başarı ölçümü:

- replay / live simülasyonda net edge
- ücret ve slippage sonrası dayanıklılık
- farklı dönem ve enstrümanlarda bozulmayan davranış
- aşırı işlem üretmeyen, seçici karar kalitesi

### coin-oracle

Soru:

> Üretilen trade planı uygulanabilir emir yaşam döngüsüne çevrilebiliyor mu?

Rol:

- entry tetikleme
- pozisyon açma
- stop / target / exit yönetimi
- replay ve execution ölçümü
- fee, slippage, latency ve operational constraints

Bu katman trade planını uygular. Yapısal latent öğrenimin ana yeri değildir.

### Stage 2 — Tradability / Quality

Soru:

> Teknik veya öngörülen break işlem alınmaya değer kalite taşıyor mu?

Rol:

- follow-through tahmini
- MFE / MAE profili
- risk / ödül potansiyeli
- path davranışı
- setup kalitesi
- `trade` vs `no_trade` ayrımı

Stage 2 pozisyon yönetim motoru değildir. Görevi, execution öncesi kalite ve risk/ödül proxy'lerini üretmektir.

### Stage 1B — Forward Structure Forecasting

Soru:

> Mevcut yapı yakın gelecekte hangi yapısal davranışa evrilebilir?

Rol:

- future break direction
- break proximity
- structural pressure
- continuation / rejection / failure path
- horizon-based structure shift

Stage 1B, trade kalitesini tek başına belirlemez. Bir break'in olası olması, onun işlem alınmaya değer olduğu anlamına gelmez.

### Stage 1A — Structure Representation

Soru:

> Mevcut pencere MSB semantiği açısından anlamlı bir latent temsile dönüştürülebiliyor mu?

Rol:

- `bullish / bearish / intact` label'ları ile latent geometriyi hizalamak
- coin identity shortcut'ını sınırlamak
- short / long bağlamı ortak anchor'da temsil etmek
- hard-case davranışını incelemek
- downstream Stage 1B / Stage 2 görevlerine bilgi taşıyacak representation üretmek

Stage 1A'nın başarısı trade başarısı değildir. Stage 1A'nın başarısı, trade planı zincirinin yapısal algı temelini güçlendirmesidir.

---

## Bottom-Up ve Top-Down Değerlendirme

Bugüne kadar ana değerlendirme çoğunlukla bottom-up yapıldı:

```text
latent iyi mi?
classifier F1 nasıl?
nearest-neighbor label agreement arttı mı?
symbol leakage azaldı mı?
```

Bu gerekli ama yeterli değil.

Top-down soru şudur:

```text
Bu sonuç, ileride daha iyi TradePlan üretmeye hizmet ediyor mu?
```

Bu nedenle ara metrikler program hedefiyle şu şekilde bağlanmalıdır:

- Stage 1A latent geometry iyi olmalı, çünkü Stage 1B / Stage 2'nin yapısal girdisi buradan gelir.
- Stage 1B forward tahmini iyi olmalı, çünkü TradePlan zamanlama ve yön bilgisini buradan alır.
- Stage 2 kalite ayrımı iyi olmalı, çünkü her doğru yapı trade edilebilir değildir.
- coin-oracle replay sonucu iyi olmalı, çünkü nihai edge ancak execution dahil ölçülebilir.

---

## Karar İlkeleri

1. Yerel metrik, program hedefinden kopuk okunmamalı.
2. Stage 1A sonuçları doğrudan PnL ile yargılanmamalı.
3. PnL sonucu kötü olan bir deney, Stage 1A temsilini otomatik olarak geçersiz kılmaz; aradaki Stage 1B / Stage 2 / execution halkaları ayrıca incelenmelidir.
4. Yüksek classifier F1, iyi representation veya iyi trade planı anlamına gelmez.
5. Bir break'in doğru tespit edilmesi, onun trade edilmeye değer olduğu anlamına gelmez.
6. Programın nihai çıktısı sinyal değil, `no_trade` veya uygulanabilir `TradePlan` olmalıdır.
7. Yeni repo veya stage açıldığında, hangi `TradePlan` alanını daha iyi üretmeye hizmet ettiği açıkça yazılmalıdır.

---

## Açık Sorular

### Research Questions

- Stage 2 için ilk tradability proxy'leri ne olmalı?
- `quality_score` kalibrasyonu hangi replay hedefleriyle validate edilmeli?
- Stage 1A latent'in Stage 2'ye transferi hangi minimal probe ile test edilmeli?
- coin-oracle tarafında `no_trade` karar kalitesi nasıl ölçülmeli?

### Design / Contract Questions

- `TradePlan` contract hangi alanlarla versionlanmalı?
- Entry zone ve invalidation rule-based mi başlamalı, yoksa Stage 2 tarafından mı öğrenilmeli?
- `TradePlan` alanları hangi repo tarafından üretilecek, hangi repo tarafından tüketilecek?
- `reason_codes` insan okunabilir açıklama mı, yoksa makine tarafından işlenen enum seti mi olmalı?

---

## Yakın Dönem Kullanım

Bu belge, mevcut Stage 1A çalışmalarını durdurmak için değil, onları daha doğru bağlama oturtmak için kullanılmalıdır.

Pratik kullanım:

- yeni deney planında "program hedefine bağlantı" kısa yazılmalı
- Stage 1A readout'larında sadece yerel metrik değil, downstream anlamı da belirtilmeli
- Stage 1B / Stage 2 tasarımına geçerken `Draft TradePlan Contract` referans alınmalı
- repo sınırı tartışmalarında `program / repo / stage` ayrımı korunmalı
