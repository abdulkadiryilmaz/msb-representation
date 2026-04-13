# Worklog: 2026-04-12 — Stage 1 MSB Representation Framing

**Status**: In Progress  
**Goal**: Stage 1'i saf MSB classification probleminden çıkarıp MSB-oriented representation learning olarak yeniden çerçevelemek

---

## Problem Statement

Mevcut yapı uzun süredir `intact / bullish / bearish` MSB classification etrafında düşünülüyordu. Bu framing şu soruyu doğurdu:

- Eğer MSB matematiksel olarak zaten tanımlanabiliyorsa, bunu neden ayrıca model öğreniyor?

Bu itiraz haklı. Teknik olarak:
- swing high / swing low tespiti
- break zone içinde kırılım
- confirmation horizon

kurallarla üretilebiliyorsa, `MSB var mı?` sorusu büyük ölçüde deterministic hale geliyor. Bu durumda Stage 1'i doğrudan aynı label'ı tahmin eden classifier olarak kurmak öğrenme açısından zayıf bir problem haline geliyor.

Ayrıca ikinci bir karışıklık ortaya çıktı:
- `MSB oluştu mu?`
- `Bu MSB trade almaya değer mi?`

Bu iki soru aynı değil. Başarısız trade'i Stage 1'e gömersek, Stage 1 ile Stage 2 ayrımı bozuluyor.

---

## Context

Mevcut proje hedefi saf teknik event detection'dan daha ileri:

- kısa ve uzun bağlamı kullanarak yapı kırılımını anlamak
- fakeout benzeri durumları ayırmak
- nihayetinde `tradable structure break` üretmek

Bu bağlamda üç ayrı seviye netleşti:

1. `MSB detection`
2. `Tradable MSB`
3. `Execution-aware signal`

Mevcut karışıklığın kaynağı, Stage 1'e hem 1. hem 2. seviyeyi aynı anda yükleme eğilimiydi.

---

## Investigation

### 1. Stage 1 saf classification olarak zayıf kalıyor

Eğer target yalnızca:
- `0 = intact`
- `1 = bullish`
- `2 = bearish`

ve bu target kuralla üretiliyorsa, modelin bunu öğrenmesinin değeri sınırlı olabilir. Bu durumda model çoğu zaman kurallı label'ın örüntüsel karşılığını taklit eder.

Bu, representation learning açısından tatmin edici değil.

### 2. Asıl değer latent temsilde olabilir

Stage 1'in daha anlamlı versiyonu:

- modelin final ürünü bir sınıf kararı değil
- model, MSB açısından anlamlı bir latent space öğrensin
- `bullish / bearish / intact` etiketleri latent'i şekillendiren supervision olarak kullanılsın

Bu framing ile şunlar incelenebilir:

- bullish / bearish / intact latent uzayda ayrışıyor mu
- kümelenme oluşuyor mu
- farklı coinlerde benzer yapılar hizalanıyor mu
- latent, sonraki tahmin görevlerine faydalı mı

### 3. Fakeout ve tradability ayrımı

`Fakeout` iki farklı anlamda kullanılıyordu:

1. `Yapısal fakeout`
   - teknik break oluşmamış
   - Stage 1'de `intact`
2. `Trading fakeout` veya `tradable olmayan gerçek break`
   - teknik break var
   - follow-through zayıf veya işlem kalitesi düşük
   - Stage 1'de MSB olarak kalmalı
   - Stage 2'de elenmeli

Bu ayrım kritik. Aksi halde Stage 1 teknik yapı yerine trade sonucunu öğrenmeye başlar.

### 4. Semantic representation ile classifier output farklı şeyler

Net ayrım:

- `latent vector`: encoder çıktısı
- `classifier output`: final sınıf tahmini
- `semantic representation`: iç temsilin görev açısından anlamlı olması

Bu worklog kapsamında alınan pozisyon:

- Stage 1'in amacı zorunlu olarak classifier output üretmek değil
- Stage 1'in amacı MSB-semantic internal representation öğrenmek olabilir

---

## Decision

Stage 1 aşağıdaki şekilde yeniden çerçevelendi:

- Stage 1 bir `MSB-oriented representation learning` problemidir
- `bullish / bearish / intact` etiketleri matematiksel olarak üretilecektir
- Bu etiketler final ürün olarak zorunlu classification başlığı üretmek için değil, latent uzayı hizalamak için kullanılacaktır

Başka deyişle:

- amaç: `MSB var mı?` classifier'ı kurmak değil
- amaç: MSB ile ilişkili yapısal durumu taşıyan latent temsil öğrenmek

Bu temsil daha sonra iki amaç için değerlendirilecektir:

1. Latent analizleri
   - class separability
   - clustering
   - coinler arası hizalanma
   - regime dayanıklılığı
2. Downstream görevler
   - geleceğe dönük break tahmini
   - tradability / trade quality
   - exit veya path davranışı

Ek karar:

- Mevcut short ve long AE çalışmaları korunacak, ancak Stage 1A için yeni encoder hattı ayrı geliştirilecek
- Mevcut AE latent'leri `bootstrap / baseline` olarak değerlidir, fakat yeni hedef için doğrudan nihai latent kabul edilmeyecektir

---

## Revised Stage Breakdown

### Stage 1A — Structure Representation

Amaç:
- mevcut pencerenin yapısal durumunu MSB açısından anlamlı latent uzayda temsil etmek

Not:
- bu aşamada final amaç trade kararı değildir
- `bullish / bearish / intact` etiketi supervision olarak kullanılabilir

### Stage 1A Latent Spec

Stage 1A latent'inin taşıması beklenen ana sinyaller:

- `Yapı durumu`
  - intact mı, kırılgan mı, break'e yaklaşmış mı
- `Yönlü yapı`
  - bullish break baskısı mı, bearish break baskısı mı
- `Swing ilişkisi`
  - son anlamlı swing high/low'a göre konum
  - bu seviyelere yakınlık ve baskı
- `Compression -> expansion geçişi`
  - sıkışma sonrası displacement işareti
- `Break kalitesi için ön sinyal`
  - wick-dominant mı, close-dominant mı
  - impulsif mi, sönük mü
- `Çok ölçekli bağlam`
  - kısa hareketin daha geniş trend/rejim içindeki yeri
- `Coinler arası ortak yapı`
  - farklı coinlerde aynı yapısal davranışın benzer latent bölgelere düşebilmesi

Stage 1A latent'inde baskın olmaması istenen shortcut'lar:

- `Salt zaman kimliği`
  - hour/day feature'larının latent geometriyi domine etmesi
- `Coin identity shortcut`
  - latent'in esasen coin ismi taşıması
- `Saf volatility shortcut`
  - her sert hareketin MSB-benzeri görünmesi
- `Gürültülü hacim anomalileri`
  - structure ile zayıf bağlı aşırı non-stationary sinyaller
- `Reconstruction'a yararlı ama görev dışı detaylar`
  - yapısal anlam taşımayan waveform ayrıntıları

Bu latent'ten beklenen temel ayrımlar:

- `intact` vs `bullish`
- `intact` vs `bearish`
- `bullish` vs `bearish`
- `high-vol intact` vs `true break`
- `wick sweep` vs `close-confirmed break`

Tek cümlelik hedef:

- Stage 1A latent'i, `bu pencere yapısal olarak ne durumda ve hangi yönde kırılma mantığı taşıyor?` sorusuna cevap verebilmelidir

### Stage 1A Input Principles

Stage 1A input'u indicator-merkezli değil, `price-structure` merkezli olmalıdır.

Çekirdek görüş:

- MSB yapısal bir fiyat olayıdır
- Bu nedenle model ham candle geometrisine erişmelidir
- Özellikle wick vs close ayrımı, displacement şekli ve swing topolojisi yalnızca türetilmiş indicator'larla tam taşınmayabilir

Bu yüzden Stage 1A için `raw OHLC` merkezde tutulacaktır.

İlk çekirdek input ailesi:

- `open`
- `high`
- `low`
- `close`
- `log_return`
- `open_close_return`
- `hl_range`
- `atr_norm`
- `price_position`
- `bb_position`

İkinci dalga, opsiyonel context feature'ları:

- `vwap_dev`
- `rsi`
- `macd`
- `macd_signal`

Üçüncü dalga, dikkatli yönetilecek shortcut-risk feature'ları:

- `volume_zscore`
- `volume_ratio`
- `hour_sin`
- `hour_cos`
- `dow_sin`
- `dow_cos`

Temel karar:

- Raw OHLC çıkarılmayacak, tersine merkezde olacak
- Derived feature'lar yardımcı rol üstlenecek
- Volume ve time feature'ları ilk tasarımda ya dışarıda tutulacak ya da kontrollü yardımcı grup olarak ele alınacak

### Feature Group Strategy

Feature'ları tek düz liste olarak değil, risk ve rol bazlı gruplar halinde düşünmek daha doğru bulundu.

Önerilen gruplar:

1. `Price-structure group`
   - OHLC
   - return/range tabanlı türevler
   - temel position feature'ları
2. `Context group`
   - momentum ve pozisyonel özetler
   - RSI/MACD/VWAP benzeri sinyaller
3. `Risky shortcut group`
   - volume
   - time

Bu gruplama şu nedenle önemli:

- tüm feature'lara eşit statü vermek istemiyoruz
- bazı feature'lar yardımcı olabilir, ama latent geometriyi yanlış eksenlerde organize etme riski taşır

İkinci ve üçüncü dalga feature'ların eklenmesi şu üç eksende değerlendirilecek:

1. `Temsil kalitesine katkı`
   - class separability artıyor mu
   - `high-vol intact` vs `true break` ayrımı iyileşiyor mu
   - coinler arası hizalanma düzeliyor mu
2. `Shortcut riski`
   - latent coin/time/volume rejimine göre mi ayrışıyor
   - feature kaldırılınca model çöküyor mu
   - yapı yerine activity veya saat bilgisi mi baskın hale geliyor
3. `Downstream fayda`
   - Stage 1B future break görevi için latent faydalı mı
   - Stage 2 tradability/quality için katkı sağlıyor mu

Bu grupları tamamen çıkarma yerine etkilerini sınırlamak için şu yöntemler aday olarak belirlendi:

- `Ayrı branch / ayrı encoder`
  - price-structure, context ve risky feature'ların ayrı işlenmesi
- `Daha düşük kapasite`
  - risky branch'e daha düşük latent boyut ayrılması
- `Feature dropout / stochastic masking`
  - volume veya time feature'larının eğitimde bazen kapatılması

Bu yaklaşımların ortak amacı:

- yardımcı feature'ları tamamen kaybetmeden kullanmak
- ancak modelin bunlara aşırı bağımlı ve shortcut-açık hale gelmesini önlemek

### Window Size Principles

Window size, Stage 1A latent'inin neyi göreceğini ve neyi kaçıracağını doğrudan belirler. Bu nedenle üç ayrı karar olarak ele alınacaktır:

1. `short window`
2. `long window`
3. `short-long ölçek ilişkisi`

#### Short window'un rolü

Short pencere şu sinyalleri taşımalıdır:

- lokal swing yapısı
- wick vs close davranışı
- compression -> break geçişi
- kısa vadeli displacement
- anlık yapısal kırılganlık

Short pencere fazla kısa olursa:

- swing yapısı eksik kalabilir
- break öncesi bağlam zayıflar

Fazla uzun olursa:

- lokal yapı bulanıklaşır
- rejim bilgisi short latent'i domine eder

Stage 1A için aday short aralığı:

- `32 bar` (8h)
- `48 bar` (12h)
- `64 bar` (16h)

İlk güçlü prior:

- `48 bar`

#### Long window'un rolü

Long pencere şu sinyalleri taşımalıdır:

- daha büyük trend/rejim
- short penceredeki olayın bağlamı
- son günlerdeki yapısal bias
- aynı short pattern'in hangi zeminde oluştuğu

Long pencere fazla kısa olursa:

- short branch'in tekrarına dönüşür

Fazla uzun olursa:

- gereksiz tarih yükü taşır
- güncel yapı sinyali zayıflar
- öğrenme zorlaşır

Stage 1A için aday long aralığı:

- `192 bar` (48h)
- `288 bar` (72h)
- `384 bar` (96h)

İlk güçlü prior:

- `288 bar`

#### Short-long ölçek ilişkisi

Long branch, short branch'in yalnızca biraz büyüğü olmamalıdır. Gerçekten farklı zaman ölçeği taşımalıdır.

Örnek:

- `short=48`, `long=64` -> zayıf ölçek ayrımı
- `short=48`, `long=192` veya `288` -> anlamlı ölçek ayrımı

İlk aday set:

- `short=48`, `long=192`
- `short=48`, `long=288`
- `short=64`, `long=256`

#### Window size seçim ölçütleri

Pencere boyları yalnızca loss veya tek bir downstream metriğe göre seçilmeyecek. Üç eksen birlikte değerlendirilecek:

1. `Latent davranışı`
   - class separability
   - `high-vol intact` vs `true break` ayrımı
   - bullish / bearish / intact geometri
2. `Ölçek katkısı`
   - short ve long latent aşırı korele mi
   - long branch gerçekten ek bilgi getiriyor mu
   - fusion sonrası temsil zenginleşiyor mu
3. `Forward utility`
   - Stage 1B future structure görevlerine transfer
   - farklı rejimlerde dayanıklılık

Çalışma varsayılanı:

- `short=48`
- `long=288`

Bu seçim nihai kabul edilmeyecek; Stage 1A için güçlü başlangıç hipotezi olarak ele alınacak

### Window Alignment and Overlap

Stage 1A için varsayılan hizalama:

- short ve long pencereler aynı `anchor bar`da bitecek
- başka deyişle her iki branch de aynı karar anını temsil edecek
- hizalama `same end timestamp / right-aligned` olacak

Bu yaklaşım şu nedenlerle tercih edildi:

- gelecek verisine dayanmaz
- short ve long aynı olayı temsil eder
- short = yakın geçmiş, long = daha geniş geçmiş ayrımı korunur

Alternatifler değerlendirildi:

- `center align` -> gelecek bilgisi gerektirir, leakage üretir
- `left align` -> aynı karar anı bozulur, fusion zayıflar
- `lagged long context` -> teorik olarak mümkün, ancak güncel bağlamı zayıflatır ve ilk tasarım için gereksiz karmaşıklık ekler

Overlap konusunda alınan pozisyon:

- long pencerenin short'u tamamen içermesi başlangıç için kabul edilebilir
- bu, `aynı anın daha geniş bağlamda temsili` ile uyumludur
- sorun overlap'ın kendisi değil, long branch'in yalnızca short'un tekrarı haline gelmesidir

Bu nedenle başlangıç tasarımı:

- `full overlap`
- `same end timestamp`
- ayrı branch / ayrı encoder

Redundancy problemi aşağıdaki işaretlerle izlenecek:

- short-only vs short+long katkısı düşük mü
- short ve long latent korelasyonu yüksek mi
- fused latent, short latent'e göre gerçekten yeni ayrım gücü getiriyor mu

Gerekirse ileride şu alternatifler deney olarak değerlendirilebilir:

- `gapli long context`
- `downsampled / dilated long`
- `reduced-overlap long`

Ancak ilk Stage 1A tasarımı için bunlar default olmayacak

### Shared-Core Feature Principle

Önceki dual-window deneylerinde short ve long için fazla farklı feature aileleri kullanılması, iki branch'in `aynı anın iki ölçeği` olmaktan çıkıp iki ayrı modaliteye dönüşme riskini gösterdi.

Bu nedenle Stage 1A için şu ilke benimsendi:

- `same anchor, same semantic family, different temporal scale`

Bu ne demek:

- short ve long branch aynı yapısal dünyanın farklı çözünürlükleri olmalı
- fark, mümkün olduğunca feature ontolojisinden değil zaman ölçeği ve çözünürlükten gelmeli

Karar:

- short ve long ortak bir çekirdek feature ailesi paylaşacak
- ölçeğe özel farklılaşma gerekiyorsa bu öncelikle zaman çözünürlüğü, pencere boyu, smoothing, encoder kapasitesi veya küçük ek feature setleri ile yapılacak
- tamamen ayrık feature dillerinden kaçınılacak

Bu yaklaşımın amacı:

- `aynı anı daha geniş bağlamda temsil etme` fikrini korumak
- short ve long arasında semantik köprü kurmak
- fusion öğrenmesini kolaylaştırmak
- long branch'in yalnızca tekrar veya bambaşka modalite olmasını önlemek

### Candidate Short and Long Feature Sets

#### Shared core

Short ve long branch için ortak çekirdek feature ailesi:

- `open`
- `high`
- `low`
- `close`
- `log_return`
- `open_close_return`
- `hl_range`
- `atr_norm`
- `price_position`
- `bb_position`

Bu çekirdek aile:

- ham candle geometrisini
- lokal hareketin normalize edilmiş biçimini
- pencere içindeki yapısal konumu

aynı semantik düzlemde taşır.

#### Candidate short feature set (v1)

Stage 1A short branch için ilk aday:

- `open`
- `high`
- `low`
- `close`
- `log_return`
- `open_close_return`
- `hl_range`
- `atr_norm`
- `price_position`
- `bb_position`

Gerekirse kısa vadeli detay için sonraki aday ekler:

- üst/alt wick oranları
- candle body oranı
- close location within bar

Bu ekler ilk sürümde zorunlu değil; önce çekirdek yapı dili test edilecek.

#### Candidate long feature set (v1)

Stage 1A long branch için ilk aday:

- `open`
- `high`
- `low`
- `close`
- `log_return`
- `open_close_return`
- `hl_range`
- `atr_norm`
- `price_position`
- `bb_position`

Long branch'te farklılaşma öncelikle şu yollarla gelecek:

- daha uzun pencere
- gerekirse downsampling / dilation
- gerekirse smoothing
- daha düşük latent kapasitesi veya farklı encoder kapasitesi

Yani long branch'in ana farkı feature semantiğinden değil, zaman ölçeğinden gelecektir.

#### Optional scale-specific extras

İlk çekirdek sürümden sonra küçük, ölçeğe özel ekler değerlendirilebilir:

Short için olası ekler:

- wick/body odaklı detail feature'lar
- micro-displacement veya close-confirmation feature'ları

Long için olası ekler:

- daha yavaş volatility / trend özetleri
- yumuşatılmış position veya range feature'ları

Bu ekler ancak shared-core yaklaşımı oturduktan sonra değerlendirilecek.

### Temporal Resolution Options

Stage 1A için zaman çözünürlüğü tartışmasında üç seçenek değerlendirildi:

1. `15m long`
2. `1h long`
3. `4h macro context`

#### 15m long

Örnek:

- short = `15m x 48 = 12h`
- long = `15m x 288 = 72h`

Artılar:

- same-anchor hizalama çok temiz
- short ve long aynı piyasa dilini konuşur
- wick, displacement ve intraday yapı korunur
- shared-core feature ilkesiyle en doğal başlangıçtır

Eksiler:

- redundancy riski daha yüksektir
- long branch bazen "çok uzun short pencere" gibi davranabilir

Karar:

- Stage 1A için `ilk varsayılan long çözünürlüğü 15m` olacak

#### 1h long

Örnek:

- short = `15m`
- long = `1h`

Artılar:

- bağlamı daha temiz özetler
- mikro gürültüyü azaltır
- short ile redundancy'yi düşürür
- trader'ın kısa vade + bağlam okumasına daha yakındır

Eksiler:

- hizalama ve feature üretimi daha dikkatli tasarım ister
- bazı geçiş detayları kaybolabilir

Karar:

- `1h long`, 15m long oturduktan sonra ilk ciddi alternatif olarak değerlendirilecek

#### 4h macro context

Örnek:

- short = `15m`
- long = `15m` veya `1h`
- ek macro branch = `4h`

Artılar:

- makro rejim / trend zemini çok daha net taşınabilir
- short ve long ile rol ayrımı güçlüdür

Eksiler:

- Stage 1A long branch için fazla kaba olabilir
- local-intermediate köprüsünü zayıflatabilir
- long branch'in yerine geçtiğinde semantik kopukluk yaratabilir

Karar:

- `4h`, Stage 1A'de mevcut long branch'in yerine geçmeyecek
- ileride gerekirse ayrı bir `macro context branch` olarak değerlendirilecek

Bu bölüm için özet yol haritası:

1. `15m short + 15m long` ile başla
2. Ardından `15m short + 1h long` karşılaştır
3. Daha sonra gerekirse `4h macro branch` ekle

### Stage 1A Evaluation Framework

Stage 1A'de başarı, reconstruction kalitesi ile değil, temsilin görev açısından ne kadar anlamlı olduğu ile ölçülecek.

Bu nedenle evaluation dört katmanda ele alınacak:

#### 1. Representation geometry

Soru:

- latent uzay yapısal bilgi taşıyor mu

Bakılacaklar:

- `class separability`
  - bullish / bearish / intact ayrımı
- `intra-class compactness`
  - aynı sınıf örnekleri ne kadar yakın
- `inter-class distance`
  - sınıflar arası mesafe
- `nearest-neighbor purity`
  - komşular aynı yapısal sınıftan mı geliyor
- `coin mixing`
  - latent kümeleri coin kimliğine göre mi, yapıya göre mi oluşuyor

Araçlar:

- PCA / UMAP / t-SNE
- class centroid distance
- silhouette score
- kNN purity

#### 2. Probe evaluation

Encoder dondurulacak ve latent üzerine basit probe'lar takılacak:

- linear classifier
- logistic regression
- küçük MLP

Amaç:

- `bullish / bearish / intact` ayrımı latent üzerinde ne kadar kolay

Karşılaştırmalar:

- short latent
- long latent
- fused latent

Beklenti:

- fused latent en güçlü temsil olmalı
- short ve long farklı katkılar üretmeli

Metrikler:

- macro F1
- class-wise precision / recall / F1
- confusion matrix

#### 3. Hard-case evaluation

Stage 1A'in gerçekten yapısal bilgi taşıyıp taşımadığını görmek için özel örnek kümeleri değerlendirilecek.

Özellikle:

- `high-vol intact` vs `true break`
- `wick sweep` vs `close-confirmed break`
- `local bullish` vs `broader-context recovery`
- gürültülü bullish / bearish ayrımları

Amaç:

- latent'in kolay volatility ayrımı mı yaptığı
- yoksa gerçekten structure semantics taşıdığı mı

Bu katmanda:

- curated örnek setleri
- nearest-neighbor incelemeleri
- probe hata analizi

kullanılacak.

#### 4. Transfer / downstream evaluation

En güçlü temsil testi, latent'in gelecekteki görevlerde işe yarayıp yaramadığıdır.

Aday görevler:

- future break direction
- break proximity
- path class
- tradability / quality score

Amaç:

- Stage 1A latent'in sadece mevcut yapıyı ayırmakla kalmayıp ileri görevler için de bilgi taşıdığını doğrulamak

#### Primary evaluation bundle

Stage 1A için temel başarı paketi:

- probe `macro F1`
- class-wise F1
- nearest-neighbor purity
- coin leakage / coin shortcut kontrolü

İkincil destekleyici sinyaller:

- silhouette / centroid distance
- PCA/UMAP görselleştirmeleri
- hard-case sonuçları

Üçüncül doğrulama:

- Stage 1B transfer performansı
- Stage 2 transfer performansı

Sonuç ilkesi:

- iyi temsil = iyi reconstruction değil
- iyi temsil = label hizalaması, coinler arası semantik yapı, hard-case ayrımı ve downstream faydayı birlikte taşıyan latent

### Stage 1A Objective Candidates

Stage 1A için amaç doğrudan classifier üretmek değil, latent'i `bullish / bearish / intact` etrafında görev açısından anlamlı biçimde organize etmektir.

Bu nedenle objective seçenekleri reconstruction merkezli değil, representation merkezli ele alınacaktır.

#### 1. Auxiliary cross-entropy

Yapı:

- encoder latent üretir
- küçük bir head latent'ten `bullish / bearish / intact` tahmini yapar
- cross-entropy loss kullanılır

Artılar:

- en stabil ve en basit başlangıç
- uygulaması kolay
- sınıf ayrımı için güçlü temel sağlar

Eksiler:

- latent geometriyi doğrudan düzenlemez
- classifier-friendly ama representation olarak sığ bir kod üretebilir
- semantic manifold yerine decision boundary odaklı bir ayrım oluşabilir

Rol:

- güçlü baseline

#### 2. Supervised contrastive loss

Amaç:

- aynı sınıftaki örnekleri latent uzayda yakınlaştırmak
- farklı sınıfları uzaklaştırmak

Artılar:

- doğrudan latent geometry'yi şekillendirir
- representation learning hedefiyle doğal uyumludur
- probe ve transfer görevleri için daha semantik latent üretebilir

Eksiler:

- eğitim daha hassas olabilir
- batch composition önemlidir
- noisy label durumunda daha kırılgan olabilir
- `intact` sınıfı çok heterojen olduğu için tek kümeye zorlamak yapay olabilir

Rol:

- representation-centric güçlü aday

#### 3. CE + supervised contrastive hibriti

Birleşik loss:

- `L = lambda_ce * L_ce + lambda_supcon * L_supcon`

Artılar:

- CE sınıf ayrımını ve training stabilitesini korur
- supervised contrastive latent geometry'yi düzenler
- direct classifier ile saf metric learning arasında dengeli yaklaşım sunar

Eksiler:

- loss dengesi ayar ister
- `intact` sınıfı için contrastive baskı dikkatli tasarlanmalıdır

Rol:

- Stage 1A için ana aday objective

#### 4. Triplet / prototype / center tabanlı seçenekler

Bu aileler alternatif olarak değerlidir ancak ilk iterasyon için ikincil tutulacaktır.

Sebep:

- triplet mining karmaşıktır
- prototype / center yaklaşımı heterojen `intact` sınıfı için kaba kalabilir
- erken aşamada gereksiz tasarım yükü ekleyebilir

#### Objective yol haritası

İlk deney sırası:

1. `CE only` baseline
2. `SupCon only` deney
3. `CE + SupCon` ana aday

Beklenti:

- `CE only` -> stabil ama daha sığ temsil
- `SupCon only` -> daha semantik ama daha kırılgan
- `CE + SupCon` -> en dengeli temsil

Özel not:

- `intact` sınıfı muhtemelen heterojen bir manifold olacak
- bu nedenle contrastive objective tasarlanırken bullish/bearish cluster'larını daha güçlü şekillendirip intact tarafını daha gevşek bırakma ihtiyacı doğabilir

### Encoder Symmetry vs Asymmetry

Stage 1A için short ve long encoder'ların tamamen aynı olması gerekmiyor; ancak aynı mimari ailesinden gelmeleri güçlü bir tercih olarak belirlendi.

Temel gerekçe:

- short ve long branch aynı yapısal dünyanın iki ölçeğini temsil ediyor
- bu nedenle tamamen farklı modalitelere dönüşmeleri istenmiyor
- aynı block ailesi ve aynı genel tasarım dili, fusion ve semantik hizalama açısından avantaj sağlıyor

Ancak short ve long branch'in çözmesi gereken iş birebir aynı değil:

#### Short encoder'ın rolü

- lokal swing yapısını çözmek
- wick vs close farkını görmek
- kısa vadeli displacement'i yakalamak
- yüksek zaman çözünürlüğündeki detayları taşımak

Bu nedenle short branch:

- daha yüksek detay korumalı
- downsampling konusunda daha dikkatli olmalı
- gerekirse daha yüksek kapasite veya daha fazla blok taşımalı

#### Long encoder'ın rolü

- daha geniş bağlamı özetlemek
- birkaç günün rejim ve yapı zeminini taşımak
- mikro detaydan çok orta ölçek örüntüyü temsil etmek

Bu nedenle long branch:

- daha agresif temporal reduction tolere edebilir
- daha hafif bir encoder olabilir
- aynı block ailesi içinde daha az blok veya daha küçük latent ile çalışabilir

Karar ilkesi:

- `architecturally related, not identical`

Başka deyişle:

- short ve long aynı encoder ailesinden gelecek
- ama blok sayısı, kapasite, subsampling stratejisi ve latent boyutu birebir aynı olmak zorunda olmayacak

Bu çerçevede önceki deneylerde kullanılan örüntü:

- short için daha derin yapı
- long için daha hafif yapı
- long'da subsampling sonrası blok azaltımı

yeni Stage 1A için de mantıklı bir başlangıç prior'ı olarak kabul edildi; ancak artık gerekçesi reconstruction değil, `local detail` vs `broader context` ayrımıdır.

### Stage 1B — Forward Structure Forecasting

Amaç:
- Stage 1A temsilinin yakın gelecekteki yapısal kırılım davranışını ne kadar taşıdığını test etmek

Örnek sorular:
- bu pencere bullish/bearish break'e ne kadar yakın
- hangi yönde kırılım potansiyeli var
- yapısal gerilim veya çözülme sinyali taşıyor mu

### Stage 2 — Tradability / Quality

Amaç:
- teknik veya öngörülen break'in işlem alınmaya değer olup olmadığını değerlendirmek

Bu aşama şunları içerebilir:
- follow-through
- MFE/MAE
- risk/ödül
- exit davranışı

---

## Key Learnings

- `MSB var mı?` sorusu, matematiksel tanım netse büyük ölçüde rule-based çözülebilir.
- Öğrenme açısından daha değerli problem, MSB açısından anlamlı latent temsil öğrenmektir.
- `Yapısal fakeout` ile `tradable olmayan gerçek MSB` aynı şey değildir; Stage 1 ve Stage 2 burada ayrılmalıdır.
- Label kullanmak, final ürünü classifier yapmak zorunda değildir; latent uzayı hizalamak için supervision olarak kullanılabilir.
- `Tradable structure break` projenin gerçek üst hedefidir, ancak Stage 1 bu hedefin yalnızca yapı tarafını ele almalıdır.

## Next Steps

- [ ] Stage 1A için kesin target/supervision rolünü yaz: label nerede kullanılıyor, nerede kullanılmıyor
- [ ] Latent'i etiketlerle hizalamak için aday loss'ları karşılaştır: auxiliary CE vs supervised contrastive vs metric-based
- [ ] Stage 1A için analiz setini tanımla: PCA/UMAP, class separability, coinler arası nearest-neighbor
- [ ] Stage 1B için forward target seçeneklerini yaz: future break, path class, horizon-based structure shift
- [ ] Stage 2 için `tradable break` proxy tasarımlarını ayrı worklog veya ek update ile netleştir

---

## Update: Session Outcome

Bu oturum sonunda önemli bir framing değişikliği yapıldı:

- Önceki düşünce: Stage 1 = MSB classifier
- Yeni düşünce: Stage 1 = MSB-aware representation learner

Bu değişiklik, projeyi rule-based event detection ile representation learning arasında daha tutarlı bir zemine taşıyor. Özellikle short ve long encoder'ların downstream değerini anlamak için latent space analizi artık ikincil değil, merkezi araç haline geliyor.

İkinci önemli karar:

- Mevcut short/long AE çalışmaları terk edilmeyecek
- Ancak yeni hedef farklı olduğu için Stage 1A encoder hattı sıfırdan ve ayrı geliştirilecek
- Eski latent'ler, yeni hattın karşılaştırmalı baseline'ı olarak korunacak

Stage 1A için latent tanımı da netleşti:

- amaç yalnızca `MSB oldu mu?` bilgisini kodlamak değil
- amaç, yapısal durum, yönlü baskı, swing ilişkisi ve çok ölçekli bağlam taşıyan bir temsil öğrenmek
- aynı zamanda volatility, coin identity veya zaman shortcut'larının latent geometriyi domine etmesi istenmiyor

Input tarafı için de ikinci bir karar alındı:

- Stage 1A input'u `price-structure first` tasarlanacak
- raw OHLC çekirdek rol alacak
- derived feature'lar yardımcı dalgalar halinde eklenecek
- volume ve time feature'ları ilk aşamada kontrollü, sınırlı veya ayrı branch mantığıyla değerlendirilecek

Window tarafında da başlangıç çerçevesi netleşti:

- Stage 1A için iki ölçekli temsil korunacak
- `short=48`, `long=288` güçlü başlangıç prior'ı olarak kullanılabilir
- ancak pencere boyları latent geometri, ölçek katkısı ve forward utility üzerinden ayrıca doğrulanacak

Hizalama ve feature semantiği için de iki ilke netleşti:

- short ve long aynı end timestamp'te hizalanacak
- short ve long, ortak çekirdek feature ailesini paylaşacak; fark esas olarak zaman ölçeğinden gelecektir

Temporal resolution için de başlangıç kararı alındı:

- Stage 1A long branch ilk sürümde `15m` çözünürlükte kalacak
- `1h long` ilk büyük alternatif olarak saklanacak
- `4h` ise long'ın yerine değil, ileride eklenebilecek ayrı `macro context branch` adayı olarak not edilecek

Evaluation için de yeni çerçeve sabitlendi:

- Stage 1A encoder'ı reconstruction ile değil, representation geometry + probe + hard-case + transfer eksenlerinde değerlendirilecek

Objective tarafında da başlangıç çerçevesi netleşti:

- reconstruction ana objective olmayacak
- `CE only`, `SupCon only` ve özellikle `CE + SupCon` hibriti ilk adaylar olacak

Encoder tarafında da mimari ilke netleşti:

- short ve long encoder'lar aynı aileden gelecek
- ancak short için daha detay odaklı, long için daha bağlam odaklı asimetrik kapasite kabul edilecek

### Fusion Candidates

Fusion'ın rolü, short ve long temsilini yalnızca teknik olarak birleştirmek değil, `local structure` ile `broader context` ilişkisini tek bir MSB-oriented temsilde toplamak olarak tanımlandı.

#### 1. Concatenation baseline

En basit yapı:

- `z_fused = [z_short ; z_long]`

Artılar:

- temiz baseline
- bilgi kaybı yok
- uygulaması ve analizi kolay
- neyin hangi branch'ten geldiği açık

Eksiler:

- short ve long arasındaki ilişkiyi sonraki katmanın kendi başına öğrenmesini bekler
- bağlamsal yorumlama için özel inductive bias vermez

Karar:

- `concat`, Stage 1A için zorunlu baseline olacak

#### 2. Gated fusion

Mantık:

- short latent ana taşıyıcıdır
- long latent bağlamsal kapı / modülatör üretir
- context, local structure'ı hangi boyutlarda nasıl okumamız gerektiğini etkiler

Örnek şema:

- `g = sigmoid(W z_long)`
- `z_short_ctx = z_short * g`
- `z_fused = [z_short_ctx ; z_long]`

Artılar:

- "aynı short yapı farklı context'te farklı anlam taşır" fikrine doğal uyum sağlar
- long branch'in rolünü daha aktif hale getirir
- concat'e göre daha uygun inductive bias sunar

Eksiler:

- biraz daha karmaşık
- yanlış ayarda long branch short'u fazla bastırabilir

Karar:

- `gated fusion`, concat'ten sonra ilk güçlü ana aday olacak

#### 3. Daha karmaşık fusion (cross-attention vb.)

Bu aile teorik olarak güçlü olsa da ilk iterasyon için gereksiz karmaşık görüldü.

Sebep:

- yorumlaması zor
- debug maliyeti yüksek
- erken aşamada representation problemini gereksiz yere ağırlaştırabilir

Karar:

- ilk iterasyonda kullanılmayacak

#### Fusion ilkesi

Stage 1A için mevcut tasarım ilkesi:

- `short latent` = ana taşıyıcı
- `long latent` = bağlamsal modülatör

Bu, şu zihinsel modele dayanır:

- short branch: "şu an ne oluyor?"
- long branch: "olan şeyi hangi hikâye içinde okumalıyım?"

İlk deney sırası:

1. `concat baseline`
2. `gated fusion`
3. gerekirse daha karmaşık fusion

### Latent Dimension Principles

Latent boyut seçimi yalnızca kapasite meselesi değil, short ve long branch'in temsil içindeki rol dağılımını da belirler.

Temel ilke:

- `z_short > z_long`

Gerekçe:

- short branch ana taşıyıcıdır
- local structure, wick/close farkı, displacement ve kısa swing ilişkileri daha yüksek çözünürlüklü bilgi taşır
- long branch ise bağlamsal, daha özet bilgi taşır

Bu nedenle boyut dağılımı da bu hiyerarşiyi yansıtmalıdır.

#### İlk aday boyutlar

Stage 1A için aday dağılımlar:

- `z_short = 64`, `z_long = 32`
- `z_short = 48`, `z_long = 24`
- `z_short = 64`, `z_long = 16`

İlk güçlü prior:

- `z_short = 64`
- `z_long = 32`

Concat baseline için:

- `z_fused = 96`

Bu boyut, ilk iterasyon için yeterli ama aşırı geniş olmayan temsil alanı sunar.

#### Neden short daha büyük?

Short encoder'ın taşıdığı bilgiler:

- lokal yapı
- kısa vadeli swing ilişkileri
- wick vs close farkı
- displacement detayları

Long encoder'ın taşıdığı bilgiler:

- daha geniş bağlam
- rejim / bias
- context özeti

Bu yüzden same-width yaklaşımı zorunlu görülmedi; hatta çoğu durumda gereksiz kapasite yaratabilir.

#### Boyut riskleri

Latent fazla büyük olursa:

- shortcut öğrenme kolaylaşır
- coin identity / volatility / zaman bilgisi daha rahat taşınır
- representation semantiği zayıflayabilir

Latent fazla küçük olursa:

- local structure yeterince temsil edilemez
- hard-case ayrımı zayıflar
- long bağlam anlamsız derecede sıkışabilir

Bu nedenle amaç:

- yapısal bilgiyi kaybetmeden sıkıştırmak
- ama aşırı geniş kapasite verip shortcut alanı açmamak

#### Long latent neden daha küçük olabilir?

Long branch'in görevi ince mikro detay değil, bağlamsal özet taşımaktır. Bu yüzden daha küçük latent:

- long branch'i gerçekten özet üretmeye zorlar
- local bilgiyi kopyalama riskini azaltır
- short ve long rol ayrımını güçlendirir

İlk karar:

- Stage 1A v1 için `64 + 32` güçlü varsayılan olacak
- sonraki adımda kontrollü küçültme deneyleri ile long latent'in gerçekten 32 boyut gerektirip gerektirmediği test edilecek

### Global Vector vs Sequence/Token Representation

Stage 1A için encoder çıktısı iki şekilde düşünüldü:

1. `Global vector`
2. `Short token / sequence representation`

#### Global vector

Yapı:

- short pencere -> tek latent vektör
- long pencere -> tek latent vektör

Artılar:

- daha sade tasarım
- fusion, objective ve probe kurulumları daha kolay
- ilk iterasyon için debug ve yorumlama maliyeti daha düşük

Karar:

- Stage 1A v1 global vector ile başlayacak

#### Sequence / token representation

Yapı:

- encoder, tüm pencereyi tek vektöre sıkıştırmak yerine
- pencerenin birkaç bölümünü temsil eden kısa bir token dizisi üretebilir

Örnek düşünce:

- short -> `K` adet kısa token
- long  -> `M` adet kısa token

Bu yaklaşımın motivasyonu:

- MSB gibi yapısal problemler yalnızca "hangi bilgi var" ile değil, "zaman içinde hangi sırayla oldu" ile de ilgilidir
- tüm zamanı çok erken tek vektöre sıkıştırmak, pencerenin iç yapısını fazla düzleştirebilir

Bu nedenle sequence/token yaklaşımı ileride ciddi alternatif olarak not edildi; ancak ilk iterasyonda ek karmaşıklık nedeniyle ertelendi.

#### Ne zaman yeniden gündeme alınacak

Şu belirtiler görülürse token/sequence temsili yeniden değerlendirilecek:

- global latent hard-case ayrımlarında zayıf kalırsa
- pencere içi olay sırası önemli görünür ama global vector bunu taşıyamazsa
- gated/concat fusion sonrası hâlâ yeterli yapı sinyali oluşmazsa

#### Pencere içi zaman bilgisi

Mevcut tasarımda pencere içi zaman bilgisi tamamen kayıp değildir:

- short ve long pencereler sıralı zaman dizisi olarak encoder'a girer
- yani bar sırası korunur
- ayrıca tüm pencereler aynı `end timestamp`e hizalanır; bu yüzden pencerenin sonuna yakın barların "karar anına daha yakın" olduğu örtük olarak korunur

Ancak ilk Stage 1A çekirdek setinde şu tür `explicit clock-time` feature'lar merkezde olmayacaktır:

- `hour_sin`
- `hour_cos`
- `dow_sin`
- `dow_cos`

Bu feature'lar shortcut riski nedeniyle ilk aşamada ya dışarıda tutulacak ya da kontrollü yardımcı grup olarak ele alınacaktır.

Sonuç:

- `relative time within window` korunuyor
- `absolute clock/calendar time` ilk sürümde ana sinyal olmayacak

### Temporal Position Handling

Stage 1A için pencere içindeki göreli zaman bilgisi önemlidir:

- olayın pencerenin başında mı sonunda mı olduğu
- break'in son barlara ne kadar yakın gerçekleştiği
- squeeze -> sweep -> breakout gibi sıralı yapılar

CNN tabanlı encoder'lar bu bilgiyi tamamen kaybetmez:

- giriş sıralı zaman dizisidir
- pencereler sabit uzunluktadır
- tüm örnekler aynı `end timestamp` mantığı ile hizalanır
- dolayısıyla pencerenin sonuna yakın pattern'ler örtük olarak farklı anlam taşıyabilir

Ancak bu positional bilgi ilk aşamada `explicit` değil, `implicit` taşınır.

Bu yüzden karar:

- Stage 1A v1'de ekstra positional encoding eklenmeden başlanacak
- fixed-length, same-end aligned windows + CNN başlangıç için yeterli kabul edilecek

Ne zaman yeniden gündeme alınacak:

- model pencerenin sonundaki teyit / kırılım olaylarını yeterince ayıramazsa
- hard-case'lerde olayın konumu belirleyici görünürse
- aynı pattern farklı pencere konumlarında farklı davranıyor ama latent bunu yakalayamıyorsa

Bu durumda ilk aday çözüm, ağır positional encoding değil, basit ek zaman kanalları olacak:

- `relative_position` channel
  - pencere içinde 0 -> 1
- `distance_to_end` channel
  - karar anına kaç bar kaldığı

İlke:

- önce implicit temporal bias ile başla
- ihtiyaç olursa explicit ama sade pozisyon kanalı ekle

### Encoder Family Choice

Stage 1A için ilk encoder ailesi olarak `temporal CNN / TCN-benzeri` yapı seçildi.

Karar gerekçesi:

- problem sabit uzunluklu pencere verisi üstünde çalışıyor
- local shape ve kısa yapısal örüntüler kritik
- training stabilitesi önemli
- erken aşamada modelin ne öğrenip öğrenmediğini yorumlamak kolay olmalı

Bu nedenle ilk tercih:

- residual temporal convolution blocks
- gerekirse kontrollü dilation

#### Neden CNN / TCN

- candle şekli, wick/body ilişkisi, squeeze -> breakout gibi lokal örüntüler için doğal
- sabit pencere verisinde verimli
- short/long branch tasarımına doğal uyum sağlıyor
- concat / gated fusion ile sade entegrasyon sunuyor
- mevcut proje deneyimi ve kod tabanı ile uyumlu

#### Neden ilk aşamada RNN değil

- eğitim ve tuning maliyeti daha yüksek olabilir
- paralellik ve verim daha zayıf
- lokal shape yakalama çoğu zaman convolution kadar doğal değildir

#### Neden ilk aşamada Transformer değil

- positional handling daha merkezi ve hassas hale gelir
- gereksiz erken karmaşıklık ekler
- debug ve yorumlama maliyeti yükselir
- representation problemi henüz yeterince sade kurulmadan model ailesini ağırlaştırmak risklidir

Karar:

- Stage 1A v1 için `temporal convolution family` ile başlanacak
- Transformer veya daha ağır sequence modelleri ancak global-vector CNN hattı sınırına dayanırsa ikinci nesil seçenek olarak ele alınacak

### Projection Head Role

Stage 1A'de contrastive / representation learning objective'i doğrudan ana latent üzerinde çalıştırmak yerine, küçük bir projection head kullanmak tercih edildi.

Temel ayrım:

1. `encoder / fused latent`
   - ana temsil
   - probe, transfer ve downstream görevlerde değerlendirilecek uzay
2. `projection head output`
   - contrastive objective için yardımcı uzay
3. `classification head`
   - auxiliary CE loss için küçük başlık

Önerilen akış:

- `z_enc` = short + long fusion sonrası ana latent
- `z_proj` = projection MLP(`z_enc`)
- `L_ce` -> class head(`z_enc`)
- `L_supcon` -> `z_proj`

Bu ayrımın faydası:

- contrastive objective'in sert geometrik baskısı doğrudan ana latent'i kilitlemez
- ana latent daha transfer-edilebilir ve çok amaçlı kalabilir
- CE ve SupCon rollerini ayırmak kolaylaşır

Karar:

- Stage 1A objective'lerinde projection head tasarıma dahil edilecek
- projection head küçük ve yardımcı rol üstlenecek
- nihai değerlendirilen temsil `z_proj` değil, `z_enc` olacak

Related: [2026-04-10-msb-label-improvements.md](2026-04-10-msb-label-improvements.md)  
Related: [2026-04-11-multi-symbol-dual-window.md](2026-04-11-multi-symbol-dual-window.md)  
Related: [MULTI_SCALE_MSB_INTERPRETATION.md](../MULTI_SCALE_MSB_INTERPRETATION.md)  
Related: [TEMPORAL_CNN_BASICS.md](../TEMPORAL_CNN_BASICS.md)  
See: [WORKLOG_GUIDE.md](../WORKLOG_GUIDE.md)

---

**Status**: In Progress
