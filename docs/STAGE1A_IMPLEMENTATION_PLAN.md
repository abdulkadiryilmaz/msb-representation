# Stage 1A Implementation Plan

Bu doküman, `Stage 1A` için üzerinde uzlaşılan ilk uygulama sırasını özetler. Amaç, tasarım tartışmasını worklog'dan ayırıp uygulama aşamasında takip edilebilir bir plan bırakmaktır.

İlgili tasarım worklog'u:

- [2026-04-12-stage1-msb-representation-framing.md](worklogs/2026-04-12-stage1-msb-representation-framing.md)

## Scope

Stage 1A amacı:

- `MSB-oriented representation learning`
- final ürün olarak doğrudan classifier üretmek değil
- `bullish / bearish / intact` label'larını latent hizalama supervision'ı olarak kullanmak

Bu plan, yalnızca ilk çalışan sürümü hedefler.

## Stage 1A Dataset Contract

Stage 1A hattı mevcut `data/windows/` çıktısını doğrudan mutate etmeyecek. Ayrı bir dataset root kullanılacak:

```text
data/stage1a/<exchange>/<timeframe>/
  spec.json
  symbols/
    BTC_USDT_15m.npz
    ETH_USDT_15m.npz
    XRP_USDT_15m.npz
```

`spec.json` içeriği:

- exchange / timeframe
- sembol listesi
- short / long window parametreleri
- label prior parametreleri
- opsiyonel train / val split sınırları
- short / long feature kolonları

Her sembol dosyası (`symbols/<symbol>.npz`) şu alanları taşıyacak:

- `short_windows`: `(N, F_short, 48)`
- `long_windows`: `(N, F_long, 288)`
- `labels`: `(N,)`
- `timestamps`: `(N,)`

Karar gerekçesi:

- Stage 1A hattı ayrı bir namespace altında kalır
- short / long / label hizalaması tek dosyada tutulur
- loader tarafında inner-join veya çok dosya yönetimi tekrarlanmaz
- mevcut AE / joint veri hattı bozulmaz
- dataset build her zaman explicit symbol allowlist ile yapılır; `raw/` altındaki tüm parquet'leri otomatik toplamak Stage 1A deney sözleşmesini gevşetir
- tercih edilen yol named dataset profile kullanmaktır; böylece sembol evreni versiyonlu ve tekrar üretilebilir kalır

V1'de normalizasyon bu dataset dosyalarına gömülmeyecek.

- dataset unnormalized feature window taşıyacak
- normalizer yalnızca train split üstünde fit edilecek
- checkpoint ile birlikte saklanacak

## Feature Freeze

V1 shared-core feature ailesi:

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

Karar:

- short ve long branch aynı feature ailesiyle başlayacak
- long branch farkını esas olarak pencere uzunluğu taşıyacak
- scale-specific extra feature'lar ilk sürüme alınmayacak

## Label Freeze

V1 label, `event-style` değil `state-style` olacak.

- structure zone: ilk `36` bar
- recent zone: son `12` bar
- pivot lookback: `3`
- min break prior: `max(0.2%, 0.5 * ATR / close)`
- label kuralı: recent zone içinde yeterli sayıda close son confirmed swing üstünde / altında kalıyorsa bullish / bearish

Not:

- aynı state birkaç ardışık anchor'da tekrar edebilir
- bu davranış ilk sürüm için kabul edilir
- confirmation horizon bu aşamada yoktur; Stage 1A event trigger değil temsil katmanıdır

## V1 Design Freeze

İlk uygulama için sabitlenecek varsayılanlar:

- short window: `15m x 48`
- long window: `15m x 288`
- aynı `end timestamp` hizalaması
- shared-core feature ailesi
- temporal CNN / TCN-benzeri encoder ailesi
- `z_short = 64`
- `z_long = 32`
- fusion baseline: `concat`
- objective baseline:
  - önce `CE only`
  - sonra `CE + SupCon`
- projection head dahil
- ana temsil: `z_enc`

Stage 1A supervision yönü:

- `event-style` değil, `state-style`
- label, final output değil; latent hizalama supervision'ı
- sınıflar:
  - `0 = intact`
  - `1 = bullish`
  - `2 = bearish`

İlk label prior'ı:

- `current-structure oriented`
- trade outcome veya tradability içermeyecek
- yapısal durum supervision'ı verecek

## Uygulama Sırası

### 1. Veri hattını kur

Yeni Stage 1A hattı için:

- short window builder
- long window builder
- shared-core feature seti
- aynı end timestamp hizalaması
- train/val/test split

Not:

- `bullish / bearish / intact` label'ları veri hazırlama aşamasında rule-based üretilir
- modelin içinde üretilmez

İlk Stage 1A label prior parametreleri:

- short window = `48 bar`
- `structure zone = first 36 bars`
- `recent zone = last 12 bars`
- pivot lookback `N = 3`
- reference structure = structure zone içindeki son confirmed swing high / swing low
- minimum break margin başlangıç prior'ı:
  - `min_break_pct = 0.2%`
  - `atr_factor = 0.5`

Not:

- `recent zone` state-style supervision için kullanılır
- aynı yapının birkaç ardışık anchor'da aynı state'i taşıması bu aşamada kabul edilebilir
- event / transition mantığı Stage 1B veya signal layer tarafında ele alınacaktır

Sanity check notu:

- NEAR son günler üstünde yapılan ilk kural taraması, bullish / bearish state bölgelerinin makul biçimde çıktığını gösterdi
- aynı zamanda state-style supervision'ın beklenildiği gibi `persistent` olduğu da görüldü
- yani aynı bullish veya bearish yapı birkaç ardışık anchor boyunca aynı label'ı taşıyabiliyor

Bu davranış ilk sürüm için kabul edildi.

Sebep:

- Stage 1A amacı event trigger üretmek değil
- current structural state temsili öğrenmektir

İlk tuning eksenleri:

- `recent zone` uzunluğu
- `k close` sayısı
- margin seviyesi
- son close filtresinin katılığı

Split tercihi:

- ana split ekseni `zaman` olacak
- `train`, `val`, `test` kronolojik ayrılacak
- tüm seçili eğitim coin'leri her split döneminde bulunabilecek
- `test`, train ve val'de kullanılmamış `OOS (out-of-sample)` dönem olacak

Coin-held-out transfer testleri:

- ana split değil
- ikinci faz ek evaluation olarak ele alınacak

Rejim anomalileri:

- Luna çöküşü, sert correction dönemleri, panic regime gibi gerçek piyasa anomalileri varsayılan olarak dataset'ten çıkarılmayacak
- ancak bu dönemler ayrı slice / tag / analiz başlığı olarak izlenecek
- gerekirse sonraki iterasyonlarda sample weighting, capped sampling veya ablation ile etkileri kontrol edilecek
- yalnızca veri bozukluğu, exchange outage veya piyasa dışı açık veri hataları gibi durumlar varsayılan çıkarma adayı sayılacak

### 2. Dataset / loader yapısını kur

İlk dataset yapısı:

- short window
- long window
- label
- timestamp
- symbol

Amaç:

- training ve eval için tek, temiz veri arayüzü sağlamak

Bu adımın somut kod karşılığı:

- `src/msb_repr/stage1a/config.py`
- `src/msb_repr/stage1a/features.py`
- `src/msb_repr/stage1a/labels.py`
- `src/msb_repr/stage1a/dataset.py`
- `scripts/build_stage1a_dataset.py`

Örnek build komutu:

```bash
python scripts/build_stage1a_dataset.py \
  --data-root data \
  --exchange binance \
  --timeframe 15m \
  --bars-per-day 96 \
  --dataset-profile core4 \
  --train-end 2025-09-30 \
  --val-end 2025-12-31
```

İlk named profile'lar:

- `core4` -> `BTC_USDT_15m ETH_USDT_15m SOL_USDT_15m XRP_USDT_15m`
- `altcoin6` -> `BTC_USDT_15m ETH_USDT_15m SOL_USDT_15m XRP_USDT_15m NEAR_USDT_15m INJ_USDT_15m`
- `pipeline3` -> `BTC_USDT_15m XRP_USDT_15m NEAR_USDT_15m`

### 3. Minimum model iskeleti

İlk çalışan sürüm:

- short encoder
- long encoder
- concat fusion
- auxiliary class head
- projection head

İlk deneme:

- `CE only`

Amaç:

- shape / alignment / training loop / basic metrics hattını ayağa kaldırmak

Bu adımın somut kod karşılığı:

- `src/msb_repr/stage1a/model.py`
- `scripts/train_stage1a.py`

V1 model kontratı:

- short encoder -> `z_short = 64`
- long encoder -> `z_long = 32`
- fusion -> `concat`
- fused latent -> auxiliary class head
- fused latent -> projection head
- ana temsil -> `z_fused`

İlk teslimat kriteri:

- dataset root okunabiliyor olmalı
- bir batch short/long forward pass çalışmalı
- `z_short`, `z_long`, `z_fused`, `z_proj`, `logits` tensor shape'leri doğrulanmalı

### 4. İlk evaluation araçları

Model ile birlikte erken aşamada kurulması gereken araçlar:

- latent export
- probe evaluation
- PCA / UMAP görselleştirme
- class-wise F1
- nearest-neighbor inceleme
- short vs long vs fused karşılaştırma

Bu adım sona bırakılmayacak; Stage 1A'in başarısı zaten bu araçlarla okunacak.

### 5. Baseline deney sırası

İlk deney sırası:

1. `CE only + concat`
2. `CE + SupCon + concat`
3. `CE + SupCon + gated fusion`

Bu sırada başka eksenler sabit kalacak:

- window
- feature set
- latent boyut
- resolution

### 6. Hard-case örnek seti hazırla

Küçük ama elle seçilmiş örnek havuzu:

- `high-vol intact`
- `true break`
- `wick sweep`
- `close-confirmed break`

Amaç:

- modelin volatility mi yoksa structure mı öğrendiğini erken görmek

### 7. İlk karar noktası

İlk üç deneyden sonra şu sorular cevaplanacak:

- fused latent short-only'den daha iyi mi
- long branch gerçekten ek bilgi getiriyor mu
- `CE + SupCon` geometry'yi iyileştiriyor mu
- gated fusion anlamlı katkı veriyor mu

Bu noktada ancak yeterli sinyal varsa ikinci nesil fikirler açılacak.

## İkinci Faz Adayları

V1 sonrası adaylar:

- `1h long` varyantı
- explicit temporal position channel
- sequence / token representation
- Stage 1B future structure task
- ileride `4h macro branch`

## Uygulama Prensibi

- önce sade ama ölçülebilir sistem
- sonra tek değişkenli deneyler
- tasarım tartışmasını uygulama sırasında genişletmemek

## Not

MSB label üretiminin ayrıntıları bu dokümanda açılmıyor. Uygulama sırasında gerekirse ayrı worklog veya teknik not ile belgelenecek.
