# Worklog: 2026-04-14 — CE-Only Bucket Readout on core4 Val

**Status**: Complete ✓  
**Goal**: domain bucket sonuçlarını ayrı bir araştırma notu olarak toplamak ve her bucket'ın ne anlama geldiğini açıkça belgelemek

---

## Neden Ayrı Bir Not

Bucket altyapısı daha önce eklendi, ancak iki ayrı ihtiyaç vardı:

- bucket'ların teknik olarak nasıl üretildiğini belgelemek
- bucket sonuçlarının araştırma açısından ne söylediğini ayrı bir yerde okumak

Bu not ikinci ihtiyaca cevap verir.

Amaç:

- her bucket'ın domain anlamını netleştirmek
- bucket'ların hangi tip örneği temsil ettiğini açıklamak
- `CE-only` baseline'ın bu bucket'larda nasıl davrandığını özetlemek

## Kaynak

Okuma şu çıktı üstünden yapıldı:

- `data/stage1a/binance/15m/checkpoints/ce_only/analysis/val_latents/symbol_bucket_full_distribution.csv`

Dataset:

- universe: `core4`
- symbols: `BTC / ETH / SOL / XRP`
- split: `val`
- objective: `CE-only`

## Bucket Sözlüğü

### `clean_intact`

Anlamı:

- örnek `intact`
- belirgin wick taşması veya kırılım eşiğine çok yaklaşma yok

Nasıl değerlendirilmeli:

- bu bucket, "normal intact state" referansıdır
- burada hata düşükse model intact state'in merkezini iyi öğrenmiştir
- burada hata yüksekse modelin temel state ayrımı zayıf demektir

Val okuması:

- BTC: `4.8%`
- ETH: `8.8%`
- SOL: `7.5%`
- XRP: `6.3%`

Yorum:

- bu bucket genel olarak zor değil
- model intact çekirdeğini makul biçimde öğrenmiş

### `high_vol_intact`

Anlamı:

- örnek hâlâ `intact`
- ama kısa pencere volatilitesi yüksek

Nasıl değerlendirilmeli:

- bu bucket, "volatil ama henüz gerçek break değil" bölgesidir
- hata artıyorsa model volatiliteyi break ile karıştırıyor olabilir

Val okuması:

- BTC: `6.7%`
- ETH: `13.2%`
- SOL: `14.7%`
- XRP: `12.4%`

Yorum:

- clean intact'a göre daha zor
- özellikle ETH / SOL / XRP tarafında yüksek volatilite sınır bulanıklığı yaratıyor

### `high_vol_intact_wick_sweep`

Anlamı:

- örnek `intact`
- volatilite yüksek
- aynı anda wick-led failed break işareti var

Nasıl değerlendirilmeli:

- bu bucket pratikte en "tuzak" intact örneklerinden biridir
- wick, fiyatı seviyenin ötesine taşır ama close teyidi gelmez
- modelin bunu gerçek break sanma riski yüksektir

Val okuması:

- BTC: `66.7%`
- ETH: `31.6%`
- SOL: `22.6%`
- XRP: `21.7%`

Yorum:

- bu bucket en problemli sınırlardan biri
- özellikle BTC'de örnek sayısı az ama hata oranı çok yüksek
- failed break + volatility kombinasyonu belirgin zorlayıcı alan

### `wick_sweep_up`

Anlamı:

- yukarı yönde wick ile seviye aşılmış
- close ile teyit yok
- label hâlâ `intact`

Nasıl değerlendirilmeli:

- bullish false break / liquidity sweep benzeri örnekler burada toplanır
- model bunları bullish break ile karıştırıyorsa local wick davranışı tam çözülmemiştir

Val okuması:

- BTC: `7.5%`
- ETH: `13.1%`
- SOL: `10.0%`
- XRP: `9.2%`

Yorum:

- orta zorlukta bucket
- true break kadar temiz değil, ama borderline bucket kadar da ağır değil

### `wick_sweep_down`

Anlamı:

- aşağı yönde wick ile seviye aşılmış
- close teyidi yok
- label hâlâ `intact`

Nasıl değerlendirilmeli:

- bearish false break / downside liquidity sweep bölgesidir
- model bearish state'e aşırı hevesliyse burada hata artar

Val okuması:

- BTC: `5.3%`
- ETH: `11.5%`
- SOL: `9.9%`
- XRP: `10.9%`

Yorum:

- wick_sweep_up ile benzer ama bazı coinlerde biraz daha temiz
- yine de intact sınırının kolay bölgesi değil

### `borderline_intact_break`

Anlamı:

- örnek resmen `intact`
- fakat close count, break teyidine sadece bir adım uzak

Nasıl değerlendirilmeli:

- bu bucket Stage 1A için en kritik sınır bölgesidir
- tam olarak "henüz break mi değil mi?" gri alanını temsil eder
- burada hata artması beklenir; asıl mesele bunun ne kadar arttığıdır

Val okuması:

- BTC: `20.1%`
- ETH: `24.0%`
- SOL: `32.6%`
- XRP: `26.1%`

Yorum:

- en net sorun bucket'ı bu
- dört sembolde de belirgin biçimde zor
- özellikle SOL burada ciddi sınır belirsizliği gösteriyor

### `close_confirmed_break_up`

Anlamı:

- bullish break, close sayısıyla teyit edilmiş
- ama displacement çok güçlü olmayabilir

Nasıl değerlendirilmeli:

- bu bucket "geçerli ama merkezî olmayan bullish break" gibi okunmalı
- burada hata varsa model bullish state'i tanıyor ama yeterince temiz ayıramıyor olabilir

Val okuması:

- BTC: `12.8%`
- ETH: `11.7%`
- SOL: `12.1%`
- XRP: `11.5%`

Yorum:

- orta düzey zorluk
- bullish teyitli break öğrenilmiş, ama tamamen temiz değil

### `close_confirmed_break_down`

Anlamı:

- bearish break, close sayısıyla teyit edilmiş
- displacement güçlü olmayabilir

Nasıl değerlendirilmeli:

- bearish state için teyitli ama merkezî olmayan break alanıdır
- close-confirmed bullish ile birlikte ana break kütlesini oluşturur

Val okuması:

- BTC: `15.1%`
- ETH: `13.9%`
- SOL: `10.1%`
- XRP: `11.8%`

Yorum:

- bullish muadiliyle benzer zorlukta
- BTC tarafında biraz daha zor
- yön karışıklığından çok intact sınırına yakınlık etkili olabilir

### `true_break_up`

Anlamı:

- bullish break teyitli
- aynı zamanda displacement güçlü

Nasıl değerlendirilmeli:

- bu bucket "merkezî bullish örnekler" gibi düşünülmeli
- model burada zorlanıyorsa temsil çok ciddi sorunludur

Val okuması:

- BTC: `0.0%`
- ETH: `0.0%`
- SOL: `3.3%`
- XRP: `2.9%`

Yorum:

- çok temiz bucket
- model güçlü bullish structural shift'i iyi öğrenmiş

### `true_break_down`

Anlamı:

- bearish break teyitli
- displacement güçlü

Nasıl değerlendirilmeli:

- bearish tarafın merkezî örnekleri
- burada düşük hata, modelin bearish structural shift'i sahiden tanıdığını gösterir

Val okuması:

- BTC: `1.7%`
- ETH: `0.0%`
- SOL: `0.0%`
- XRP: `1.3%`

Yorum:

- en temiz bucket'lardan biri
- `CE-only` baseline bearish true break state'ini de iyi öğrenmiş

## Toplu Okuma

Bucket tarafında çıkan ana tablo şu:

- en kolay alan: `true_break_up` / `true_break_down`
- orta zorluk: `close_confirmed_break_up` / `close_confirmed_break_down`
- zor alan: `wick_sweep_*`
- en zor alan: `borderline_intact_break` ve özellikle `high_vol_intact_wick_sweep`

Bu çok önemli çünkü:

- sorun `bullish vs bearish` yön ayrımı değil
- sorun "henüz intact mı, yoksa artık state shift başladı mı?" sınırı

Başka deyişle:

- classifier head güçlü break örneklerini gayet iyi ayırıyor
- semantik zorluk, teyit öncesi ve failed-break alanında toplanıyor

## Coin Bazlı Kısa Notlar

BTC:

- genel olarak temiz
- ama `high_vol_intact_wick_sweep` çok sert problemli

ETH:

- dağılım dengeli
- `borderline_intact_break` ve volatil-intact bucket'larında zorluk belirgin

SOL:

- en belirgin sınır problemi burada
- `borderline_intact_break` hatası `32.6%`

XRP:

- teyitli break bucket'ları makul
- borderline ve volatil intact alanları yine zor

## Ne Anlama Geliyor

Bu bucket okuması sonrası daha net söyleyebiliriz:

- `CE-only` baseline başarısız değil
- hatta güçlü true-break state'leri iyi öğrenmiş
- fakat Stage 1A'nın asıl açık problemi, break teyidi öncesindeki gri alan

Bu da sonraki adımlar için yön verir:

- `SupCon` eklenecekse hedef genel accuracy değil
- özellikle `borderline_intact_break`, `wick_sweep_*`, `high_vol_intact_wick_sweep` bucket'larında latent geometry'yi temizlemek olmalı

## Related

- `docs/worklogs/2026-04-13-hard-case-domain-buckets.md`
- `docs/worklogs/2026-04-14-ce-only-embedding-view-readout.md`
- `data/stage1a/binance/15m/checkpoints/ce_only/analysis/val_latents/symbol_bucket_full_distribution.csv`

---

**Status**: Complete ✓
