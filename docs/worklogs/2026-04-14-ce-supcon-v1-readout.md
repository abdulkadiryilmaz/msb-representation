# Worklog: 2026-04-14 — CE + SupCon V1 Readout

**Status**: Complete ✓  
**Goal**: İlk `CE + SupCon` koşusunun classifier, latent geometry ve bucket davranışı açısından `CE-only` baseline'a göre ne getirdiğini kayda geçirmek

---

## Experiment Setup

İlk `CE + SupCon` koşusu şu ayarla çalıştırıldı:

- dataset: `core4`
- timeframe: `15m`
- short window: `48`
- long window: `288`
- objective: `CE + 0.05 * SupCon(z_proj, y)`
- temperature: `0.1`
- batch size: `64`
- checkpoint dir: `data/stage1a/binance/15m/checkpoints/ce_supcon_v1`

Eğitim `mps` üzerinde koştu ve early stopping ile epoch `30`'da durdu.

En iyi val macro-F1:

- `CE-only`: `0.8901`
- `CE + SupCon v1`: `0.8908`

Dolayısıyla ilk koşu classifier tarafında açık bir regression üretmedi. Çok küçük de olsa pozitif bir fark verdi.

## Neye Bakıldı

Bu koşu sonrası hem `val` hem `test` split için aynı latent analysis hattı çalıştırıldı:

- latent export
- nearest-neighbor readout
- hard-case ve bucket çıktıları
- `z_short`, `z_long`, `z_fused` embedding compare

Bu notta asıl karar ekseni şuydu:

- `SupCon`, özellikle `z_long` ve `z_fused` tarafında daha semantik komşuluk üretiyor mu?
- bunu yaparken symbol shortcut etkisini azaltıyor mu, yoksa büyütüyor mu?

## Val Readout

### 1. `z_short` büyük ölçüde aynı kaldı

Val tarafında `z_short` için tablo neredeyse yatay:

- mean NN label agreement: `0.8436 -> 0.8417`
- top-1 label match: `0.8442 -> 0.8479`
- mean NN symbol agreement: `0.2793 -> 0.2720`

Bu şu anlama geliyor:

- short branch bozulmadı
- ama belirgin bir semantik sıçrama da gelmedi

### 2. `z_long` daha label-aware oldu

Val tarafında en büyük değişim `z_long` üstünde geldi:

- mean NN label agreement: `0.4442 -> 0.4746`
- top-1 label match: `0.5692 -> 0.6709`

Bu güçlü bir sinyal. Yani contrastive objective, long branch'in latent düzenini daha sınıf-duyarlı hale getirdi.

### 3. Aynı anda `z_long` daha symbol-heavy hale geldi

Aynı `z_long` için:

- mean NN symbol agreement: `0.9093 -> 0.9493`
- top-1 symbol match: `0.9678 -> 0.9906`

Yani semantik kazanç, shortcut baskısı azalarak gelmedi. Tam tersine coin-level kümelenme daha da güçlendi.

### 4. `z_fused` tarafında net kazanım yok

Val tarafında `z_fused` için:

- mean NN label agreement: `0.8437 -> 0.8431`
- top-1 label match: `0.8466 -> 0.8553`
- mean NN symbol agreement: `0.5739 -> 0.6001`

Bu tabloyu olumlu okumak zor:

- top-1 label match biraz artıyor
- ama genel label agreement yatay
- symbol agreement ise yükseliyor

Yani fused latent tarafında henüz temiz bir geometry iyileşmesi yok.

## Test Readout

Test split, val bulgusunu büyük ölçüde doğruladı.

### 1. `z_short` testte hafif geriledi

- mean NN label agreement: `0.8438 -> 0.8380`
- top-1 label match: `0.8438 -> 0.8389`
- symbol agreement hafif düştü

Bu küçük bir fark, ama `SupCon` kısa branch'te açık bir fayda getirmedi.

### 2. `z_long` testte de daha label-aware

- mean NN label agreement: `0.4504 -> 0.4838`
- top-1 label match: `0.5689 -> 0.6779`

Val'deki artış OOS test'te de korunuyor. Bu önemli çünkü `z_long` üzerindeki etki sadece val-içi bir artefakt gibi görünmüyor.

### 3. Ama `z_long` testte de daha symbol-heavy

- mean NN symbol agreement: `0.9115 -> 0.9470`
- top-1 symbol match: `0.9707 -> 0.9893`

Dolayısıyla test tarafında da aynı gerilim korunuyor:

- daha iyi label-aware komşuluk
- ama aynı anda daha güçlü symbol shortcut

### 4. `z_fused` testte ikna edici değil

- mean NN label agreement: `0.8445 -> 0.8397`
- top-1 label match: `0.8511 -> 0.8406`
- mean NN symbol agreement: `0.5970 -> 0.6346`

Bu, test tarafında fused latent için net negatif bir okuma veriyor:

- semantik komşuluk hafif zayıflamış
- symbol bias ise artmış

## Bucket Readout

Asıl hedef bucket'lar:

- `borderline_intact_break`
- `high_vol_intact_wick_sweep`
- `wick_sweep_up`
- `wick_sweep_down`
- `close_confirmed_break_up`
- `close_confirmed_break_down`

### 1. `z_long` bucket düzeyinde de toparlanıyor

Hem val hem test'te `z_long` için şu desen tekrarlandı:

- `borderline_intact_break` top-1 label match artıyor
- `wick_sweep_*` top-1 label match artıyor
- `high_vol_intact_wick_sweep` hata oranı düşebiliyor
- `close_confirmed_break_*` bucket'larında label agreement artıyor

Bu, contrastive objective'in long branch'e gerçekten bir düzen verdiğini düşündürüyor.

### 2. Fakat aynı bucket'larda symbol agreement da yükseliyor

Neredeyse tüm kritik bucket'larda:

- `z_long` NN symbol agreement artıyor
- `z_fused` NN symbol agreement de çoğunlukla artıyor

Yani model bu bucket'larda daha doğru komşular buluyor olabilir, ama bunu kısmen coin identity'yi daha sert kullanarak yapıyor olabilir.

### 3. `borderline_intact_break` hâlâ ana açık problem

En kritik sınır bucket'ı olan `borderline_intact_break` için:

- `z_long` label match artıyor
- ama `z_fused` tarafında iyileşme yok
- test'te `z_fused` top-1 label match düşüyor

Bu önemli çünkü asıl araştırma hedefi fused latent davranışı. Orada henüz çözülmüş bir ilerleme yok.

### 4. Bazı zor bucket'larda yerel artılar var

Özellikle:

- `high_vol_intact_wick_sweep`
- bazı `wick_sweep_*` dilimleri

için hata oranında veya top-1 label match'te iyileşme görülüyor.

Bu yüzden deney tamamen negatif değil. Ama bu iyileşmelerin çoğu fused uzaya temiz ve shortcut-free biçimde taşınmış görünmüyor.

## Overall Decision

İlk `CE + SupCon` koşusu için en dürüst karar şu:

- classifier performansı korunuyor
- `z_long` branch semantik olarak belirgin biçimde iyileşiyor
- fakat aynı anda symbol shortcut etkisi de büyüyor
- `z_fused` tarafında ise henüz ikna edici bir genel kazanç yok

Başka deyişle:

- deney başarısız değil
- ama hedeflenen problemi tam çözmüş de değil

Bu koşu, `SupCon` sinyalinin yararlı olabileceğini gösterdi. Fakat mevcut positive tanımı ve uygulama noktası ile elde edilen fayda, symbol shortcut etkisinden yeterince ayrışmıyor.

## What This Means

Bu not sonrası artık daha net biliyoruz:

- `SupCon` tamamen yanlış yönde değil
- contrastive signal özellikle `z_long`'u harekete geçiriyor
- fakat salt label-based batch positive tanımı, multi-symbol setup içinde shortcut baskısını da büyütebiliyor

Bu yüzden bir sonraki varyant için asıl soru artık:

- `SupCon` kalsın mı kalmasın mı?

değil.

Asıl soru:

- `SupCon`'u hangi temsil üstünde ve hangi positive/negative tanımıyla uygularsak fused latent'te gerçek semantik kazanç alırız?

## Recommended Next Directions

Bu koşudan sonra iki mantıklı yön var:

1. branch-aware varyant

- `SupCon`'u yalnızca `z_long` ya da branch-specific projection üstünde denemek

2. pair definition varyant

- positive eşleşmeleri yalnızca label ile değil, symbol shortcut baskısını azaltacak ek kısıtlarla kurmak

Örneğin:

- batch sampling dengesini değiştirmek
- symbol-aware weighting eklemek
- aynı symbol içi positive etkisini sınırlamak

## Related

- `docs/worklogs/2026-04-14-ce-supcon-experiment-plan.md`
- `docs/worklogs/2026-04-14-ce-only-synthesis.md`
- `docs/worklogs/2026-04-14-ce-only-embedding-view-readout.md`
- `docs/worklogs/2026-04-14-ce-only-bucket-readout.md`
- `docs/worklogs/2026-04-14-ce-only-test-split-readout.md`

---

**Status**: Complete ✓
