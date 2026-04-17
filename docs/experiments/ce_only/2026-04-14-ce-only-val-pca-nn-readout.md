# Worklog: 2026-04-14 — CE-Only Val PCA / UMAP / Nearest-Neighbor Readout

**Status**: Complete ✓  
**Goal**: daha önce üretilmiş val latent artefaktlarını gerçekten okuyup `CE-only` latent geometry hakkında açık bir yorum yazmak

---

## Neden Bu Not

Val latent export ve analiz çıktıları üretilmişti, ancak bunların araştırma okuması eksikti.

Bu not şu sorulara cevap verir:

- val latent uzayı gerçekten organize görünüyor mu
- PCA tarafında yapı ne kadar düşük boyuta sıkışıyor
- nearest-neighbor davranışı ne söylüyor
- hata örnekleri hangi tip komşulara çekiliyor

## Kaynak

Kullanılan artefaktlar:

- `data/stage1a/binance/15m/checkpoints/ce_only/analysis/val_latents/summary.json`
- `.../pca_projection.csv`
- `.../umap_projection.csv`
- `.../nearest_neighbors.csv`
- `.../embedding_compare/embedding_overall_summary.csv`

## First Read

Val özet:

- accuracy: `0.8885`
- mean confidence: `0.8897`
- mean neighbor label agreement: `0.8437`

PCA explained variance:

- PC1: `0.5499`
- PC2: `0.3793`
- ilk iki bileşen toplamı: yaklaşık `0.9292`

Bu önemli çünkü:

- fused latent varyansının çok büyük kısmı ilk iki eksende taşınıyor
- bu, latent uzayın tamamen dağınık olmadığını gösteriyor
- state ayrımının düşük boyutlu baskın eksenler üzerinden örgütlendiği düşünülebilir

Bu noktada dikkat:

- burada gerçek scatter görseli üretip manuel cluster çizimi yapmadık
- elimizdeki okuma CSV + explained variance + NN davranışından geliyor

## UMAP Notu

`umap_projection.csv` başarıyla üretildi.

Bu da şu anlama geliyor:

- artefakt hazır
- ama bu notta UMAP koordinatları üzerinden manuel görsel cluster yorumu yapılmadı

Başka deyişle:

- UMAP output mevcut
- fakat bugüne kadar asıl okuma nearest-neighbor ve bucket bazlı analiz üstünden yapıldı

## Nearest-Neighbor Readout

İlk misclassified örnekler özellikle şunu gösteriyor:

- bazı bullish label örnekleri latent uzayda yoğun intact komşulara çekiliyor
- bazı intact örnekler de bullish komşulara yakınlaşıyor

Bu, önceki confusion ve bucket sonuçlarıyla tutarlı:

- ana sorun `bullish vs bearish` yön ayrımı değil
- ana sorun `intact <-> break` sınırı

Val NN örneklerinde görülen desen:

- yanlış sınıflandırılmış bullish örneğin ilk komşuları çoğu zaman `intact`
- üstelik bu komşular bazen cross-symbol geliyor
- yani latent geometry tamamen coin içine kapanmıyor

Bu olumlu bir sinyal:

- en azından `z_fused` düzeyinde cross-symbol semantik komşuluk var

Ama aynı anda şu risk de var:

- semantik olarak sınırda olan bullish örnekler kolayca intact manifold'una düşebiliyor

## Embedding View Context

Val embedding compare sonuçları:

- `z_short`: label agreement `0.8436`, symbol agreement `0.2793`
- `z_long`: label agreement `0.4442`, symbol agreement `0.9093`
- `z_fused`: label agreement `0.8437`, symbol agreement `0.5739`

Bu nedenle val PCA / NN okuması tek başına değil, şu bağlamda anlaşılmalı:

- `z_short` asıl semantik taşıyıcı
- `z_long` coin/context ağırlıklı
- `z_fused` short semantiğini koruyor, ama context de topluyor

Nearest-neighbor readout da bu tabloya uyuyor.

## What It Means

Val tarafında şu sonucu daha rahat söyleyebiliriz:

- latent uzay rastgele değil
- güçlü state örnekleri organize
- hata örnekleri büyük ölçüde intact-break sınırında
- cross-symbol semantik yakınlık mevcut
- ama borderline örnekler intact manifold'una kayabiliyor

Bu, `CE-only` için olumlu ama sınırlı bir sonuçtur:

- geometry çalışıyor
- fakat henüz sınır örneklerinde yeterince sert değil

## Conclusion

Val PCA / UMAP / NN tarafında ana mesaj:

- `CE-only` latent uzayı anlamlı bir organizasyon gösteriyor
- güçlü break örnekleri için temsil temiz
- en kritik eksik, intact-break sınırında manifold ayrımı

Yani bugünkü okuma, bucket ve embedding compare bulgularını destekliyor; onlarla çelişmiyor.

## Related

- `docs/experiments/ce_only/2026-04-14-ce-only-bucket-readout.md`
- `docs/experiments/ce_only/2026-04-14-ce-only-embedding-view-readout.md`

---

**Status**: Complete ✓
