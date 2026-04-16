# Worklog: 2026-04-14 — CE-Only Embedding View Readout (`z_short` / `z_long` / `z_fused`)

**Status**: Complete ✓  
**Goal**: `CE-only` baseline'da short branch, long branch ve fused latent'in gerçekten ne taşıdığını netleştirmek

---

## Problem Statement

`CE-only` baseline iyi classification verdi, ancak şu soru açıktı:

- bu ayrımı esas olarak hangi latent taşıyor?

Özellikle görmek istediğimiz şey:

- `z_short` gerçekten structural state semantiğini mi taşıyor
- `z_long` broader context mi taşıyor, yoksa coin identity shortcut'una mı kayıyor
- `z_fused`, short bilgisini koruyup anlamlı bağlam ekliyor mu

Bu nedenle aynı val export üstünde `z_short`, `z_long` ve `z_fused` nearest-neighbor davranışı karşılaştırıldı.

## Setup

Kullanılan kaynak:

- latent export: `data/stage1a/binance/15m/checkpoints/ce_only/analysis/val_latents.npz`
- karşılaştırma script'i: `scripts/compare_stage1a_embedding_views.py`

Ana metrikler:

- mean nearest-neighbor label agreement
- mean nearest-neighbor symbol agreement
- top-1 label match rate
- top-1 symbol match rate

Okuma mantığı:

- yüksek `label agreement` -> latent structural state semantiğini iyi taşıyor
- yüksek `symbol agreement` -> latent coin identity / symbol-specific context'e yakın davranıyor

## Overall Result

Genel sonuç:

- `z_short`
  - mean NN label agreement: `0.8436`
  - mean NN symbol agreement: `0.2793`
  - top-1 label match: `0.8442`
  - top-1 symbol match: `0.3135`
- `z_long`
  - mean NN label agreement: `0.4442`
  - mean NN symbol agreement: `0.9093`
  - top-1 label match: `0.5692`
  - top-1 symbol match: `0.9678`
- `z_fused`
  - mean NN label agreement: `0.8437`
  - mean NN symbol agreement: `0.5739`
  - top-1 label match: `0.8466`
  - top-1 symbol match: `0.6906`

Ana okuma:

- `z_short`, state semantiğini net biçimde taşıyor
- `z_long`, label semantiğinden çok symbol kimliğine yakın davranıyor
- `z_fused`, `z_short` kadar iyi label semantiği koruyor ama daha fazla symbol/context bilgisi topluyor

## Interpretation

Bu sonuç önemli çünkü:

- kısa pencere branch'i gerçekten Stage 1A'nın ana yapısal ayrımını öğrenmiş görünüyor
- uzun pencere branch'i faydasız değil, ama ilk sürümde semantik ayrımdan çok coin / regime context taşıyor
- fusion, short semantiğini bozmadan context ekliyor; fakat aynı anda symbol bias'ını da artırıyor

Başka deyişle:

- `z_short` = "şu an local structure ne?"
- `z_long` = "bu yapı daha çok hangi coin/rejim bağlamında oluşuyor?"
- `z_fused` = "local structure + broader context"

## Bucket-Level Readout

Tam dağılım tarafında `true_break_*` bucket'ları en temiz örnekler olarak kaldı:

- BTC `true_break_up`: hata `0.0%`
- ETH `true_break_up`: hata `0.0%`
- ETH `true_break_down`: hata `0.0%`
- SOL `true_break_down`: hata `0.0%`
- XRP `true_break_down`: hata `1.3%`

Bu bize şunu söylüyor:

- güçlü teyitli structural shift örnekleri latent ve classifier için kolay
- problem alanı hâlâ sınır bölgeleri

Sınır bucket'larında hata belirgin artıyor:

- BTC `borderline_intact_break`: `20.1%`
- ETH `borderline_intact_break`: `24.0%`
- SOL `borderline_intact_break`: `32.6%`
- XRP `borderline_intact_break`: `26.1%`

Bu da önceki gözlemi güçlendiriyor:

- ana zorluk `bullish vs bearish` yön ayrımı değil
- asıl zorluk `intact -> break` sınırında

## What It Means For Stage 1A

Bu karşılaştırma sonrası daha net söyleyebildiğimiz şeyler:

- Stage 1A V1'de local branch kritik taşıyıcıdır
- long branch'in şu anki katkısı semantik ayrım değil, bağlamsal / symbol-specific bilgi
- fused latent kullanılabilir, ancak downstream işlerde symbol shortcut riskine karşı dikkatli okunmalıdır

Bu yüzden `CE + SupCon` için artık daha net bir hedef var:

- amaç genel accuracy kurtarmak değil
- özellikle `z_long` ve `z_fused` tarafında semantik komşuluğu artırmak
- symbol shortcut etkisini düşürmek
- borderline ve wick-heavy örneklerde latent geometry'yi iyileştirmek

## Decision

Bu aşamada çıkarılan karar:

- `CE-only` baseline yeterince güçlü bir referans çizgisi verdi
- `SupCon` için gerekçe artık daha net: latent geometry'yi symbol bias'tan daha semantik yapıya çekmek
- sonraki deneyler `z_short` semantiğini korurken `z_long` / `z_fused` tarafında label-aware komşuluğu artırmaya odaklanmalı

## Related Artifacts

- `data/stage1a/binance/15m/checkpoints/ce_only/analysis/val_latents/embedding_compare/embedding_overall_summary.csv`
- `data/stage1a/binance/15m/checkpoints/ce_only/analysis/val_latents/embedding_compare/embedding_bucket_comparison.csv`
- `data/stage1a/binance/15m/checkpoints/ce_only/analysis/val_latents/symbol_bucket_full_distribution.csv`
- `docs/worklogs/2026-04-13-ce-only-baseline-core4.md`
- `docs/worklogs/2026-04-14-embedding-view-comparison.md`

---

**Status**: Complete ✓
