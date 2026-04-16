# Worklog: 2026-04-14 — CE-Only Test Split Readout on core4

**Status**: Complete ✓  
**Goal**: `CE-only` baseline'ın OOS `test` split'te ne kadar korunduğunu görmek

---

## Neden Bu Not

Val split üstünde oldukça fazla analiz yapıldı.

Ama bunların hepsi zaman olarak `val` dönemi içindeydi.

Bu yüzden kritik soru şuydu:

- aynı kalite `test` split'te de korunuyor mu?

Bu not test split export, analyze ve embedding compare çıktılarının ilk okumasını toplar.

## Setup

Çalıştırılan akış:

```bash
PYTHONPATH=src python scripts/export_stage1a_latents.py \
  --dataset-root data/stage1a/binance/15m \
  --checkpoint-dir data/stage1a/binance/15m/checkpoints/ce_only \
  --split test

PYTHONPATH=src python scripts/analyze_stage1a_latents.py \
  --latent-path data/stage1a/binance/15m/checkpoints/ce_only/analysis/test_latents.npz \
  --skip-umap

PYTHONPATH=src python scripts/compare_stage1a_embedding_views.py \
  --latent-path data/stage1a/binance/15m/checkpoints/ce_only/analysis/test_latents.npz
```

Test sample sayısı:

- `9704`

## Overall Result

Test özet:

- accuracy: `0.8879`
- mean confidence: `0.8919`
- mean neighbor label agreement: `0.8445`

Val ile karşılaştırma:

- val accuracy: `0.8885`
- test accuracy: `0.8879`

Yani accuracy neredeyse aynı kaldı.

Nearest-neighbor tarafı da benzer:

- val NN label agreement: `0.8437`
- test NN label agreement: `0.8445`

Bu güçlü bir sinyal:

- `CE-only` çözüm val'e özel dar bir ezber gibi görünmüyor
- OOS dönemde de aynı bant korunuyor

## PCA

Test PCA explained variance:

- PC1: `0.5633`
- PC2: `0.3604`
- ilk iki bileşen toplamı: yaklaşık `0.9236`

Val ile çok yakın.

Bu, latent yapının test döneminde de benzer biçimde sıkıştığını düşündürüyor.

## Embedding View Comparison

Test embedding compare:

- `z_short`
  - label agreement: `0.8438`
  - symbol agreement: `0.2731`
- `z_long`
  - label agreement: `0.4504`
  - symbol agreement: `0.9115`
- `z_fused`
  - label agreement: `0.8445`
  - symbol agreement: `0.5970`

Val ile kıyas:

- `z_short` neredeyse aynı
- `z_long` yine symbol-heavy
- `z_fused` yine short semantiğini koruyor, ama symbol/context bias taşıyor

Önemli nokta:

- embedding yapısı test split'te de değişmiyor
- yani `z_long` problemi de, `z_short` gücü de OOS dönemde aynı kalıyor

## Bucket Readout

Test bucket davranışı da val ile uyumlu.

Temiz alanlar:

- `true_break_up`
  - BTC: `0.0%`
  - ETH: `1.3%`
  - SOL: `4.0%`
  - XRP: `0.0%`
- `true_break_down`
  - BTC: `0.0%`
  - ETH: `1.0%`
  - SOL: `0.0%`
  - XRP: `1.1%`

Zor alanlar:

- `borderline_intact_break`
  - BTC: `24.3%`
  - ETH: `25.3%`
  - SOL: `25.0%`
  - XRP: `26.8%`
- `high_vol_intact_wick_sweep`
  - BTC: `24.1%`
  - ETH: `31.4%`
  - SOL: `29.1%`
  - XRP: `30.9%`

Bu ne demek:

- ana hata modu test split'te de değişmedi
- model hâlâ true break state'lerinde temiz
- asıl zor alan yine break teyidi öncesi ve failed-break bölgesi

## Nearest-Neighbor Examples

Test nearest-neighbor örnekleri de val ile aynı hikâyeyi anlatıyor:

- bazı bullish label örnekleri ilk komşularında yoğun intact örnekler taşıyor
- bazı intact örnekler bullish manifold'una yaklaşıyor

Bu da aynı sonucu destekliyor:

- OOS dönemde de ana sınır intact-break sınırı
- problem yön karışıklığı değil

## Decision

Test split okuması sonrası şu karar daha rahat verilebilir:

- `CE-only` baseline yalnızca val içi iyi görünen bir çözüm değil
- OOS dönemde de oldukça stabil
- bu yüzden sonraki objective değişikliği aceleyle değil, bilinçli gerekçeyle yapılmalı

Başka deyişle:

- `SupCon` hâlâ mantıklı olabilir
- ama artık bu, performans kurtarma adımı değil
- latent geometry refinement adımıdır

## Conclusion

Test split değerlendirmesi `CE-only` baseline'ı güçlendiriyor.

Şu anda elimizdeki en net resim:

- classifier kalitesi val ve test'te benzer
- latent komşuluk kalitesi val ve test'te benzer
- short branch ana semantik taşıyıcı
- long branch symbol/context ağırlıklı
- ana açık problem intact-break sınırı

Bu, Stage 1A için güçlü ve dürüst bir ara sonuçtur.

## Related

- `docs/worklogs/2026-04-14-ce-only-val-pca-nn-readout.md`
- `docs/worklogs/2026-04-14-ce-only-bucket-readout.md`
- `docs/worklogs/2026-04-14-ce-only-embedding-view-readout.md`

---

**Status**: Complete ✓
