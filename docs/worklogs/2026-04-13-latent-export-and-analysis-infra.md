# Worklog: 2026-04-13 — Latent Export ve Val Analysis Altyapısı

**Status**: Complete ✓  
**Goal**: `best.pt` checkpoint'inden latent export alıp val split üstünde PCA / UMAP, nearest-neighbor ve hard-case incelemesini tekrar üretilebilir hale getirmek

---

## Neden Bu Adım

`CE-only` baseline güçlü bir sınıflandırma sonucu verdi, ancak bu tek başına representation kalitesini kanıtlamıyor.

Bir sonraki araştırma soruları artık şunlar:

- `z_fused` semantik olarak düzenli mi
- aynı structural state farklı coinlerde yakın komşu oluyor mu
- en zor örnekler hangi hata modlarında toplanıyor
- `SupCon` gerçekten latent geometry'yi iyileştirir mi

Bu soruları elle notebook içinde değil, repo içinde tekrar üretilebilir CLI araçlarıyla açmak gerekiyordu.

## Eklenen Altyapı

Yeni ortak modül:

- `src/msb_repr/stage1a/analysis.py`

Bu modül şunları sağlıyor:

- checkpoint bundle yükleme
- saved normalizer'ları split dataset'e uygulama
- `z_short`, `z_long`, `z_fused`, `z_proj`, logits ve probability export
- PCA projection
- cosine nearest-neighbor çıkarımı
- hard-case skorlaması

Yeni script'ler:

- `scripts/export_stage1a_latents.py`
- `scripts/analyze_stage1a_latents.py`

## Çıktı Sözleşmesi

Latent export `.npz` dosyası şunları içerir:

- `z_short`
- `z_long`
- `z_fused`
- `z_proj`
- `logits`
- `probs`
- `labels`
- `preds`
- `timestamps`
- `symbols`
- `label_names`
- `export_metadata_json`

Analiz script'i bu export üstünden şu çıktıları üretir:

- `summary.json`
- `pca_projection.csv`
- `umap_projection.csv` (`umap-learn` kuruluysa)
- `nearest_neighbors.csv`
- `hard_cases.csv`

## Hard-Case Yaklaşımı

İlk sürüm hard-case skoru üç sinyali birleştiriyor:

- yanlış sınıflandırma
- düşük confidence margin
- nearest-neighbor label uyumsuzluğu

Bu kasıtlı olarak basit tutuldu.

Amaç:

- önce en sorunlu örnekleri tekrar üretilebilir biçimde yüzeye çıkarmak
- daha sonra domain-spesifik bucket'ları (`wick sweep`, `close-confirmed break`, `high-vol intact`) bunun üstüne eklemek

## Kararlar

- latent export ile analiz script'i ayrıldı; böylece bir kez export alınıp farklı embedding alanları üstünde (`z_fused`, `z_short`, `z_long`, `z_proj`) tekrar analiz yapılabilir
- checkpoint yükleme için eğitim script'ine bağımlı olmayan ortak helper eklendi
- nearest-neighbor tarafı notebook bağımlılığı yerine repo içi CLI çıktısı üretecek şekilde tasarlandı
- UMAP opsiyonel bırakıldı; `.[viz]` kurulumu yoksa altyapı PCA + NN + hard-case ile yine çalışıyor

## Kullanım

Latent export:

```bash
python scripts/export_stage1a_latents.py \
  --dataset-root data/stage1a/binance/15m \
  --checkpoint-dir data/stage1a/binance/15m/checkpoints/ce_only \
  --split val
```

Analiz:

```bash
python scripts/analyze_stage1a_latents.py \
  --latent-path data/stage1a/binance/15m/checkpoints/ce_only/analysis/val_latents.npz
```

## Sonraki Adım

- `hard_cases.csv` üstünden domain-tagging katmanı eklemek
- `z_fused` yanında `z_short` ve `z_long` karşılaştırmalı raporlamak
- test split için aynı hattı çalıştırmak
- sonra `CE + SupCon` koşusunu aynı analiz hattından geçirmek

---

**Status**: Complete ✓
