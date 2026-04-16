# Worklog: 2026-04-15 — Branch-Aware SupCon Implementation

**Status**: Complete ✓  
**Goal**: `SupCon` sinyalini fused projection yerine seçilebilir bir temsil üstüne taşıyarak özellikle `z_long` branch'i hedefleyebilen ikinci deney yüzeyini açmak

---

## Neden Bu Değişiklik Yapıldı

İlk `CE + SupCon v1` koşusu şu tabloyu verdi:

- classifier tarafı bozulmadı
- `z_long` label-aware komşuluk açısından belirgin iyileşti
- ama aynı anda symbol shortcut etkisi de büyüdü
- `z_fused` tarafında ikna edici genel iyileşme görülmedi

Bu yüzden sonraki mantıklı soru şuydu:

- contrastive baskıyı doğrudan fused projection üstünde tutmak yerine, yalnızca problemli branch'e yöneltirsek daha kontrollü bir etki elde edebilir miyiz?

Bu worklog, o amaçla açılan `branch-aware SupCon` implementasyonunu kaydeder.

## Tasarım Kararı

Mevcut akış korunuyor:

- `z_short`
- `z_long`
- `z_fused`
- `z_proj = projection_head(z_fused)`

Buna ek olarak yeni bir opsiyonel projection head eklendi:

- `z_long_proj = long_projection_head(z_long)`

Bu sayede `SupCon` kaynağı seçilebilir hale geldi:

- `z_proj`
- `z_long_proj`

İlk branch-aware varyant için önerilen kullanım:

- classifier aynı şekilde `z_fused` üstünde kalır
- contrastive loss yalnızca `z_long_proj` üstünde hesaplanır

Yani loss artık şu şekilde kurulabilir:

- `CE(logits(z_fused), y) + lambda * SupCon(z_long_proj, y)`

## Kod Değişiklikleri

### 1. Model

`Stage1AModel` artık opsiyonel `long_projection_dim` alıyor.

Eğer bu değer verilirse model:

- `long_projection_head`
- `z_long_proj`

çıktısını da üretir.

Eski checkpoint'lerle geriye dönük uyumluluk korunur:

- eski metadata içinde `long_projection_dim` yoksa head kurulmaz

### 2. Trainer

Trainer config'e yeni alan eklendi:

- `supcon_embedding_key`

Bu alan sayesinde `SupCon` şu embedding key'lerden biri üstünde çalışabilir:

- `z_proj`
- `z_long_proj`

Varsayılan davranış değişmedi:

- varsayılan hâlâ `z_proj`

### 3. Train CLI

`train_stage1a.py` artık şunları destekliyor:

- `--supcon-embedding-key {z_proj,z_long_proj}`
- `--long-projection-dim`

Pratik kolaylık olarak:

- eğer `--supcon-embedding-key z_long_proj` seçilir
- ve `--long-projection-dim` verilmezse
- otomatik olarak `z_long` boyutu kullanılır

Varsayılan checkpoint adı da buna göre ayrıştırılır:

- fused projection SupCon için: `ce_supcon_v1`
- long projection SupCon için: `ce_supcon_long_v1`

### 4. Analysis / Export

Latent export hattı artık opsiyonel olarak:

- `z_long_proj`

çıktısını da `.npz` içine yazar.

Bu sayede branch-aware deney sonrası yeni projection uzayı da aynı analiz hattıyla export edilebilir.

### 5. Embedding Compare

`compare_stage1a_embedding_views.py` artık:

- latent export içinde varsa `z_long_proj`

embedding'ini de karşılaştırma kapsamına alabilir.

Varsayılan embedding seti şu mantıkla çalışır:

- her zaman `z_short`, `z_long`, `z_fused`
- varsa ayrıca `z_long_proj`

## Doğrulama

İlgili testler çalıştırıldı:

```bash
PYTHONPATH=src pytest -q tests/test_stage1a_analysis.py tests/test_stage1a_dataset.py tests/test_stage1a_losses.py
```

Sonuç:

```text
12 passed in 0.57s
```

## İlk Önerilen Deney

İlk branch-aware koşu için önerilen komut:

```bash
source .venv/bin/activate
python scripts/train_stage1a.py \
  --dataset-root data/stage1a/binance/15m \
  --epochs 50 \
  --batch-size 64 \
  --checkpoint-dir data/stage1a/binance/15m/checkpoints/ce_supcon_long_v1 \
  --use-supcon \
  --supcon-weight 0.05 \
  --supcon-temperature 0.1 \
  --supcon-embedding-key z_long_proj
```

Bu koşuda:

- fused classifier korunur
- contrastive baskı yalnızca long branch projection üstünde çalışır

## Beklenen Okuma

Bu varyanttan sonra özellikle şu sorulara bakılmalı:

- `z_long` ve `z_long_proj` daha semantik hale geliyor mu?
- bu etki `z_fused` tarafına pozitif taşınıyor mu?
- `z_fused` symbol shortcut etkisi artmadan kalabiliyor mu?
- `borderline_intact_break` ve `wick` bucket'larında gerçek iyileşme geliyor mu?

Başarı ölçütü yine yalnızca macro-F1 değil.

Asıl karar ekseni:

- fused latent geometry gerçekten daha iyi oldu mu?

## Related

- `docs/worklogs/2026-04-14-ce-supcon-experiment-plan.md`
- `docs/worklogs/2026-04-14-ce-supcon-v1-readout.md`

---

**Status**: Complete ✓
