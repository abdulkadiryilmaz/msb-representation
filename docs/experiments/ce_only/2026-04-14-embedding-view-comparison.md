# Worklog: 2026-04-14 — Embedding View Comparison (`z_short` / `z_long` / `z_fused`)

**Status**: Complete ✓  
**Goal**: aynı domain bucket dağılımı üstünde farklı embedding görünümlerinin komşuluk kalitesini karşılaştırmak

---

## Neden Gerekliydi

`symbol_bucket_full_distribution.csv` yararlı ama tek başına embedding karşılaştırması yapmıyor.

Sebep:

- bucket ataması fiyat yapısından geliyor
- aynı örnek için bucket, `z_short` ya da `z_long` seçimine göre değişmiyor

Dolayısıyla `z_short` ve `z_long` karşılaştırması için şu tip metrikler gerekiyor:

- bucket bazında nearest-neighbor label agreement
- bucket bazında same-symbol neighbor oranı
- top-1 neighbor label match
- top-1 neighbor symbol match

## Eklenen Araç

Yeni script:

- `scripts/compare_stage1a_embedding_views.py`

Bu script tek latent export üstünden varsayılan olarak:

- `z_short`
- `z_long`
- `z_fused`

için karşılaştırma üretir.

## Üretilen Çıktılar

- `embedding_bucket_comparison.csv`
- `embedding_overall_summary.csv`
- `summary.json`

Temel okuma biçimi:

- `symbol x bucket x embedding_key`
- her satırda count sabit kalır
- değişen şey komşuluk kalitesidir

## Karar

Bu yaklaşım, “hangi branch daha iyi?” sorusunu ham accuracy yerine:

- hangi yapı tipinde daha temiz komşuluk veriyor
- hangi bucket’ta cross-symbol genelleme daha güçlü
- fusion gerçekten short/long üstüne ek değer katıyor mu

eksenine taşıyor.

---

**Status**: Complete ✓
