# Worklog: 2026-05-12 — Directional Borderline Intact Buckets

**Status**: Complete  
**Scope**: Stage 1A analysis bucket semantics

---

## Problem

`borderline_intact_break` bucket'ı şu örnekleri topluyordu:

- label hâlâ `intact`
- close confirmation sayısı break teyidine bir adım uzak

Pressure probe sonrası bu bucket'ın neredeyse tamamen `up_pressure` / `down_pressure` örneklerinden oluştuğu görüldü. Bu, eski bucket adının doğru gri alanı yakaladığını ama yön bilgisini kaybettiğini gösterdi.

Eski okuma:

- `borderline_intact_break` zor bir `intact` / `break` sınırı

Daha rafine okuma:

- `borderline_intact_break_up`: bullish teyide bir adım uzak intact
- `borderline_intact_break_down`: bearish teyide bir adım uzak intact
- `borderline_intact_break_mixed`: iki yönde de teyide bir adım uzak intact

---

## Change

`scripts/analyze_stage1a_latents.py` içindeki domain bucket ataması güncellendi.

Yeni bucket'lar:

- `borderline_intact_break_up`
- `borderline_intact_break_down`
- `borderline_intact_break_mixed`

Bu değişiklik:

- Stage 1A dataset label'larını değiştirmez
- model eğitim hedefini değiştirmez
- yalnızca analiz ve readout bucket semantiğini rafine eder

---

## Rationale

Aggregate `borderline_intact_break` hata oranı tek başına yanıltıcıydı.

Çünkü modelin yaptığı hata çoğu zaman direction ile uyumlu:

- up-pressure borderline intact örnekleri `bullish` diye yanlışlanıyor
- down-pressure borderline intact örnekleri `bearish` diye yanlışlanıyor

Bu durum modelin tamamen kör olmadığını, fakat Stage 1A classifier sınırının yönsel baskıyı bazen erken break olarak yorumladığını gösteriyor.

Yeni alt bucket'lar bu ayrımı doğrudan görünür kılar.

---

## Verification

Çalıştırılan test:

```bash
PYTHONPATH=src pytest -q tests/test_stage1a_analysis.py
```

Sonuç:

```text
9 passed
```

`ce_supcon_long_v1` için val/test latent analizleri ve embedding compare çıktıları yeni bucket adlarıyla yeniden üretildi.

