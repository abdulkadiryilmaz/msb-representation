# Worklog Writing Guide

## Amaç

Worklog'lar oturum bazlı geliştirme sürecini kaydeder. Amaç yalnızca **ne** yapıldığını değil, **neden** o kararın alındığını da görünür kılmaktır.

Not:

- `docs/worklogs/` karar, implementasyon ve yön değişikliği notları içindir
- araştırma framing / kavramsal temel belgeleri `docs/foundations/` altında tutulur
- deney sonuç okumaları `docs/experiments/` altında tutulur
- üst seviye deney özeti için `docs/EXPERIMENT_INDEX.md` kullanılır

## Ne Zaman Worklog Açılmalı

Şu durumlarda worklog aç:

- anlamlı feature / model / objective değişiklikleri
- önemli içgörü üreten debugging oturumları
- mimari kararlar veya tasarım yön değişimleri
- veri sözleşmesi, normalizasyon, pencere tasarımı değişiklikleri
- çok adımlı araştırma akışları
- mimari sonucu etkileyen experiment plan veya decision notları

Şu durumlarda worklog açma:

- trivial bug fix
- sadece dokümantasyon düzeltmesi
- küçük config değişikliği
- rutin veri fetch veya tekrar çalıştırma

## Format

### Dosya Adı

```text
docs/worklogs/YYYY-MM-DD-{topic}.md
```

Örnek:

- `2026-04-15-ce-only-baseline-review.md`
- `2026-04-20-supcon-objective-v1.md`

### Belge Yapısı

```markdown
# Worklog: YYYY-MM-DD — Kısa Başlık

**Status**: In Progress | Complete ✓ | Blocked
**Goal**: Tek cümlelik hedef

---

## Problem Statement

Problemi veya motivasyonu açık biçimde yaz.

## Context

Gerekliyse arka planı ver.

## Investigation

Toplanan sinyaller, bakılan opsiyonlar, görülen içgörüler.

## Decision

Alınan karar ve gerekçesi.

## Implementation

Yapılan teknik değişikliğin yüksek seviyeli özeti.

## Key Learnings

- önemli çıkarım
- önemli çıkarım

## Next Steps

- ✓ tamamlananlar
- bekleyenler

---

**Status**: Final durum
```

## Yazım Stili

- anlatı değil, karar odaklı yaz
- mümkünse somut metrik kullan
- kısa paragraf düzeni tercih et
- başarısız denemeleri de kaydet
- komut dökümü veya uzun kod blokları ekleme

## Güncelleme Kuralı

- aynı gün ve aynı konu sürüyorsa mevcut worklog güncellenebilir
- farklı problem veya farklı deney ekseni açıldıysa yeni worklog aç

## Entegrasyon

İlgili dosyalara link ver:

- `docs/STAGE1A_IMPLEMENTATION_PLAN.md`
- `docs/MULTI_SCALE_MSB_INTERPRETATION.md`
- `docs/TEMPORAL_CNN_BASICS.md`
- `AGENTS.md`
- `docs/EXPERIMENT_INDEX.md`

## Experiments Ayrımı

Deney notları için önerilen yapı:

```text
docs/experiments/
  ce_only/
  ce_supcon/
```

Bu klasörlerde şu tür dosyalar tutulur:

- baseline sonucu
- readout
- comparison
- synthesis
- experiment plan

Özet kural:

- "araştırma çerçevesi / kavramsal temel ne?" -> `docs/foundations/`
- "ne implement edildi ve neden?" -> `docs/worklogs/`
- "deneyde ne oldu?" -> `docs/experiments/`

## Hızlı Kontrol Listesi

- [ ] dosya adı doğru mu
- [ ] status ve goal dolu mu
- [ ] problem açık yazıldı mı
- [ ] karar ve gerekçe yazıldı mı
- [ ] sonraki adımlar net mi
- [ ] uzun komut / kod dökümü yok mu
