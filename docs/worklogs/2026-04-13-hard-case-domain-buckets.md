# Worklog: 2026-04-13 — Hard-Case Domain Bucket Katmanı

**Status**: Complete ✓  
**Goal**: `hard_cases.csv` çıktısını salt skor listesinden çıkarıp MSB-diline daha yakın hata modlarıyla etiketlemek

---

## Problem

İlk latent analysis sürümü en zor örnekleri çıkarıyordu, ama şu soruya cevap vermiyordu:

- bu örnek neden zor?

Skor sıralaması araştırma için gerekli ama yeterli değil.

Özellikle `CE-only` sonrası görmek istediğimiz şey:

- model en çok `wick sweep` benzeri failed break örneklerinde mi zorlanıyor
- `intact` ile gerçek break sınırı hangi volatilite rejimlerinde bulanıklaşıyor
- güçlü close-confirmed break örnekleri ile borderline örnekler ayrışıyor mu

## Eklenen Katman

Latent export sırasında short window içinden domain özeti çıkarıldı:

- recent `atr_norm` ortalaması
- recent `hl_range` ortalaması
- recent body / range özeti
- swing-derived bullish / bearish break level
- close confirmation sayıları
- wick-only break sayıları
- final close displacement

Bu özetler export `.npz` içine eklendi.

Ardından `analyze_stage1a_latents.py`, `hard_cases.csv` üretirken bu sinyallerden bucket atıyor.

## İlk Bucket Seti

- `high_vol_intact`
- `high_vol_intact_wick_sweep`
- `wick_sweep_up`
- `wick_sweep_down`
- `close_confirmed_break_up`
- `close_confirmed_break_down`
- `true_break_up`
- `true_break_down`
- `borderline_intact_break`
- `borderline_bullish_break`
- `borderline_bearish_break`
- `clean_intact`

Bu bucket'lar yeni eğitim label'ı değil.

Analiz etiketi olarak kullanılıyorlar.

## Kural Felsefesi

İlk sürüm kasıtlı olarak heuristik.

Amaç:

- hard case listesinde manuel okuma maliyetini düşürmek
- hata modlarını MSB açısından gruplayabilmek
- `CE + SupCon` sonrası aynı bucket setinde karşılaştırma yapabilmek

Bu nedenle bucket kuralları:

- dataset içi quantile threshold
- label prior ile uyumlu close count mantığı
- wick vs close teyidi ayrımı

üstüne kuruldu.

## Beklenen Kullanım

Analiz sonrası artık yalnızca “zor örnekler” listesi değil, aynı zamanda:

- hangi hard-case bucket baskın
- hangi sembolde hangi bucket yoğun
- `intact` sınırı mı yoksa teyitli break sınırı mı daha sorunlu

soruları açılabilir.

Bir sonraki doğal adım:

- bucket dağılımını sembol bazında raporlamak
- `z_short` vs `z_long` için bucket-sensitive karşılaştırma yapmak
- heuristik bucket'ları manuel gözden geçirmeyle rafine etmek

---

**Status**: Complete ✓
