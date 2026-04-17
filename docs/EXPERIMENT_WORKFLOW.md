# Experiment Workflow

Bu belge, Stage 1A deneyleri için ortak çalışma akışını tanımlar.

Amaç:

- deney planı, koşu, analiz ve dokümantasyon adımlarını standartlaştırmak
- yeni oturumlarda bağlam kaybını azaltmak
- hangi artefaktların zorunlu olduğunu açık hale getirmek

## 1. Yeni Deney Ne Zaman Açılır

Yeni bir experiment plan aç:

- yeni objective varyantı deneniyorsa
- aynı objective içinde yeni masking / sampling / batch strategy deneniyorsa
- farklı checkpoint ailesi oluşuyorsa
- başarı kriteri veya baseline to beat değişiyorsa

Aynı deney zincirinin küçük tekrarları için mevcut plan güncellenebilir.

## 2. Deney Dokümanı Nerede Yaşar

Deney planı ve readout notları:

- `docs/experiments/<family>/`

Örnek:

- `docs/experiments/ce_only/`
- `docs/experiments/ce_supcon/`

Karar / implementasyon notları ise:

- `docs/worklogs/`

## 3. Naming Convention

Plan:

```text
YYYY-MM-DD-<variant>-plan.md
```

Sonuç:

```text
YYYY-MM-DD-<variant>-readout.md
```

Örnek:

- `2026-04-17-ce-supcon-long-symbol-v2-plan.md`
- `2026-04-17-ce-supcon-long-symbol-v2-readout.md`

## 4. Run Öncesi Checklist

Her yeni koşudan önce:

1. baseline to beat net mi
2. deney hipotezi tek cümlede yazılı mı
3. değişen şey tekil ve anlaşılır mı
4. checkpoint adı açık mı
5. plan notunda:
   - method
   - command
   - success criteria
   - planned readout
   yazıyor mu
6. `docs/EXPERIMENT_INDEX.md` içinde bu koşu `in_progress` olarak görünüyor mu

## 5. Eğitim Koşusu Sırasında

Koşu sırasında not alınması gereken minimum bilgi:

- training device
- early stopping epoch
- best val macro-F1
- dikkate değer training davranışı

Bu notlar sonradan readout içinde özetlenecek.

## 6. Run Sonrası Zorunlu Artefaktlar

Bir deney tamamlandıktan sonra minimum olarak:

1. eğitim özeti

- `metadata.json`
- `history.json`

2. val artefaktları

- latent export
- analyze
- embedding compare

3. test artefaktları

- latent export
- analyze
- embedding compare

4. gerekiyorsa notebook / manuel inceleme

## 7. Standart Post-Run Checklist

Koşu bittikten sonra varsayılan sıra:

1. eğitim özetini oku
2. val latent export al
3. val analyze çalıştır
4. val embedding compare çalıştır
5. test latent export al
6. test analyze çalıştır
7. test embedding compare çalıştır
8. baseline to beat ile yan yana kıyas yap
9. readout dokümanı yaz
10. `EXPERIMENT_INDEX.md` güncelle
11. gerekiyorsa family README timeline güncelle

## 8. Standart Kıyas Eksenleri

Her readout'ta mümkün olduğunca şu eksenlere bak:

### classifier

- best val macro-F1
- test genelleme sinyali

### embedding

- `z_short`
- `z_long`
- `z_fused`
- varsa branch-specific projection (`z_long_proj` gibi)

İzlenecek ana metrikler:

- mean NN label agreement
- mean NN symbol agreement
- top-1 label match
- top-1 symbol match

### bucket

Özellikle:

- `borderline_intact_break`
- `wick_sweep_up`
- `wick_sweep_down`
- `high_vol_intact_wick_sweep`
- `close_confirmed_break_up`
- `close_confirmed_break_down`

## 9. Readout Yazım Kuralı

Bir readout notu mutlaka şunları içermeli:

- experiment setup
- classifier summary
- val readout
- test readout
- bucket readout
- overall decision
- next direction

Kritik kural:

- yalnızca macro-F1 ile karar verilmez
- latent geometry ana karar eksenidir

## 10. Experiment Index Ne Zaman Güncellenir

Index şu durumlarda güncellenir:

- yeni koşu başladığında `in_progress` satırı eklenir veya güncellenir
- readout tamamlandığında status:
  - `reference`
  - `best current`
  - `mixed`
  - `rejected`
  olarak netleştirilir

## 11. Family README Ne Zaman Güncellenir

Family README şu durumda güncellenir:

- yeni readout sonrası deney zincirinin anlatısı değişiyorsa
- yeni varyant timeline'a ekleniyorsa
- “best current” varyant değişiyorsa

## 12. Worklog Ne Zaman Gerekir

Ek olarak `worklog` aç:

- yeni objective / masking / sampling mekanizması implemente edildiyse
- latent analysis altyapısı değiştiyse
- araştırma yönünü değiştiren tasarım kararı alındıysa

Özet:

- implementasyon / karar -> `worklog`
- deney planı / deney sonucu -> `experiments`

## 13. Minimum Okuma Sırası

Yeni bir oturumda hızlı toparlama için:

1. `docs/EXPERIMENT_INDEX.md`
2. ilgili family README
3. aktif plan notu
4. baseline to beat readout
5. son başarısız / son başarılı varyant readout'ları
