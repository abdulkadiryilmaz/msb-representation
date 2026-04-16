# Worklog: 2026-04-13 — CE-Only Baseline on core4

**Status**: Complete ✓  
**Goal**: Stage 1A `CE-only` baseline'ının gerçekten çalışıp çalışmadığını, ne öğrendiğini ve `CE + SupCon` öncesi hangi açık soruların kaldığını netleştirmek

---

## Problem Statement

Stage 1A hattı için ilk çalışan sürüm kurulduktan sonra şu soru açıktı:

- `CE-only` supervision, `bullish / bearish / intact` state-style label'ları anlamlı biçimde ayırmaya yeterli mi?

Bu soru kritik çünkü:

- eğer `CE-only` zaten güçlü bir ayrışma veriyorsa, `SupCon` eklemenin gerekçesi daha net tanımlanmalı
- eğer `CE-only` erken aşamada collapse ediyorsa, önce objective tarafına müdahale etmek gerekir

Bu nedenle `core4` dataset universe üstünde ilk temiz baseline koşusu çalıştırıldı.

## Context

Kullanılan sabitler:

- dataset profile: `core4`
- semboller: `BTC_USDT_15m`, `ETH_USDT_15m`, `SOL_USDT_15m`, `XRP_USDT_15m`
- short window: `48`
- long window: `288`
- step size: `4`
- short features: `10`
- long features: `10`
- latent boyutları: `z_short = 64`, `z_long = 32`
- fusion: `concat`
- objective: `CE only`

Split:

- train end: `2025-09-30`
- val end: `2025-12-31`

## Dataset Summary

Aligned sample sayıları:

- BTC: `37415`
- ETH: `37331`
- SOL: `37415`
- XRP: `37415`

Toplam dataset büyüklüğü:

- train: `131040`
- val: `8832`

Global label dağılımı:

- `0 = intact`: `73606`
- `1 = bullish`: `38498`
- `2 = bearish`: `37472`

Bu dağılım `intact` lehine dengesiz, ancak aşırı değil. Bullish ve bearish sınıfları da güçlü örnek sayısına sahip.

## Investigation

### Eğitim eğrisi

Koşu `36` epoch sürdü ve early stopping ile kapandı.

En iyi `val_loss`:

- `0.2651`
- epoch `26`

En iyi `val_macro_f1`:

- `0.8901`
- epoch `34`

Bu iki noktanın farklı olması önemli:

- model loss açısından en iyi noktaya daha erken ulaşıyor
- sınıf ayrışması ve karar kalitesi birkaç epoch daha iyileşmeye devam ediyor

Bu, classification geometry ile calibrated loss minimization'ın tamamen aynı optimumu paylaşmadığını gösteriyor.

### Validation confusion

En iyi checkpoint (`best.pt`) val split üstünde tekrar değerlendirildi.

Confusion matrix:

```text
[[3673  229  250]
 [ 236 2046   12]
 [ 246   12 2128]]
```

Class-wise metrikler:

- class `0 / intact`: precision `0.8840`, recall `0.8846`, F1 `0.8843`
- class `1 / bullish`: precision `0.8946`, recall `0.8919`, F1 `0.8933`
- class `2 / bearish`: precision `0.8904`, recall `0.8919`, F1 `0.8911`

Önemli gözlem:

- belirgin tek-sınıf collapse yok
- `bullish` ve `bearish` sınıfları birbirine çok az karışıyor
- ana hata modu `intact <-> bullish` ve `intact <-> bearish`

Bu beklenen bir yapı:

- yönlü kırılım state'leri kendi aralarında iyi ayrışıyor
- asıl sınır, “gerçek structural state shift mi yoksa halen intact mi?” ekseninde oluşuyor

### Per-symbol validation behavior

Val doğruluğu:

- BTC: `0.8958`
- ETH: `0.8813`
- SOL: `0.8877`
- XRP: `0.8890`

Yorum:

- dört sembolde de performans yakın
- tek bir sembol tüm sonucu sürüklemiyor
- BTC hafif daha kolay, ETH hafif daha zor görünüyor
- yine de aradaki fark Stage 1A baseline için problemli büyüklükte değil

## Decision

`CE-only` baseline başarısız değil; tersine güçlü bir ilk referans çizgi üretti.

Bu aşamada alınan karar:

- hemen `CE + SupCon`'a geçmeden önce `CE-only` koşusunu temsil kalitesi açısından okumak
- bir sonraki adım olarak latent export / probe / nearest-neighbor / hard-case inceleme araçlarını açmak
- `SupCon` ekleme gerekçesini “classification zaten iyi” gerçeğini dikkate alarak yeniden tanımlamak

Başka deyişle:

- `SupCon` artık “baseline kurtarma hamlesi” değil
- latent geometry'yi daha semantik ve daha transfer-edilebilir hale getirme adayıdır

## What It Learned

Bu baseline'ın öğrendiği şeyler:

- state-style `intact / bullish / bearish` ayrımı pratikte öğrenilebilir
- short + long concat yapısı, ilk sürümde çalışır durumda
- `bullish` ve `bearish` yön ayrımı güçlü biçimde öğreniliyor
- multi-symbol `core4` universe üstünde tek bir sembole aşırı bağımlı olmayan bir çözüm var
- label prior, ilk sürüm için kullanılabilir kalitede

## What It Has Not Proven Yet

Bu koşu henüz şu soruları cevaplamıyor:

- latent uzay gerçekten semantik olarak iyi organize oldu mu
- aynı yapılar coinler arasında yakın komşu oluyor mu
- high-vol intact ile true break ayrımı latent seviyede ne kadar temiz
- wick sweep vs close-confirmed break ayrımı ne kadar taşınıyor
- OOS `test` dönemi üstünde genelleme ne durumda
- downstream görevler için `z_fused` ne kadar faydalı

Kritik nokta:

- yüksek `val_macro_f1`, representation kalitesini tek başına kanıtlamaz
- bu koşu şimdilik yalnızca classifier head'in state-style supervision'ı iyi optimize ettiğini kanıtlıyor

## Key Learnings

- `CE-only` Stage 1A için yeterince güçlü bir baseline; ilk objective olarak doğru seçim oldu
- ana hata ekseni yön karışıklığı değil, `intact` sınırında belirsizlik
- loss optimumu ile macro-F1 optimumu farklı epoch'larda oluşuyor; checkpoint seçimi yalnızca loss ile yapılmamalı
- `core4` universe ilk baseline için dengeli ve kullanılabilir görünüyor
- `SupCon` için gerekçe artık “performans yetersiz” değil, “latent geometry'yi daha bilinçli şekillendirme” olmalı

## Next Steps

- [ ] `best.pt` için latent export aracı yaz
- [ ] val split üstünde PCA / UMAP ve nearest-neighbor inceleme yap
- [ ] hard-case örnek seti hazırla: `high-vol intact`, `true break`, `wick sweep`, `close-confirmed break`
- [ ] OOS `test` split değerlendirmesi ekle
- [ ] sonra `CE + SupCon` varyantını aynı dataset profile üstünde çalıştır

Related: [2026-04-12-stage1-msb-representation-framing.md](2026-04-12-stage1-msb-representation-framing.md)  
Related: [STAGE1A_IMPLEMENTATION_PLAN.md](../STAGE1A_IMPLEMENTATION_PLAN.md)  
Related: [MULTI_SCALE_MSB_INTERPRETATION.md](../MULTI_SCALE_MSB_INTERPRETATION.md)

---

**Status**: Complete ✓
