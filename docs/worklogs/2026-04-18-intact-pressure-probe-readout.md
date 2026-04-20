# Worklog: 2026-04-18 — Intact Pressure Probe Readout

**Status**: Complete ✓  
**Goal**: Stage 1A latent'inin `intact` altkümesinde `directional pressure` semantiğini taşıyıp taşımadığını probe ile ölçmek ve `ce_only` ile `ce_supcon_long_v1` koşularını karşılaştırmak

---

## Problem Statement

Stage 1A için ana supervision `bullish / bearish / intact` state-style label'larıdır.

Bu kurulum:

- current structural state'i taşımayı hedefler
- ancak `intact` içindeki `up-pressure / down-pressure / neutral` semantiğini doğrudan öğretmez

Araştırma sorusu şuydu:

- mevcut Stage 1A latent'i, ana eğitim hedefinde açıkça optimize edilmemiş olsa bile `intact` içi yönsel baskı bilgisini taşıyor mu?

İkinci soru:

- bu sinyal `ce_only` ile `ce_supcon_long_v1` arasında özellikle short / long / fused temsillerde nasıl değişiyor?

---

## Context

Bu oturum öncesinde:

- latent export'lar için analysis-only `pressure_labels_v1.parquet` artefact'ı üretildi
- bu label'lar yalnızca `base_label == intact` örneklerde `neutral / up_pressure / down_pressure` ayrımı veriyor
- ana Stage 1A dataset contract'ı değiştirilmedi

Probe kuralları:

- train split olarak `val_latents`
- eval split olarak `test_latents`
- embedding'ler: `z_short`, `z_long`, `z_fused`
- baseline: majority class
- probe'lar:
  - linear probe = `LogisticRegression`
  - nonlinear probe = küçük `MLP`

Bu okuma için ana metrik:

- `macro F1`

Sebep:

- sınıf dağılımı belirgin dengesiz
- accuracy tek başına yanıltıcı olur

---

## Investigation

### 1. Majority baseline düşük ama yanıltıcı olmayan referans veriyor

Test split `intact` pressure dağılımı:

- `neutral`: baskın sınıf
- `up_pressure`: azınlık sınıf
- `down_pressure`: azınlık sınıf

Bu yüzden majority baseline:

- accuracy'de görece yüksek
- macro F1'da düşük

Referans:

- majority macro F1: `0.3015`

### 2. `ce_supcon_long_v1` pressure semantiğini gerçekten taşıyor

`ce_supcon_long_v1` linear probe sonuçları:

- `z_short`: macro F1 `0.5257`
- `z_long`: macro F1 `0.4842`
- `z_fused`: macro F1 `0.5407`

Bu üç sonuç da majority baseline'ın belirgin biçimde üstünde.

İlk ana çıkarım:

- `intact` içi directional pressure semantiği latent'te gerçekten var

İkinci çıkarım:

- en güçlü okuma `z_fused`
- ikinci en güçlü okuma `z_short`
- `z_long` da anlamlı sinyal taşıyor

### 3. `ce_only` ile fark en çok `z_long` tarafında açılıyor

`ce_only` linear probe sonuçları:

- `z_short`: macro F1 `0.5227`
- `z_long`: macro F1 `0.2067`
- `z_fused`: macro F1 `0.5254`

`ce_supcon_long_v1` ile karşılaştırınca:

- `z_short`: `0.5227 -> 0.5257`
- `z_long`: `0.2067 -> 0.4842`
- `z_fused`: `0.5254 -> 0.5407`

Ana fark:

- short tarafta koşular birbirine çok yakın
- fused tarafta `ce_supcon_long_v1` lehine küçük ama net artış var
- asıl sıçrama long branch'te

Bu okuma şu anlama geliyor:

- `ce_only` long latent'te pressure semantiğini neredeyse taşıyamıyor
- `ce_supcon_long_v1` long latent'i bu bilgiyi belirgin biçimde daha okunabilir hale getiriyor

### 4. Confusion matrix okuması: long branch'te büyük yapısal iyileşme var

`ce_only / z_long` linear confusion matrix:

- `neutral` örneklerin büyük kısmı `up_pressure` veya `down_pressure` olarak dağılmış
- sınıf sınırları kararsız
- genel geometri pressure açısından zayıf

`ce_supcon_long_v1 / z_long` linear confusion matrix:

- `neutral` örneklerde doğru kalma sayısı belirgin artıyor
- `up_pressure` ve `down_pressure` örnekleri de daha dengeli okunuyor
- long latent, pressure semantiğini ilk kez kullanılabilir seviyede taşıyor

Bu, yalnızca küçük bir skor artışı değil; long branch'in iç düzeninde anlamlı bir değişim işareti.

### 5. MLP probe bu okumada ana referans olamadı

Nonlinear `MLP` probe sonuçları:

- çoğu koşuda `neutral` sınıfına çökme eğilimi gösterdi
- `macro F1` tarafında linear probe'dan zayıf kaldı

Bu yüzden bu readout için güvenilir ana araç:

- `linear probe`

Yorum:

- burada "bilgi var ama yalnızca nonlinear model kazıyabiliyor" tipi bir durum yok
- tersine, en değerli sinyal lineer ayrışabilirlikte görünüyor

---

## Decision

Bu oturum sonrası alınan yorum kararı:

- Stage 1A latent'i `intact` içi directional pressure semantiğini taşıyor
- bu semantik özellikle `ce_supcon_long_v1` koşusunda daha belirgin
- farkın ana kaynağı `z_long` branch'teki iyileşme

Başka deyişle:

- `directional pressure` tamamen Stage 1B'ye bırakılmış ve Stage 1A'de hiç oluşmayan bir bilgi değil
- en azından `ce_supcon_long_v1` için bu bilgi Stage 1A representation içinde okunabilir hale gelmiş

Bu nedenle sonraki tartışmalarda şu cümle savunulabilir:

- `ce_supcon_long_v1`, long branch'i sadece state supervision açısından değil, `intact` içi pressure semantiği açısından da daha semantik hale getirmiş olabilir

---

## Implementation

Bu oturumda eklenen analysis altyapısı:

- latent export'lardan `pressure_labels_v1.parquet` üretimi
- `intact` altkümesinde pressure label türetimi
- `z_short / z_long / z_fused` için linear ve MLP probe script'i

Ana script'ler:

- `scripts/generate_stage1a_pressure_labels.py`
- `scripts/probe_stage1a_pressure.py`

İlgili analysis artefact'ları:

- her checkpoint altındaki `analysis/*_latents/pressure_labels_v1.parquet`
- her probe koşusu altındaki `pressure_probe_<embedding>/summary.json`

---

## Key Learnings

- Stage 1A ana supervision'ı pressure bilgisini doğrudan öğretmese de, bu semantik latent'te oluşabiliyor
- `macro F1`, bu dengesiz sınıf yapısında accuracy'den daha anlamlı probe metriği verdi
- `ce_supcon_long_v1` ile en belirgin kazanç `z_long` tarafında görüldü
- `z_fused` tarafındaki artış küçük ama tutarlı
- current pressure label tanımı altında linear readout, nonlinear readout'tan daha güvenilir sinyal verdi

---

## Next Steps

- ✓ `intact` pressure analysis label üretimini analysis artefact olarak ekle
- ✓ `ce_supcon_long_v1` ve `ce_only` için pressure probe çalıştır
- ✓ `ce_supcon_long_symbol_v2` için aynı pressure probe hattını çalıştır
- `ce_supcon_long_v1` linear confusion matrix'i sınıf bazlı daha detaylı oku
- pressure tanımını `v2` ile refine etmeden önce mevcut `v1` label dağılımını ve zor örneklerini incele
- gerekirse `ce_supcon_long_symbol_v1` koşusunu da aynı probe ile karşılaştır

---

## Addendum — `ce_supcon_long_symbol_v2` Readout

Bu koşu daha önce genel latent kalite açısından başarısız olarak işaretlenmişti. Pressure probe bunu daha parçalı okuma imkanı verdi.

`ce_supcon_long_symbol_v2` sonuçları:

- `z_short` linear macro F1: `0.5476`
- `z_long` linear macro F1: `0.2278`
- `z_fused` linear macro F1: `0.5408`

Karşılaştırma:

- `z_short` tarafında bu koşu, hem `ce_only` hem `ce_supcon_long_v1` üstünde
- `z_fused` tarafında `ce_supcon_long_v1` ile neredeyse aynı
- `z_long` tarafında ise belirgin biçimde zayıf ve `ce_only`e yakın

Bu yüzden bu koşu için daha doğru yorum:

- failure homojen değil
- short ve fused temsiller pressure semantiğini taşımaya devam ediyor
- asıl kırılma long branch'te

Başka deyişle:

- `ce_supcon_long_symbol_v2`, pressure semantiği açısından tamamen çökmüş bir temsil üretmiyor
- ancak long branch'i pressure yönünde semantik hale getiremiyor
- bu da genel başarısızlık okumasının büyük kısmının long taraflı veya fusion'ın başka eksenlerde bozulmasıyla ilişkili olabileceğini düşündürüyor

---

**Status**: Complete ✓
